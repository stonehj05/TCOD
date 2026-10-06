"""Teacher gating and scoring inside the trainer (workflow_args.defer_teacher: true).

With `defer_teacher`, OPD_gated_alfworld_workflow_agree_lookahead leaves every teacher call
to the trainer: the explorer only plays the game and attaches the gate's inputs to each turn
(`info["opd_deferred"]`). An explore step produces several hundred turns and the trainer
samples `train_batch_size` of them, so asking the teacher here does the agreement prompt,
the progress prompt and the logprob scoring for the sampled turns only.

`DeferredTeacherGate.annotate` does for a sampled batch what the workflow's passes 2 and 3 do
for an episode, with the workflow module's own prompt builders, parser and weight function:
  1. agreement question for every turn;
  2. progress question for the turns that have a forward window (in disagree_required mode
     only where the teacher disagreed); the final `window_size` turns use the outcome;
  3. weight per turn;
  4. teacher logprobs of the student's tokens, blended toward the student's own by the weight
     and stored as `teacher_logprobs`, exactly as the workflow stores them.

Vanilla OPD (OPD_workflow_fullmemory.py with defer_teacher) has no gate: its turns carry
gate_mode "always", get no question, and only step 4 runs, storing the teacher's logprobs as
they are.

The teacher engines are the explorer's auxiliary models (created and placed as before);
they are reached by their Ray actor names. Settings, all in the taskset's workflow_args:
  trainer_teacher_parallel_prompts  requests in flight per teacher engine (default 32)
  skip_zero_weight_scoring          do not score turns whose weight is 0 (default false).
                                    Their blended teacher logprobs equal the student's own
                                    whatever the teacher says, so training is unchanged;
                                    only the raw-KL diagnostic then covers fewer turns.
"""

import asyncio
import time
from typing import Dict, List, Optional

import ray
import torch

from trinity.common.config import Config
from trinity.common.experience import Experience
from trinity.common.workflows.envs.TCOD.alfworld.OPD_gated_workflow_agree_lookahead import (
    agree_messages,
    gate_weight,
    parse_yes_no,
    payload_memory,
    progress_messages,
)
from trinity.utils.log import get_logger

PAYLOAD_KEY = "opd_deferred"
DEFAULT_PARALLEL_PER_TEACHER = 32
CALL_ATTEMPTS = 3


class DeferredTeacherGate:
    def __init__(self, config: Config, teachers: Optional[List] = None):
        self.logger = get_logger(__name__, in_ray_actor=True)
        # Config validation moves `explorer_input.taskset` into `tasksets`.
        explorer_input = config.buffer.explorer_input
        tasksets = ([explorer_input.taskset] if explorer_input.taskset else []) + list(explorer_input.tasksets or [])
        args = next((t.workflow_args for t in tasksets if (t.workflow_args or {}).get("defer_teacher")), {})
        self.enabled = bool(args.get("defer_teacher", False))
        self.parallel = max(1, int(args.get("trainer_teacher_parallel_prompts", DEFAULT_PARALLEL_PER_TEACHER)))
        self.skip_zero_weight_scoring = bool(args.get("skip_zero_weight_scoring", False))
        aux = config.explorer.auxiliary_models
        self.teacher_names = [
            f"{config.explorer.name}_auxiliary_model_0_{j}" for j in range(aux[0].engine_num)
        ] if aux else []
        self.namespace = config.ray_namespace
        self.teachers = teachers  # Ray actor handles; looked up on first use
        self._limits: List[asyncio.Semaphore] = []
        self._next = 0

    def _connect(self) -> None:
        if self.teachers is None:
            if not self.teacher_names:
                raise RuntimeError("defer_teacher needs a teacher in explorer.auxiliary_models")
            self.teachers = [ray.get_actor(n, namespace=self.namespace) for n in self.teacher_names]
        if not self._limits:
            self._limits = [asyncio.Semaphore(self.parallel) for _ in self.teachers]

    async def _call(self, method: str, *args, **kwargs):
        """One teacher request, round-robin over the engines, bounded per engine."""
        i = self._next % len(self.teachers)
        self._next += 1
        async with self._limits[i]:
            for attempt in range(CALL_ATTEMPTS):
                try:
                    return await getattr(self.teachers[i], method).remote(*args, **kwargs)
                except Exception as e:  # engine hiccup: retry, then let the trainer stop loudly
                    if attempt == CALL_ATTEMPTS - 1:
                        raise
                    self.logger.warning(f"teacher {method} failed ({e!r}); retrying")
                    await asyncio.sleep(2.0)

    async def _ask(self, messages: List[Dict[str, str]], temperature: float, max_tokens: int):
        responses = await self._call(
            "chat", messages, lora_request=None, temperature=temperature, max_tokens=max_tokens, n=1
        )
        text = responses[0].response_text or ""
        return parse_yes_no(text), text

    async def annotate(self, exps: List[Experience]) -> Dict[str, float]:
        """Gate and score the deferred turns of a sampled batch in place; returns metrics."""
        turns = [e for e in exps if e.info and PAYLOAD_KEY in e.info]
        if not turns:
            return {}
        self._connect()
        t0 = time.time()
        payloads = [e.info[PAYLOAD_KEY] for e in turns]
        n = len(turns)
        # gate_mode "always" (vanilla OPD, OPD_workflow_fullmemory.py): no question, weight 1.
        gated = [i for i in range(n) if payloads[i]["gate"]["gate_mode"] != "always"]
        memories = {i: payload_memory(payloads[i]) for i in gated}

        def ask_agree(i):
            g = payloads[i]["gate"]
            return self._ask(
                agree_messages(memories[i], payloads[i]["step"], payloads[i]["action"]),
                g["consistency_temperature"], g["consistency_max_tokens"],
            )

        def ask_progress(i):
            g, t = payloads[i]["gate"], payloads[i]["step"]
            return self._ask(
                progress_messages(memories[i], t, t + g["window_size"] - 1),
                g["progress_temperature"], g["progress_max_tokens"],
            )

        windowed = [i for i in gated if payloads[i]["has_window"]]
        sum_mode = [i for i in windowed if payloads[i]["gate"]["gate_mode"] == "sum"]
        # "sum" needs both answers for every turn: ask them together with the agreement prompts.
        agree_list, early = await asyncio.gather(
            asyncio.gather(*[ask_agree(i) for i in gated]),
            asyncio.gather(*[ask_progress(i) for i in sum_mode]),
        )
        agree_answers = dict(zip(gated, agree_list))
        progress_answers = dict(zip(sum_mode, early))
        # disagree_required: progress only matters where the teacher disagreed.
        late = [
            i for i in windowed
            if payloads[i]["gate"]["gate_mode"] != "sum" and agree_answers[i][0] is not True
        ]
        progress_answers.update(zip(late, await asyncio.gather(*[ask_progress(i) for i in late])))

        weights, n_disagree, n_agree_bad, n_prog_bad, n_not_progress, n_prog_known = [], 0, 0, 0, 0, 0
        for i, exp in enumerate(turns):
            if i not in agree_answers:
                weights.append(1.0)
                continue
            agrees = agree_answers[i][0]
            n_agree_bad += agrees is None
            disagree = agrees is not True  # "No" or unparseable (fail-safe)
            not_progress = None
            if i in progress_answers:
                says_progress = progress_answers[i][0]
                n_prog_bad += says_progress is None
                not_progress = says_progress is not True
            elif not payloads[i]["has_window"]:
                not_progress = not bool(payloads[i]["final_reward"])
            weight = gate_weight(payloads[i]["gate"], disagree, not_progress)
            weights.append(weight)
            n_disagree += disagree
            if exp.metrics is None:
                exp.metrics = {}
            exp.metrics["opd_gate_disagree"] = float(disagree)
            exp.metrics["opd_gate_weight"] = weight
            if not_progress is not None:
                n_prog_known += 1
                n_not_progress += not_progress
                exp.metrics["opd_gate_not_progress"] = float(not_progress)

        t_prompts = time.time() - t0
        score = [i for i in range(n) if weights[i] != 0.0 or not self.skip_zero_weight_scoring]
        scored = await asyncio.gather(*[
            self._call("logprobs", turns[i].tokens.tolist(), temperature=payloads[i]["gate"]["temperature"])
            for i in score
        ])
        teacher_logprobs = dict(zip(score, scored))
        raw_kl = []
        for i, exp in enumerate(turns):
            student = exp.logprobs
            if i in teacher_logprobs:
                teacher = teacher_logprobs[i][exp.prompt_length - 1:]
                assert len(teacher) == len(student), (
                    f"Length mismatch: teacher_logprobs={len(teacher)}, student_logprobs={len(student)}. "
                    f"tokens={len(exp.tokens)}, prompt_length={exp.prompt_length}"
                )
                raw_kl.append((student - teacher).sum().item())
                # ungated turns keep the teacher's logprobs untouched, as the vanilla workflow stores them
                exp.teacher_logprobs = (
                    student + weights[i] * (teacher - student) if i in agree_answers else teacher
                )
            else:  # weight 0, not scored: blending would give the student's own logprobs anyway
                exp.teacher_logprobs = student.clone()
            exp.teacher_logprobs_valid_mask = torch.full((len(student),), True, dtype=torch.bool)
            del exp.info[PAYLOAD_KEY]

        metrics = {
            "time/teacher_gate": time.time() - t0,
            "time/teacher_gate_prompts": t_prompts,  # agreement + progress questions
            "time/teacher_gate_scoring": time.time() - t0 - t_prompts,
            "teacher_gate/turns": n,
            "teacher_gate/scored_turns": len(score),
        }
        if raw_kl:
            metrics["teacher_gate/raw_kl_per_turn"] = sum(raw_kl) / len(raw_kl)
        if not gated:
            return metrics
        g_n = len(gated)
        g_w = [weights[i] for i in gated]
        full = [
            g["single_criterion_weight"] * 2 if g["gate_mode"] == "sum" else g["disagree_no_progress_weight"]
            for g in (payloads[i]["gate"] for i in gated)
        ]
        metrics.update({
            "teacher_gate/agree_prompts": g_n,
            "teacher_gate/progress_prompts": len(progress_answers),
            "teacher_gate/opd_gate_apply_rate": sum(g_w) / g_n,
            "teacher_gate/opd_gate_full_rate": sum(w == f for w, f in zip(g_w, full)) / g_n,
            "teacher_gate/opd_gate_half_rate": sum(0.0 < w < f for w, f in zip(g_w, full)) / g_n,
            "teacher_gate/opd_gate_none_rate": sum(w == 0.0 for w in g_w) / g_n,
            "teacher_gate/opd_gate_disagree_rate": n_disagree / g_n,
            "teacher_gate/consistency_parse_success_rate": 1.0 - n_agree_bad / g_n,
        })
        if n_prog_known:
            metrics["teacher_gate/opd_gate_not_progress_rate"] = n_not_progress / n_prog_known
        if progress_answers:
            metrics["teacher_gate/progress_parse_success_rate"] = 1.0 - n_prog_bad / len(progress_answers)
        return metrics
