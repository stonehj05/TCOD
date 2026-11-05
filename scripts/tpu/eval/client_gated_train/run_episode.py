"""Shared single-episode rollout loop for ALFWorld, used by both pipeline
stages -- mirrors webshop_ts_probe/run_episode.py's structure, adapted for
ALFWorld's env API.

Key differences from the WebShop version:
- No available_actions/validate_action gating: TCOD's own ALFWorld code
  always calls env.step(action) with whatever got parsed, no pre-validation
  or error-message feedback loop (unlike WebShop). Replicated faithfully
  here -- every turn is "applied", so there's no env_rounds/loop_iterations
  split the way WebShop needed one. `action_admissible` is still recorded per
  step as a diagnostic (was the parsed action literally one of the game's
  admissible_commands), but never gates stepping.
- Success is `done` (ALFWorld/textworld signals a win via done=True; TCOD's
  own OPD_workflow.py sets final_reward=1.0 exactly when done, 0.0 otherwise
  -- replicated here), not a reward==1.0 threshold on a continuous score like
  WebShop.
- Stage 2 continuation carries `info` (admissible_commands for the *next*
  turn) forward in addition to observation/history, since ALFWorld's action
  space is read from `info`, not re-derived from the observation HTML.
"""

from typing import Dict, List, Optional

from alfworld_agent_utils import (
    HISTORY_LENGTH,
    ALFWORLD_TEMPLATE,
    ALFWORLD_TEMPLATE_NO_HIS,
    _format_admissible_commands,
    _format_history,
    format_observation,
    parse_action,
)

# template="release_think": the FutureBridge-OPD release's prompts (TCOD's original, with the
# mandatory <think> tags), its parse_action, and its switch to the with-history template after
# HISTORY_LENGTH turns (not HISTORY_LENGTH + 1) -- what futurebridge_workflow.py trains on.
# template="tcod_train": the default prompts, but switching to the with-history template after
# HISTORY_LENGTH turns, as TCOD's own training/eval workflows (TCOD_b2f_workflow.py) do.
# template="uopd": UOPD's own rollout (trinity/common/workflows/envs/UOPD/alfworld/OPD_workflow.py,
# copied unchanged from github.com/onepounchman/UOPD): prompts, admissible-action list, history
# lines and parse_action from its utils.py; one user message per turn holding the last HISTORY_LENGTH (2) steps,
# as the release (no full conversation, whatever --full-memory says); the no-history template on the first turn
# only; success = the game's `won` flag (the game is created with UOPD's _create_alfworld_env, see 12_evaluate_checkpoint.py).
# "dash": DASH-OPD's own eval rollout (github.com/Lucian1115/DASH-OPD, its TCOD_alfworld_eval_workflow with the
# settings of configurations/alfworld/dash_opd_bench.yaml): build_alfworld_prompt, parse_action, _format_history and
# format_observation from its utils.py (trinity/common/workflows/envs/DASH/alfworld/utils.py, copied unchanged);
# full chat history; history_length 2; the no-history template for the first 3 turns; success = the episode ends.
# "gated_train": exactly the student rollout our gated workflow trains on
# (OPD_gated_workflow_teacher_lookahead._rollout / build_user_content): templates, parse_action and
# HISTORY_LENGTH from trinity/common/workflows/envs/TCOD/alfworld/utils.py, the with-history template from
# turn 3 (len(history) >= HISTORY_LENGTH), full memory, and the training's prompt limit: a chat whose prompt is
# longer than GATED_PROMPT_LIMITS[template] (model.max_prompt_tokens, 10240) gets no generation, its "response"
# being the one token at that position (trinity/common/models/model.py _handle_prompt_truncation).
# "gated_train_p40960": the same with a 40960-token prompt limit (user, 2026-11-02), the default eval's context.
TEMPLATES = ("paper", "release_think", "tcod_train", "uopd", "dash", "gated_train", "gated_train_p40960")
DASH_HISTORY_LENGTH, DASH_INITIAL_NO_HISTORY_TURNS = 2, 3  # dash_opd_bench.yaml
GATED_PROMPT_LIMITS = {
    "gated_train": 10240,  # opd_gated_qwen3_1.7b_8b_t07_s03_deferred_8chip_tpu.yaml model.max_prompt_tokens
    "gated_train_p40960": 40960,
}
GATED_TOKENIZER = "/dev/shm/models/Qwen3-1.7B"  # the student's tokenizer and chat template, as in training
_gated = {}


def gated_train_utils():
    if "utils" not in _gated:
        import importlib.util
        import os

        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../../../trinity/common/workflows/envs/TCOD/alfworld/utils.py",
        )
        spec = importlib.util.spec_from_file_location("tcod_train_alfworld_utils", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _gated["utils"] = module
    return _gated["utils"]


def _gated_prompt_ids(memory):
    """Token ids of the prompt as Trinity builds it (apply_chat_template, enable_thinking=False, then tokenize)."""
    if "tok" not in _gated:
        from transformers import AutoTokenizer

        _gated["tok"] = AutoTokenizer.from_pretrained(GATED_TOKENIZER)
    tok = _gated["tok"]
    prompt = tok.apply_chat_template(memory, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok, tok(prompt, truncation=False)["input_ids"]


def dash_utils():
    import importlib.util
    import os

    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../../../trinity/common/workflows/envs/DASH/alfworld/utils.py",
    )
    spec = importlib.util.spec_from_file_location("dash_alfworld_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_episode_dash(env, model_client, task_description, observation, info, max_steps):
    """DASH-OPD's TCODAlfworldEvalWorkflow._run_episode loop, student only."""
    d = dash_utils()
    history: List[str] = []
    memory: List[Dict[str, str]] = []
    step_log: List[Dict] = []
    actions: List[str] = []
    done, final_reward, rounds = False, 0.0, max_steps
    for step in range(max_steps):
        formatted_observation = d.format_observation(observation)
        admissible_commands = info.get("admissible_commands", [])
        if admissible_commands and isinstance(admissible_commands[0], list):
            admissible_commands = admissible_commands[0]
        formatted_actions = "\n ".join(f"'{a}'" for a in admissible_commands if a != "help")
        user_content = d.build_alfworld_prompt(
            task_description=task_description,
            step_count=step,
            history=history,
            history_length=DASH_HISTORY_LENGTH,
            initial_no_history_turns=DASH_INITIAL_NO_HISTORY_TURNS,
            current_observation=formatted_observation,
            admissible_actions=formatted_actions,
            use_initial_no_history_template=True,  # retain_full_chat_history: true
        )
        memory = memory + [{"role": "user", "content": user_content}]
        response_text, raw = model_client.chat(memory, available_actions={"commands": admissible_commands})
        response_text = response_text or ""
        memory.append({"role": "assistant", "content": response_text})
        finish_reason = None
        if raw and raw.get("choices"):
            finish_reason = raw["choices"][0].get("finish_reason")
        action = d.parse_action(response_text)
        history.append(d._format_history(formatted_observation, step + 1, action, think=None))
        actions.append(action)
        observation, reward, done, info = env.step(action)
        step_log.append({"step": step, "prompt": user_content, "response": response_text,
                         "finish_reason": finish_reason, "parsed_action": action,
                         "action_admissible": action in admissible_commands,
                         "reward": reward, "done": done})
        if done:
            rounds, final_reward = step + 1, 1.0
            break
    return {"actions": actions, "env_rounds": rounds, "final_reward": final_reward, "done": done,
            "step_log": step_log, "final_observation": observation, "final_info": info}


def uopd_utils():
    import importlib.util
    import os

    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../../../trinity/common/workflows/envs/UOPD/alfworld/utils.py",
    )
    spec = importlib.util.spec_from_file_location("uopd_alfworld_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_episode_uopd(env, model_client, task_description, observation, info, max_steps):
    """UOPD's OPD_workflow._run_episode loop (one message per turn), student only (as its eval workflow)."""
    u = uopd_utils()
    history: List[str] = []
    step_log: List[Dict] = []
    actions: List[str] = []
    won, final_reward, rounds = False, 0.0, max_steps
    for r in range(max_steps):
        format_obs = u.format_observation(observation)
        admissible_commands = info.get("admissible_commands", [])
        if admissible_commands and isinstance(admissible_commands[0], list):
            admissible_commands = admissible_commands[0]
        reformatted_admissible = "\n ".join(f"'{s}'" for s in admissible_commands if s != "help")
        if len(history) == 0:
            user_content = u.ALFWORLD_TEMPLATE_NO_HIS.format(
                current_observation=format_obs, admissible_actions=reformatted_admissible)
        else:
            user_content = u.ALFWORLD_TEMPLATE.format(
                task_description=task_description,
                step_count=r,
                history_length=min(u.HISTORY_LENGTH, len(history)),
                action_history="\n".join(history[-u.HISTORY_LENGTH:]),
                current_step=r + 1,
                current_observation=format_obs,
                admissible_actions=reformatted_admissible,
            )
        memory = [{"role": "user", "content": user_content}]
        response_text, raw = model_client.chat(memory, available_actions={"commands": admissible_commands})
        finish_reason = None
        if raw and raw.get("choices"):
            finish_reason = raw["choices"][0].get("finish_reason")
        action = u.parse_action(response_text)
        history.append(u._format_history(format_obs, r + 1, action))
        actions.append(action)
        observation, reward, done, info = env.step(action)
        step_log.append({"step": r, "prompt": user_content, "response": response_text,
                         "finish_reason": finish_reason, "parsed_action": action,
                         "action_admissible": action in admissible_commands,
                         "reward": reward, "done": done})
        if done:
            won = bool(info.get("won", False))
            rounds, final_reward = r + 1, 1.0 if won else 0.0
            break
    return {"actions": actions, "env_rounds": rounds, "final_reward": final_reward, "done": won,
            "step_log": step_log, "final_observation": observation, "final_info": info}


def _release_utils():
    import importlib.util
    import os

    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../../../trinity/common/workflows/envs/TCOD/alfworld/ftb_release_utils.py",
    )
    spec = importlib.util.spec_from_file_location("ftb_release_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_episode(
    env,
    model_client,
    task_description: str,
    observation: str,
    info: dict,
    history: Optional[List[str]] = None,
    start_step: int = 0,
    max_steps: int = 30,
    template: str = "paper",
) -> Dict:
    if template == "uopd":
        return _run_episode_uopd(env, model_client, task_description, observation, info, max_steps)
    if template == "dash":
        return _run_episode_dash(env, model_client, task_description, observation, info, max_steps)
    if template == "release_think":
        release = _release_utils()
        template_no_his, template_his = release.ALFWORLD_TEMPLATE_NO_HIS, release.ALFWORLD_TEMPLATE
        parse, no_his_turns = release.parse_action, HISTORY_LENGTH
    elif template in GATED_PROMPT_LIMITS:
        g = gated_train_utils()
        template_no_his, template_his = g.ALFWORLD_TEMPLATE_NO_HIS, g.ALFWORLD_TEMPLATE
        parse, no_his_turns = g.parse_action, g.HISTORY_LENGTH
    else:
        template_no_his, template_his = ALFWORLD_TEMPLATE_NO_HIS, ALFWORLD_TEMPLATE
        parse = parse_action
        no_his_turns = HISTORY_LENGTH if template == "tcod_train" else HISTORY_LENGTH + 1
    history = list(history) if history else []
    step_log: List[Dict] = []
    actions: List[str] = []
    memory: List[Dict[str, str]] = []

    done = False
    final_reward = 0.0
    last_step_index = start_step - 1

    for r in range(start_step, start_step + max_steps):
        last_step_index = r
        formatted_observation = format_observation(observation)
        admissible_commands = info.get("admissible_commands", [])
        if admissible_commands and isinstance(admissible_commands[0], list):
            admissible_commands = admissible_commands[0]
        formatted_actions = _format_admissible_commands(info)

        # NOTE: `<=` not `<` -- see the matching note in webshop_ts_probe's
        # run_episode.py. Switches to the with-history template starting turn
        # 4 (r=3, 0-indexed), matching the DASH-OPD paper's stated protocol.
        if len(history) < no_his_turns:
            user_content = template_no_his.format(
                current_observation=formatted_observation,
                admissible_actions=formatted_actions,
            )
        else:
            action_history_str = "\n".join(history[-HISTORY_LENGTH:])
            user_content = template_his.format(
                task_description=task_description,
                step_count=r,
                history_length=min(HISTORY_LENGTH, len(history)),
                action_history=action_history_str,
                current_step=r + 1,
                current_observation=formatted_observation,
                admissible_actions=formatted_actions,
            )

        memory = memory + [{"role": "user", "content": user_content}]
        finish_reason = None
        prompt_truncated = False
        if template in GATED_PROMPT_LIMITS:
            limit = GATED_PROMPT_LIMITS[template]
            tok, ids = _gated_prompt_ids(memory)
            prompt_truncated = len(ids) > limit
        if prompt_truncated:
            response_text, raw, finish_reason = tok.decode(ids[limit]), None, "prompt_truncated"
        else:
            response_text, raw = model_client.chat(
                memory, available_actions={"commands": admissible_commands}
            )
        memory.append({"role": "assistant", "content": response_text})
        if raw and raw.get("choices"):
            finish_reason = raw["choices"][0].get("finish_reason")

        action = parse(response_text)
        action_admissible = action in admissible_commands
        history.append(_format_history(formatted_observation, r + 1, action))
        actions.append(action)

        observation, reward, done, info = env.step(action)

        step_log.append(
            {
                "step": r,
                "prompt": user_content,
                "response": response_text,
                "finish_reason": finish_reason,
                "parsed_action": action,
                "action_admissible": action_admissible,
                "reward": reward,
                "done": done,
            }
        )

        if done:
            final_reward = 1.0
            break

    return {
        "actions": actions,
        "env_rounds": last_step_index - start_step + 1,
        "final_reward": final_reward,
        "done": done,
        "step_log": step_log,
        "final_observation": observation,
        "final_info": info,
    }
