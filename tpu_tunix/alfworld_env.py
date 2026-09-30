"""ALFWorld episodes for the Tunix OPD port.

Prompt construction mirrors OPD_alfworld_workflow
(trinity/common/workflows/envs/TCOD/alfworld/OPD_workflow.py): every turn is a
single self-contained user message holding the task, a capped 2-step history,
the current observation and the admissible actions. Templates and env helpers
are loaded from the original utils.py so both code paths stay in sync.
"""

import importlib.util
import os
from typing import List, Optional

_UTILS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "trinity", "common", "workflows", "envs", "TCOD", "alfworld", "utils.py",
)
# Loaded by path: importing it through the `trinity` package would pull in
# torch/verl, which are not installed in the TPU environment.
_spec = importlib.util.spec_from_file_location("tcod_alfworld_utils", _UTILS_PATH)
U = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(U)


class AlfworldEpisode:
    """One ALFWorld game, stepped turn by turn by the batched rollout loop.

    `expert_actions`/`start_step` replay a prefix of expert actions before the
    student takes over (the TCOD-b2f checkpoint mechanism); `max_turns` caps
    how many student turns are played (the TCOD-f2b distill window). Plain OPD
    uses neither.
    """

    def __init__(
        self,
        game_file: str,
        max_env_steps: int = 30,
        expert_actions: Optional[List[str]] = None,
        start_step: int = 0,
        max_turns: Optional[int] = None,
    ):
        self.game_file = game_file
        self.max_env_steps = max_env_steps
        self.done = False
        self.reward = 0.0
        self.truncated = False  # prompt exceeded the rollout prompt budget
        self.turns = 0
        if start_step > 0 and expert_actions:
            (self.env, self.observation, self.info, self.history, self.task_description,
             self.step_idx, replay_done) = U._create_alfworld_env_with_checkpoint(
                game_file, expert_actions, start_step)
            if replay_done:
                self.done, self.reward = True, 1.0
        else:
            self.env = U._create_alfworld_env(game_file)
            self.observation, self.info = self.env.reset()
            self.task_description = U._extract_task(self.observation)
            self.history: List[str] = []
            self.step_idx = 0
        remaining = max_env_steps - self.step_idx
        self.turn_budget = remaining if max_turns is None else min(max_turns, remaining)
        if self.turn_budget <= 0:
            self.done = True

    @property
    def active(self) -> bool:
        return not self.done and self.turns < self.turn_budget

    def user_content(self) -> str:
        format_obs = U.format_observation(self.observation)
        admissible = self.info.get("admissible_commands", [])
        if admissible and isinstance(admissible[0], list):
            admissible = admissible[0]
        admissible_str = "\n ".join(f"'{s}'" for s in admissible if s != "help")
        r = self.step_idx
        if len(self.history) < U.HISTORY_LENGTH:
            return U.ALFWORLD_TEMPLATE_NO_HIS.format(
                current_observation=format_obs, admissible_actions=admissible_str)
        return U.ALFWORLD_TEMPLATE.format(
            task_description=self.task_description,
            step_count=r,
            history_length=min(U.HISTORY_LENGTH, len(self.history)),
            action_history="\n".join(self.history[-U.HISTORY_LENGTH:]),
            current_step=r + 1,
            current_observation=format_obs,
            admissible_actions=admissible_str,
        )

    def messages(self) -> List[dict]:
        return [{"role": "user", "content": self.user_content()}]

    def step(self, response_text: str) -> None:
        format_obs = U.format_observation(self.observation)
        action = U.parse_action(response_text)
        self.history.append(U._format_history(format_obs, self.step_idx + 1, action))
        self.observation, _, done, self.info = self.env.step(action)
        self.step_idx += 1
        self.turns += 1
        if done:
            self.done, self.reward = True, 1.0
        elif self.step_idx >= self.max_env_steps:
            self.done = True

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:  # textworld may already have torn the env down
            pass
