# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from TCOD's trinity/common/workflows/envs/TCOD/alfworld/utils.py
# for a standalone, Trinity/Ray-free teacher/student rollout pipeline --
# mirrors webshop_ts_probe/webshop_agent_utils.py's approach.
#
# Prompt templates below match the DASH-OPD paper's (arXiv:2607.29078)
# Appendix-quoted ALFWorld prompt verbatim, NOT TCOD's released code (which
# added a "<think></think>" wrapping instruction absent from the paper, and
# dropped the paper's "Do not output any other text besides your reasoning
# and the action." constraint) -- same mismatch pattern found and fixed for
# WebShop this session, applied here proactively from the start.

import re
from typing import List, Optional


# --------------------- ALFWorld --------------------- #
ALFWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are:
[{admissible_actions}].

Now it's your turn to take an action.
You should first reason about the current situation.
Once you've finished your reasoning, you should choose the best admissible action for the current step and present it within <action> </action> tags.
Do not output any other text besides your reasoning and the action.
"""

ALFWORLD_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are:
[{admissible_actions}].

Now it's your turn to take an action.
You should first reason about the current situation.
Once you've finished your reasoning, you should choose the best admissible action for the current step and present it within <action> </action> tags.
Do not output any other text besides your reasoning and the action.
"""

HISTORY_LENGTH = 2
MEMORY_FORMAT = "[Observation {step_num}: '{obs}', Action {step_num}: '{act}']"


def parse_action(response: str) -> str:
    # Same robustness fixes established for WebShop this session, applied
    # proactively here since it's the same underlying code pattern (and same
    # error-feedback-text-gets-quoted-back risk would apply if we ever add
    # error feedback -- we don't here, matching TCOD's un-validated stepping,
    # but a model can still reference "<action>" in its own prose while
    # reasoning about the task, e.g. restating the instructions).
    try:
        if "<action>" in response:
            return response.rsplit("<action>", 1)[-1].split("</action>")[0].strip()
        if "[action]" in response:
            return response.rsplit("[action]", 1)[-1].split("[/action]")[0].strip()
        return ""
    except Exception as e:
        print(f"Error parsing action: {e}, response = {response}")
        return ""


def format_observation(observation: str) -> str:
    return observation.strip()


def _extract_task(text_obs: str) -> str:
    """Extract the task description from the text observation."""
    task_start = text_obs.find("Your task is to: ")
    if task_start != -1:
        return text_obs[task_start + len("Your task is to: ") :].strip()
    raise ValueError("Task description not found in text observation.")


def _format_history(observation: str, step_num: int, act: str) -> str:
    return MEMORY_FORMAT.format(step_num=step_num, obs=observation.strip(), act=act)


def _format_admissible_commands(info: dict) -> str:
    admissible_commands = info.get("admissible_commands", [])
    if admissible_commands and isinstance(admissible_commands[0], list):
        admissible_commands = admissible_commands[0]
    return "\n ".join(f"'{s}'" for s in admissible_commands if s != "help")


def _create_alfworld_env(game_file_path: str):
    """Create AlfWorld textworld environment for one specific game file.

    Unlike WebShop's single shared catalog env, ALFWorld envs are inherently
    per-game -- there's no "build once, reuse across many tasks via
    env.reset(session=task_id)" optimization available here. Each task/attempt
    creates (and should env.close()) its own env.
    """
    try:
        import textworld
        import textworld.gym
        from alfworld.agents.environment.alfred_tw_env import (
            AlfredDemangler,
            AlfredExpert,
            AlfredExpertType,
        )

        expert = AlfredExpert(expert_type=AlfredExpertType.HANDCODED)
        request_infos = textworld.EnvInfos(
            description=True, inventory=True, admissible_commands=True
        )
        env_id = textworld.gym.register_game(
            game_file_path, request_infos, wrappers=[AlfredDemangler(), expert]
        )
        return textworld.gym.make(env_id)
    except Exception as e:
        raise ImportError(
            f"Error creating AlfWorld env: {e}. "
            "Ensure alfworld is installed: https://github.com/alfworld/alfworld"
        ) from e
