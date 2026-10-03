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
# Copied verbatim from TCOD/trinity/common/workflows/envs/TCOD/alfworld/
# utils.py, kept in sync with that file's fixes this session: the memory-
# list fix + defensive parse_action, AND (most recently) the prompt-
# template fix that brought TCOD's own template in line with the DASH-OPD
# paper's (arXiv:2607.29078) Appendix H wording -- removed the "<think>
# </think>" requirement (the paper explicitly states its templates "do not
# require <thought> or <think> tags") and added the paper's "Do not output
# any other text besides your reasoning and the action." constraint, absent
# from TCOD's original template.
#
# One remaining, deliberate difference from alfworld_agent_utils.py (used by
# the other probe scripts in this directory): the history-length threshold.
# TCOD's actual code (and this file) switches to the with-history template
# once len(history) is NOT < HISTORY_LENGTH (i.e. starting turn 3); the
# paper's own Appendix H text says "from the fourth turn onward" (turn 4),
# which alfworld_agent_utils.py implements via `<=` instead of `<`. This is
# a pre-existing, one-turn discrepancy in TCOD's own shipped code, not
# something introduced by either probe pipeline -- kept faithful to the
# actual training code here since this file's purpose is evaluating TCOD-
# trained checkpoints under the exact conditions they were trained on.
from typing import List


# --------------------- ALFWorld (TCOD's exact template) --------------------- #
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


def parse_action(response):
    # Same defensive guard applied to TCOD's own utils.py this session --
    # only treats it as an action if "<action>" is actually present, rather
    # than an unconditional response.split("<action>")[1] that raises
    # IndexError (and spams stderr) when the model narrates in plain prose
    # without emitting the tag.
    try:
        if "<action>" in response:
            return response.rsplit("<action>", 1)[-1].split("</action>")[0].strip()
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
    """Create AlfWorld textworld environment for one specific game file."""
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
