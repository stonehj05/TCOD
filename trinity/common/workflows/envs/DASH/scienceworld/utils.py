# Modified for DASH-OPD, 2026.
# Changes to the upstream implementation are maintained in this repository.

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --------------------- ScienceWorld --------------------- #
SCIWORLD_SYSTEM_PROMPT = """
You are an expert agent operating in the ScienceWorld text environment.

At each step, You should first reason about the current situation.
Then output exactly one environment action within <action> </action> tags.
Do not talk to the user. Solve the task by interacting with the environment.
"""

SCIWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ScienceWorld text environment.
{task_description}
Your current observation is: {current_observation}
Available action commands: [{action_templates}]
Available objects you can interact with: [{objects}]

Now it's your turn to take an action. Combine an action command with appropriate object(s) to form a valid action.
You should first reason about the current situation.
Once you've finished your reasoning, you should choose the best valid action for the current step and present it within <action> </action> tags.
Do not output any other text besides your reasoning and the action.
"""

SCIWORLD_TEMPLATE = """
You are an expert agent operating in the ScienceWorld text environment.
{task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Available action commands: [{action_templates}]
Available objects you can interact with: [{objects}]

Now it's your turn to take an action. Combine an action command with appropriate object(s) to form a valid action.
You should first reason about the current situation.
Once you've finished your reasoning, you should choose the best valid action for the current step and present it within <action> </action> tags.
Do not output any other text besides your reasoning and the action.
"""


def parse_action(response: str) -> str:
    try:
        return response.split("<action>")[1].split("</action>")[0].strip()
    except Exception as e:
        print(f"Error parsing action: {e}, response = {response}")
        return ""


def format_observation(observation: str) -> str:
    return observation.strip()


HISTORY_LENGTH = 2
INITIAL_NO_HISTORY_TURNS = 2
MEMORY_FORMAT = "[Observation {step_num}: '{obs}', Action {step_num}: '{act}']"
MEMORY_FORMAT_WITH_THINK = (
    "[Observation {step_num}: '{obs}', Think {step_num}: '{think}', " "Action {step_num}: '{act}']"
)


def parse_explorer_schedule_step(batch_id) -> Tuple[int, bool]:
    """Map Explorer's 1-based batch id to a zero-based curriculum step.

    This intentionally matches the ALFWorld TCOD curriculum convention.  The
    first Explorer batch therefore uses schedule step 0, so a curriculum with
    ``checkpoint_steps=N`` changes after N complete batches rather than at the
    start of batch N.
    """
    try:
        if isinstance(batch_id, bool):
            raise ValueError("boolean batch id is not a training step")
        if isinstance(batch_id, int):
            parsed = batch_id
        elif isinstance(batch_id, str):
            parsed = int(batch_id.split("/", maxsplit=1)[0].strip())
        else:
            raise TypeError(f"unsupported batch id type: {type(batch_id).__name__}")
    except (TypeError, ValueError):
        return 0, True
    return max(parsed - 1, 0), False


def build_scienceworld_prompt(
    *,
    task_description: str,
    step_count: int,
    history: Sequence[str],
    history_length: int,
    initial_no_history_turns: int = INITIAL_NO_HISTORY_TURNS,
    current_observation: str,
    action_templates: str,
    objects: str,
    use_initial_no_history_template: bool = False,
) -> str:
    """Build one ScienceWorld prompt with an explicit history policy.

    The default full-chat policy keeps the original ScienceWorld behavior: the
    first two turns use ``SCIWORLD_TEMPLATE_NO_HIS`` and later turns include the
    two most recent structured observation/action entries.
    """
    if history_length < -1:
        raise ValueError(f"history_length must be -1 or non-negative, got {history_length}")
    if isinstance(initial_no_history_turns, bool) or not isinstance(initial_no_history_turns, int):
        raise TypeError(
            "initial_no_history_turns must be an integer, got "
            f"{type(initial_no_history_turns).__name__}"
        )
    if initial_no_history_turns < 0:
        raise ValueError(
            "initial_no_history_turns must be non-negative, got " f"{initial_no_history_turns}"
        )

    if use_initial_no_history_template and step_count < initial_no_history_turns:
        return SCIWORLD_TEMPLATE_NO_HIS.format(
            task_description=task_description,
            current_observation=current_observation,
            action_templates=action_templates,
            objects=objects,
        )

    if history_length == -1:
        recent_history = list(history)
    elif history_length == 0:
        recent_history = []
    else:
        recent_history = list(history[-history_length:])
    return SCIWORLD_TEMPLATE.format(
        task_description=task_description,
        step_count=step_count,
        history_length=len(recent_history),
        action_history="\n".join(recent_history),
        current_step=step_count + 1,
        current_observation=current_observation,
        action_templates=action_templates,
        objects=objects,
    )


def _extract_task(task_description: str) -> str:
    return task_description.strip()


def parse_think(response: str) -> str:
    """Extract complete think blocks before the first action block."""
    action_start = response.find("<action>")
    search_end = action_start if action_start >= 0 else len(response)
    thoughts = []
    search_start = 0
    while True:
        open_start = response.find("<think>", search_start, search_end)
        if open_start < 0:
            break
        content_start = open_start + len("<think>")
        content_end = response.find("</think>", content_start, search_end)
        if content_end < 0:
            break
        thought = response[content_start:content_end].strip()
        if thought:
            thoughts.append(thought)
        search_start = content_end + len("</think>")
    return "\n".join(thoughts)


def _format_history(
    observation: str,
    step_num: int,
    act: str,
    think: Optional[str] = None,
) -> str:
    if think:
        return MEMORY_FORMAT_WITH_THINK.format(
            step_num=step_num,
            obs=observation.strip(),
            think=think.strip(),
            act=act,
        )
    return MEMORY_FORMAT.format(step_num=step_num, obs=observation.strip(), act=act)


def _parse_task_config(task_desc: Any) -> Dict[str, Any]:
    if isinstance(task_desc, dict):
        return task_desc
    if isinstance(task_desc, str):
        return json.loads(task_desc)
    raise TypeError(f"Unsupported ScienceWorld task description type: {type(task_desc)}")


def _get_admissible_commands(info: Dict[str, Any]) -> List[str]:
    commands = info.get("valid", [])
    if isinstance(commands, str):
        return [commands] if commands else []
    if commands and isinstance(commands[0], list):
        commands = commands[0]
    return [cmd for cmd in commands if cmd]


def _get_compact_action_info(env) -> Tuple[List[str], List[str]]:
    """Get compact action representation: templates + objects separately.

    ScienceWorld's get_valid_action_object_combinations() returns ~1162 full
    combinations per step, which easily exceeds token limits. Instead, we use
    get_possible_actions() (~23 templates) and get_possible_objects() (~20 objects)
    which together are ~50 items — about 25x smaller.
    """
    try:
        actions = env.get_possible_actions()
    except AttributeError:
        actions = env.getPossibleActions()

    try:
        objects = env.get_possible_objects()
    except AttributeError:
        objects = env.getPossibleObjects()

    if isinstance(actions, str):
        actions = [actions] if actions else []
    if isinstance(objects, str):
        objects = [objects] if objects else []

    return actions, objects


def _create_scienceworld_env(
    task_desc: Any,
    *,
    max_env_steps: int = 30,
    generate_gold_path: bool = False,
):
    task_config = _parse_task_config(task_desc)

    try:
        from scienceworld import ScienceWorldEnv
    except Exception as e:
        raise ImportError(
            f"Error importing ScienceWorldEnv: {e}. "
            "Please make sure the scienceworld package is installed successfully: "
            "https://github.com/allenai/ScienceWorld"
        ) from e

    task_name = task_config["task_name"]
    var_num = task_config["var_num"]
    jar_path = task_config.get("jar_path", "")
    simplification_str = task_config.get("simplification_str", "easy")
    env_step_limit = max(task_config.get("env_step_limit", 100), max_env_steps + 5)

    env = ScienceWorldEnv("", jar_path, envStepLimit=env_step_limit)
    env.load(task_name, var_num, simplification_str, generateGoldPath=generate_gold_path)
    return env


def _reset_scienceworld_env(env) -> Tuple[str, Dict[str, Any], str]:
    observation, info = env.reset()
    task_description = _extract_task(env.get_task_description())
    return observation, info, task_description


def _create_scienceworld_env_with_checkpoint(
    task_desc: Any,
    actions: List[str],
    checkpoint_step: int,
    *,
    max_env_steps: int = 30,
):
    env = _create_scienceworld_env(
        task_desc,
        max_env_steps=max_env_steps,
        generate_gold_path=True,
    )
    observation, info, task_description = _reset_scienceworld_env(env)

    history: List[str] = []
    best_score = info.get("score", 0)

    done = False
    for step in range(min(checkpoint_step, len(actions))):
        action = actions[step]
        format_obs = format_observation(observation)
        history.append(_format_history(format_obs, step + 1, action))
        observation, reward, done, info = env.step(action)
        best_score = max(best_score, info.get("score", best_score + reward))
        if done:
            break

    return (
        env,
        observation,
        info,
        history,
        task_description,
        len(history),
        done,
        best_score / 100.0,
    )
