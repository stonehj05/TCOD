# Modified for DASH-OPD, 2026.
# Changes to the upstream implementation are maintained in this repository.

from typing import List, Optional, Sequence, Tuple

# --------------------- ALFWorld --------------------- #
ALFWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason about the current situation.
Once you've finished your reasoning, you should choose the best admissible action for the current step and present it within <action> </action> tags.
Do not output any other text besides your reasoning and the action.
"""

ALFWORLD_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason about the current situation.
Once you've finished your reasoning, you should choose the best admissible action for the current step and present it within <action> </action> tags.
Do not output any other text besides your reasoning and the action.
"""

def parse_action(response):
    try:
        # parse the action within the <action> </action> tag
        action = response.split("<action>")[1].split("</action>")[0].strip()
        return action
    except Exception as e:
        print(f"Error parsing action: {e}, response = {response}")
        return ""

def format_observation(observation: str):
    return observation.strip()


HISTORY_LENGTH = 2
INITIAL_NO_HISTORY_TURNS = 2
MEMORY_FORMAT = "[Observation {step_num}: '{obs}', Action {step_num}: '{act}']"
MEMORY_FORMAT_WITH_THINK = (
    "[Observation {step_num}: '{obs}', Think {step_num}: '{think}', "
    "Action {step_num}: '{act}']"
)


def parse_explorer_schedule_step(batch_id) -> Tuple[int, bool]:
    """Map Explorer's 1-based batch id to a zero-based curriculum step.

    Returns ``(schedule_step, parse_failed)``. Evaluation batch ids such as
    ``"5/task_name"`` are accepted so every ALFWorld curriculum uses the same
    step convention.
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


def build_alfworld_prompt(
    *,
    task_description: str,
    step_count: int,
    history: Sequence[str],
    history_length: int,
    initial_no_history_turns: int = INITIAL_NO_HISTORY_TURNS,
    current_observation: str,
    admissible_actions: str,
    use_initial_no_history_template: bool = False,
) -> str:
    """Build the self-contained ALFWorld prompt used in training and evaluation.

    ``history_length=-1`` includes all available structured interaction history.
    Non-negative values include at most that many recent entries. When full chat
    history is retained, ``use_initial_no_history_template`` enables the TCOD
    warm-up behavior: the first ``initial_no_history_turns`` turns use the
    observation-only template because the first observation already contains
    the task.
    """
    if history_length < -1:
        raise ValueError(
            f"history_length must be -1 or non-negative, got {history_length}"
        )
    if isinstance(initial_no_history_turns, bool) or not isinstance(
        initial_no_history_turns, int
    ):
        raise TypeError(
            "initial_no_history_turns must be an integer, got "
            f"{type(initial_no_history_turns).__name__}"
        )
    if initial_no_history_turns < 0:
        raise ValueError(
            "initial_no_history_turns must be non-negative, got "
            f"{initial_no_history_turns}"
        )

    if (
        use_initial_no_history_template
        and step_count < initial_no_history_turns
    ):
        return ALFWORLD_TEMPLATE_NO_HIS.format(
            current_observation=current_observation,
            admissible_actions=admissible_actions,
        )

    if history_length == -1:
        recent_history = list(history)
    elif history_length == 0:
        recent_history = []
    else:
        recent_history = list(history[-history_length:])
    return ALFWORLD_TEMPLATE.format(
        task_description=task_description,
        step_count=step_count,
        history_length=len(recent_history),
        action_history="\n".join(recent_history),
        current_step=step_count + 1,
        current_observation=current_observation,
        admissible_actions=admissible_actions,
    )


def _extract_task(text_obs: str) -> str:
    """Extract the task description from the text observation."""
    task_start = text_obs.find("Your task is to: ")
    if task_start != -1:
        return text_obs[task_start + len("Your task is to: ") :].strip()
    raise ValueError("Task description not found in text observation.")


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
    """Format one structured history entry, optionally including model reasoning."""
    if think:
        return MEMORY_FORMAT_WITH_THINK.format(
            step_num=step_num,
            obs=observation.strip(),
            think=think.strip(),
            act=act,
        )
    return MEMORY_FORMAT.format(step_num=step_num, obs=observation.strip(), act=act)


def _create_alfworld_env(game_file_path: str):
    """Create AlfWorld textworld environment."""
    try:
        import textworld
        import textworld.gym
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler

        request_infos = textworld.EnvInfos(
            description=True, inventory=True, admissible_commands=True
        )
        # These workflows use dataset-provided actions for B2F checkpoints and
        # never consume AlfredExpert's extra.expert_plan. Mounting that wrapper
        # would nevertheless run its hand-coded policy after every environment
        # step, which can raise and abort an otherwise valid model rollout.
        env_id = textworld.gym.register_game(
            game_file_path, request_infos, wrappers=[AlfredDemangler()]
        )
        return textworld.gym.make(env_id)
    except Exception as e:
        raise ImportError(
            f"Error creating AlfWorld env: {e}. "
            "Ensure alfworld is installed: https://github.com/alfworld/alfworld"
        ) from e


def _create_alfworld_env_with_checkpoint(
    game_file_path: str, actions: List[str], checkpoint_step: int
):
    """Create AlfWorld environment and execute actions up to checkpoint_step.

    Args:
        game_file_path: Path to the game file
        actions: List of predefined expert actions
        checkpoint_step: Number of actions to execute before letting model take over

    Returns:
        tuple: (
            env,
            observation,
            info,
            history,
            task_description,
            current_step,
            done,
        )
    """
    env = _create_alfworld_env(game_file_path)
    observation, info = env.reset()

    task_description = _extract_task(observation)
    history: List[str] = []

    # Execute predefined actions up to checkpoint_step
    done = False
    for step in range(min(checkpoint_step, len(actions))):
        action = actions[step]
        format_obs = format_observation(observation)
        history.append(_format_history(format_obs, step + 1, action))
        observation, reward, done, info = env.step(action)

        if done:
            # If task completes before checkpoint, return the final state
            break

    return env, observation, info, history, task_description, len(history), done
