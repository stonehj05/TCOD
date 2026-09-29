from typing import List

# --------------------- ALFWorld --------------------- #
# Fixed to match the DASH-OPD paper's (arXiv:2607.29078) Appendix H prompt
# templates verbatim -- the paper explicitly states "[t]he templates request
# reasoning before the action but do not require <thought> or <think> tags"
# and its own template box ends with "Do not output any other text besides
# your reasoning and the action." TCOD's original shipped template did the
# opposite on both counts (required a <think></think> block, had no
# anti-rambling instruction) -- a real, previously-identified discrepancy
# between the actual training codebase and the paper's documented design,
# separate from the memory-list fix. Plausibly a contributing cause of the
# response-length-blowup failure mode: an open-ended "<think>" requirement
# with no length/stopping constraint is a natural setup for a model to
# never converge to an action. See alfworld_ts_probe/alfworld_agent_utils.py
# for the standalone-probe-pipeline copy this was already fixed in (built
# before this training-code copy was updated to match).
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

def parse_action(response):
    # Guarded against a missing <action> tag (e.g. the model narrates in
    # plain prose and never emits the tag) -- the previous unconditional
    # response.split("<action>")[1] raised IndexError in that case, caught
    # by the broad except below and logged as noise on every occurrence.
    # rsplit(..., 1) takes the LAST <action> tag if the model's own
    # reasoning happens to mention the literal string "<action>" earlier
    # in its response. Matches the equivalent fix already applied in
    # alfworld_ts_probe/alfworld_agent_utils.py's parse_action.
    try:
        if "<action>" in response:
            return response.rsplit("<action>", 1)[-1].split("</action>")[0].strip()
        return ""
    except Exception as e:
        print(f"Error parsing action: {e}, response = {response}")
        return ""

def format_observation(observation: str):
    return observation.strip()


HISTORY_LENGTH = 2
MEMORY_FORMAT = "[Observation {step_num}: '{obs}', Action {step_num}: '{act}']"


def _extract_task(text_obs: str) -> str:
    """Extract the task description from the text observation."""
    task_start = text_obs.find("Your task is to: ")
    if task_start != -1:
        return text_obs[task_start + len("Your task is to: ") :].strip()
    raise ValueError("Task description not found in text observation.")


def _format_history(observation: str, step_num: int, act: str) -> str:
    """Format observation and action for action_history."""
    return MEMORY_FORMAT.format(step_num=step_num, obs=observation.strip(), act=act)


def _create_alfworld_env(game_file_path: str):
    """Create AlfWorld textworld environment."""
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

