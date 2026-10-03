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


def run_episode(
    env,
    model_client,
    task_description: str,
    observation: str,
    info: dict,
    history: Optional[List[str]] = None,
    start_step: int = 0,
    max_steps: int = 30,
) -> Dict:
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
        if len(history) <= HISTORY_LENGTH:
            user_content = ALFWORLD_TEMPLATE_NO_HIS.format(
                current_observation=formatted_observation,
                admissible_actions=formatted_actions,
            )
        else:
            action_history_str = "\n".join(history[-HISTORY_LENGTH:])
            user_content = ALFWORLD_TEMPLATE.format(
                task_description=task_description,
                step_count=r,
                history_length=min(HISTORY_LENGTH, len(history)),
                action_history=action_history_str,
                current_step=r + 1,
                current_observation=formatted_observation,
                admissible_actions=formatted_actions,
            )

        memory = memory + [{"role": "user", "content": user_content}]
        response_text, raw = model_client.chat(
            memory, available_actions={"commands": admissible_commands}
        )
        memory.append({"role": "assistant", "content": response_text})
        finish_reason = None
        if raw and raw.get("choices"):
            finish_reason = raw["choices"][0].get("finish_reason")

        action = parse_action(response_text)
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
