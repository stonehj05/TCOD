"""Single-episode ALFWorld rollout loop matching TCOD's actual trained-on
prompt/context format -- companion to run_episode.py (which deliberately
matches the DASH-OPD paper's protocol instead, for the other probe scripts).

Key difference from run_episode.py: builds a fresh, single-turn `messages`
list each step (`[{"role": "user", "content": user_content}]`) instead of
accumulating a growing `memory` list across turns. This mirrors the fix
applied to TCOD's OPD_gated_workflow.py this session -- the model was
trained seeing ONLY the current turn's self-contained prompt (which already
embeds the capped HISTORY_LENGTH=2 textual summary), never the raw
conversation history. Evaluating with an accumulating `memory` (as
run_episode.py does, by design, for its own paper-fidelity purposes) would
score this checkpoint on an out-of-distribution context shape.

Also uses tcod_alfworld_utils's `<` (not `<=`) history-length threshold, and
its `<think></think>`-instructing templates -- both are exact matches to
what the checkpoint actually saw during training (OPD_gated_workflow.py /
OPD_workflow.py), not the DASH-OPD paper's quoted variant.
"""

from typing import Dict, List, Optional

from tcod_alfworld_utils import (
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

        # TCOD's own threshold: `< HISTORY_LENGTH`, not `<=` -- the
        # with-history template kicks in starting turn 3 (r=2), not turn 4.
        if len(history) < HISTORY_LENGTH:
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

        # Single self-contained turn -- no growing `memory` list. See module
        # docstring / OPD_gated_workflow.py's fix for why.
        messages = [{"role": "user", "content": user_content}]
        response_text, raw = model_client.chat(
            messages, available_actions={"commands": admissible_commands}
        )
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
