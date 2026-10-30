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

# Modified for DASH-OPD, 2026.
# Changes to the upstream implementation are maintained in this repository.

import re
from typing import List, Optional, Sequence, Tuple

# --------------------- WebShop --------------------- #
WEBSHOP_TEMPLATE_NO_HIS = """
You are an expert autonomous agent operating in the WebShop e-commerce environment.
Your task is to: {task_description}.
Your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason about the current situation, then think which admissible action best advances the shopping goal.
Once you've finished your reasoning, you should choose the best admissible action for the current step and present it within <action> </action> tags.
Do not output any other text besides your reasoning and the action.
"""

WEBSHOP_TEMPLATE = """
You are an expert autonomous agent operating in the WebShop e-commerce environment.
Your task is to: {task_description}.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason about the current situation, then think which admissible action best advances the shopping goal.
Once you've finished your reasoning, you should choose the best admissible action for the current step and present it within <action> </action> tags.
Do not output any other text besides your reasoning and the action.
"""

HISTORY_LENGTH = 2
INITIAL_NO_HISTORY_TURNS = 2
MEMORY_FORMAT = "[Observation {step_num}: '{obs}', Action {step_num}: '{act}']"
MEMORY_FORMAT_WITH_THINK = (
    "[Observation {step_num}: '{obs}', Think {step_num}: '{think}', " "Action {step_num}: '{act}']"
)


def parse_explorer_schedule_step(batch_id) -> Tuple[int, bool]:
    """Map Explorer's 1-based batch id to the shared zero-based curriculum step."""

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


def build_webshop_prompt(
    *,
    task_description: str,
    step_count: int,
    history: Sequence[str],
    history_length: int,
    initial_no_history_turns: int = INITIAL_NO_HISTORY_TURNS,
    current_observation: str,
    available_actions: str,
    use_initial_no_history_template: bool = False,
) -> str:
    """Build a WebShop prompt under the shared explicit history policy."""

    if history_length < -1:
        raise ValueError(f"history_length must be -1 or non-negative, got {history_length}")
    if isinstance(initial_no_history_turns, bool) or not isinstance(initial_no_history_turns, int):
        raise TypeError(
            "initial_no_history_turns must be an integer, got "
            f"{type(initial_no_history_turns).__name__}"
        )
    if initial_no_history_turns < 0:
        raise ValueError(
            f"initial_no_history_turns must be non-negative, got {initial_no_history_turns}"
        )

    if use_initial_no_history_template and step_count < initial_no_history_turns:
        return WEBSHOP_TEMPLATE_NO_HIS.format(
            task_description=task_description,
            current_observation=current_observation,
            available_actions=available_actions,
        )

    selected_history = list(history) if history_length == -1 else list(history[-history_length:])
    return WEBSHOP_TEMPLATE.format(
        task_description=task_description,
        step_count=step_count,
        history_length=len(selected_history),
        action_history="\n".join(selected_history),
        current_step=step_count + 1,
        current_observation=current_observation,
        available_actions=available_actions,
    )


def parse_action(response: str) -> str:
    try:
        return response.split("<action>")[1].split("</action>")[0].strip()
    except Exception as e:
        print(f"Error parsing action: {e}, response = {response}")
        return ""


def validate_action(action: str, available_actions: dict):
    pattern = re.compile(r"(.+)\[(.+)\]")
    match = re.match(pattern, action)
    if match is None:
        action_name = action
        action_arg = None
    else:
        action_name, action_arg = match.groups()

    if action_arg is not None:
        action_arg = action_arg.lower()

    if (
        action_name == "search"
        and action_arg is not None
        and action_arg != ""
        and available_actions.get("has_search_bar", False)
    ):
        return True, ""
    if action_name == "click" and action_arg in available_actions.get("clickables", []):
        return True, ""

    if action_name == "search":
        if action_arg == "" or action_arg is None:
            return (
                False,
                "Invalid action, please type in the query you want to search in the square brackets here.",
            )
        return (
            False,
            "Can not perform search action without search bar. Please click the Back to Search button first.",
        )

    if action_name == "click" and action_arg not in available_actions.get("clickables", []):
        return (
            False,
            "Incorrect action format, make sure you have the correct button name that is "
            f"within the current page. The buttons you can click now are: {available_actions.get('clickables', [])}",
        )

    return (
        False,
        "Invalid action. You should wrap your action with the <action> </action> tag and follow the action format.",
    )


def format_observation(observation: str) -> str:
    return observation.strip()


def _compact_text(text: str) -> str:
    return " ".join(line.strip() for line in text.splitlines() if line.strip())


def _extract_task_description(observation: str) -> str:
    """Extract the instruction text from a WebShop observation."""
    lines = [line.strip() for line in observation.splitlines()]
    try:
        instruction_idx = lines.index("Instruction:")
    except ValueError:
        return _compact_text(observation)

    task_lines = []
    for line in lines[instruction_idx + 1 :]:
        if not line:
            continue
        if line.startswith("[button]"):
            break
        task_lines.append(line)

    return " ".join(task_lines).strip() or _compact_text(observation)


def parse_think(response: str) -> str:
    """Extract optional reasoning from either common reasoning tag."""

    for start_tag, end_tag in (("<think>", "</think>"), ("<thought>", "</thought>")):
        if start_tag in response and end_tag in response:
            return response.split(start_tag, 1)[1].split(end_tag, 1)[0].strip()
    return ""


def _format_history(
    observation: str,
    step_num: int,
    action: str,
    *,
    think: Optional[str] = None,
) -> str:
    values = {
        "step_num": step_num,
        "obs": _compact_text(observation),
        "act": action,
        "think": think or "",
    }
    template = MEMORY_FORMAT_WITH_THINK if think is not None else MEMORY_FORMAT
    return template.format(**values)


def _format_available_actions(available_actions: dict) -> str:
    formatted_actions = []
    if available_actions.get("has_search_bar", False):
        formatted_actions.append("search[<query>]")
    for clickable in available_actions.get("clickables", []):
        formatted_actions.append(f"click[{clickable}]")
    return "\n".join(formatted_actions)


def _create_webshop_env():
    try:
        import gym  # type: ignore[import-not-found]
        from web_agent_site.envs import (
            WebAgentTextEnv,
        )  # type: ignore[import-not-found]  # noqa: F401
    except Exception as e:
        raise ImportError(
            f"Error importing WebAgentTextEnv {e}. "
            "Please make sure you have installed the web_agent_site package, "
            "following the instructions in https://github.com/princeton-nlp/WebShop"
        ) from e

    return gym.make(
        "WebAgentTextEnv-v0",
        observation_mode="text_rich",
        num_products=None,
        human_goals=True,
    )


def _get_webshop_gold_action_sequence(
    env,
    session_id: int,
    *,
    max_search_pages: int = 5,
) -> List[str]:
    """Build and verify a successful trajectory from WebShop's session goal.

    WebShop exposes the target ASIN, canonical search query, and requested product
    options through the simulated server.  This is the WebShop counterpart of
    ScienceWorld's runtime ``get_gold_action_sequence`` path.  The trajectory is
    executed before it is returned, so an unreachable target or an option mismatch
    fails loudly instead of silently turning B2F into vanilla OPD.

    The environment is reset to the initial state before returning (and after an
    error), allowing the caller to replay a curriculum prefix on the same instance.
    """
    if max_search_pages <= 0:
        raise ValueError(f"max_search_pages must be positive, got {max_search_pages}")

    env.reset(session=session_id)
    try:
        server = getattr(env, "server", None)
        env_session = str(getattr(env, "session", session_id))
        sessions = getattr(server, "user_sessions", None)
        if not isinstance(sessions, dict) or env_session not in sessions:
            raise RuntimeError(
                "WebShop runtime expert generation requires env.server.user_sessions "
                f"to contain session {env_session!r}."
            )
        goal = sessions[env_session].get("goal") or {}
        query = goal.get("query")
        target_asin = str(goal.get("asin") or "").strip().lower()
        goal_options = goal.get("goal_options") or {}
        if isinstance(query, (list, tuple)):
            query = " ".join(str(token) for token in query)
        query = str(query or "").strip()
        if isinstance(goal_options, dict):
            option_items = list(goal_options.items())
        elif isinstance(goal_options, (list, tuple)):
            option_items = [
                (f"option_{index}", value) for index, value in enumerate(goal_options)
            ]
        else:
            option_items = []
        if not query or not target_asin or not isinstance(
            goal_options, (dict, list, tuple)
        ):
            raise RuntimeError(
                "WebShop session goal must provide query, asin, and goal_options; "
                f"got query={query!r}, asin={target_asin!r}, "
                f"goal_options={goal_options!r}."
            )

        product = getattr(server, "product_item_dict", {}).get(target_asin.upper(), {})
        product_title = str(product.get("Title") or "").strip()
        title_tokens = list(
            dict.fromkeys(
                token.lower()
                for token in re.findall(r"[A-Za-z0-9]+", product_title)
                if len(token) >= 4
            )
        )
        informative_tokens = sorted(title_tokens, key=lambda token: (-len(token), token))[:8]
        search_candidates = [
            product_title,
            " ".join(informative_tokens[:3]),
            " ".join(informative_tokens[:2]),
            *informative_tokens,
            " ".join(str(value) for value in goal.get("attributes", [])),
            goal.get("instruction_text"),
            product.get("name"),
            goal.get("name"),
            query,
            target_asin,
        ]
        search_candidates = list(
            dict.fromkeys(
                str(candidate).strip()
                for candidate in search_candidates
                if str(candidate or "").strip()
            )
        )

        actions: List[str] = []

        def execute(action: str):
            actions.append(action)
            return env.step(action)

        requested_options = [
            str(value).strip().lower()
            for value in (
                goal_options.values() if isinstance(goal_options, dict) else goal_options
            )
        ]

        def perfect_purchase_for(clickables):
            lowered_clickables = {str(value).lower() for value in clickables}
            if target_asin in lowered_clickables:
                return target_asin, {
                    str(option_name): str(option_value).strip().lower()
                    for option_name, option_value in option_items
                }

            product_dict = getattr(server, "product_item_dict", {})
            if not product_dict:
                return None

            from thefuzz import fuzz  # type: ignore[import-not-found]
            from web_agent_site.engine.goal import (  # type: ignore[import-not-found]
                get_reward,
            )

            prices = getattr(server, "product_prices", {})
            result_asins = [
                value.upper()
                for value in clickables
                if str(value).upper() in product_dict
            ]
            result_asins.sort(key=lambda asin: asin != target_asin.upper())
            for asin in result_asins:
                candidate_product = product_dict[asin]
                selected_options = {}
                unused_option_names = set(candidate_product.get("options", {}))
                for requested in requested_options:
                    best_match = None
                    for option_name in unused_option_names:
                        for option_value in candidate_product["options"][option_name]:
                            score = fuzz.token_set_ratio(str(option_value).lower(), requested)
                            if best_match is None or score > best_match[0]:
                                best_match = (score, option_name, str(option_value).lower())
                    if best_match is None or best_match[0] <= 85:
                        selected_options = None
                        break
                    _, option_name, option_value = best_match
                    selected_options[option_name] = option_value
                    unused_option_names.remove(option_name)
                if selected_options is None:
                    continue
                score = get_reward(
                    candidate_product,
                    goal,
                    price=prices.get(asin),
                    options=selected_options,
                )
                if float(score) >= 1.0:
                    return asin.lower(), selected_options
            return None

        selected_query = None
        selected_purchase = None
        for candidate_index, candidate in enumerate(search_candidates):
            if candidate_index > 0:
                env.reset(session=session_id)
            actions.clear()
            _, _, done, _ = execute(f"search[{candidate}]")
            if done:
                raise RuntimeError("WebShop expert search unexpectedly terminated the episode.")

            for page_index in range(max_search_pages):
                available = env.get_available_actions()
                clickables = {
                    str(value).lower() for value in available.get("clickables", [])
                }
                selected_purchase = perfect_purchase_for(clickables)
                if selected_purchase is not None:
                    selected_query = candidate
                    break
                if "next >" not in clickables or page_index + 1 >= max_search_pages:
                    break
                _, _, done, _ = execute("click[next >]")
                if done:
                    raise RuntimeError(
                        "WebShop expert pagination unexpectedly terminated the episode."
                    )
            if selected_query is not None:
                break
        if selected_query is None:
            raise RuntimeError(
                "WebShop could not find a clickable perfect-score product within "
                f"{max_search_pages} result pages for any verified search candidate "
                f"{search_candidates!r}."
            )

        selected_asin, selected_options = selected_purchase
        _, _, done, _ = execute(f"click[{selected_asin}]")
        if done:
            raise RuntimeError("WebShop expert product selection unexpectedly terminated the episode.")

        for option_name, option_value in sorted(selected_options.items()):
            option_value = str(option_value).strip().lower()
            clickables = {
                str(value).lower()
                for value in env.get_available_actions().get("clickables", [])
            }
            if option_value not in clickables:
                raise RuntimeError(
                    f"WebShop goal option {option_name!r}={option_value!r} is not "
                    f"clickable for selected ASIN {selected_asin!r}."
                )
            _, _, done, _ = execute(f"click[{option_value}]")
            if done:
                raise RuntimeError(
                    f"WebShop expert option {option_name!r} unexpectedly terminated the episode."
                )

        clickables = {
            str(value).lower() for value in env.get_available_actions().get("clickables", [])
        }
        if "buy now" not in clickables:
            raise RuntimeError(
                f"WebShop Buy Now action is unavailable for selected ASIN {selected_asin!r}."
            )
        _, reward, done, _ = execute("click[buy now]")
        if not done or float(reward) < 1.0:
            raise RuntimeError(
                "WebShop runtime expert trajectory did not achieve a perfect score: "
                f"done={done}, reward={reward}, actions={actions!r}."
            )
        return actions
    finally:
        env.reset(session=session_id)


def _reset_webshop_env_with_checkpoint(
    env,
    session_id: int,
    actions: List[str],
    checkpoint_step: int,
):
    """Reset an existing WebShop environment and replay an expert prefix."""
    env.reset(session=session_id)
    observation = env.observation
    history: List[str] = []
    task_description = _extract_task_description(observation)
    done = False
    reward = -0.1

    for step in range(min(checkpoint_step, len(actions))):
        action = actions[step]
        formatted_observation = format_observation(observation)
        history.append(_format_history(formatted_observation, step + 1, action))
        observation, reward, done, _ = env.step(action)
        if done:
            break

    return env, observation, history, task_description, len(history), done, reward


def _create_webshop_env_with_checkpoint(
    session_id: int,
    actions: List[str],
    checkpoint_step: int,
):
    """Create WebShop environment and execute actions up to checkpoint_step."""
    env = _create_webshop_env()
    return _reset_webshop_env_with_checkpoint(env, session_id, actions, checkpoint_step)
