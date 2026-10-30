# -*- coding: utf-8 -*-
"""Shared WebShop environments: one loaded product catalog serves many games.

WebShop's WebAgentTextEnv loads the whole catalog (1.18M products, the full dataset that TCOD's
4096 + 100 session ids need; the 1000-product set has 13 human goals) into every env it creates,
which is why TCOD's README asks for ~1TB of memory. Here a few Ray actors each load it once
(`WebShopEnvServer`) and host any number of games on it: WebAgentTextEnv(server=...) with its own
session_prefix per game, so two games of the same session id (a student's game and a teacher
look-ahead replaying it) never share state. Rollout runners and look-ahead replays reach the
actors by name (`RemoteWebShopEnv`); the game itself is WebShop's code, unchanged.

Every actor builds the same shop: `random` is seeded before the catalog is loaded, because
WebShop draws the price of a product with a price range and the price cap of each human goal
(the "price lower than ... dollars" of the instruction) from `random` while loading. Unseeded,
each env instance gets different caps for the same session id (as in TCOD's runs). The goals
(products, attributes, options, instruction wording) do not depend on it.
"""

import os
import random
import shutil
import zlib
from typing import Dict, Optional, Tuple

ACTOR_PREFIX = "webshop_env_server"
DEFAULT_NUM_SERVERS = 4
DEFAULT_SEED = 0


def _java_home() -> Optional[str]:
    java = shutil.which("java")
    if java is None:
        return None
    return os.path.dirname(os.path.dirname(os.path.realpath(java)))


class WebShopEnvServer:
    """One WebShop SimServer (catalog, search index, goals) and the games played on it."""

    def __init__(self, seed: int = DEFAULT_SEED):
        # pyserini (the search engine) starts a JVM through pyjnius, which needs JAVA_HOME.
        if "JAVA_HOME" not in os.environ and _java_home():
            os.environ["JAVA_HOME"] = _java_home()
        from web_agent_site.envs.web_agent_text_env import SimServer, WebAgentTextEnv
        from web_agent_site.utils import DEFAULT_FILE_PATH

        self._env_cls = WebAgentTextEnv
        random.seed(seed)
        # = what gym.make("WebAgentTextEnv-v0", observation_mode="text_rich", num_products=None,
        #   human_goals=True) builds inside each env (TCOD's _create_webshop_env)
        self.server = SimServer(
            "http://127.0.0.1:3000", DEFAULT_FILE_PATH, None, -1, None, True, False
        )
        self.envs: Dict[str, object] = {}
        self._next = 0

    def ready(self) -> int:
        return len(self.server.goals)

    def new_env(self, observation_mode: str = "text_rich") -> str:
        env_id = f"g{self._next}_"
        self._next += 1
        self.envs[env_id] = self._env_cls(
            observation_mode=observation_mode, server=self.server, session_prefix=env_id,
            num_products=None, human_goals=True,
        )
        return env_id

    def _drop_session(self, env) -> None:
        if getattr(env, "session", None) is not None:
            self.server.user_sessions.pop(env.session, None)

    def reset(self, env_id: str, session: int) -> Tuple[str, Dict]:
        env = self.envs[env_id]
        self._drop_session(env)  # the previous game's session state
        env.reset(session=session)
        return env.observation, env.get_available_actions()

    def step(self, env_id: str, action: str) -> Tuple[str, float, bool, Dict]:
        env = self.envs[env_id]
        observation, reward, done, _ = env.step(action)
        return observation, float(reward), bool(done), env.get_available_actions()

    def close(self, env_id: str) -> None:
        env = self.envs.pop(env_id, None)
        if env is not None:
            self._drop_session(env)


def get_server(index: int, num_servers: int = DEFAULT_NUM_SERVERS, seed: int = DEFAULT_SEED):
    """The named server actor `index % num_servers`, created on this node if it does not exist."""
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    name = f"{ACTOR_PREFIX}_{index % num_servers}"
    try:
        return ray.get_actor(name, namespace=ACTOR_PREFIX)
    except ValueError:
        pass
    env_vars = {k: os.environ[k] for k in ("JAVA_HOME", "TMPDIR") if k in os.environ}
    return (
        ray.remote(WebShopEnvServer)
        .options(
            name=name,
            namespace=ACTOR_PREFIX,
            get_if_exists=True,
            lifetime="detached",  # outlives the runner that created it; ended with the cluster
            num_cpus=1,
            max_concurrency=1,
            max_restarts=-1,
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                ray.get_runtime_context().get_node_id(), soft=True
            ),
            runtime_env={"env_vars": env_vars},
        )
        .remote(seed)
    )


def server_index(key: str) -> int:
    return zlib.crc32(key.encode())


class RemoteWebShopEnv:
    """A game on a WebShopEnvServer, with the env methods TCOD's WebShop workflows use
    (reset(session=), observation, get_available_actions(), step(), close()), as coroutines."""

    def __init__(self, key: str, num_servers: int = DEFAULT_NUM_SERVERS, seed: int = DEFAULT_SEED,
                 observation_mode: str = "text_rich"):
        self.server = get_server(server_index(key), num_servers, seed)
        self.observation_mode = observation_mode  # "text": UOPD's WebShop workflows
        self.env_id: Optional[str] = None
        self.observation: Optional[str] = None
        self._available: Dict = {}

    async def reset(self, session: int) -> str:
        if self.env_id is None:
            self.env_id = await self.server.new_env.remote(self.observation_mode)
        self.observation, self._available = await self.server.reset.remote(self.env_id, int(session))
        return self.observation

    def get_available_actions(self) -> Dict:
        return self._available

    async def step(self, action: str):
        observation, reward, done, self._available = await self.server.step.remote(self.env_id, action)
        self.observation = observation
        return observation, reward, done, {}

    async def close(self) -> None:
        if self.env_id is not None:
            env_id, self.env_id = self.env_id, None
            await self.server.close.remote(env_id)
