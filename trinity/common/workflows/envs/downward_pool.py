"""Reuse fast_downward library copies across ALFWorld games instead of leaking one per game.

fast_downward.load_lib() copies libdownward.so (~35MB) into a temporary directory and loads the copy, so that
planners in one process do not share the library's global state. textworld's PddlEnv calls it once per env and
nothing ever unloads it (dlclose does not reliably unmap it and crashes at exit), so each game keeps ~35MB of a
deleted /dev/shm file mapped for the life of the process. Over a long run this filled the workers' tmpfs and
killed training (gated ALFWorld Gemma run, 2026-10-30).

install() keeps a per-process pool instead: PddlEnv.close() hands its copy back, and the next PddlEnv takes a
free copy before making a new one. PddlEnv.reset() already reloads a fresh problem into the same copy
(load_sas), and a copy is only reused once its env is closed, so no two live envs share one.
"""
import threading

_installed = False


def install() -> None:
    global _installed
    if _installed:
        return
    try:
        import fast_downward
        from textworld.envs.pddl import pddl as tw_pddl
    except ImportError:
        return
    free, lock, load_copy = [], threading.Lock(), fast_downward.load_lib

    def load_lib():
        with lock:
            if free:
                return free.pop()
        return load_copy()

    orig_close = tw_pddl.PddlEnv.close

    def close(self):
        orig_close(self)
        lib = self.__dict__.pop("downward_lib", None)
        if lib is not None:
            self.__dict__.pop("_pddl_state", None)
            with lock:
                free.append(lib)

    fast_downward.load_lib = load_lib
    tw_pddl.PddlEnv.close = close
    _installed = True
