import logging
import os
import sys

pin_memory_num_threads = 1

trace_dataset_function: list[str] = []

TRACE_IO = os.environ.get("TRACE_IO", "1") == "1"

CACHE_PICKLE_LOAD = os.environ.get("TINYTRACE_CACHE", "0") == "1"

flow_info: list[list[str]] = [
    [
        "raw_collate",
        "trace_collate",
        "loader_profiler",
        "pickler",
        "pin_memory",
        "train_time",
    ],
]

trace_level = os.environ.get("TINYTRACE_LEVEL", "info")

tinytrace_log_level = os.environ.get("TINYTRACE_LOG_LEVEL", "info")
if tinytrace_log_level == "debug":
    log_level = logging.DEBUG
else:
    log_level = logging.INFO

logger = logging.getLogger("TinyTrace")
logger.setLevel(log_level)
logger.propagate = False
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    fmt="%(asctime)s %(name)s [%(levelname)s]: %(message)s",
    datefmt="%m/%d %H:%M:%S",
)
handler.setFormatter(formatter)
logger.addHandler(handler)


def add_trace_dataset_function(func):
    trace_dataset_function.append(func)


def set_log_level(level):
    logger.setLevel(level)


def set_trace_level(level):
    global trace_level
    trace_level = level


def add_flow_chain(chain: list[str]):
    flow_info.append(chain)


def set_trace_io(enable: bool):
    global TRACE_IO
    TRACE_IO = enable


def set_cache_pickle(enable: bool):
    global CACHE_PICKLE_LOAD
    CACHE_PICKLE_LOAD = enable
