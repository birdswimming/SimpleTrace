import builtins
import io
import json
import os
import pickle
import time
from functools import wraps

from . import config, runtime_state
from .default_decorator import default_func_decorator
from .trace_type import Trace_Data


def get_file_size(file, obj):
    if file:
        try:
            return os.path.getsize(file)
        except (OSError, FileNotFoundError):
            pass

    if obj is not None:
        if isinstance(obj, (io.BytesIO, io.StringIO)):
            return len(obj.getvalue())

        if hasattr(obj, "seek") and hasattr(obj, "tell"):
            try:
                current_pos = obj.tell()
                obj.seek(0, os.SEEK_END)
                size = obj.tell()
                obj.seek(current_pos)
                return size
            except Exception:
                pass

    return 0


def open_decorator(open_func):
    @wraps(open_func)
    def wrapper(file, *args, **kwargs):
        origin_trace_data = runtime_state.trace_data
        childs = dict()
        runtime_state.trace_data = childs
        start = time.perf_counter_ns()
        file_obj = open_func(file, *args, **kwargs)
        end = time.perf_counter_ns()
        size = get_file_size(file, file_obj)
        if "open" not in origin_trace_data:
            origin_trace_data["open"] = Trace_Data()
        origin_trace_data["open"].append(
            start=start,
            end=end,
            pid=os.getpid(),
            childs=childs,
            args={"file_name": f"{file}", "size": size},
        )
        runtime_state.trace_data = origin_trace_data
        return file_obj

    return wrapper


origin_open = None
origin_pickle_load = None
origin_pickle_loads = None
origin_json_load = None
origin_json_loads = None
origin_exists = None
# open_wrapper = open_decorator(origin_open)
# pickle_load_wrapper = load_decorator(origin_pickle_load)
# pickle_loads_wrapper = default_decorator(origin_pickle_loads, "pickle.loads")
# json_load_wrapper = default_decorator(origin_json_load, "json.load")
# json_loads_wrapper = default_decorator(origin_json_loads, "json.loads")
# exists_wrapper = default_decorator(origin_exists, "exists")


def start_io_trace():
    if config.trace_level == "profiler_only":
        return
    global origin_open, origin_pickle_load, origin_pickle_loads, origin_json_load, origin_json_loads, origin_exists
    origin_open = builtins.open
    origin_pickle_load = pickle.load
    origin_pickle_loads = pickle.loads
    origin_json_load = json.load
    origin_json_loads = json.loads
    origin_exists = os.path.exists
    open_wrapper = open_decorator(origin_open)
    pickle_load_wrapper = default_func_decorator(origin_pickle_load, "pickle.load")
    pickle_loads_wrapper = default_func_decorator(origin_pickle_loads, "pickle.loads")
    json_load_wrapper = default_func_decorator(origin_json_load, "json.load")
    json_loads_wrapper = default_func_decorator(origin_json_loads, "json.loads")
    exists_wrapper = default_func_decorator(origin_exists, "exists")
    builtins.open = open_wrapper
    pickle.load = pickle_load_wrapper
    pickle.loads = pickle_loads_wrapper
    json.load = json_load_wrapper
    json.loads = json_loads_wrapper
    os.path.exists = exists_wrapper


def end_io_trace():
    if config.trace_level == "profiler_only":
        return
    builtins.open = origin_open
    pickle.load = origin_pickle_load
    pickle.loads = origin_pickle_loads
    json.load = origin_json_load
    json.loads = origin_json_loads
    os.path.exists = origin_exists
