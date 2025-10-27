import os
import time
from functools import wraps

from . import runtime_state
from .trace_type import Trace_Data


# decorator for standalone functions
def default_func_decorator(func, name):
    @wraps(func)
    def wrapper(*arg, **kwargs):
        origin_trace_data = runtime_state.trace_data
        childs = dict()
        runtime_state.trace_data = childs
        start = time.perf_counter_ns()
        result = func(*arg, **kwargs)
        end = time.perf_counter_ns()
        if name not in origin_trace_data:
            origin_trace_data[name] = Trace_Data()
        origin_trace_data[name].append(start, end, os.getpid(), childs)
        runtime_state.trace_data = origin_trace_data
        return result

    return wrapper


# decorator for methods of an instance
def default_method_decorator(method, name):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        origin_trace_data = runtime_state.trace_data
        childs = dict()
        runtime_state.trace_data = childs
        start = time.perf_counter_ns()
        result = method(self, *args, **kwargs)
        end = time.perf_counter_ns()
        if name not in origin_trace_data:
            origin_trace_data[name] = Trace_Data()
        origin_trace_data[name].append(start, end, os.getpid(), childs)
        runtime_state.trace_data = origin_trace_data
        return result

    return wrapper


# decortaor for special class methods lick __getitem__, __add__, __setitem__ etc.
# Example:

# MyDataset A
# A[i] will call type(A).__getitem__(i), rather than A.__getitem__(i)


def default_special_method_decorator(method, name):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        origin_trace_data = runtime_state.trace_data
        childs = dict()
        runtime_state.trace_data = childs
        start = time.perf_counter_ns()
        result = method(self, *args, **kwargs)
        end = time.perf_counter_ns()
        if name not in origin_trace_data:
            origin_trace_data[name] = Trace_Data()
        origin_trace_data[name].append(start, end, os.getpid(), childs)
        runtime_state.trace_data = origin_trace_data
        return result

    return wrapper
