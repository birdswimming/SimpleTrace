import os
import time
import types
from collections.abc import Sequence
from functools import wraps

from . import config, runtime_state
from .config import trace_dataset_function
from .default_decorator import default_func_decorator, default_method_decorator
from .trace_type import Trace_Data


def get_nested_attr(obj, attr_path: str, default=None):
    """获取对象的多级嵌套属性，若中途某一级不存在则返回 default"""
    attrs = attr_path.split(".")
    for attr in attrs[:-1]:
        if not hasattr(obj, attr):
            return default
        obj = getattr(obj, attr)
    return obj, attrs[-1]


def make_wrapper(attr, full_name=None):
    if isinstance(attr, types.MethodType):
        wrapped = default_method_decorator(attr, full_name)
        return types.MethodType(wrapped, attr.__self__)
    else:
        wrapped = default_func_decorator(attr, full_name)
        return wrapped


def get_name(t):
    return getattr(t, "__name__", f"{t.__class__.__name__}")


def patch_nested_callable(obj, attr_path):
    obj, attr_name = get_nested_attr(obj, attr_path)
    full_path = f"{get_name(obj)}.{attr_path}"
    target = None
    if obj is not None and hasattr(obj, attr_name):
        target = getattr(obj, attr_name)
    if callable(target):
        wrapped = make_wrapper(target, full_name=f"{get_name(target)}({full_path})")
        setattr(obj, attr_name, wrapped)

    elif isinstance(target, Sequence) and not isinstance(target, str):
        patched = []
        for i, item in enumerate(target):
            if callable(item):
                wrapped = make_wrapper(
                    item, full_name=f"{get_name(item)}({full_path}[{i}])"
                )
                patched.append(wrapped)
            else:
                patched.append(item)
        setattr(obj, attr_name, type(target)(patched))

    else:
        config.logger.warning(
            f"{attr_path} is not recognized as callable or callable container."
        )


def patch_map_dataset(dataset):
    # 修饰 __getitem__
    if hasattr(dataset, "__getitems__"):
        raise NotImplementedError
    elif hasattr(dataset, "__getitem__"):
        original_class = dataset.__class__  # 保存原始类
        # decortaor for special class methods lick __getitem__, __add__, __setitem__ etc.
        # Example:

        # MyDataset A
        # A[i] will call type(A).__getitem__(i), rather than A.__getitem__(i)
        class PatchedDataset(original_class):
            patched = True

            @wraps(original_class.__getitem__)
            def __getitem__(self, *args, **kwargs):
                name = f"{self.__class__.__name__}.__getitem__"
                origin_trace_data: dict[str, Trace_Data] = {}

                # start tracing target function called during __getitem__
                childs = dict()
                runtime_state.trace_data = childs

                start = time.perf_counter_ns()
                result = super().__getitem__(*args, **kwargs)
                end = time.perf_counter_ns()

                origin_trace_data[name] = Trace_Data()
                origin_trace_data[name].append(start, end, os.getpid(), childs)

                runtime_state.trace_data_list.append(origin_trace_data)
                return result

        PatchedDataset.__name__ = f"{original_class.__name__}"
        dataset.__class__ = PatchedDataset

    for func in trace_dataset_function:
        patch_nested_callable(dataset, func)

    return dataset
