import os
import time
from functools import wraps

import torch
from torch.utils.data import get_worker_info

from . import config, runtime_state
from .collate_decorator import collate_decorator
from .dataset_decorator import patch_map_dataset
from .io_decorator import end_io_trace, start_io_trace
from .trace_type import Batch_Trace_Data, Collated_Trace_Data


def trace_fuse(trace_data: dict[str, Collated_Trace_Data], start_time: float = None):
    fused_trace: list[dict] = []
    origin_keys = list(trace_data.keys())
    for func in origin_keys:
        func_trace_data = trace_data[func]
        if not isinstance(func_trace_data, Collated_Trace_Data):
            continue
        trace_data.pop(func)

        starts = func_trace_data.start
        ends = func_trace_data.end
        pids = func_trace_data.pid
        childs = func_trace_data.child

        trace = {
            "start": starts.min().item(),
            "name": func,
            "duration": (ends - starts).sum().item(),
            "pid": pids[0],
            "childs": childs,
            "args": {},
        }

        if "size" in func_trace_data.args:
            sizes = func_trace_data.args["size"]
            trace["args"]["size"] = torch.tensor([sizes.sum().item()])

        fused_trace.append(trace)

    sorted_trace = sorted(fused_trace, key=lambda x: x["start"])
    for trace in sorted_trace:
        func = trace["name"]
        duration = trace["duration"]
        pid = trace["pid"]
        args = trace["args"]
        childs = trace["childs"]
        if start_time is None:
            start_time = trace["start"]

        fused_name = f"batch_{func}"

        trace_data[fused_name] = Collated_Trace_Data(
            start=torch.tensor([start_time]),
            end=torch.tensor([start_time + duration]),
            pid=torch.tensor([pid]),
            child=childs,
            args=args,
        )

        trace_fuse(childs, start_time + 1000)
        start_time = start_time + duration


def map_fetcher_init_decorator(fetcher_init):
    @wraps(fetcher_init)
    def wrapper(self, dataset, auto_collation, collate_fn, drop_last):
        runtime_state.init_distribute((get_worker_info() is not None))
        if hasattr(dataset, "patched") and dataset.patched:
            pass
        else:
            dataset = patch_map_dataset(dataset)
        collate_fn = collate_decorator(collate_fn)
        return fetcher_init(self, dataset, auto_collation, collate_fn, drop_last)

    return wrapper


def default_fetch(self, fetch, possibly_batched_index):
    if config.TRACE_IO:
        start_io_trace()
    raw_data = fetch(self, possibly_batched_index)
    if config.TRACE_IO:
        end_io_trace()
    return raw_data


def profiled_fetch(self, fetch, possibly_batched_index, index_iter):
    prefix = f"iter_{index_iter}"

    config.logger.info(f"{prefix}: profile dataloader")

    json_trace_path = os.path.join(
        runtime_state.profile_dir,
        f"{runtime_state.task_id}_rank_{runtime_state.rank}_dataloader_{prefix}.json",
    )

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], with_stack=True
    ) as profiler:
        config.logger.info(f"{prefix}: dataloader profiler is profiling...")

        raw_data = default_fetch(self, fetch, possibly_batched_index)

        start = time.perf_counter_ns()

        config.logger.info(f"{prefix}: dataloader profiler is exiting...")

    config.logger.info(f"{prefix}: exporting profile json to {json_trace_path}")
    profiler.export_chrome_trace(json_trace_path)

    end = time.perf_counter_ns()
    runtime_state.trace_data["loader_profiler"] = Batch_Trace_Data(
        start=start,
        end=end,
        pid=os.getpid(),
    )

    return raw_data


def map_fetcher_fetch_decorator(fetch):
    @wraps(fetch)
    def wrapper(self, possibly_batched_index):
        index_iter = possibly_batched_index.get("index_iter", -1)
        enable_profile = possibly_batched_index.get("enable_profile", False)
        profile_mode = possibly_batched_index.get("profile_mode", -1)
        possibly_batched_index = possibly_batched_index.get(
            "raw_index", possibly_batched_index
        )

        dataloader_profile_enable = enable_profile and profile_mode in [0, 2]
        train_profile_enable = enable_profile and profile_mode in [0, 1]
        skip = profile_mode == 0 and torch.utils.data.get_worker_info() is None
        runtime_state.trace_data = {}
        runtime_state.trace_data_list = []

        if dataloader_profile_enable and not skip:
            raw_data = profiled_fetch(self, fetch, possibly_batched_index, index_iter)
        else:
            raw_data = default_fetch(self, fetch, possibly_batched_index)

        if config.trace_level != "debug" and config.trace_level != "profiler_only":
            trace_fuse(runtime_state.trace_data)

        if config.trace_level == "profiler_only":
            result = {
                "raw_data": raw_data,
                "train_profile_enable": train_profile_enable,
            }
        else:
            result = {
                "raw_data": raw_data,
                "train_profile_enable": train_profile_enable,
                "trace_data": runtime_state.trace_data,
            }
        return result

    return wrapper
