import os
import time

import torch
from torch.utils.data._utils.collate import collate, default_collate_fn_map

from . import runtime_state
from .trace_type import (
    Batch_Trace_Data,
    Collated_Trace_Data,
    Trace_Container,
    Trace_Data,
)

custom_collate_fn_map = default_collate_fn_map.copy()


def collate_trace_container_fn(batch: list[Trace_Container], *, collate_fn_map=None):
    if len(batch) == 0:
        raise ValueError("get empty batch")
    elem = batch[0][0]
    data_list = []
    for trace_data in batch:
        data_list.extend(trace_data)
    if isinstance(elem, float):
        collated_data = torch.tensor(data_list, dtype=torch.float64)
    elif isinstance(elem, int):
        collated_data = torch.tensor(data_list)
    elif isinstance(elem, dict):
        # Used for children in trace container
        collated_data = collate(data_list, collate_fn_map=custom_collate_fn_map)
    else:
        collated_data = data_list
    return collated_data


def colate_trace_data_fn(batch: list[Trace_Data], *, collate_fn_map=None):
    fields = ["start", "end", "pid", "child", "args"]
    grouped = {f: [getattr(trace_data, f) for trace_data in batch] for f in fields}

    collated = {
        f: collate(values, collate_fn_map=collate_fn_map)
        for f, values in grouped.items()
    }

    return Collated_Trace_Data(**collated)


def collate_child_dict_fn(batch: list[dict], *, collate_fn_map=None):
    assert all(isinstance(d, dict) for d in batch)
    all_keys = set()
    for d in batch:
        all_keys.update(d.keys())
    result = {
        key: collate([d[key] for d in batch if key in d], collate_fn_map=collate_fn_map)
        for key in all_keys
    }
    return result


custom_collate_fn_map[Trace_Container] = collate_trace_container_fn
custom_collate_fn_map[Trace_Data] = colate_trace_data_fn
custom_collate_fn_map[dict] = collate_child_dict_fn


def collate_decorator(collate_fn):
    def wrapper(batch):
        trace_batch = runtime_state.trace_data_list
        raw_start = time.perf_counter_ns()
        result = collate_fn(batch)
        raw_end = time.perf_counter_ns()
        if len(trace_batch) > 0:
            start = time.perf_counter_ns()
            runtime_state.trace_data = collate(
                trace_batch, collate_fn_map=custom_collate_fn_map
            )
            end = time.perf_counter_ns()
            runtime_state.trace_data["trace_collate"] = Batch_Trace_Data(
                start=start,
                end=end,
                pid=os.getpid(),
            )

        runtime_state.trace_data["raw_collate"] = Batch_Trace_Data(
            start=raw_start,
            end=raw_end,
            pid=os.getpid(),
        )
        return result

    return wrapper
