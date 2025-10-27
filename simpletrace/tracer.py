import itertools
import os
import queue
import threading
import time

import torch

from . import config, runtime_state
from .config import flow_info, trace_level
from .trace_type import Batch_Trace_Data, Collated_Trace_Data


def print_trace_data(trace_data, prefix: str = ""):
    next_prefix = prefix + "  "
    if isinstance(trace_data, dict):
        for k, v in trace_data.items():
            print(f"{prefix}{k}({type(v)}):{{")
            print_trace_data(v, next_prefix)
            print(f"{prefix}}}")
    elif isinstance(trace_data, torch.Tensor):
        if trace_data.dim() == 0:
            print(f"{next_prefix}{trace_data}")
        else:
            if hasattr(trace_data, "is_nested") and trace_data.is_nested:
                print(f"{prefix}nested:([")
            else:
                print(f"{prefix}tensor:([")
            for t in trace_data:
                print_trace_data(t, next_prefix)
            print(f"{prefix}])")
    elif isinstance(trace_data, Collated_Trace_Data):
        attrs = ["start", "end", "pid", "child", "args"]
        for attr in attrs:
            value = getattr(trace_data, attr)
            print(f"{prefix}{attr}({type(value)}):{{")
            print_trace_data(value, next_prefix)
            print(f"{prefix}}}")
    else:
        print(f"{prefix}{trace_data}")


def nested_to_list(nested):
    if isinstance(nested, torch.Tensor) and nested.is_nested:
        return torch.cat([t.flatten() for t in nested]).tolist()
    else:
        return list(itertools.chain.from_iterable(nested))


class Tracer:
    def __init__(self):
        self.main_pid = -1
        self.initialized = False

        self.write_flag = True
        self.task_queue = queue.Queue()
        self.worker_thread = threading.Thread(
            target=self._async_write_file, daemon=True
        )
        self.worker_thread.start()
        self.trace_count = 0
        self.trace_count_period = 400000
        self.tid_offset = {-1: 0}
        self.padding_content = []

    def before_train_init(self):
        self.main_pid = os.getpid()
        self.trace_file = os.path.join(
            runtime_state.trace_dir,
            f"{runtime_state.task_id}_rank_{runtime_state.rank}_split_{self.trace_count//self.trace_count_period}_iter_start_{runtime_state.train_iter}_trace.json",
        )
        self.initialized = True
        self._write_trace("[", 0)
        for content, line_count in self.padding_content:
            self._write_trace(content, line_count)

    def _async_write_file(self):
        while self.write_flag or not self.task_queue.empty():
            filepath, content = self.task_queue.get()
            with open(filepath, "a") as f:
                f.write(content + "\n")
            self.task_queue.task_done()

    def _write_trace(self, content, line_count):
        if not self.initialized:
            # self.padding_content.append((content, line_count))
            return
        self.trace_count += line_count
        if self.trace_count > self.trace_count_period:
            self.task_queue.put((self.trace_file, "{}]"))
            self.trace_file = os.path.join(
                runtime_state.trace_dir,
                f"{runtime_state.task_id}_rank_{runtime_state.rank}_split_{self.trace_count//self.trace_count_period}_iter_start_{runtime_state.train_iter}_trace.json",
            )
            self.task_queue.put((self.trace_file, "["))
            self.trace_count = line_count
        self.task_queue.put((self.trace_file, content))

    def _write_dist_trace(self, name, start_time, end_time, info: dict = {}):
        info_args = ""
        for k, v in info.items():
            info_args += f', "{k}":"{v}"'
        args = f'{{"iter": {runtime_state.train_iter}{info_args}}}'
        start_time = int(start_time * 1e6)
        end_time = int(end_time * 1e6)
        trace_start_str = f'{{"name": "{name}", "ph": "X", "ts": {start_time}, "dur": {end_time - start_time}, "pid": {runtime_state.rank}, "tid": {self.main_pid}, "args": {args}}}'
        trace_str = trace_start_str + "," + "\n"
        self._write_trace(trace_str, 1)

    def _trace_duration(self, labels, start_times, end_times, pids, args):
        line_count = 0
        trace_str = ""
        for label, start_time, end_time, pid, arg in zip(
            labels, start_times, end_times, pids, args
        ):
            trace_start_str = f'{{"name": "{label}", "ph": "X", "ts": {start_time}, "dur": {end_time - start_time}, "pid": {runtime_state.rank}, "tid": {pid}, "args": {arg}}}'
            trace_str += trace_start_str + "," + "\n"
            line_count += 1
        self._write_trace(trace_str, line_count)

    def _trace_flow(self, start_time, start_tid, end_time, end_tid, arg):
        trace_str = ""
        trace_start_str = f'{{"name": "flow", "ph": "s", "ts": {start_time}, "pid": {runtime_state.rank}, "tid": {start_tid}, "id":{end_time}, "args": {arg}}}'
        trace_end_str = f'{{"name": "flow", "ph": "t", "ts": {end_time}, "pid": {runtime_state.rank}, "tid": {end_tid}, "id":{end_time}, "args": {arg}}}'
        trace_str += trace_start_str + "," + trace_end_str + "," + "\n"
        self._write_trace(trace_str, 1)

    def export_item_trace(self, trace_data: dict):
        for func, func_trace_data in trace_data.items():
            if isinstance(func_trace_data, Collated_Trace_Data):
                starts = func_trace_data.start
                ends = func_trace_data.end
                pids = func_trace_data.pid
                args = func_trace_data.args
                names = [func] * len(starts)

                arg_strs = [f'"iter": {runtime_state.train_iter}'] * len(starts)
                for key, value in args.items():
                    for i in range(len(starts)):
                        arg_strs[i] += f', "{key}": "{value[i]}"'
                args = [f"{{{arg_str}}}" for arg_str in arg_strs]

                self._trace_duration(names, starts, ends, pids, args)

                childs_trace = func_trace_data.child
                self.export_item_trace(childs_trace)

    def export_batch_trace(self, trace_data: dict, flow_info: list[list[str]]):
        trace_info: dict[str, dict] = dict()

        for func, func_trace_data in trace_data.items():
            if not isinstance(func_trace_data, Batch_Trace_Data):
                continue
            start = func_trace_data.start
            end = func_trace_data.end
            pid = func_trace_data.pid
            tid = func_trace_data.tid
            if tid in self.tid_offset:
                tid_offset = self.tid_offset[tid]
            elif tid not in self.tid_offset:
                tid_offset = len(self.tid_offset)
                self.tid_offset[tid] = tid_offset
            self._trace_duration(
                [func],
                [start],
                [end],
                [pid + tid_offset],
                [f'{{"iter": {runtime_state.train_iter}}}'],
            )
            trace_info[func] = {"start": start, "end": end, "pid": pid + tid_offset}

        for chain in flow_info:
            last_node = None
            for node in chain:
                if node in trace_info:
                    if last_node is not None:
                        self._trace_flow(
                            trace_info[last_node]["end"],
                            trace_info[last_node]["pid"],
                            trace_info[node]["start"],
                            trace_info[node]["pid"],
                            f'{{"iter": {runtime_state.train_iter}}}',
                        )
                    last_node = node
                else:
                    continue

    def export_trace(self, trace_data):
        if trace_level == "profiler_only":
            return

        train_time_start = (
            runtime_state.train_start_time
            if runtime_state.train_start_time != -1
            else runtime_state.data_end_time
        )
        train_time_end = (
            runtime_state.train_end_time
            if runtime_state.train_end_time != -1
            else time.perf_counter_ns()
        )
        trace_data["train_time"] = Batch_Trace_Data(
            start=train_time_start,
            end=train_time_end,
            pid=self.main_pid,
        )

        trace_data["data_time"] = Batch_Trace_Data(
            start=runtime_state.data_start_time,
            end=runtime_state.data_end_time,
            pid=self.main_pid,
        )

        if runtime_state.profile_export:
            trace_data["profiler"] = Batch_Trace_Data(
                start=runtime_state.profile_start_time,
                end=runtime_state.profile_end_time,
                pid=self.main_pid,
            )

        config.logger.debug(f"tracer_iter: {runtime_state.train_iter}")

        self.export_item_trace(trace_data)

        self.export_batch_trace(trace_data, flow_info)
