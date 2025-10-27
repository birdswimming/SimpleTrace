import multiprocessing.reduction
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Empty

import torch
from torch.utils.data._utils.pin_memory import (
    MP_STATUS_CHECK_INTERVAL,
    ExceptionWrapper,
    pin_memory,
)

from .config import pin_memory_num_threads
from .trace_type import check_data

_ForkingPickler = multiprocessing.reduction.ForkingPickler


def custom_get(self, block=True, timeout=None):
    if self._closed:
        raise ValueError(f"Queue {self!r} is closed")
    if block and timeout is None:
        with self._rlock:
            res = self._recv_bytes()
        self._sem.release()
    else:
        if block:
            deadline = time.monotonic() + timeout
        if not self._rlock.acquire(block, timeout):
            raise Empty
        try:
            if block:
                timeout = deadline - time.monotonic()
                if not self._poll(timeout):
                    raise Empty
            elif not self._poll():
                raise Empty
            res = self._recv_bytes()
            self._sem.release()
        finally:
            self._rlock.release()
    # unserialize the data after having released the lock
    pickler_start = time.perf_counter_ns()
    result = _ForkingPickler.loads(res)
    pickler_end = time.perf_counter_ns()
    return result, pickler_start, pickler_end


def modified_pin_memory_loop(in_queue, out_queue, device_id, done_event, device):
    torch.set_num_threads(1)

    if device == "cuda":
        torch.cuda.set_device(device_id)
    elif device == "xpu":
        torch.xpu.set_device(device_id)  # type: ignore[attr-defined]
    elif device == torch._C._get_privateuse1_backend_name():
        custom_device_mod = getattr(torch, torch._C._get_privateuse1_backend_name())
        custom_device_mod.set_device(device_id)

    def do_one_step(thread_idx, pid):
        try:
            r, pickler_start, pickler_end = in_queue.custom_get(
                timeout=MP_STATUS_CHECK_INTERVAL
            )
        except queue.Empty:
            return
        idx, data = r
        if not done_event.is_set() and not isinstance(data, ExceptionWrapper):
            if check_data(data):
                data["trace_data"]["pickler"] = {
                    "start": pickler_start,
                    "end": pickler_end,
                    "pid": pid,
                    "tid": threading.get_native_id(),
                }
            pin_start = time.perf_counter_ns()
            try:
                data = pin_memory(data, device)
                pin_end = time.perf_counter_ns()
                if check_data(data):
                    data["trace_data"]["pin_memory"] = {
                        "start": pin_start,
                        "end": pin_end,
                        "pid": pid,
                        "tid": threading.get_native_id(),
                    }
            except Exception:
                data = ExceptionWrapper(
                    where=f"in pin memory thread for device {device_id}"
                )

            r = (idx, data)
        while not done_event.is_set():
            try:
                out_queue.put(r, timeout=MP_STATUS_CHECK_INTERVAL)
                break
            except queue.Full:
                continue

    def worker(thread_idx):
        pid = os.getpid()
        while not done_event.is_set():
            do_one_step(thread_idx, pid)

    with ThreadPoolExecutor(max_workers=pin_memory_num_threads) as executor:
        _ = [executor.submit(worker, i + 1) for i in range(pin_memory_num_threads)]

        # 等待 done_event 被设置，done_event 通常由父线程或其他地方设置
        done_event.wait()
