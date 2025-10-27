import io
import os
import queue
import socket
import threading
import time

import torch
import torch.distributed as dist
import torch.profiler as profiler
import yaml

from . import config, runtime_state

pipe_path = "/tmp/simpletrace"
notify_queue = queue.Queue()
config_queue = queue.Queue()

profile_iter_start = -1
profile_iter_end = -1
profile_ranks = []

# 0: profile both dataloader and train
# 1: only profile train
# 2: only profile dataloader
profile_mode = 0

profiling_iter_start = -1

train_profiler = None


def pipe_listener():
    config.logger.info("listener: waiting for config file...")
    with open(pipe_path, "r") as fifo:
        while True:
            line = fifo.readline()
            if line:
                config.logger.info(f"listener: receive config file: {line.strip()}")
                notify_queue.put(line.strip())  # 推送到主线程消息队列
            else:
                # 管道被关闭（发送端退出）
                time.sleep(0.1)


def load_config_to_tensor(config_path: str, rank_num: int) -> torch.Tensor:
    with open(config_path, "r") as f:
        profile_config = yaml.safe_load(f)
    mode = profile_config.get("mode", 0)
    wait = profile_config.get("schedule_wait", 0)
    active = profile_config.get("schedule_active", 0)
    ranks = profile_config.get("ranks", [])
    relative = profile_config.get("relative", True)
    padded_ranks = ranks[:rank_num] + [-1] * (rank_num - len(ranks))
    tensor = torch.tensor(
        [mode, wait, active, relative] + padded_ranks, dtype=torch.int32
    )
    return tensor


def parse_tensor_to_config(config_tensor: torch.Tensor) -> dict:
    config_tensor = config_tensor.cpu()
    config_list = config_tensor.tolist()
    mode = config_list[0]
    wait = config_list[1]
    active = config_list[2]
    relative = config_list[3]
    ranks = [r for r in config_list[4:] if r != -1]
    config = {
        "mode": mode,
        "schdule_wait": wait,
        "schdule_active": active,
        "relative": relative,
        "ranks": ranks,
    }
    return config


def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    if tensor.device != torch.device("cpu"):
        tensor = tensor.cpu()
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return buffer.getvalue()


def bytes_to_tensor(data: bytes) -> torch.Tensor:
    buffer = io.BytesIO(data)
    return torch.load(buffer)


def send_tensor(ip: str, port: int, tensor: torch.Tensor, timeout: float = 10.0):
    data = tensor_to_bytes(tensor)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect((ip, port))
            s.sendall(len(data).to_bytes(8, "big"))
            s.sendall(data)
    except socket.timeout:
        config.logger.error(f"Send to {ip}:{port} timed out after {timeout} seconds")
    except socket.error as e:
        config.logger.error(f"Failed to send tensor to {ip}:{port}: {e}")


def recv_tensor(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", port))
        s.listen(1)
        conn, _ = s.accept()
        with conn:
            length = int.from_bytes(conn.recv(8), "big")
            buf = b""
            while len(buf) < length:
                chunk = conn.recv(min(4096, length - len(buf)))
                if not chunk:
                    break
                buf += chunk
            return bytes_to_tensor(buf)


def custom_broadcast(tensor: torch.Tensor):
    rank = dist.get_rank()
    num_rank = dist.get_world_size()
    if rank == 0:
        for target_rank in range(1, num_rank):
            send_tensor(
                runtime_state.ips[target_rank],
                runtime_state.base_port + target_rank,
                tensor,
            )
    else:
        config.logger.debug(f"rank[{rank}]: waiting config")
        tensor.copy_(recv_tensor(runtime_state.base_port + rank))
        config.logger.debug(f"rank[{rank}]: config received")


def communicater():
    # [0]: mode [1]:iter_start [2]:active [3]:relative [4]:ranks(fill with -1)
    config_tensor = torch.zeros(4 + runtime_state.rank_num, dtype=torch.int32)
    while True:
        if runtime_state.distribute:
            if runtime_state.rank == 0:
                config_path = notify_queue.get()
                if not os.path.exists(config_path):
                    config.logger.warning(
                        f"[Rank {runtime_state.rank}] Config file does not exist: {config_path}"
                    )
                    continue
                new_tensor = load_config_to_tensor(config_path, runtime_state.rank_num)
                if new_tensor is None:
                    continue
                config_tensor.copy_(new_tensor)
                custom_broadcast(config_tensor)
            else:
                custom_broadcast(config_tensor)
        else:
            config_path = notify_queue.get()
            if not os.path.exists(config_path):
                config.logger.warning(
                    f"[Rank {runtime_state.rank}] Config file does not exist: {config_path}"
                )
                continue
            new_tensor = load_config_to_tensor(config_path, runtime_state.rank_num)
            if new_tensor is None:
                continue
            config_tensor.copy_(new_tensor)

        config_queue.put(config_tensor.clone())


def enable_profile(index_iter):
    check_rank = runtime_state.rank in profile_ranks
    iter = index_iter

    check_iter = iter >= profile_iter_start and iter < profile_iter_end
    enable = check_rank and check_iter
    return enable


def profile_running():
    iter = runtime_state.index_iter
    running = iter >= profile_iter_start and iter < profile_iter_end
    return running


def print_config():
    if runtime_state.rank != 0:
        return
    config.logger.info("===== Profiling Configuration =====")
    mode_desc = {
        0: "both dataloader and train",
        1: "train only",
        2: "dataloader only",
    }.get(profile_mode, "unknown")

    config.logger.info(f"Profile Mode      : {profile_mode} ({mode_desc})")
    config.logger.info(
        f"Profile Iteration : [{profile_iter_start},{profile_iter_end}) (duration = {profile_iter_end - profile_iter_start})"
    )
    config.logger.info(f"Profile Ranks     : {profile_ranks}")
    config.logger.info("===================================")


def set_config(profile_config: dict):
    global profile_mode, profile_iter_start, profile_iter_end, profile_ranks

    config.logger.info(f"rank[{runtime_state.rank}]: receive config: {profile_config}")
    mode = profile_config.get("mode", 0)
    wait = profile_config.get("schdule_wait", 0)
    active = profile_config.get("schdule_active", 0)
    ranks = profile_config.get("ranks", [])
    relative = profile_config.get("relative", True)

    if mode not in [0, 1, 2]:
        config.logger.warning(
            f"Invalid config: 'mode' must be 0 (both), 1 (train), or 2 (dataloader), but got '{mode}'"
        )
        return

    if active < 0:
        config.logger.warning(
            f"Invalid config: 'active' must be non-negative, but got '{active}'"
        )
        return

    if len(ranks) == 0:
        config.logger.warning("Invalid config: 'ranks' list cannot be empty")
        return

    for rank in ranks:
        if rank < 0 or rank >= runtime_state.rank_num:
            config.logger.warning(
                f"Invalid config: rank value '{rank}' is out of valid range [0, {runtime_state.rank_num - 1}]"
            )
            return

    current_iter = max(runtime_state.train_iter, runtime_state.index_iter)

    if relative:
        if wait < 0:
            config.logger.warning(
                f"Invalid config: 'wait' must be non-negative, but got '{wait}'"
            )
            return
        profile_iter_start = current_iter + wait
    else:
        if wait < current_iter:
            config.logger.warning(
                f"Invalid config: 'wait' must bigger than current iter {current_iter}, but got '{wait}'"
            )
            return
        profile_iter_start = wait

    profile_mode = mode
    profile_iter_end = profile_iter_start + active
    profile_ranks = ranks
    print_config()


def try_set_config():
    if profile_running():
        return

    try:
        config_tensor = config_queue.get_nowait()  # 非阻塞获取
        config_dict = parse_tensor_to_config(config_tensor)
        set_config(config_dict)
    except queue.Empty:
        pass


def try_start_train_profiler(train_profile_enable: bool):
    global train_profiler, profiling_iter_start
    if train_profile_enable:
        if train_profiler is None:
            profiling_iter_start = runtime_state.train_iter
            train_profiler = profiler.profile(
                activities=[
                    profiler.ProfilerActivity.CPU,
                    profiler.ProfilerActivity.CUDA,
                ],
                with_stack=True,
            )
            train_profiler.__enter__()
            config.logger.info("train profiler is profiling...")


def try_end_train_profiler(train_profile_enable: bool):
    global train_profiler
    runtime_state.profile_start_time = time.perf_counter_ns()
    if not train_profile_enable:
        if train_profiler is not None:
            train_profiler.__exit__(None, None, None)
            json_trace_path = os.path.join(
                runtime_state.profile_dir,
                f"{runtime_state.task_id}_rank_{runtime_state.rank}_train_iter_{profiling_iter_start}_{runtime_state.train_iter}.json",
            )
            config.logger.info(f"train profiler exporting to {json_trace_path}")
            train_profiler.export_chrome_trace(json_trace_path)
            train_profiler = None
            runtime_state.profile_end_time = time.perf_counter_ns()
            runtime_state.profile_export = True
            return
    runtime_state.profile_end_time = time.perf_counter_ns()
    runtime_state.profile_export = False
    return


def start_listener():
    if runtime_state.rank == 0:
        if not os.path.exists(pipe_path):
            os.mkfifo(pipe_path)
        listener_thread = threading.Thread(target=pipe_listener, daemon=True)
        listener_thread.start()
    communication_thread = threading.Thread(target=communicater, daemon=True)
    communication_thread.start()
