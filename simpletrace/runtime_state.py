import os
import socket
import struct
import subprocess
from datetime import datetime

import torch
import torch.distributed as dist

from . import config
from .trace_type import Trace_Data

work_dir = os.environ.get("WORK_DIR", "./trace_data")
task_id = os.environ.get(
    "MLP_TASK_ID", f"t-{datetime.now().strftime('%Y%m%d%H%M%S')}-local"
)
os.makedirs(work_dir, exist_ok=True)
trace_dir = os.path.join(work_dir, "trace")
os.makedirs(trace_dir, exist_ok=True)
profile_dir = os.path.join(work_dir, "profile")
os.makedirs(profile_dir, exist_ok=True)

index_iter = 0
train_iter = 0

rank = -1
rank_num = -1
distribute = False

ips = []
base_port = 34567


def get_self_ip():
    master_ip = os.environ["MASTER_ADDR"]
    result = subprocess.run(
        ["ip", "route", "get", master_ip], capture_output=True, text=True
    )
    output = result.stdout.strip()
    for token in output.split():
        if token == "src":
            return output.split()[output.split().index("src") + 1]
    return None


def ip_to_uint32(ip_str):
    packed_ip = socket.inet_aton(ip_str)
    return struct.unpack("!I", packed_ip)[0]


def uint32_to_ip(uint32):
    packed_ip = struct.pack("!I", uint32)
    return socket.inet_ntoa(packed_ip)


def report_ip():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self_ip = get_self_ip()
    ip_tensor = torch.tensor([ip_to_uint32(self_ip)], dtype=torch.int32, device=device)
    gathered = [
        torch.empty(1, dtype=torch.int32, device=device) for _ in range(rank_num)
    ]
    dist.all_gather(tensor_list=gathered, tensor=ip_tensor)
    for u in gathered:
        ips.append(uint32_to_ip(int(u[0])))
    config.logger.debug(f"rank[{rank}]: get ips:{ips}")


def init_distribute(from_worker=False):
    global rank, rank_num, distribute
    if rank != -1:
        return

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        rank_num = dist.get_world_size()
        distribute = True
        if not from_worker:
            report_ip()
    else:
        rank = 0
        rank_num = 1
        distribute = False


loop_start_time = -1
data_start_time = -1
data_end_time = -1
train_start_time = -1
train_end_time = -1
loop_end_time = -1
profile_start_time = -1
profile_end_time = -1
profile_export = False


trace_data: dict[str, Trace_Data] = {}
trace_data_list: list[dict] = []
