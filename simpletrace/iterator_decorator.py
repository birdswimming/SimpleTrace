from functools import wraps

from torch.utils.data.dataloader import DataLoader, _BaseDataLoaderIter

from . import profile, runtime_state
from .loop_manager import Loop_Manager
from .trace_type import check_data


def before_run_trace(origin_next, manager: Loop_Manager):
    @wraps(origin_next)
    def new_next(self):
        manager.data_start()
        result = origin_next(self)
        if check_data(result):
            manager.trace_data = result.get("trace_data", dict())
            manager.train_profile_enable = result.get("train_profile_enable", dict())
            result = result["raw_data"]
        manager.data_end()
        return result

    return new_next


def next_index_decorator(next_index):
    @wraps(next_index)
    def wrapper(self: _BaseDataLoaderIter, *args, **kwargs):
        raw_index = next_index(self, *args, **kwargs)
        enable_profile = profile.enable_profile(self.index_count)
        patched_index = {
            "index_iter": self.index_count,
            "enable_profile": enable_profile,
            "profile_mode": profile.profile_mode,
            "raw_index": raw_index,
        }
        self.index_count += 1
        runtime_state.index_iter = self.index_count
        return patched_index

    return wrapper


def init_decorator(init):
    @wraps(init)
    def wrapper(self: _BaseDataLoaderIter, loader: DataLoader):
        self.index_count = 0
        init(self, loader)

    return wrapper
