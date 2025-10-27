import multiprocessing.queues

import torch.utils.data._utils.pin_memory
from torch.utils.data._utils.fetch import _MapDatasetFetcher
from torch.utils.data.dataloader import _BaseDataLoaderIter

from . import config
from .config import (
    add_flow_chain,
    add_trace_dataset_function,
    set_cache_pickle,
    set_log_level,
    set_trace_io,
    set_trace_level,
)
from .fetcher_decorator import map_fetcher_fetch_decorator, map_fetcher_init_decorator
from .iterator_decorator import (
    before_run_trace,
    init_decorator,
    next_index_decorator,
)
from .loop_manager import Loop_Manager
from .pinmemory_hooks import custom_get, modified_pin_memory_loop

if config.trace_level != "profiler_only":
    _MapDatasetFetcher.__init__ = map_fetcher_init_decorator(
        _MapDatasetFetcher.__init__
    )
    multiprocessing.queues.Queue.custom_get = custom_get
    torch.utils.data._utils.pin_memory._pin_memory_loop = modified_pin_memory_loop

manager = Loop_Manager()
_MapDatasetFetcher.fetch = map_fetcher_fetch_decorator(_MapDatasetFetcher.fetch)
_BaseDataLoaderIter.__next__ = before_run_trace(_BaseDataLoaderIter.__next__, manager)
_BaseDataLoaderIter._next_index = next_index_decorator(_BaseDataLoaderIter._next_index)
_BaseDataLoaderIter.__init__ = init_decorator(_BaseDataLoaderIter.__init__)

__all__ = [
    "add_trace_dataset_function",
    "add_flow_chain",
    "set_trace_level",
    "set_log_level",
    "cycle_dataloaders",
    "set_trace_io",
    "set_cache_pickle",
]
