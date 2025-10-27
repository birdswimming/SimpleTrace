import logging
import sys
import time

from . import config, profile, runtime_state
from .tracer import Tracer, print_trace_data


class Loop_Manager:
    def __init__(self):
        self.step_position = "data_start"
        self.trace_data: dict = {}
        self.train_profile_enable: bool = False
        self.initialized = False
        self.tracer = Tracer()

    def before_train_init(self):
        runtime_state.init_distribute()
        self.tracer.before_train_init()
        self.initialized = True
        profile.start_listener()

    # mamully called at the beginning of train loop
    def loop_start(self):
        self.step_position = "loop_start"
        self.step()
        runtime_state.loop_start_time = time.perf_counter_ns()
        config.logger.debug("Loop start.")

    # auto called at the beginning of next(dataloader)
    # export_trace() must called before data_end() or new trace_data will cover old one
    # so this is the lastest time to call export_trace()
    def data_start(self):
        if self.step_position == "data_start":
            self.step()
        runtime_state.data_start_time = time.perf_counter_ns()
        config.logger.debug("Data start.")

    # auto called at the end of next(dataloader)
    def data_end(self):
        runtime_state.data_end_time = time.perf_counter_ns()
        config.logger.debug(f"train_profile_enable: {self.train_profile_enable}")
        profile.try_end_train_profiler(self.train_profile_enable)
        profile.try_set_config()
        profile.try_start_train_profiler(self.train_profile_enable)
        if config.logger.getEffectiveLevel() == logging.DEBUG:
            with open("trace.txt", "w") as f:
                old_out, sys.stdout = sys.stdout, f
                try:
                    print_trace_data(self.trace_data)
                finally:
                    sys.stdout = old_out
        config.logger.debug("Data end.")
        runtime_state.train_start_time = time.perf_counter_ns()

    # mamully called at the begin of model train
    def train_start(self):
        runtime_state.train_start_time = time.perf_counter_ns()
        config.logger.debug("Train start.")

    # mamully called at the end of model train
    def train_end(self):
        runtime_state.train_end_time = time.perf_counter_ns()
        config.logger.debug("Train end.")

    # mamully called at the end of train loop
    def loop_end(self):
        runtime_state.loop_end_time = time.perf_counter_ns()
        config.logger.debug("Loop end.")

    def step(self):
        if self.initialized:
            self.tracer.export_trace(self.trace_data)
            runtime_state.train_iter += 1

        if not self.initialized:
            self.before_train_init()
