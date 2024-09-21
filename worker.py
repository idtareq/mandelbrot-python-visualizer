import logging
import time
import numpy as np
import multiprocessing as mp
import threading
from enum import Enum
from typing import Any
from dataclasses import dataclass
from contextlib import contextmanager
from util import divide_into_ranges

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class WorkerType(Enum):
    PROCESS = "process"
    THREAD = "thread"


@dataclass
class Worker:
    id: int
    viz: Any
    syncer: "WorkerSynchronizer"

    def __call__(self):
        pixels_a = np.ndarray(
            (self.viz.shared_memory.screen_width, self.viz.shared_memory.screen_height),
            dtype=np.uint32,
            buffer=self.viz.shared_memory.pixels.buf,
        )

        ranges = divide_into_ranges(self.viz.screen_height, self.viz.number_of_workers)
        while not self.syncer.is_terminated:
            self.syncer.worker_before_hook()
            # Perform computation
            self.viz.worker_function(
                pixels_a,
                self.viz.shared_memory.cx_a,
                self.viz.shared_memory.cy_a,
                self.viz.shared_memory.max_iters.value,
                ranges[self.id][0],
                ranges[self.id][1],
            )
            self.syncer.worker_after_hook()


class WorkerManager:
    def __init__(self, viz) -> None:
        self.viz = viz
        self.syncer = None

    def initialize_workers(self):
        if self.viz.controls.has_switched_workers:
            return
        self.viz.controls.has_switched_workers = True

        logger.debug(f"Initializing {self.viz.controls.worker_type.value} workers.")
        self.terminate_workers()
        if self.viz.controls.worker_type == WorkerType.PROCESS:
            self.syncer = WorkerSynchronizer(self.viz.number_of_workers, False)
            for id in range(self.viz.number_of_workers):
                worker_args = (
                    id,
                    self.viz,
                )
                mp.Process(
                    target=Worker(*worker_args, self.syncer),
                    daemon=True,
                ).start()
        if self.viz.controls.worker_type == WorkerType.THREAD:
            self.syncer = WorkerSynchronizer(self.viz.number_of_workers, True)
            mp.Process(
                target=thread_workers_process,
                args=(self.viz, self.syncer),
                daemon=True,
            ).start()

    def terminate_workers(self):
        if self.syncer:
            self.syncer.terminate_workers()

    @property
    def has_initialized(self):
        if self.syncer:
            return self.syncer.has_initialized
        return False


def thread_workers_process(viz, thread_syncer: "WorkerSynchronizer"):
    for id in range(viz.number_of_workers):
        worker_args = (id, viz)
        threading.Thread(
            target=Worker(*worker_args, thread_syncer),
            daemon=True,
        ).start()
    while not thread_syncer._terminate.value:
        continue


class WorkerSynchronizer:
    def __init__(self, number_of_workers: int, is_threading: bool):
        self.number_of_workers = number_of_workers
        self._barrier = (
            threading.Barrier(self.number_of_workers)
            if is_threading
            else mp.Barrier(self.number_of_workers)
        )
        self._busy_flag = mp.Value("b", False)
        self._continue_flag = mp.Value("b", False)
        self._done_flag = mp.Value("b", False)
        self._terminate = mp.Value("b", False)
        self.has_initialized = False

    def continue_workers(self):
        if not self.has_initialized:
            self.has_initialized = True
        self._done_flag.value = False
        self._continue_flag.value = True

    def worker_before_hook(self):
        """Called by the worker"""
        self._barrier.wait()
        self._continue_flag.value = False
        self._busy_flag.value = True

    def worker_after_hook(self):
        """Called by the worker"""
        self._barrier.wait()
        self._busy_flag.value = False
        self._done_flag.value = True
        while not self._continue_flag.value and not self._terminate.value:
            time.sleep(0.05)
            continue

    def terminate_workers(self):
        self._terminate.value = True
        self._continue_flag.value = True

    @property
    def is_done(self):
        return bool(self._done_flag.value)

    @property
    def is_busy(self):
        return bool(self._busy_flag.value)

    @property
    def is_terminated(self):
        return bool(self._terminate.value)
