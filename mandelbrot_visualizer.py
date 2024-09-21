import logging
from typing import Callable
import numpy as np
from contextlib import contextmanager

from shared_memory import SharedMemory
from worker import WorkerManager
from controls import Controls

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class MandelbrotVisualizer:
    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        number_of_workers: int,
        max_iters: int,
        controls: Controls,
        worker_function: Callable,
    ):
        self.screen_width = screen_width
        self.number_of_workers = number_of_workers
        self.screen_height = screen_height
        self.has_switched_workers = True
        self.shared_memory = SharedMemory(
            self.screen_width, self.screen_height, max_iters
        )
        self.controls = controls
        self.worker_function = worker_function
        self.worker_manager = WorkerManager(self)
        self.worker_manager.initialize_workers()

    def update(self):
        if self.worker_manager.syncer and not self.worker_manager.syncer.is_busy:
            self.worker_manager.initialize_workers()

            if self.shared_memory.max_iters.value != self.controls.max_iters:
                self.shared_memory.max_iters.value = self.controls.max_iters

            zoomX = self.controls.zoom + self.controls.zoom * (
                self.screen_height / self.screen_width
            )
            self.shared_memory.cx_a[:] = np.linspace(
                self.controls.centerX - zoomX,
                self.controls.centerX + zoomX,
                self.screen_width,
            )
            self.shared_memory.cy_a[:] = np.linspace(
                self.controls.centerY - self.controls.zoom,
                self.controls.centerY + self.controls.zoom,
                self.screen_height,
            )

    @contextmanager
    def get_pixels(self):
        pixels = None
        continue_workers = False
        try:
            if self.worker_manager.syncer:
                if not self.worker_manager.has_initialized:
                    pixels = np.zeros(
                        (self.screen_width, self.screen_height), dtype=np.uint32
                    )
                if self.worker_manager.syncer.is_done:
                    pixels = self.shared_memory.pixels_a
                    continue_workers = True
            yield pixels
        finally:
            if continue_workers and self.worker_manager.syncer:
                self.worker_manager.syncer.continue_workers()

    def get_texts(self):
        return [
            f"{self.controls.worker_type} (press c to change)",
            f"Iters: {self.shared_memory.max_iters.value} (press right/left arrow to change)",
        ]

    def terminate(self):
        self.worker_manager.terminate_workers()
        self.shared_memory.clean_up_memory()
