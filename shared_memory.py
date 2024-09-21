import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np


class SharedMemory:
    """Handles shared memory arrays and values."""

    def __init__(self, screen_width: int, screen_height: int, max_iters: int):
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Create shared memory blocks
        self.cx = shared_memory.SharedMemory(
            create=True, size=screen_width * np.float64().nbytes
        )
        self.cy = shared_memory.SharedMemory(
            create=True, size=screen_height * np.float64().nbytes
        )
        self.pixels = shared_memory.SharedMemory(
            create=True, size=screen_width * screen_height * np.uint32().nbytes
        )

        # Create NumPy arrays backed by shared memory
        self.cx_a = np.ndarray((screen_width,), dtype=np.float64, buffer=self.cx.buf)
        self.cy_a = np.ndarray((screen_height,), dtype=np.float64, buffer=self.cy.buf)
        self.pixels_a = np.ndarray(
            (screen_width, screen_height), dtype=np.uint32, buffer=self.pixels.buf
        )

        self.max_iters = mp.Value("i", max_iters)

    def clean_up_memory(self):
        self.cx.close()
        self.cx.unlink()
        self.cy.close()
        self.cy.unlink()
        self.pixels.close()
        self.pixels.unlink()
