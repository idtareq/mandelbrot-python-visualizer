import logging
import pygame as pg
import numpy as np
import multiprocessing as mp
from enum import Enum
from dataclasses import dataclass
import threading
from mandelbrot import generate_mandelbrot_set
from util import divide_into_ranges, text_drop_shadow

numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class SharedMemory:
    """Handles shared memory arrays and values."""

    def __init__(self, screen_width: int, screen_height: int, max_iters: int):
        self.cx = mp.RawArray("d", screen_width)
        self.cx_a = np.frombuffer(self.cx, dtype=np.double)
        self.cy = mp.RawArray("d", screen_height)
        self.cy_a = np.frombuffer(self.cy, dtype=np.double)
        self.pixels = mp.RawArray("d", screen_width * screen_height)
        self.pixels_a = np.frombuffer(self.pixels, dtype=np.double).reshape(
            screen_width, screen_height
        )
        self.max_iters = mp.RawValue("i", max_iters)


class WorkerSynchronizer:
    """
    Synchronizes worker processes or threads with the main thread during Mandelbrot set computation.

    This class uses synchronization primitives to coordinate the start and end of computation cycles
    between workers and the main thread. It ensures that all workers complete their computation
    before the main thread proceeds and that workers wait for the main thread before starting the
    next cycle.
    """

    def __init__(self, number_of_workers: int):
        self._barrier = mp.Barrier(number_of_workers)
        self.busy = mp.Event()
        self.next = mp.Event()
        self.terminate = mp.Event()

    def start(self):
        """Prepares the workers to start a new computation cycle."""
        # Ensure workers will wait after finishing computation until 'next' is set.
        self.next.clear()
        # Indicate that workers are now busy computing.
        self.busy.set()

    def end(self):
        """Synchronizes workers at the end of their computation cycle."""
        # Wait until all workers have finished their computation.
        self._barrier.wait()
        # Indicate that workers have finished computing.
        self.busy.clear()
        # Wait for the main thread to signal the start of the next computation cycle.
        self.next.wait()


class WorkerType(Enum):
    PROCESS = "process"
    THREAD = "thread"


@dataclass
class Worker:
    id: int
    screen_width: int
    screen_height: int
    number_of_workers: int
    shared_memory: SharedMemory
    syncer: WorkerSynchronizer

    def __call__(self):
        pixels_a = np.frombuffer(self.shared_memory.pixels, dtype=np.double).reshape(
            self.screen_width, self.screen_height
        )
        ranges = divide_into_ranges(self.screen_height, self.number_of_workers)
        while not self.syncer.terminate.is_set():
            self.syncer.start()
            # Perform computation
            generate_mandelbrot_set(
                pixels_a,
                self.shared_memory.cx,
                self.shared_memory.cy,
                self.shared_memory.max_iters.value,
                ranges[self.id][0],
                ranges[self.id][1],
            )
            self.syncer.end()


class MandelbrotVisualizer:
    def __init__(
        self,
        screen_width,
        screen_ratio,
        number_of_workers,
        worker_type,
        max_iters,
    ):
        self.screen_width = screen_width
        self.number_of_workers = number_of_workers
        self.screen_ratio = screen_ratio
        self.screen_height = int(screen_ratio * screen_width)
        self.worker_type = worker_type
        self.shared_memory = SharedMemory(
            self.screen_width, self.screen_height, max_iters
        )
        self._process_syncer = WorkerSynchronizer(self.number_of_workers)
        self._thread_syncer = WorkerSynchronizer(self.number_of_workers)
        self._threads: list[threading.Thread] = []
        self._processes: list[mp.Process] = []

    def initialize_workers(self):
        """Initializes worker processes or threads based on the worker type."""

        # Check if previous workers need to be terminated
        if self._process_syncer.terminate.is_set():
            for process in self._processes:
                if process.is_alive():
                    return
            self._process_syncer.terminate.clear()
            self._processes = []

        if self._thread_syncer.terminate.is_set():
            for thread in self._threads:
                if thread.is_alive():
                    return
            self._thread_syncer.terminate.clear()
            self._threads = []

        if self.worker_type == WorkerType.PROCESS and not self._processes:
            logger.debug("Initializing process workers.")
            for id in range(self.number_of_workers):
                worker_args = (
                    id,
                    self.screen_width,
                    self.screen_height,
                    self.number_of_workers,
                    self.shared_memory,
                )
                self._processes.append(
                    mp.Process(
                        target=Worker(*worker_args, self._process_syncer),
                        daemon=True,
                    )
                )
                self._processes[-1].start()

        if self.worker_type == WorkerType.THREAD and not self._threads:
            logger.debug("Initializing thread workers.")
            for id in range(self.number_of_workers):
                worker_args = (
                    id,
                    self.screen_width,
                    self.screen_height,
                    self.number_of_workers,
                    self.shared_memory,
                )
                self._threads.append(
                    threading.Thread(
                        target=Worker(*worker_args, self._thread_syncer),
                        daemon=True,
                    )
                )
                self._threads[-1].start()

    @property
    def is_loading(self):
        return (
            self._process_syncer.terminate.is_set()
            or self._thread_syncer.terminate.is_set()
        )

    @property
    def syncer(self):
        if self.worker_type == WorkerType.PROCESS:
            return self._process_syncer
        elif self.worker_type == WorkerType.THREAD:
            return self._thread_syncer
        else:
            assert ValueError("Invalid worker type")

    def terminate(self):
        self._thread_syncer.terminate.set()
        self._process_syncer.terminate.set()
        self._thread_syncer.next.set()
        self._process_syncer.next.set()

    def update(self, centerX, centerY, zoom):
        if not self.syncer.busy.is_set():
            zoomX = zoom + zoom * self.screen_ratio
            self.shared_memory.cx_a[:] = np.linspace(
                centerX - zoomX, centerX + zoomX, self.screen_width
            )
            self.shared_memory.cy_a[:] = np.linspace(
                centerY - zoom, centerY + zoom, self.screen_height
            )

    def draw_mandelbrot(self, surface: pg.Surface):
        if not self.syncer.busy.is_set() and not self.is_loading:
            pixels_r = pg.surfarray.pixels2d(surface)
            pixels_r[:] = self.shared_memory.pixels_a
            del pixels_r
            self.syncer.next.set()
        if self.is_loading:
            surface.fill((0, 0, 0))

    def draw_texts(self, surface: pg.Surface, font: pg.font.Font, fps: int):
        surface.fill((0, 0, 0, 0))

        text_color = 255, 255, 255
        shadow_color = 0, 0, 100

        surface.blit(
            text_drop_shadow(
                font,
                f"{self.worker_type} (press c to change)",
                3,
                text_color,
                shadow_color,
            ),
            (10, 10),
        )
        surface.blit(
            text_drop_shadow(
                font,
                f"Iters: {self.shared_memory.max_iters.value} (press right/left arrow to change)",
                3,
                text_color,
                shadow_color,
            ),
            (10, 30),
        )
        surface.blit(
            text_drop_shadow(
                font,
                f"fps: {fps}",
                3,
                text_color,
                shadow_color,
            ),
            (10, 50),
        )


def handle_input(
    viz: MandelbrotVisualizer,
    centerX,
    centerY,
    zoom,
    is_panning,
    pan_start_pos,
    delta_time,
):
    pressed_key = pg.key.get_pressed()
    quit = False
    PAN_SENSITIVITY = 2

    if pressed_key[pg.K_RIGHT]:
        viz.shared_memory.max_iters.value += 2
    if pressed_key[pg.K_LEFT]:
        viz.shared_memory.max_iters.value -= 2
    if pressed_key[pg.K_a]:
        centerX -= zoom * delta_time
    if pressed_key[pg.K_d]:
        centerX += zoom * delta_time
    if pressed_key[pg.K_w]:
        centerY -= zoom * delta_time
    if pressed_key[pg.K_s]:
        centerY += zoom * delta_time
    if pressed_key[pg.K_UP]:
        zoom -= zoom * delta_time
    if pressed_key[pg.K_DOWN]:
        zoom += zoom * delta_time
    if pressed_key[pg.K_ESCAPE]:
        quit = True

    for event in pg.event.get():
        if event.type == pg.QUIT:
            quit = True
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_c:
                viz.worker_type = (
                    WorkerType.PROCESS
                    if viz.worker_type == WorkerType.THREAD
                    else WorkerType.THREAD
                )
        elif event.type == pg.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                is_panning = True
                pan_start_pos = event.pos
            elif event.button == 4:  # Scroll up (on some systems)
                # Zoom in
                zoom_factor = 0.9
                mouse_x, mouse_y = event.pos
                centerX += (
                    (mouse_x - viz.screen_width / 2)
                    * (zoom / viz.screen_width)
                    * (1 - zoom_factor)
                )
                centerY += (
                    (mouse_y - viz.screen_height / 2)
                    * (zoom / viz.screen_height)
                    * (1 - zoom_factor)
                )
                zoom *= zoom_factor
            elif event.button == 5:  # Scroll down (on some systems)
                # Zoom out
                zoom_factor = 1.1
                mouse_x, mouse_y = event.pos
                centerX += (
                    (mouse_x - viz.screen_width / 2)
                    * (zoom / viz.screen_width)
                    * (1 - 1 / zoom_factor)
                )
                centerY += (
                    (mouse_y - viz.screen_height / 2)
                    * (zoom / viz.screen_height)
                    * (1 - 1 / zoom_factor)
                )
                zoom *= zoom_factor
        elif event.type == pg.MOUSEBUTTONUP:
            if event.button == 1:  # Left mouse button
                is_panning = False
        elif event.type == pg.MOUSEMOTION:
            if is_panning:
                dx = event.pos[0] - pan_start_pos[0]
                dy = event.pos[1] - pan_start_pos[1]
                pan_start_pos = event.pos

                scale_x = zoom / viz.screen_width
                scale_y = zoom / viz.screen_height
                centerX -= dx * scale_x * PAN_SENSITIVITY
                centerY -= dy * scale_y * PAN_SENSITIVITY
        elif event.type == pg.MOUSEWHEEL:
            # For newer versions of Pygame
            if event.y > 0:  # Scroll up
                zoom_factor = 0.9
            else:  # Scroll down
                zoom_factor = 1.1
            mouse_x, mouse_y = pg.mouse.get_pos()
            centerX += (
                (mouse_x - viz.screen_width / 2)
                * (zoom / viz.screen_width)
                * (1 - zoom_factor)
            )
            centerY += (
                (mouse_y - viz.screen_height / 2)
                * (zoom / viz.screen_height)
                * (1 - zoom_factor)
            )
            zoom *= zoom_factor

    viz.shared_memory.max_iters.value = np.clip(
        viz.shared_memory.max_iters.value, 50, 500
    )
    centerX = np.clip(centerX, -2, 2)
    centerY = np.clip(centerY, -2, 2)
    zoom = np.clip(zoom, -2, 2)

    return centerX, centerY, zoom, is_panning, pan_start_pos, quit


def main():
    pg.init()
    info = pg.display.Info()

    viz = MandelbrotVisualizer(
        screen_width=int(info.current_w // 1.3),
        screen_ratio=info.current_h / info.current_w,
        number_of_workers=mp.cpu_count(),
        worker_type=WorkerType.PROCESS,
        max_iters=80,
    )

    screen = pg.display.set_mode((viz.screen_width, viz.screen_height))
    mandelbrot_surface = pg.Surface((screen.get_width(), screen.get_height()))
    text_surface = pg.Surface((screen.get_width(), screen.get_height()), pg.SRCALPHA)
    pg.display.set_caption("Mandelbrot Python Visualizer")
    clock = pg.time.Clock()
    font = pg.font.Font(None, 25)
    centerX = 0
    centerY = 0
    zoom = 2
    is_panning = False
    pan_start_pos = (0, 0)
    quit = False

    while True:
        delta_time = clock.tick(30) / 1000.0
        centerX, centerY, zoom, is_panning, pan_start_pos, quit = handle_input(
            viz, centerX, centerY, zoom, is_panning, pan_start_pos, delta_time
        )

        viz.initialize_workers()
        viz.update(centerX, centerY, zoom)
        viz.draw_mandelbrot(mandelbrot_surface)
        viz.draw_texts(text_surface, font, round(clock.get_fps(), 1))
        screen.fill((0, 0, 0))
        screen.blit(mandelbrot_surface, (0, 0))
        screen.blit(text_surface, (0, 0))

        pg.display.flip()

        if quit:
            logger.debug("Quitting.")
            viz.terminate()
            pg.quit()
            break


if __name__ == "__main__":
    main()
