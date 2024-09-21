import numpy as np
from dataclasses import dataclass
from worker import WorkerType


@dataclass
class Controls:
    worker_type: WorkerType
    screen_width = 0
    screen_height = 0
    centerX: float = 0
    centerY: float = 0
    zoom: float = 2
    is_panning = False
    pan_start_pos = (0, 0)
    quit = False
    max_iters = 80
    PAN_SENSITIVITY = 2
    SPEED = 0.0075
    has_switched_workers = False

    def left(self):
        self.centerX -= self.zoom * self.SPEED
        self.centerX = np.clip(self.centerX, -2, 2)

    def right(self):
        self.centerX += self.zoom * self.SPEED
        self.centerX = np.clip(self.centerX, -2, 2)

    def up(self):
        self.centerY -= self.zoom * self.SPEED
        self.centerY = np.clip(self.centerY, -2, 2)

    def down(self):
        self.centerY += self.zoom * self.SPEED
        self.centerY = np.clip(self.centerY, -2, 2)

    def zoomin(self, speed=1):
        self.zoom -= self.zoom * self.SPEED * speed
        self.zoom = np.clip(self.zoom, -2, 2)

    def zoomout(self, speed=1):
        self.zoom += self.zoom * self.SPEED * speed
        self.zoom = np.clip(self.zoom, -2, 2)

    def increase_iters(self):
        self.max_iters += 2
        self.max_iters = np.clip(self.max_iters, 50, 500)

    def decrease_iters(self):
        self.max_iters -= 2
        self.max_iters = np.clip(self.max_iters, 50, 500)

    def switch_worker(self):
        self.worker_type = (
            WorkerType.PROCESS
            if self.worker_type == WorkerType.THREAD
            else WorkerType.THREAD
        )
        self.has_switched_workers = False

    def start_pan(self, x, y):
        self.is_panning = True
        self.pan_start_pos = x, y

    def stop_pan(self):
        self.is_panning = False

    def move_pan(self, x, y, screen_width, screen_height):
        if self.is_panning:
            dx = x - self.pan_start_pos[0]
            dy = y - self.pan_start_pos[1]
            self.pan_start_pos = x, y

            scale_x = self.zoom / screen_width
            scale_y = self.zoom / screen_height
            self.centerX -= dx * scale_x * self.PAN_SENSITIVITY
            self.centerY -= dy * scale_y * self.PAN_SENSITIVITY
