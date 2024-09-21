import numpy as np
from abc import ABC, abstractmethod

from controls import Controls


class Renderer(ABC):
    def __init__(self, controls: Controls):
        self.controls = controls

    @abstractmethod
    def render_pixels(self, pixels: np.ndarray): ...

    @abstractmethod
    def render_texts(self, texts: list[str]): ...

    @abstractmethod
    def display(self): ...

    @abstractmethod
    def handle_input(self): ...

    @abstractmethod
    def quit(self): ...
