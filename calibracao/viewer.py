from abc import ABC, abstractmethod
from typing import Set, Callable
import sys

class Viewer(ABC):

    @property
    def keyboard_listeners(self) -> Set[Callable[[str], None]]:
        return self._keyboard_listeners

    @property
    def mouse_listeners(self) -> Set[Callable[[int, int], None]]:
        return self._mouse_listeners

    def __init__(self):
        self._keyboard_listeners = set()
        self._mouse_listeners = set()

    # @abstractmethod
    # def read(self):
    #     """Read a frame from the device"""
    #     pass

    @abstractmethod
    def set_image(self, image):
        """Set a frame into the displaying device"""
        pass

    @abstractmethod
    def set_loop(self, loop: Callable[[None], None]):
        """Set a function to be called by the GUI loop"""
        pass

    @abstractmethod
    def start(self):
        """Begin showing the GUI"""
        pass

    def add_keyboard_listener(self, func: Callable[[str], None]):
        self._keyboard_listeners.add(func)

    def add_mouse_listener(self, func: Callable[[int, int], None]):
        self._mouse_listeners.add(func)

    def remove_keyboard_listener(self, func: Callable):
        self._keyboard_listeners.discard(func)

    def remove_mouse_listener(self, func: Callable):
        self._mouse_listeners.discard(func)


class PyGLViewer(Viewer):

    def __init__(self):
        super().__init__()
        sys.path.append('./python_glview')
        import pyglview

        self._glviewer = pyglview.Viewer(
            keyboard_listener=self._proxy_kb_listener)

    def set_image(self, image):
        self._glviewer.set_image(image)

    def set_loop(self, func):
        self._glviewer.set_loop(func)

    def start(self):
        self._glviewer.start()

    def _proxy_kb_listener(self, key: str, x: int, y: int):
        for listener in self.keyboard_listeners.copy():
            listener(key)

    def _proxy_mouse_listener(self, button, state, x: int, y: int):
        for listener in self.mouse_listeners.copy():
            listener(button, state, x, y)
