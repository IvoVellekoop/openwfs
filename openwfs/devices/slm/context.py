import threading
import weakref

from .. import safe_import

glfw = safe_import("glfw", "opengl")

SLM = "slm.SLM"


class Context:
    """Holds a weak reference to an SLM window

    The context is used to activate the OpenGL context of the window.
    It can be used as `with context:` to activate the OpenGL context of the SLM window.
    Just to be sure, it also globally locks all access to OpenGL, so that only
     one thread can use OpenGL at the same time.
    This class holds a weak ref to the SLM object, so that the SLM object can be garbage collected.
    """

    _lock = threading.RLock()

    def __init__(self, slm):
        if isinstance(slm, self.__class__):
            # copy constructor
            self._slm = slm._slm
            if slm.slm is None:
                raise ValueError("The SLM object has been deleted")
        else:
            # construct a new Context object
            # note that adding the same slm to the set multiple times
            # is not a problem, because _all_slms is a set
            self._slm = weakref.ref(slm)

    def __enter__(self):
        self._lock.acquire()
        slm = self.slm
        if slm is not None:
            glfw.make_context_current(slm._window)  # noqa: ok to use _window
        return slm

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._lock.release()

    @property
    def slm(self) -> SLM:
        return self._slm()
