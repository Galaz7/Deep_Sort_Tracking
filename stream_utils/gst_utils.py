import gi

from typing import List

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
gi.require_version("GstVideo", "1.0")

# Although unused, GstApp is needed for using appsink instead of the default sink
# PyCharm may say it can't find a reference to the below modules
# Use Alt+Enter -> generate stubs for binary module X
from gi.repository import Gst, GstApp, GObject


def get_by_cls(pipeline: Gst.Pipeline, cls: GObject.GType) -> List[Gst.Element]:
    """Returns all the elements of the GStreamer pipe of the selected type

    Args:
        pipeline: gst pipeline object
        cls: the type of element to retrieve

    Returns:
        elements: retrieved elements of the desired type

    """
    elements = pipeline.iterate_elements()
    if isinstance(elements, Gst.Iterator):
        # Patch "TypeError: ‘Iterator’ object is not iterable."
        # For versions, we have to get a python iterable object from Gst iterator
        _elements = []
        while True:
            ret, el = elements.next()
            if ret == Gst.IteratorResult(1):  # GST_ITERATOR_OK
                _elements.append(el)
            else:
                break
        elements = _elements

    elements = [e for e in elements if isinstance(e, cls)]
    return elements


def get_appsink(pipeline: Gst.Pipeline):
    """Returns the appsink object of the GStreamer stream
    """
    return get_by_cls(pipeline, GstApp.AppSink).pop(0)


def get_appsrc(pipeline: Gst.Pipeline):
    """Returns the appsink object of the GStreamer stream
    """
    return get_by_cls(pipeline, GstApp.AppSrc).pop(0)
