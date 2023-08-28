from queue import Queue
from threading import Thread
from typing import Tuple

import gi

from fractions import Fraction

import numpy as np
from gstreamer import GstContext, GstPipeline, GstApp, Gst, GstVideo, GLib, GstVideoSink
import gstreamer.utils as utils

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

from .gst_utils import get_appsrc

class RTPStreamSender:
    def __init__(self, source_img_size: Tuple[int, int], framerate: int = 25, host_ip: str = '127.0.0.1',
                 host_port: int = 5003, max_queue_size: int = 100):
        """Sets up stream parameters. The creation of the pipeline is actually done (once) in the first call to
        write_frame when a frame needs to be sent

        Args:
            framerate: the frame at which the frames are sent
            host_ip: IP address to send the frames to
            host_port: port to send the frames to

        """
        self.host_ip = host_ip
        self.host_port = host_port

        self.source_img_size = source_img_size
        self.fps = Fraction(framerate)

        self.pts = 0
        self.duration = 10 ** 9 / (self.fps.numerator / self.fps.denominator)  # frame duration

        self.main_loop = GLib.MainLoop()
        self.pipeline = None
        self.appsrc = None

        self.start_rtp_stream(*self.source_img_size)

        self.stream_queue = Queue(maxsize=max_queue_size)
        self.thread = Thread(target=self.write_frame, args=(), daemon=True)
        self.thread.start()

    def construct_gst_send_command(self, height: int, width: int) -> str:
        """Constructs the GStreamer command to stream through RTP

        Args:
            height: height of the sent frame
            width: width of the sent frame

        Returns:
            gst_command: the full GStreamer command as a single string

        """
        src = f'appsrc emit-signals=True is-live=True block=true format=GST_FORMAT_TIME ' \
              f'caps=video/x-raw,format=BGR,width={width},height={height}'  # ,framerate={self.fps}/1'

        queue = 'queue max-size-buffers=4'
        video_convert = 'videoconvert ! video/x-raw,format=I420'
        encodings = 'x264enc'
        rtp_to_payload = 'rtph264pay'
        send_packages = f'udpsink host={self.host_ip} port={self.host_port}'

        gst_command = ' ! '.join([src, queue, video_convert, encodings, rtp_to_payload, send_packages])

        return gst_command

    def put_frame_in_queue(self, frame: np.ndarray):
        """Puts the given frame inside a queue from which the frame is read and sent through the gstreamer pipeline
        If the queue is full, pops the oldest frame and places the new frame at the end of the queue

        Args:
            frame: frame to be put inside the queue

        """
        if self.stream_queue.full():
            self.stream_queue.get()
        self.stream_queue.put_nowait(frame)

    def start_rtp_stream(self, height: int, width: int):
        """Initializes the GStreamer pipeline

        Args:
            height: height of the sent frame
            width: width of the sent frame

        """
        gst_command = self.construct_gst_send_command(height, width)

        self.pipeline = GstPipeline(gst_command)
        self.pipeline.startup()
        self.appsrc = get_appsrc(self.pipeline.pipeline)

    def write_frame(self):
        """Retrieves a frame from the queue and sends it through the GStreamer pipeline.
        Calls start_rtp_stream to initializes the pipeline in case it hasn't been initialized yet (first frame)

        """
        while True:
            frame = self.stream_queue.get()
            if self.pipeline is None or not self.pipeline.is_active:
                self.start_rtp_stream(*frame.shape[:2])

            gst_buffer = utils.ndarray_to_gst_buffer(frame)

            # set pts and duration to be able to record video, calculate fps
            self.pts += self.duration  # Increase pts by duration
            gst_buffer.pts = self.pts
            gst_buffer.duration = self.duration

            # emit <push-buffer> event with Gst.Buffer
            self.appsrc.emit("push-buffer", gst_buffer)

    def __del__(self):
        """Shuts down the GStreamer mainloop. Also shuts down the pipeline if it wasn't shut down yet

        """
        if self.pipeline:
            self.pipeline.shutdown()
        self.main_loop.quit()


if __name__ == "__main__":
    pass
