import time
from threading import Thread
from typing import List, Union
from queue import Queue, Full

import numpy as np
import cv2
import gi
import torch

# Configures to get the specified version of the namespace
# Read more at: https://pygobject.readthedocs.io/en/latest/guide/api/api.html#gi.require_version
gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
gi.require_version("GstVideo", "1.0")

from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import LOGGER

# Although unused, GstApp is needed for using appsink instead of the default sink
# PyCharm may say it can't find a reference to the below modules
# Use Alt+Enter -> generate stubs for binary module X
from gi.repository import Gst, GLib, GstApp, GObject, GstVideo

from .gst_utils import get_appsink

_ = GstApp


class RTPDataLoader:
    out_path = 'runs/exp'

    def __init__(self, gst_command: str, img_size: Union[int, tuple], stride: int = 32, queue_max_frames: int = 100,
                 auto: bool = True, transforms=None):
        """Initializes the following:
            - GStreamer pipeline which reads the input frames
            - The thread that reads from the pipeline, where the first frame is read in a blocking manner
                to make sure the tracker doesn't attempt to start working before the stream starts being received
            - Queue into which the frames are dumped and later read by the tracker

        Args:
            gst_command: the GStreamer command of a pipeline which reads input video from an incoming stream
            img_size: the size to resize the input frame to
                int: the size of the larger side of the video frame - the other side is scaled accordingly to match
                        the side size ratio of the original frame
                tuple: the sizes to resize to
            stride: the multiplicative number by which the side sizes should be able to be divided
            queue_max_frames: maximal number of frames to be stored at any point in time inside the frames queue
            auto: allows the usage of the above-mention stride parameter. Otherwise, resizes just by scaling down to the
                    desired (int) frame size (see img_size)
            transforms: additional data transformations

        """
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride
        self.transforms = transforms
        self.auto = auto
        self._source_img_size = None

        self.imgs_queue, self.frames, self.thread = Queue(maxsize=queue_max_frames), 0, None

        Gst.init(None)
        self.main_loop = GLib.MainLoop()
        self.pipeline = Gst.parse_launch(gst_command)
        self.appsink = get_appsink(self.pipeline)

        self.start_reading_stream()

    def start_reading_stream(self):
        """Starts the run of the daemon thread that reads frames from the RTP stream
        daemon thread = background service thread that runs as a low priority thread and performs
                            background operations

        """
        self.pipeline.set_state(Gst.State.PLAYING)

        # TODO: add some assert to check if pipeline is streaming

        self.update(first_sample=True)
        self.thread = Thread(target=self.update, args=(), daemon=True)
        self.thread.start()

    def update(self, frame_keep_frequency: int = 1, timeout: int = Gst.SECOND, first_sample: bool = False):
        """Continuously reads frames from the appsink of the RTP stream
        Also could be run to read a single frame, which is meant to be run at the start of the reading stream session.
        This is done to block the run of the script until the first frame is received

        Args:
            frame_keep_frequency: keeps every frame_keep_frequency-th frame.
                                    frame_keep_frequency=1 means keeping all frames
            timeout: timeout for waiting for a new frame to be read from the appsink
            first_sample: whether this call is meant only to wait and read the first frame or read the whole stream

        """
        frame_count = 0
        while True:
            sample_frame = self.appsink.try_pull_sample(timeout)  # get sample from sink
            if not sample_frame:
                continue

            frame_count += 1
            if frame_count % frame_keep_frequency == 0:
                frame = RTPDataLoader.gst_to_numpy(sample_frame=sample_frame)
                if self.imgs_queue.full():
                    self.imgs_queue.get()
                self.imgs_queue.put_nowait(frame)
                if first_sample:
                    self._source_img_size = frame.shape[:2]
                    break

            time.sleep(0.0)  # wait time

    @staticmethod
    def gst_to_numpy(sample_frame: Gst.Sample) -> np.ndarray:
        """Converts GStream sample object which represents a video frame to a numpy array
        the video frames are received as YUV420 single channel frames, which are later
        converted to a 3 channel RGB frames

        Args:
            sample_frame: GStream sample object representing a single video frame

        Returns:
            frame_np: the video frame as a numpy array

        """
        buffer = sample_frame.get_buffer()
        caps = sample_frame.get_caps().get_structure(0)

        height, width = caps.get_value('height'), caps.get_value('width')

        dtype = RTPDataLoader.get_np_dtype_from_caps(caps=caps)

        # taking extra 0.5*height since in YUV420 (I420) format, the frame comes as a single channeled frame,
        # but Y takes the 0-100% rows, U takes 101-125% rows and V takes the 126-150% rows
        yuv420_frame = np.ndarray(shape=(height * 3 // 2, width, 1),
                                  buffer=buffer.extract_dup(0, buffer.get_size()),
                                  dtype=dtype)

        rgb_frame = cv2.cvtColor(yuv420_frame, cv2.COLOR_YUV420p2RGB)
        return rgb_frame

    @staticmethod
    def get_np_dtype_from_caps(caps):
        """Returns the dtype of the video frame that was received through the GStreamer

        Args:
            caps: GStreamer caps object, containing frame metadata

        Returns:
            dtype: the data type of the frame received by the stream

        """
        _DTYPES = {
            16: np.int16,
        }

        format_str = caps.get_value('format')
        video_format = GstVideo.VideoFormat.from_string(format_str)

        format_info = GstVideo.VideoFormat.get_info(video_format)
        dtype = _DTYPES.get(format_info.bits, np.uint8)
        return dtype

    @property
    def source_img_size(self):
        return self._source_img_size

    def __iter__(self):
        return self

    def __next__(self) -> (str, np.ndarray, np.ndarray, ):
        """Retrieves a frame from the received frames queue, performs the relevant transformation on the frame

        Returns:
            output_path: path to the folder to save the outputs of the model
            im: transformed numpy frame
            im0: original numpy frame

        """
        if not self.thread.is_alive() or cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        im0 = self.imgs_queue.get()
        if self.transforms:
            im = np.stack([self.transforms(cv2.cvtColor(x, cv2.COLOR_BGR2RGB)) for x in im0])  # transforms
        else:
            im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]  # resize
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous

        return RTPDataLoader.out_path, im, im0, None, ''

    def __len__(self):
        return 1

    def __del__(self):
        torch.cuda.empty_cache()
        self.main_loop.quit()
