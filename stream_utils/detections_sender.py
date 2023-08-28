import numpy as np

import json
import socket
import time

from queue import Queue
from threading import Thread


class BboxSender:
    """This class handles sending the detections through a socket
    """

    def __init__(self, dest_ip: str, dest_port: int, queue_max_size: int = 100, timeout: float = 10,
                 max_connect_attempts: int = 5):
        self.dest_ip = dest_ip
        self.dest_port = dest_port
        self.sender_socket = None
        self.timeout = timeout
        self.max_connect_attempts = max_connect_attempts

        self.frame_detections_queue = Queue(maxsize=queue_max_size)
        self.create_socket(timeout=timeout)

        self.sender_thread = Thread(target=self.send_bboxes, args=(), daemon=True)
        self.sender_thread.start()

    def create_socket(self, timeout: float) -> None:
        """Tries to create a socket through which the detections will be sent.
        Blocks until the connection is successfully established

        Args:
            timeout: timeout (seconds) for socket operations

        """
        for i in range(self.max_connect_attempts):
            try:
                print(f'\nAttempt ({i + 1}) to connect to {self.dest_ip}:{self.dest_port}')

                self.sender_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sender_socket.settimeout(timeout)
                self.sender_socket.connect((self.dest_ip, self.dest_port))

                print(f'Successfully connected to {self.dest_ip}:{self.dest_port}')
                return
            except socket.error as error:
                print(f"Socket connect to {self.dest_ip}:{self.dest_port} failed with error:", str(error))
                time.sleep(5)

        raise TimeoutError(
            f'\nFailed all {self.max_connect_attempts} attempts to connect to {self.dest_ip}:{self.dest_port}')

    def put_detections_in_queue(self, detections: np.ndarray) -> None:
        """Puts new detections in the detections queue.
        If the queue is full, removes the oldest detections set and puts the new detections set in the queue

        Args:
            detections: numpy array, expected to be of shape (num of bboxes, 7)
                        where for every row we have:
                        - [:4] are (top, left, bottom, right) coordinates
                        - [4] id of the tracked target
                        - [5] label of the tracked target
                        - [6] detection confidence

        """
        if self.frame_detections_queue.full():
            self.frame_detections_queue.get()
        self.frame_detections_queue.put(detections)

    def send_bboxes(self) -> None:
        """Sends through the socket all the detection jsons in the order were put in the queue.
        If for some reason the socket is closed, creates a new socket.

        """
        while True:
            frame_bboxes = self.frame_detections_queue.get()
            self.send_detections_as_json(detections=frame_bboxes)

    def send_detections_as_json(self, detections: np.ndarray) -> None:
        """Sends a json of detections encoded to bytes through a socket

        Args:
            detections: numpy array, expected to be of shape (num of bboxes, 7)
                        where for every row we have:
                        - [:4] are (top, left, bottom, right) coordinates
                        - [4] id of the tracked target
                        - [5] label of the tracked target
                        - [6] detection confidence

        """
        data_payload = BboxSender.convert_np_to_json_bytes(detections=detections)

        self.sender_socket.send(data_payload)

    @staticmethod
    def convert_np_to_json_bytes(detections: np.ndarray) -> bytes:
        """Converts a numpy array to a json packet encoded to bytes

        Args:
            detections: numpy array, expected to be of shape (num of bboxes, 7)
                        where for every row we have:
                        - [:4] are (top, left, bottom, right) coordinates
                        - [4] id of the tracked target
                        - [5] label of the tracked target
                        - [6] detection confidence

        Returns:
            json_bytes: a json encoded into a bytes array.
                        {'tracks': [
                            {'bbox: (top, left, bottom, right),
                            'track_id': int,
                            'label': int,
                            'conf': float},

                            {'bbox: (top, left, bottom, right),
                            'track_id': int,
                            'label': int,
                            'conf': float}]
                        }

        """
        json_dict = {
            'tracks': []
        }

        for detection in detections:
            entry = dict()
            entry['bbox'] = tuple(detection[:4].astype(int).tolist())
            entry['track_id'] = int(detection[4])
            entry['label'] = int(detection[5])
            entry['conf'] = detection[6]

            json_dict['tracks'].append(entry)

        json_string = json.dumps(json_dict)
        json_bytes = json_string.encode()

        return json_bytes
