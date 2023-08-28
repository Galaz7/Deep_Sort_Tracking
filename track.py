import os
import sys

# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import logging
import torch
import numpy as np
import torch.backends.cudnn as cudnn

from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strong_sort'))  # add strong_sort ROOT to PATH
if str(ROOT / 'trackers' / 'ocsort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'ocsort'))  # add strong_sort ROOT to PATH
if str(ROOT / 'trackers' / 'strong_sort' / 'deep' / 'reid' / 'torchreid') not in sys.path:
    sys.path.append(
        str(ROOT / 'trackers' / 'strong_sort' / 'deep' / 'reid' / 'torchreid'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import VID_FORMATS, LoadImages
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, increment_path, strip_optimizer, colorstr, print_args,
                                  check_file)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from trackers.multi_tracker_zoo import create_tracker

from stream_utils.RTPDataLoader import RTPDataLoader
from stream_utils.RTPStreamSender import RTPStreamSender
from stream_utils.detections_sender import BboxSender
from stream_utils.consts import ResultsSendMethod

# remove duplicated stream handler to avoid duplicated logging
logging.getLogger().removeHandler(logging.getLogger().handlers[0])


@torch.no_grad()
def run(
        source='0',
        yolo_weights=WEIGHTS / 'yolov5x.pt',  # model.pt path(s),
        reid_weights=WEIGHTS / 'best.pt',  # model.pt path,
        tracking_method='strongsort',
        image_size=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_vid=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project_name=ROOT / 'runs/track',  # save results to project_name/video_dir_name
        video_dir_name='exp',  # save results to project_name/video_dir_name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        eval=False,  # run multi-gpu eval
        stream_tracking_results=ResultsSendMethod.NO_SEND,  # Streams tracking results through RTP if True
        stream_tracking_fps=25,  # FPS of the output RTP stream, relevant only if stream_tracking_results is True
        stream_tracking_ip='127.0.0.1',  # IP address to stream the tracking results to
        stream_tracking_port=5003,  # Port to stream the tracking results to
        yolo_rate=1,  # Uses YOLO detection every 'yolo_rate' frames
        frame_queue_max_size=100,  # Maximum number of incoming frames the queue can hold
        estimate_camera_motion=False,  # use camera motion estimation algorithm to enhance tracking
):
    # Finds the type of the input
    source = str(source)
    yolo_rate = int(yolo_rate)
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_rtp_stream = source.startswith('udpsrc')

    # Makes sure that the file exists
    if is_file:
        source = check_file(source)

    # Directories
    save_dir_path = get_save_path(yolo_weights_path=yolo_weights, reid_weights_path=reid_weights,
                                  video_dir_name=video_dir_name, exist_ok=exist_ok, project_name=project_name,
                                  save_txt=save_txt)

    # Load model
    if eval:
        device = torch.device(int(device))
    else:
        device = select_device(device)
    model = DetectMultiBackend(yolo_weights, device=device, dnn=dnn, data=None, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    image_size = check_img_size(image_size, s=stride)  # check image size

    show_vid = check_imshow() and show_vid

    # Creates a StrongSort tracker instance
    tracker = create_tracker(tracking_method, reid_weights, device, half)

    if hasattr(tracker, 'model'):
        if hasattr(tracker.model, 'warmup'):
            tracker.model.warmup()

    # Dataloader
    dataset = load_dataset(source=source, image_size=image_size, stride=stride, auto_pad=pt,
                           is_rtp_stream=is_rtp_stream, frame_queue_max_size=frame_queue_max_size)

    # Results sender
    results_sender = None
    if stream_tracking_results == ResultsSendMethod.FRAMES_WITH_DETECTIONS.value:
        results_sender = RTPStreamSender(source_img_size=dataset.source_img_size, framerate=stream_tracking_fps,
                                         host_ip=stream_tracking_ip, host_port=stream_tracking_port)
    elif stream_tracking_results == ResultsSendMethod.ONLY_DETECTIONS.value:
        results_sender = BboxSender(dest_ip=stream_tracking_ip, dest_port=stream_tracking_port,
                                    queue_max_size=frame_queue_max_size)

    vid_path, vid_writer, txt_path = None, None, None

    # Run tracking
    # model.warmup(image_size=(1, 3, *image_size))  # warmup
    elapsed_time, num_frames_seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frame, prev_frame = None, None
    resized_im_size = None
    pred = None
    last_detections = None
    outputs = []
    for frame_idx, (out_path, input_frame, source_frame, vid_cap, frame_text) in enumerate(dataset):
        yolo_time = 0.0

        if not resized_im_size:
            resized_im_size = input_frame.shape[1:]

        if frame_idx % yolo_rate == 0:
            t1 = time_sync()
            transformed_frame = transform_input_frame(input_frame=input_frame, device=device, half=half)
            t2 = time_sync()
            elapsed_time[0] += t2 - t1

            # Inference
            visualize = increment_path(save_dir_path / Path(out_path[0]).stem, mkdir=True) if visualize else False
            predictions = model(transformed_frame, augment=augment, visualize=visualize)
            t3 = time_sync()
            yolo_time = t3 - t2
            elapsed_time[1] += yolo_time

            # Apply NMS
            predictions = non_max_suppression(predictions, conf_thres, iou_thres, classes, agnostic_nms,
                                              max_det=max_det)
            elapsed_time[2] += time_sync() - t3
            last_detections = predictions[
                0]  # Currently NMS works on one frame and we don't use batch processing option.
            # TODO: Check NMS batch processing options

            transformed_frame.detach().cpu()

        # Process detections
        num_frames_seen += 1
        detections = last_detections.clone()
        p, im0, _ = out_path, source_frame.copy(), getattr(dataset, 'frame', 0)
        p = Path(p)  # to Path
        # video file
        if source.endswith(VID_FORMATS):
            txt_file_name = p.stem
            save_path = str(save_dir_path / p.name)  # im.jpg, vid.mp4, ...
        # folder with imgs
        else:
            txt_file_name = p.parent.name  # get folder name containing current img
            save_path = str(save_dir_path / p.parent.name)  # im.jpg, vid.mp4, ...
        curr_frame = im0

        txt_path = str(save_dir_path / 'tracks' / txt_file_name)  # im.txt
        frame_text += '%gx%g ' % resized_im_size  # print string
        imc = im0.copy() if save_crop else im0  # for save_crop

        annotator = Annotator(im0, line_width=line_thickness, pil=not ascii)

        if estimate_camera_motion:
            if prev_frame is not None and curr_frame is not None:  # camera motion compensation
                if hasattr(tracker, 'tracker') and hasattr(tracker.tracker, 'camera_update'):
                    tracker.tracker.camera_update(prev_frame, curr_frame)

        if detections is not None and len(detections):
            # Rescale boxes from img_size to im0 size
            detections[:, :4] = scale_coords(resized_im_size, detections[:, :4], im0.shape).round()  # xyxy

            # Print results
            for c in detections[:, -1].unique():
                n = (detections[:, -1] == c).sum()  # detections per class
                frame_text += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # pass detections to strongsort
            t4 = time_sync()
            outputs = tracker.update(detections.cpu(), im0)
            t5 = time_sync()
            elapsed_time[3] += t5 - t4

            # draw boxes for visualization
            if len(outputs) > 0:
                for j, (output, conf) in enumerate(zip(outputs, detections[:, 4])):

                    bboxes = output[0:4]
                    id = output[4]
                    cls = output[5]

                    if save_txt:
                        # to MOT format
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2] - output[0]
                        bbox_h = output[3] - output[1]
                        # Write MOT compliant results to file
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                           bbox_top, bbox_w, bbox_h, -1, -1, -1))

                    if save_vid or save_crop or show_vid or stream_tracking_results:  # Add bbox to image
                        c = int(cls)  # integer class
                        id = int(id)  # integer id
                        label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                                              (
                                                                  f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                        annotator.box_label(bboxes, label, color=colors(c, True))
                        if save_crop:
                            txt_file_name = txt_file_name if (isinstance(out_path, list) and len(out_path) > 1) else ''
                            save_one_box(bboxes, imc, file=save_dir_path / 'crops' / txt_file_name / names[
                                c] / f'{id}' / f'{p.stem}.jpg', BGR=True)

            LOGGER.info(f'{frame_text}Done. yolo:({yolo_time:.3f}s), {tracking_method}:({t5 - t4:.3f}s)')

        else:
            # strongsort_list[i].increment_ages()
            outputs = np.array([])
            LOGGER.info('No detections')

        # Stream results
        im0 = annotator.result()
        if show_vid:
            cv2.imshow(str(p), im0)
            cv2.waitKey(1)  # 1 millisecond

        if stream_tracking_results == ResultsSendMethod.FRAMES_WITH_DETECTIONS.value:
            results_sender.put_frame_in_queue(frame=im0)
        elif stream_tracking_results == ResultsSendMethod.ONLY_DETECTIONS.value:
            results_sender.put_detections_in_queue(detections=outputs)

        # Save results (image with detections)
        if save_vid:
            if vid_path != save_path:  # new video
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer
                if vid_cap:  # video
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            vid_writer.write(im0)

        prev_frame = curr_frame

    # Print results
    t = tuple(x / num_frames_seen * 1E3 for x in elapsed_time)  # speeds per image
    LOGGER.info(
        f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms {tracking_method} update per image at shape {(1, 3, *image_size)}' % t)
    if save_txt or save_vid:
        frame_text = f"\n{len(list(save_dir_path.glob('tracks/*.txt')))} tracks saved to {save_dir_path / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir_path)}{frame_text}")
    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)


def get_save_path(yolo_weights_path: Path, reid_weights_path: Path, video_dir_name: str, exist_ok: bool,
                  project_name: Path, save_txt: bool):
    """Construct the path to save the resultant video to

    Args:
        yolo_weights_path: path to the yolov5 weights
        reid_weights_path: path to the ReID weights
        video_dir_name: name of the directory to save the video to
        project_name: name of the directory of the project
        save_txt: save textual output if True
        exist_ok: allow override if a run of the current number has already happened

    Returns:
        save_dir: directory path to save the resultant video to

    """
    if not isinstance(yolo_weights_path, list):  # single yolo model
        exp_name = yolo_weights_path.stem
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = video_dir_name if video_dir_name else exp_name + "_" + reid_weights_path.stem
    save_dir = increment_path(Path(project_name) / exp_name, exist_ok=exist_ok)  # increment run
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    return save_dir


def load_dataset(source: str, image_size: int, stride: int, is_rtp_stream: bool, auto_pad: bool,
                 frame_queue_max_size: int = None):
    """Loads the proper dataset for the type of input received

    Args:
        source: the source of the input (file, rtsp, RTP, etc...)
        image_size: size of the resized image to be used for inference
        stride: the stride of the model
        is_rtp_stream: True if the input video source is an RTP stream
        auto_pad:

    Returns:
        dataset: dataloader for the specified source

    """
    if is_rtp_stream:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = RTPDataLoader(source, img_size=image_size, stride=stride, auto=auto_pad,
                                queue_max_frames=frame_queue_max_size)
    else:
        dataset = LoadImages(source, img_size=image_size, stride=stride, auto=auto_pad)

    return dataset


def transform_input_frame(input_frame: np.ndarray, device, half: bool):
    transformed_frame = torch.from_numpy(input_frame).to(device)
    transformed_frame = transformed_frame.half() if half else transformed_frame.float()  # uint8 to fp16/32
    transformed_frame /= 255.0  # 0 - 255 to 0.0 - 1.0
    if len(transformed_frame.shape) == 3:
        transformed_frame = transformed_frame[None]  # expand for batch dim

    return transformed_frame


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=Path, default=WEIGHTS / 'yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--reid-weights', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--tracking-method', type=str, default='strongsort', help='strongsort, ocsort, bytetrack')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob')
    parser.add_argument('--image_size', '--img', '--img-size', nargs='+', type=int, default=[640],
                        help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project_name', default=ROOT / 'runs/track',
                        help='save results to project_name/video_dir_name')
    parser.add_argument('--video_dir_name', default='exp', help='save results to project_name/video_dir_name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--eval', action='store_true', help='run evaluation')
    parser.add_argument('--stream-tracking-results', type=int, default=0,
                        help="Results streaming mode. 0 - doesn't stream, 1 - sends frames with detections through "
                             "GStreamer, 2 - sends only detections through a socket")
    parser.add_argument('--stream-tracking-fps', type=int, default=25,
                        help='FPS of the output RTP stream, relevant only if stream_tracking_results is True')
    parser.add_argument('--stream-tracking-ip', type=str, default='127.0.0.1',
                        help='IP address to stream the tracking results to')
    parser.add_argument('--stream-tracking-port', type=int, default=5003,
                        help='Port to stream the tracking results to')
    parser.add_argument('--yolo-rate', default=1, help='The frame interval between 2 runs on the yolo')
    parser.add_argument('--frame-queue-max-size', type=int, default=25,
                        help='Maximum number of incoming frames the queue can hold')
    parser.add_argument('--estimate_camera_motion', default=False, action='store_true',
                        help='use camera motion estimation algorithm to enhance tracking')
    opt = parser.parse_args()
    opt.image_size *= 2 if len(opt.image_size) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
