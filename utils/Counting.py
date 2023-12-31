import os
import cv2
import sys
HOME = os.getcwd()
SOURCE_VIDEO_PATH = f"{HOME}/正东骑单车.mp4"
sys.path.append(f"{HOME}/ByteTrack")
sys.path.append(f"{HOME}/utils")

from IPython import display
display.clear_output()
import ultralytics
ultralytics.checks()
from IPython import display
display.clear_output()

import yolox
from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass

@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


from IPython import display
display.clear_output()

import supervision

from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.notebook.utils import show_frame_in_notebook
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator

from typing import List

import numpy as np

from utils.general import non_max_suppression

# converts Detections into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))


# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)


# matches our bounding boxes with predictions
def match_detections_with_tracks(
    detections: Detections,
    tracks: List[STrack]
) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)

    tracker_ids = [None] * len(detections)

    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids

# from ultralytics import YOLO
import torch
from models.common import DetectMultiBackend
from utils.torch_utils import select_device, smart_inference_mode

# model.fuse()

VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

# from tqdm.notebook import tqdm
from tqdm import *

def Counting(LINE_START, LINE_END, TARGET_VIDEO_PATH, reshape_size, model, CLASS_ID):

    # dict maping class_id to class_name
    CLASS_NAMES_DICT = model.model.names
    # create BYTETracker instance
    byte_tracker = BYTETracker(BYTETrackerArgs())
    # create VideoInfo instance
    video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    # create frame generator
    generator = get_video_frames_generator(SOURCE_VIDEO_PATH)
    # create LineCounter instance
    line_counter = LineCounter(start=LINE_START, end=LINE_END)
    # create instance of BoxAnnotator and LineCounterAnnotator
    box_annotator = BoxAnnotator(color=ColorPalette(), thickness=4, text_thickness=4, text_scale=2)
    line_annotator = LineCounterAnnotator(thickness=4, text_thickness=4, text_scale=2)

    # open target video file
    with VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
        # loop over video frames
        for frame in tqdm(generator, total=video_info.total_frames):
            # model prediction on single frame and conversion to supervision Detections
            
            im = frame
            interp = cv2.INTER_LINEAR
            im = cv2.resize(im, (reshape_size ,reshape_size), interp).transpose(2, 0, 1)
            frame = im.transpose(1, 2, 0)
            im = im[np.newaxis,:]
            # # 转换维度
            im = torch.from_numpy(im).to(model.device)
            results = model(im)

            # # NMS
            conf_thres = 0.25; iou_thres = 0.45; classes = None; agnostic_nms = False; max_det = 1000
            results = non_max_suppression(results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            # 设置格式
            xyxy = results[:, :4].cpu().numpy()
            confidence = results[:, 4].cpu().numpy()
            class_id = results[:, 5].cpu().numpy()

            detections = Detections(
                xyxy=xyxy,
                confidence=confidence,
                class_id=class_id.astype(int)
            )
            # 暂时注释
            # results = model(frame)
            
            # detections = Detections(
            #     xyxy=results[0].boxes.xyxy.cpu().numpy(),
            #     confidence=results[0].boxes.conf.cpu().numpy(),
            #     class_id=results[0].boxes.cls.cpu().numpy().astype(int)
            # )

            
            # filtering out detections with unwanted classes
            mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
            detections.filter(mask=mask, inplace=True)
            # tracking detections
            tracks = byte_tracker.update(
                output_results=detections2boxes(detections=detections),
                img_info=frame.shape,
                img_size=frame.shape
            )
            tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
            detections.tracker_id = np.array(tracker_id)
            # filtering out detections without trackers
            mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
            detections.filter(mask=mask, inplace=True)
            # format custom labels
            labels = [
                f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                for _, confidence, class_id, tracker_id
                in detections
            ]
            # updating line counter
            line_counter.update(detections=detections)
            # annotate and display frame
            frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
            line_annotator.annotate(frame=frame, line_counter=line_counter)
            sink.write_frame(frame)


if __name__ == "__main__":
    SOURCE_VIDEO_PATH = f"{HOME}/正东骑单车.mp4"
    weights = f"{HOME}\\runs\\train\\exp12\\weights\\best.pt"

    device = select_device(device='0')
    model = DetectMultiBackend(weights, device=device, data='yolov5s.yaml', fp16=True)

    # class_ids of interest - car, motorcycle, bus and truck
    CLASS_ID = [0]
    # settings
    LINE_START = Point(0, 310)
    LINE_END = Point(720, 310)
    reshape_size = 640

    TARGET_VIDEO_PATH = f"{HOME}/正东骑单车out.mp4"

    Counting(LINE_START, LINE_END, SOURCE_VIDEO_PATH, reshape_size, model, CLASS_ID)