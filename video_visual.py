import cv2
import numpy as np
from visualization import draw_bboxes


def save_annotated_video( input_path, detections_by_frame, output_path, fps=30, skip_frames=0 ):
    cap = cv2.VideoCapture( input_path )
    width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH ) )
    height = int( cap.get(cv2.CAP_PROP_FRAME_HEIGHT ) )

    fourcc = cv2.VideoWriter_fourcc( *'avc1' )
    out = cv2.VideoWriter( output_path, fourcc, fps, (width, height) )

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % (skip_frames + 1) != 0:
            frame_idx += 1
            continue

        if frame_idx < len( detections_by_frame ):
            detections = detections_by_frame[ frame_idx ]
            annotated_frame = draw_bboxes( frame, detections )
        else:
            annotated_frame = frame

        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        out.write( annotated_frame )
        frame_idx += 1

    cap.release()
    out.release()