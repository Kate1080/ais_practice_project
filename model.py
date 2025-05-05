from ultralytics import YOLO
from preproc import to_rgb
import cv2 as cv


class HorseDetect:
    def __init__( self, model_weights: str = "yolov8n.pt" ):
        self.model = YOLO( model_weights )
        self.class_id = 17

    def detect( self, img ):
        img_rgb = to_rgb( img )

        reses = self.model.predict(
            img_rgb,
            imgsz=640,
            classes=[ self.class_id ],
            verbose=False,
            rect=True,
            conf=0.5
        )

        return self.proc_res( reses )

    def detect_video( self, video_path, skip_frames=0, max_frames=None ):
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError( f"Не удалось открыть видеофайл: { video_path }" )

        detections_by_frame = []
        frame_count = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                current_frame = int( cap.get( cv.CAP_PROP_POS_FRAMES ) )
                if skip_frames > 0 and ( current_frame - 1 ) % ( skip_frames + 1 ) != 0:
                    continue

                detections = self.detect( frame )
                detections_by_frame.append(detections)

                frame_count += 1
                if max_frames is not None and frame_count >= max_frames:
                    break
        finally:
            cap.release()

        return detections_by_frame


    def proc_res( self, reses ):
        detections = []
        for res in reses:
            for box in res.boxes:
                detections.append({
                'bbox': box.xyxy[ 0 ].cpu().numpy().astype( int ).tolist(),
                'confidence': round( box.conf[ 0 ].cpu().item(), 2 ),
            })
        return detections

