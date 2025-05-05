import cv2
import numpy as np
from typing import List, Dict


def draw_bboxes(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    annotated_image = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det['bbox']

        # Рисуем прямоугольник
        cv2.rectangle(
            annotated_image,
            (x1, y1), (x2, y2),
            color=(0, 255, 0),
            thickness=2
        )

        # Добавляем текст с уверенностью
        label = f"Horse {det['confidence']:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        # Рисуем подложку для текста
        cv2.rectangle(
            annotated_image,
            (x1, y1 - text_height - 5),
            (x1 + text_width, y1),
            color=(0, 255, 0),
            thickness=cv2.FILLED
        )

        # Добавляем текст
        cv2.putText(
            annotated_image,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            thickness=1
        )
    return annotated_image