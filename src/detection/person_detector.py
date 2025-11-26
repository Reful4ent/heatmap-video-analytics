"""Модуль для детекции людей на видео с использованием YOLO."""

import logging
from typing import List, Optional, Tuple

import numpy as np
from ultralytics import YOLO

from src.config.settings import DetectionConfig

logger = logging.getLogger(__name__)


class PersonDetector:
    """Класс для детекции людей на кадрах с помощью YOLO."""

    def __init__(self, config: DetectionConfig) -> None:
        """Инициализирует детектор людей.

        @param config - Конфигурация детекции
        @throws {RuntimeError} При ошибке загрузки модели
        """
        self.config = config
        try:
            self.model = YOLO(config.model_name)
            logger.info(f"Модель YOLO загружена: {config.model_name}")
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки модели YOLO: {e}") from e

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Детектирует людей на кадре и возвращает bounding boxes.

        @param frame - Входной кадр (BGR формат)
        @returns {List[Tuple[int, int, int, int]]} Список bounding boxes (x1, y1, x2, y2)
        """
        results = self.model.predict(
            frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            classes=[0],  # Класс "person" в COCO
            verbose=False,
        )

        boxes_list: List[Tuple[int, int, int, int]] = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                if self.config.detection_method == "foot":
                    boxes_list.append(self._convert_to_foot_box(x1, y1, x2, y2))
                else:
                    boxes_list.append((int(x1), int(y1), int(x2), int(y2)))

        if boxes_list:
            logger.debug(f"Обнаружено людей: {len(boxes_list)}")

        return boxes_list

    def _convert_to_foot_box(self, x1: float, y1: float, x2: float, y2: float) -> Tuple[int, int, int, int]:
        """Преобразует полный bounding box в bounding box для нижней части (ноги).

        @param x1 - Левая координата
        @param y1 - Верхняя координата
        @param x2 - Правая координата
        @param y2 - Нижняя координата
        @returns {Tuple[int, int, int, int]} Bounding box для нижней части (x1, y1, x2, y2)
        """
        width = x2 - x1
        height = y2 - y1
        
        foot_height = height * 0.3
        foot_y1 = y2 - foot_height
        
        center_x = (x1 + x2) / 2
        foot_width = width * 0.4
        
        foot_x1 = center_x - foot_width / 2
        foot_x2 = center_x + foot_width / 2
        
        return (int(foot_x1), int(foot_y1), int(foot_x2), int(y2))

