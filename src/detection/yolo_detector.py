"""Модуль универсального детектора объектов на видео с использованием YOLO."""

import logging
from typing import Dict, List, Tuple

import numpy as np
from ultralytics import YOLO

from src.config.settings import DetectionConfig

logger = logging.getLogger(__name__)


class YoloDetector:
    """Детектор объектов на кадрах с помощью YOLO (COCO-классы)."""

    _TARGET_TO_CLASSES: Dict[str, List[int]] = {
        "person": [0],
        "vehicles": [1, 2, 3, 5, 7],  # bicycle, car, motorcycle, bus, truck
    }

    def __init__(self, config: DetectionConfig) -> None:
        """Инициализирует детектор YOLO.

        @param config - Конфигурация детекции
        @throws {ValueError} При неизвестной цели детекции (target)
        @throws {RuntimeError} При ошибке загрузки модели
        """
        self.config = config
        self._classes = self._get_classes_for_target(config.target)
        try:
            self.model = YOLO(config.model_name)
            logger.info(f"Модель YOLO загружена: {config.model_name}")
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки модели YOLO: {e}") from e

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Детектирует объекты на кадре и возвращает bounding boxes.

        @param frame - Входной кадр (BGR формат)
        @returns {List[Tuple[int, int, int, int]]} Список bounding boxes (x1, y1, x2, y2)
        """
        results = self.model.predict(
            frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            classes=self._classes,
            verbose=False,
        )

        boxes_list: List[Tuple[int, int, int, int]] = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                if self.config.target == "person" and self.config.detection_method == "foot":
                    boxes_list.append(self._convert_to_foot_box(x1, y1, x2, y2))
                else:
                    boxes_list.append((int(x1), int(y1), int(x2), int(y2)))

        if boxes_list:
            logger.debug(f"Обнаружено объектов: {len(boxes_list)} (target={self.config.target})")

        return boxes_list

    def _get_classes_for_target(self, target: str) -> List[int]:
        """Возвращает COCO-классы для указанной цели детекции.

        @param target - Цель детекции (например: person, vehicles)
        @returns {List[int]} Список индексов COCO-классов
        @throws {ValueError} Если цель детекции неизвестна
        """
        if target in self._TARGET_TO_CLASSES:
            return self._TARGET_TO_CLASSES[target]
        raise ValueError(
            f"Неизвестная цель детекции: {target}. "
            f"Доступно: {', '.join(sorted(self._TARGET_TO_CLASSES.keys()))}"
        )

    def _convert_to_foot_box(
        self, x1: float, y1: float, x2: float, y2: float
    ) -> Tuple[int, int, int, int]:
        """Преобразует bounding box в область нижней части объекта (foot).

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

