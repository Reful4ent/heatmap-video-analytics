"""Модуль для построения тепловых карт на основе детекций людей."""

import logging
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.config.settings import HeatmapConfig

logger = logging.getLogger(__name__)


class HeatmapGenerator:
    """Класс для построения тепловых карт из координат детекций."""

    def __init__(self, config: HeatmapConfig) -> None:
        """Инициализирует генератор тепловых карт.

        @param config - Конфигурация построения тепловых карт
        """
        self.config = config
        self.accumulator: Optional[np.ndarray] = None
        self.frame_shape: Optional[Tuple[int, int]] = None
        self.total_detections: int = 0

    def initialize(self, frame_shape: Tuple[int, int]) -> None:
        """Инициализирует внутреннюю матрицу накопления для заданного размера кадра.

        @param frame_shape - Размер кадра (ширина, высота)
        """
        width, height = frame_shape
        self.frame_shape = (int(width), int(height))
        self.accumulator = np.zeros((int(height), int(width)), dtype=np.float32)
        self.total_detections = 0

    def add_detections(self, boxes: List[Tuple[int, int, int, int]]) -> None:
        """Добавляет bounding boxes детекций в матрицу накопления.

        @param boxes - Список bounding boxes (x1, y1, x2, y2)
        @throws {RuntimeError} Если генератор не инициализирован размером кадра
        """
        if self.accumulator is None or self.frame_shape is None:
            raise RuntimeError(
                "HeatmapGenerator не инициализирован. "
                "Вызовите initialize(frame_shape) перед add_detections()."
            )

        width, height = self.frame_shape
        for x1, y1, x2, y2 in boxes:
            x1_i = max(0, min(int(x1), width))
            y1_i = max(0, min(int(y1), height))
            x2_i = max(0, min(int(x2), width))
            y2_i = max(0, min(int(y2), height))

            if x2_i > x1_i and y2_i > y1_i:
                self.accumulator[y1_i:y2_i, x1_i:x2_i] += 1.0
                self.total_detections += 1

    def clear(self) -> None:
        """Очищает накопленные данные тепловой карты."""
        if self.accumulator is not None:
            self.accumulator.fill(0.0)
        self.total_detections = 0

    def generate(
        self, frame_shape: Tuple[int, int], reference_frame: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Генерирует тепловую карту из накопленной матрицы.

        @param frame_shape - Размер кадра (ширина, высота)
        @param reference_frame - Опциональный кадр для наложения (BGR формат)
        @returns {np.ndarray} Тепловая карта в формате BGR
        @throws {ValueError} Если недостаточно детекций для построения карты
        @throws {RuntimeError} Если генератор не инициализирован
        """
        if self.accumulator is None or self.frame_shape is None:
            raise RuntimeError(
                "HeatmapGenerator не инициализирован. "
                "Вызовите initialize(frame_shape) перед generate()."
            )

        if self.total_detections < self.config.min_detections:
            raise ValueError(
                f"Недостаточно детекций для построения карты: "
                f"{self.total_detections} < {self.config.min_detections}"
            )

        if tuple(map(int, frame_shape)) != tuple(map(int, self.frame_shape)):
            logger.warning(
                f"frame_shape отличается от инициализированного значения: "
                f"{frame_shape} != {self.frame_shape}. Используется self.frame_shape."
            )

        heatmap = self.accumulator.copy()
        heatmap = cv2.GaussianBlur(
            heatmap, (self.config.blur_radius, self.config.blur_radius), 0
        )
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        colormap = plt.get_cmap(self.config.colormap)
        heatmap_colored = colormap(heatmap)[:, :, :3]
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        heatmap_bgr = cv2.cvtColor(heatmap_colored, cv2.COLOR_RGB2BGR)

        if reference_frame is not None:
            overlay = cv2.addWeighted(
                reference_frame, 1 - self.config.alpha, heatmap_bgr, self.config.alpha, 0
            )
            return overlay

        return heatmap_bgr

    def save(
        self, output_path: str, frame_shape: Tuple[int, int], reference_frame: Optional[np.ndarray] = None
    ) -> None:
        """Сохраняет тепловую карту в файл.

        @param output_path - Путь для сохранения изображения
        @param frame_shape - Размер кадра (ширина, высота)
        @param reference_frame - Опциональный кадр для наложения (BGR формат)
        @throws {ValueError} Если недостаточно детекций
        @throws {RuntimeError} Если не удалось сохранить файл
        """
        from pathlib import Path

        heatmap = self.generate(frame_shape, reference_frame)
        
        path = Path(output_path)
        if not path.suffix:
            path = path.with_suffix(".png")
            logger.warning(f"Расширение файла не указано, используется .png: {path}")
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        success = cv2.imwrite(str(path), heatmap)
        if not success:
            raise RuntimeError(
                f"Не удалось сохранить файл. Проверьте путь и права доступа: {path}"
            )
        
        logger.info(f"Тепловая карта сохранена: {path}")

