"""Worker thread для обработки видео в фоновом режиме."""

import logging
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from src.config.settings import AppConfig
from src.detection.person_detector import PersonDetector
from src.heatmap.heatmap_generator import HeatmapGenerator
from src.video.video_loader import VideoLoader

logger = logging.getLogger(__name__)


class ProcessingWorker(QThread):
    """Worker thread для обработки видео."""

    frame_ready = pyqtSignal(np.ndarray)
    heatmap_ready = pyqtSignal(np.ndarray)
    progress_updated = pyqtSignal(int, int)
    status_message = pyqtSignal(str)
    finished_signal = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, source: str, config: AppConfig, parent=None) -> None:
        """Инициализирует worker thread.

        @param source - Источник видео (путь к файлу или 'webcam')
        @param config - Конфигурация приложения
        @param parent - Родительский объект
        """
        super().__init__(parent)
        self.source = source
        self.config = config
        self._is_running = False
        self._should_stop = False

    def stop(self) -> None:
        """Останавливает обработку."""
        self._should_stop = True

    def run(self) -> None:
        """Запускает обработку видео в фоновом потоке."""
        try:
            self._is_running = True
            self._should_stop = False

            self.status_message.emit("Инициализация детектора...")
            detector = PersonDetector(self.config.detection)
            heatmap_gen = HeatmapGenerator(self.config.heatmap)

            self.status_message.emit(f"Открытие источника видео: {self.source}")
            loader = VideoLoader(self.source, self.config.video.frame_skip)
            loader.open()

            try:
                frame_size = loader.get_frame_size()
                reference_frame: Optional[np.ndarray] = None
                frame_count = 0
                total_detections = 0
                is_webcam = self.source.lower() == "webcam" or self.source.lower().startswith("webcam:")

                self.status_message.emit("Начало обработки...")

                for frame, frame_num in loader.get_frames():
                    if self._should_stop:
                        self.status_message.emit("Остановка обработки...")
                        break

                    if reference_frame is None:
                        reference_frame = frame.copy()

                    boxes = detector.detect(frame)
                    heatmap_gen.add_detections(boxes)
                    total_detections += len(boxes)
                    frame_count += 1

                    frame_with_boxes = self._draw_detections(frame.copy(), boxes)
                    self.frame_ready.emit(frame_with_boxes)

                    if frame_count % 5 == 0 or len(heatmap_gen.detections) > 0:
                        try:
                            heatmap = heatmap_gen.generate(frame_size, frame)
                            self.heatmap_ready.emit(heatmap)
                        except ValueError:
                            pass

                    self.progress_updated.emit(frame_count, total_detections)
                    self.status_message.emit(
                        f"Обработано кадров: {frame_count}, Детекций: {total_detections}"
                    )

                    if is_webcam and frame_count % 100 == 0:
                        self.status_message.emit(
                            f"Обработка в реальном времени... Кадров: {frame_count}, Детекций: {total_detections}"
                        )

                self.status_message.emit(
                    f"Обработка завершена. Кадров: {frame_count}, Детекций: {total_detections}"
                )

            finally:
                loader.close()

            self._is_running = False
            self.finished_signal.emit()

        except Exception as e:
            error_msg = f"Ошибка при обработке: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.error_occurred.emit(error_msg)
            self._is_running = False

    def _draw_detections(self, frame: np.ndarray, boxes: list) -> np.ndarray:
        """Рисует детекции на кадре.

        @param frame - Кадр для рисования
        @param boxes - Список bounding boxes (x1, y1, x2, y2)
        @returns {np.ndarray} Кадр с нарисованными детекциями
        """
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return frame

