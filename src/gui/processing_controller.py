"""Контроллер для координации обработки видео."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from src.config.settings import AppConfig
from src.gui.processing_worker import ProcessingWorker

logger = logging.getLogger(__name__)


class ProcessingController(QObject):
    """Контроллер для управления обработкой видео."""

    processing_started = pyqtSignal()
    processing_stopped = pyqtSignal()
    processing_finished = pyqtSignal()
    run_finalized = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        """Инициализирует контроллер обработки.

        @param parent - Родительский объект
        """
        super().__init__(parent)
        self.worker: Optional[ProcessingWorker] = None
        self.config = AppConfig.default()
        self.reference_frame: Optional[np.ndarray] = None
        self.frame_size: Optional[tuple] = None

    def start_processing(self, source: str) -> None:
        """Запускает обработку видео.

        @param source - Источник видео (путь к файлу или 'webcam')
        @throws {RuntimeError} Если обработка уже запущена
        """
        if self.worker and self.worker.isRunning():
            raise RuntimeError("Обработка уже запущена")

        self.worker = ProcessingWorker(source, self.config)
        self.worker.finished_signal.connect(self._on_processing_finished)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.run_finalized.connect(self.run_finalized.emit)
        self.worker.start()
        self.processing_started.emit()

    def stop_processing(self) -> None:
        """Останавливает обработку видео."""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
            self.processing_stopped.emit()

    def save_heatmap(self, output_path: str, heatmap: np.ndarray) -> None:
        """Сохраняет тепловую карту в файл.

        @param output_path - Путь для сохранения
        @param heatmap - Тепловая карта для сохранения
        @throws {ValueError} Если расширение файла не поддерживается
        """
        try:
            path = Path(output_path)
            
            if not path.suffix:
                path = path.with_suffix(".png")
                logger.warning(f"Расширение файла не указано, используется .png: {path}")
            
            supported_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
            if path.suffix.lower() not in supported_extensions:
                path = path.with_suffix(".png")
                logger.warning(
                    f"Расширение {path.suffix} может не поддерживаться, "
                    f"используется .png: {path}"
                )
            
            path.parent.mkdir(parents=True, exist_ok=True)
            import cv2

            success = cv2.imwrite(str(path), heatmap)
            if not success:
                raise RuntimeError(
                    f"Не удалось сохранить файл. Проверьте путь и права доступа: {path}"
                )
            
            logger.info(f"Тепловая карта сохранена: {path}")
        except Exception as e:
            logger.error(f"Ошибка сохранения тепловой карты: {e}", exc_info=True)
            raise

    def update_config(self, **kwargs) -> None:
        """Обновляет конфигурацию обработки.

        @param kwargs - Параметры конфигурации для обновления
        """
        target = self.config.detection.target
        if "target" in kwargs:
            target = kwargs["target"]

        if "confidence" in kwargs:
            self.config.detection.confidence_threshold = kwargs["confidence"]
        if "blur_radius" in kwargs:
            self.config.heatmap.blur_radius = kwargs["blur_radius"]
        if "alpha" in kwargs:
            self.config.heatmap.alpha = kwargs["alpha"]
        if "frame_skip" in kwargs:
            self.config.video.frame_skip = kwargs["frame_skip"]
        if "detection_method" in kwargs:
            detection_method = kwargs["detection_method"]
            if target == "vehicles" and detection_method == "foot":
                detection_method = "full_body"
            self.config.detection.detection_method = detection_method
        if "target" in kwargs:
            self.config.detection.target = kwargs["target"]

    def _on_processing_finished(self) -> None:
        """Обработчик завершения обработки."""
        self.processing_finished.emit()

    def _on_error(self, error_message: str) -> None:
        """Обработчик ошибки обработки.

        @param error_message - Сообщение об ошибке
        """
        logger.error(f"Ошибка обработки: {error_message}")
        self.processing_stopped.emit()

