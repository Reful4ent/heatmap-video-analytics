"""Worker thread для обработки видео в фоновом режиме."""

import logging
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from src.config.settings import AppConfig
from src.ai.qwen_vision_analyzer import QwenVisionAnalyzer, QwenVisionAnalyzerConfig
from src.detection.yolo_detector import YoloDetector
from src.heatmap.heatmap_generator import HeatmapGenerator
from src.video.video_loader import VideoLoader

logger = logging.getLogger(__name__)


class ProcessingWorker(QThread):
    """Worker thread для обработки видео."""

    frame_ready = pyqtSignal(np.ndarray)
    heatmap_ready = pyqtSignal(np.ndarray)
    analysis_ready = pyqtSignal(str)
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
        self._analyzer: Optional[QwenVisionAnalyzer] = None
        self._analysis_thread: Optional[threading.Thread] = None
        self._analysis_log_path = self._build_analysis_log_path(source)

    def stop(self) -> None:
        """Останавливает обработку."""
        self._should_stop = True

    def run(self) -> None:
        """Запускает обработку видео в фоновом потоке."""
        try:
            self._is_running = True
            self._should_stop = False

            self.status_message.emit("Инициализация детектора...")
            detector = YoloDetector(self.config.detection)
            heatmap_gen = HeatmapGenerator(self.config.heatmap)

            self.status_message.emit(f"Открытие источника видео: {self.source}")
            loader = VideoLoader(self.source, self.config.video.frame_skip)
            loader.open()

            try:
                frame_size = loader.get_frame_size()
                heatmap_gen.initialize(frame_size)
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
                if reference_frame is not None:
                    try:
                        last_heatmap = heatmap_gen.generate(frame_size, reference_frame)
                        self._maybe_send_analysis(frame=reference_frame, heatmap=last_heatmap, is_last_frame=True)
                    except ValueError:
                        pass

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

    def _maybe_send_analysis(self, frame: np.ndarray, heatmap: np.ndarray, is_last_frame: bool) -> None:
        """Отправляет анализ в Qwen на последнем кадре.

        @param frame - Исходный кадр (BGR)
        @param heatmap - Тепловая карта/overlay (BGR)
        @param is_last_frame - Признак финального отчёта
        """
        if not is_last_frame:
            return

        if self._analyzer is None:
            self._analyzer = QwenVisionAnalyzer(QwenVisionAnalyzerConfig())

        if self._analysis_thread is not None and self._analysis_thread.is_alive():
            return

        target = self.config.detection.target
        frame_copy = frame.copy()
        heatmap_copy = heatmap.copy()

        self.status_message.emit("AI анализ (финальный): отправка запроса (async)...")
        started_at = time.monotonic()

        def run_request() -> None:
            report = self._analyzer.analyze(
                target=target,
                frame_bgr=frame_copy,
                heatmap_bgr=heatmap_copy,
            )
            elapsed_sec = time.monotonic() - started_at
            self.status_message.emit(f"AI анализ (финальный): готово за {elapsed_sec:.1f}с")

            message = f"[AI финальный] target={target}\n{report}\n"
            self.analysis_ready.emit(message)
            self._append_analysis_log(message)

        self._analysis_thread = threading.Thread(target=run_request, daemon=True)
        self._analysis_thread.start()

    def _build_analysis_log_path(self, source: str) -> Path:
        """Строит путь до файла логов анализа.

        @param source - Источник видео
        @returns {Path} Путь до файла логов
        """
        out_dir = Path("output")
        safe_name = "webcam" if source.lower().startswith("webcam") else Path(source).stem
        return out_dir / f"analysis-{safe_name}.log"

    def _append_analysis_log(self, message: str) -> None:
        """Добавляет сообщение анализа в файл.

        @param message - Сообщение для записи
        """
        try:
            self._analysis_log_path.parent.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            with self._analysis_log_path.open("a", encoding="utf-8") as f:
                f.write(f"{ts}\n{message}\n")
        except Exception as e:
            logger.error(f"Не удалось записать analysis log: {e}", exc_info=True)

