"""Worker thread для обработки видео в фоновом режиме."""

import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from src.config.settings import AppConfig
from src.ai.qwen_vision_analyzer import QwenVisionAnalyzer, QwenVisionAnalyzerConfig
from src.detection.yolo_detector import YoloDetector
from src.heatmap.heatmap_generator import HeatmapGenerator
from src.storage import AnalysisRecord, RunMetadata, RunStorage
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
    run_finalized = pyqtSignal(str)

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
        self._storage = RunStorage()
        self._run_id: str = self._storage.generate_run_id(source)
        self._run_finalized: bool = False

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
                last_frame: Optional[np.ndarray] = None
                frame_count = 0
                total_detections = 0
                is_webcam = self.source.lower() == "webcam" or self.source.lower().startswith("webcam:")

                self.status_message.emit("Начало обработки...")

                for frame, frame_num in loader.get_frames():
                    if self._should_stop:
                        self.status_message.emit("Остановка обработки...")
                        break

                    last_frame = frame.copy()

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
                if last_frame is not None:
                    last_heatmap: Optional[np.ndarray] = None
                    try:
                        last_heatmap = heatmap_gen.generate(frame_size, last_frame)
                    except ValueError:
                        last_heatmap = None
                    self._finalize_run(
                        frame=last_frame,
                        heatmap=last_heatmap,
                        frame_size=frame_size,
                        frame_count=frame_count,
                        total_detections=total_detections,
                    )
                    if last_heatmap is not None:
                        self._maybe_send_analysis(
                            frame=last_frame, heatmap=last_heatmap, is_last_frame=True
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
            self._save_analysis_record(target=target, report=report, duration_sec=elapsed_sec)

        self._analysis_thread = threading.Thread(target=run_request, daemon=True)
        self._analysis_thread.start()

    def _finalize_run(
        self,
        frame: np.ndarray,
        heatmap: Optional[np.ndarray],
        frame_size: tuple,
        frame_count: int,
        total_detections: int,
    ) -> None:
        """Сохраняет последний кадр и метаданные прогона; тепловую карту сохраняет только при наличии.

        @param frame - Опорный кадр в BGR формате (обязателен)
        @param heatmap - Финальная тепловая карта в BGR формате или None если детекций не было
        @param frame_size - Размер кадра ширина высота
        @param frame_count - Количество обработанных кадров
        @param total_detections - Общее количество детекций
        @returns None
        """
        try:
            self._storage.save_frame(self._run_id, frame)
            if heatmap is not None:
                self._storage.save_heatmap(self._run_id, heatmap)
            metadata = RunMetadata(
                run_id=self._run_id,
                source=self.source,
                created_at=datetime.now(timezone.utc).isoformat(),
                target=self.config.detection.target,
                detection_method=self.config.detection.detection_method,
                confidence_threshold=self.config.detection.confidence_threshold,
                iou_threshold=self.config.detection.iou_threshold,
                blur_radius=self.config.heatmap.blur_radius,
                alpha=self.config.heatmap.alpha,
                frame_skip=self.config.video.frame_skip,
                frame_width=int(frame_size[0]),
                frame_height=int(frame_size[1]),
                total_frames=int(frame_count),
                total_detections=int(total_detections),
                model_name=self.config.detection.model_name,
            )
            self._storage.save_metadata(metadata)
            self._run_finalized = True
            self.run_finalized.emit(self._run_id)
            logger.info(
                f"Прогон сохранён: {self._run_id} (heatmap={'да' if heatmap is not None else 'нет'})"
            )
        except Exception as e:
            logger.error(f"Не удалось сохранить прогон {self._run_id}: {e}", exc_info=True)

    def _save_analysis_record(self, target: str, report: str, duration_sec: float) -> None:
        """Сохраняет результат AI-аналитики в JSON-файл прогона.

        @param target - Цель детекции
        @param report - Текстовый отчёт от модели
        @param duration_sec - Длительность запроса в секундах
        @returns None
        """
        try:
            record = AnalysisRecord(
                run_id=self._run_id,
                target=target,
                model_id=self._analyzer.config.model_id if self._analyzer else "",
                analysis_text=report,
                created_at=datetime.now(timezone.utc).isoformat(),
                duration_sec=float(duration_sec),
            )
            self._storage.save_analysis(record)
            self.run_finalized.emit(self._run_id)
            logger.info(f"AI-аналитика сохранена для {self._run_id}")
        except Exception as e:
            logger.error(f"Не удалось сохранить аналитику {self._run_id}: {e}", exc_info=True)

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

