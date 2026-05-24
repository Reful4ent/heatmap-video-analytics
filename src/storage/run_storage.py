"""Модуль работы с данными прогонов: запись и чтение артефактов на диск."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from src.storage.models import AnalysisRecord, RunMetadata, RunSummary

logger = logging.getLogger(__name__)


class RunStorage:
    """Сохраняет и загружает артефакты прогонов обработки видео."""

    FRAME_FILENAME = "frame.png"
    HEATMAP_FILENAME = "heatmap.png"
    METADATA_FILENAME = "metadata.json"
    ANALYSIS_FILENAME = "analysis.json"

    def __init__(self, base_dir: Union[Path, str] = "output/runs") -> None:
        """Инициализирует хранилище прогонов и создаёт корневую директорию при необходимости.

        @param base_dir - Корневая директория для всех прогонов
        @returns None
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def generate_run_id(self, source: str) -> str:
        """Генерирует уникальный идентификатор прогона на основе времени и источника.

        @param source - Путь к видеофайлу или 'webcam'
        @returns {str} run_id формата YYYYMMDD-HHMMSS_<slug>
        """
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        if source.lower().startswith("webcam"):
            slug = "webcam"
        else:
            stem = Path(source).stem
            slug = re.sub(r"[^A-Za-z0-9_-]+", "-", stem).strip("-")[:60] or "video"
        return f"{timestamp}_{slug}"

    def get_run_dir(self, run_id: str) -> Path:
        """Возвращает путь к папке прогона по идентификатору.

        @param run_id - Идентификатор прогона
        @returns {Path} Абсолютный путь к папке прогона
        """
        return self.base_dir / run_id

    def save_frame(self, run_id: str, frame_bgr: np.ndarray) -> Path:
        """Сохраняет последний кадр прогона в PNG.

        @param run_id - Идентификатор прогона
        @param frame_bgr - Кадр в BGR формате
        @returns {Path} Путь к сохранённому файлу
        @throws {RuntimeError} При ошибке записи файла
        """
        run_dir = self.get_run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / self.FRAME_FILENAME
        if not cv2.imwrite(str(path), frame_bgr):
            raise RuntimeError(f"Не удалось сохранить кадр: {path}")
        return path

    def save_heatmap(self, run_id: str, heatmap_bgr: np.ndarray) -> Path:
        """Сохраняет тепловую карту прогона в PNG.

        @param run_id - Идентификатор прогона
        @param heatmap_bgr - Тепловая карта в BGR формате
        @returns {Path} Путь к сохранённому файлу
        @throws {RuntimeError} При ошибке записи файла
        """
        run_dir = self.get_run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / self.HEATMAP_FILENAME
        if not cv2.imwrite(str(path), heatmap_bgr):
            raise RuntimeError(f"Не удалось сохранить тепловую карту: {path}")
        return path

    def save_metadata(self, metadata: RunMetadata) -> Path:
        """Сохраняет метаданные прогона в JSON.

        @param metadata - Метаданные прогона
        @returns {Path} Путь к сохранённому файлу
        """
        run_dir = self.get_run_dir(metadata.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / self.METADATA_FILENAME
        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(metadata), f, ensure_ascii=False, indent=2)
        return path

    def save_analysis(self, analysis: AnalysisRecord) -> Path:
        """Сохраняет результат AI-аналитики прогона в JSON.

        @param analysis - Результат AI-аналитики
        @returns {Path} Путь к сохранённому файлу
        """
        run_dir = self.get_run_dir(analysis.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / self.ANALYSIS_FILENAME
        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(analysis), f, ensure_ascii=False, indent=2)
        return path

    def list_runs(self) -> List[RunSummary]:
        """Возвращает список всех прогонов отсортированный по дате убыванию.

        @returns {List[RunSummary]} Список кратких описаний прогонов
        """
        summaries: List[RunSummary] = []
        if not self.base_dir.exists():
            return summaries
        for run_dir in sorted(self.base_dir.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            summary = self._build_summary(run_dir)
            if summary is not None:
                summaries.append(summary)
        return summaries

    def load_metadata(self, run_id: str) -> Optional[RunMetadata]:
        """Загружает метаданные прогона из JSON.

        @param run_id - Идентификатор прогона
        @returns {Optional[RunMetadata]} Метаданные или None если файла нет
        """
        path = self.get_run_dir(run_id) / self.METADATA_FILENAME
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return RunMetadata(**data)

    def load_analysis(self, run_id: str) -> Optional[AnalysisRecord]:
        """Загружает результат AI-аналитики прогона из JSON.

        @param run_id - Идентификатор прогона
        @returns {Optional[AnalysisRecord]} Результат или None если файла нет
        """
        path = self.get_run_dir(run_id) / self.ANALYSIS_FILENAME
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return AnalysisRecord(**data)

    def get_artifact_paths(self, run_id: str) -> Tuple[Path, Path]:
        """Возвращает пути к кадру и тепловой карте прогона.

        @param run_id - Идентификатор прогона
        @returns {Tuple[Path, Path]} Кортеж (frame_path, heatmap_path)
        """
        run_dir = self.get_run_dir(run_id)
        return run_dir / self.FRAME_FILENAME, run_dir / self.HEATMAP_FILENAME

    def _build_summary(self, run_dir: Path) -> Optional[RunSummary]:
        """Строит RunSummary из папки прогона.

        @param run_dir - Папка прогона
        @returns {Optional[RunSummary]} Сводка или None при отсутствии metadata.json
        """
        metadata_path = run_dir / self.METADATA_FILENAME
        if not metadata_path.exists():
            return None
        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Не удалось прочитать metadata.json в {run_dir}: {e}")
            return None
        return RunSummary(
            run_id=data.get("run_id", run_dir.name),
            source=data.get("source", ""),
            created_at=data.get("created_at", ""),
            target=data.get("target", ""),
            total_frames=int(data.get("total_frames", 0)),
            total_detections=int(data.get("total_detections", 0)),
            has_analysis=(run_dir / self.ANALYSIS_FILENAME).exists(),
            has_frame=(run_dir / self.FRAME_FILENAME).exists(),
            has_heatmap=(run_dir / self.HEATMAP_FILENAME).exists(),
            folder_path=str(run_dir.resolve()),
        )
