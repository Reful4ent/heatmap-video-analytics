"""Модели данных для модуля работы с данными прогонов."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RunMetadata:
    """Метаданные одного прогона обработки видео."""

    run_id: str
    source: str
    created_at: str
    target: str
    detection_method: str
    confidence_threshold: float
    iou_threshold: float
    blur_radius: int
    alpha: float
    frame_skip: int
    frame_width: int
    frame_height: int
    total_frames: int
    total_detections: int
    model_name: str


@dataclass
class AnalysisRecord:
    """Результат AI-аналитики прогона."""

    run_id: str
    target: str
    model_id: str
    analysis_text: str
    created_at: str
    duration_sec: Optional[float] = None
    error: Optional[str] = None


@dataclass
class RunSummary:
    """Краткое описание прогона для списка истории."""

    run_id: str
    source: str
    created_at: str
    target: str
    total_frames: int
    total_detections: int
    has_analysis: bool
    has_frame: bool
    has_heatmap: bool
    folder_path: str
