"""Модуль конфигурации приложения для построения тепловых карт."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DetectionConfig:
    """Конфигурация для детекции людей."""

    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    model_name: str = "yolov8n.pt"
    device: Optional[str] = None
    detection_method: str = "full_body"  # "full_body" или "foot"


@dataclass
class HeatmapConfig:
    """Конфигурация для построения тепловых карт."""

    blur_radius: int = 25
    colormap: str = "jet"
    alpha: float = 0.5
    min_detections: int = 1


@dataclass
class VideoConfig:
    """Конфигурация для обработки видео."""

    frame_skip: int = 1
    max_frames: Optional[int] = None
    webcam_index: int = 0


@dataclass
class AppConfig:
    """Основная конфигурация приложения."""

    detection: DetectionConfig
    heatmap: HeatmapConfig
    video: VideoConfig

    @classmethod
    def default(cls) -> "AppConfig":
        """Создает конфигурацию с настройками по умолчанию.

        @returns {AppConfig} Конфигурация с настройками по умолчанию
        """
        return cls(
            detection=DetectionConfig(),
            heatmap=HeatmapConfig(),
            video=VideoConfig(),
        )

