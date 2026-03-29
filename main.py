"""Главный модуль приложения для построения тепловых карт."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from src.config.settings import AppConfig
from src.detection.yolo_detector import YoloDetector
from src.heatmap.heatmap_generator import HeatmapGenerator
from src.video.video_loader import VideoLoader

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Парсит аргументы командной строки.

    @returns {argparse.Namespace} Парсированные аргументы
    """
    parser = argparse.ArgumentParser(
        description="Построение тепловых карт для ретейлинга на основе детекции людей"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Путь к видеофайлу или 'webcam' для веб-камеры",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Путь для сохранения тепловой карты (PNG/JPG)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Порог уверенности для детекции (по умолчанию: 0.5)",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Пропускать каждый N-й кадр (по умолчанию: 1)",
    )
    parser.add_argument(
        "--blur-radius",
        type=int,
        default=25,
        help="Радиус размытия для тепловой карты (по умолчанию: 25)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Прозрачность наложения тепловой карты (по умолчанию: 0.5)",
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=["person", "vehicles"],
        default="person",
        help="Цель детекции для тепловой карты: person или vehicles (по умолчанию: person)",
    )

    return parser.parse_args()


def main() -> None:
    """Главная функция приложения."""
    try:
        args = parse_arguments()

        config = AppConfig.default()
        config.detection.confidence_threshold = args.confidence
        config.heatmap.blur_radius = args.blur_radius
        config.heatmap.alpha = args.alpha
        config.video.frame_skip = args.frame_skip
        config.detection.target = args.target

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Начало обработки видео...")

        with VideoLoader(args.input, config.video.frame_skip) as loader:
            frame_size = loader.get_frame_size()
            detector = YoloDetector(config.detection)
            heatmap_gen = HeatmapGenerator(config.heatmap)
            heatmap_gen.initialize(frame_size)

            reference_frame: Optional[np.ndarray] = None
            frame_count = 0

            for frame, frame_num in loader.get_frames():
                if reference_frame is None:
                    reference_frame = frame.copy()

                boxes = detector.detect(frame)
                heatmap_gen.add_detections(boxes)
                frame_count += 1

                if frame_count % 10 == 0:
                    logger.info(f"Обработано кадров: {frame_count}")

            logger.info(f"Всего обработано кадров: {frame_count}")
            logger.info(f"Всего детекций: {heatmap_gen.total_detections}")

            if heatmap_gen.total_detections < config.heatmap.min_detections:
                logger.warning(
                    f"Недостаточно детекций для построения карты: "
                    f"{heatmap_gen.total_detections}"
                )
                sys.exit(1)

            heatmap_gen.save(str(output_path), frame_size, reference_frame)
            logger.info("Обработка завершена успешно")

    except KeyboardInterrupt:
        logger.info("Обработка прервана пользователем")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Ошибка при обработке: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
