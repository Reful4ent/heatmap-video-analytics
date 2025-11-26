"""Модуль для загрузки и обработки видео и видеопотоков."""

import logging
import time
from pathlib import Path
from typing import Generator, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoLoader:
    """Класс для загрузки видео из файлов и видеопотоков."""

    def __init__(self, source: str, frame_skip: int = 1) -> None:
        """Инициализирует загрузчик видео.

        @param source - Путь к видеофайлу или 'webcam' для веб-камеры
        @param frame_skip - Пропускать каждый N-й кадр (1 = все кадры)
        """
        self.source = source
        self.frame_skip = frame_skip
        self.cap: Optional[cv2.VideoCapture] = None
        self._frame_count = 0

    def __enter__(self) -> "VideoLoader":
        """Контекстный менеджер для открытия видео."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Контекстный менеджер для закрытия видео."""
        self.close()

    def open(self) -> None:
        """Открывает источник видео.

        @throws {RuntimeError} При ошибке открытия источника видео
        """
        if self.source.lower() == "webcam":
            self.cap = self._open_webcam(0)
        elif self.source.lower().startswith("webcam:"):
            try:
                index = int(self.source.split(":")[1])
                self.cap = self._open_webcam(index)
            except (ValueError, IndexError):
                self.cap = self._open_webcam(0)
        else:
            path = Path(self.source)
            if not path.exists():
                raise FileNotFoundError(f"Видеофайл не найден: {self.source}")
            self.cap = cv2.VideoCapture(str(path))
            if not self.cap or not self.cap.isOpened():
                raise RuntimeError(
                    f"Не удалось открыть видеофайл: {self.source}. "
                    f"Проверьте формат файла и права доступа."
                )

        logger.info(f"Видео источник открыт: {self.source}")

    def _open_webcam(self, index: int) -> cv2.VideoCapture:
        """Открывает webcam с указанным индексом.

        @param index - Индекс камеры
        @returns {cv2.VideoCapture} Объект VideoCapture
        @throws {RuntimeError} Если не удалось открыть камеру
        """
        cap = cv2.VideoCapture(index)
        
        if not cap or not cap.isOpened():
            available_cameras = self._find_available_cameras()
            if available_cameras:
                error_msg = (
                    f"Не удалось открыть камеру с индексом {index}. "
                    f"Доступные камеры: {available_cameras}. "
                    f"Возможные причины: камера занята другим приложением, "
                    f"нет прав доступа или камера не подключена."
                )
            else:
                error_msg = (
                    f"Не удалось открыть камеру с индексом {index}. "
                    f"Камеры не обнаружены. Проверьте подключение камеры и права доступа."
                )
            raise RuntimeError(error_msg)
        
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        time.sleep(0.1)
        
        ret, _ = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError(
                f"Камера {index} открыта, но не может прочитать кадры. "
                f"Проверьте, что камера не используется другим приложением."
            )
        
        return cap

    def _find_available_cameras(self) -> list[int]:
        """Находит доступные камеры.

        @returns {list[int]} Список индексов доступных камер
        """
        available = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap and cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available.append(i)
                cap.release()
        return available

    def close(self) -> None:
        """Закрывает источник видео."""
        if self.cap:
            self.cap.release()
            self.cap = None
            logger.info("Видео источник закрыт")

    def get_frames(self) -> Generator[Tuple[np.ndarray, int], None, None]:
        """Генерирует кадры из видео.

        @yields {Tuple[np.ndarray, int]} Кортеж (кадр, номер кадра)
        @throws {RuntimeError} Если видео не открыто
        """
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("Видео не открыто. Вызовите open() сначала.")

        frame_number = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            if frame_number % self.frame_skip == 0:
                yield frame, frame_number

            frame_number += 1

        logger.info(f"Обработано кадров: {frame_number}")

    def get_frame_size(self) -> Tuple[int, int]:
        """Возвращает размер кадров видео.

        @returns {Tuple[int, int]} Ширина и высота кадра
        @throws {RuntimeError} Если видео не открыто
        """
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("Видео не открыто")

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height

