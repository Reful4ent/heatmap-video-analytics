"""Виджет для отображения видео и тепловых карт."""

from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QSizePolicy


class VideoWidget(QLabel):
    """Виджет для отображения кадров видео."""

    def __init__(self, parent=None) -> None:
        """Инициализирует виджет видео.

        @param parent - Родительский виджет
        """
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setText("Ожидание видео...")
        self.setStyleSheet("background-color: black; color: white;")

    def display_frame(self, frame: np.ndarray) -> None:
        """Отображает кадр в виджете.

        @param frame - Кадр в формате BGR (OpenCV)
        """
        if frame is None or frame.size == 0:
            return

        height, width = frame.shape[:2]
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_BGR888)

        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.setPixmap(scaled_pixmap)

    def clear_display(self) -> None:
        """Очищает отображение."""
        self.clear()
        self.setText("Ожидание видео...")

