"""Виджет истории прогонов: список и просмотр артефактов."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from src.gui.video_widget import VideoWidget
from src.storage import RunStorage, RunSummary

logger = logging.getLogger(__name__)


class HistoryWidget(QWidget):
    """Отображает список прогонов и артефакты выбранного прогона."""

    def __init__(
        self, storage: Optional[RunStorage] = None, parent: Optional[QWidget] = None
    ) -> None:
        """Инициализирует виджет истории и сразу подгружает текущий список прогонов.

        @param storage - Хранилище прогонов
        @param parent - Родительский виджет
        @returns None
        """
        super().__init__(parent)
        self._storage = storage if storage is not None else RunStorage()
        self._setup_ui()
        self._connect_signals()
        self.refresh()

    def _setup_ui(self) -> None:
        """Настраивает разметку виджета."""
        root = QVBoxLayout(self)

        header = QHBoxLayout()
        header.addWidget(QLabel("История запусков:"))
        header.addStretch()
        self.refresh_button = QPushButton("Обновить")
        header.addWidget(self.refresh_button)
        root.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        self.runs_list = QListWidget()
        splitter.addWidget(self.runs_list)

        detail = QWidget()
        detail_layout = QVBoxLayout(detail)
        detail_layout.setContentsMargins(0, 0, 0, 0)

        previews = QHBoxLayout()
        self.frame_preview = VideoWidget()
        self.heatmap_preview = VideoWidget()
        previews.addWidget(self.frame_preview)
        previews.addWidget(self.heatmap_preview)
        detail_layout.addLayout(previews)

        self.metadata_label = QLabel("Выберите прогон в списке слева.")
        self.metadata_label.setWordWrap(True)
        detail_layout.addWidget(self.metadata_label)

        detail_layout.addWidget(QLabel("AI-аналитика:"))
        self.analysis_text = QTextBrowser()
        detail_layout.addWidget(self.analysis_text)

        splitter.addWidget(detail)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

    def _connect_signals(self) -> None:
        """Подключает сигналы виджета."""
        self.refresh_button.clicked.connect(self.refresh)
        self.runs_list.currentItemChanged.connect(self._on_item_changed)

    @pyqtSlot()
    def refresh(self) -> None:
        """Перечитывает список прогонов с диска и обновляет UI."""
        self.runs_list.clear()
        try:
            runs = self._storage.list_runs()
        except Exception as e:
            logger.error(f"Не удалось получить список прогонов: {e}", exc_info=True)
            return
        for run in runs:
            item = QListWidgetItem(self._format_run_label(run))
            item.setData(Qt.ItemDataRole.UserRole, run.run_id)
            self.runs_list.addItem(item)
        if self.runs_list.count() > 0:
            self.runs_list.setCurrentRow(0)
        else:
            self._clear_detail(
                "Прогонов пока нет. Запустите обработку видео на вкладке Обработка."
            )

    def _format_run_label(self, run: RunSummary) -> str:
        """Формирует подпись пункта списка для прогона.

        @param run - Сводка прогона
        @returns {str} Текст пункта списка
        """
        source_label = (
            "webcam"
            if run.source.lower().startswith("webcam")
            else Path(run.source).name
        )
        marks = []
        if run.has_analysis:
            marks.append("AI")
        marks_str = f" [{','.join(marks)}]" if marks else ""
        return (
            f"{run.created_at}  •  {source_label}  •  "
            f"кадров {run.total_frames}, детекций {run.total_detections}{marks_str}"
        )

    @pyqtSlot(QListWidgetItem, QListWidgetItem)
    def _on_item_changed(
        self,
        current: Optional[QListWidgetItem],
        _previous: Optional[QListWidgetItem],
    ) -> None:
        """Обработчик смены выбранного пункта списка.

        @param current - Новый выбранный пункт
        @param _previous - Предыдущий выбранный пункт (не используется)
        @returns None
        """
        if current is None:
            self._clear_detail("Прогон не выбран.")
            return
        run_id = current.data(Qt.ItemDataRole.UserRole)
        if not isinstance(run_id, str):
            return
        self._load_run(run_id)

    def _load_run(self, run_id: str) -> None:
        """Загружает и отображает артефакты выбранного прогона.

        @param run_id - Идентификатор прогона
        @returns None
        """
        frame_path, heatmap_path = self._storage.get_artifact_paths(run_id)
        self._display_image(self.frame_preview, frame_path)
        self._display_image(self.heatmap_preview, heatmap_path)

        metadata = self._storage.load_metadata(run_id)
        if metadata is None:
            self.metadata_label.setText("Метаданные отсутствуют.")
        else:
            self.metadata_label.setText(
                f"ID: {metadata.run_id}  |  источник: {metadata.source}  |  "
                f"target: {metadata.target}  |  модель: {metadata.model_name}  |  "
                f"размер: {metadata.frame_width}x{metadata.frame_height}  |  "
                f"conf: {metadata.confidence_threshold}  iou: {metadata.iou_threshold}  "
                f"blur: {metadata.blur_radius}  alpha: {metadata.alpha}  "
                f"frame_skip: {metadata.frame_skip}  |  "
                f"кадров: {metadata.total_frames}  детекций: {metadata.total_detections}"
            )

        analysis = self._storage.load_analysis(run_id)
        if analysis is None:
            self.analysis_text.setMarkdown(
                "_AI-аналитика для этого прогона ещё не сохранена._"
            )
        else:
            self.analysis_text.setMarkdown(analysis.analysis_text)

    def _display_image(self, widget: VideoWidget, image_path: Path) -> None:
        """Загружает PNG/JPG с диска и отображает в VideoWidget.

        @param widget - Виджет для отображения
        @param image_path - Путь к изображению
        @returns None
        """
        if not image_path.exists():
            widget.clear_display()
            return
        image = cv2.imread(str(image_path))
        if image is None:
            widget.clear_display()
            return
        widget.display_frame(image)

    def _clear_detail(self, message: str) -> None:
        """Очищает превью и устанавливает текст-заглушку в области метаданных.

        @param message - Сообщение в области метаданных
        @returns None
        """
        self.frame_preview.clear_display()
        self.heatmap_preview.clear_display()
        self.metadata_label.setText(message)
        self.analysis_text.clear()
