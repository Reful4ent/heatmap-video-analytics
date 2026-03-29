"""Главное окно GUI приложения."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSplitter,
    QSlider,
    QSpinBox,
    QStatusBar,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from src.gui.processing_controller import ProcessingController
from src.gui.video_widget import VideoWidget

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Главное окно приложения."""

    def __init__(self) -> None:
        """Инициализирует главное окно."""
        super().__init__()
        self.controller = ProcessingController()
        self.current_heatmap: Optional[np.ndarray] = None
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Настраивает интерфейс пользователя."""
        self.setWindowTitle("Тепловые карты для ретейлинга")
        self.setMinimumSize(1200, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        control_panel = self._create_control_panel()
        main_layout.addWidget(control_panel)

        self.splitter = QSplitter(Qt.Orientation.Vertical)
        main_layout.addWidget(self.splitter)

        video_container = QWidget()
        video_layout = QHBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        self.video_widget = VideoWidget()
        self.heatmap_widget = VideoWidget()
        video_layout.addWidget(self.video_widget)
        video_layout.addWidget(self.heatmap_widget)
        self.splitter.addWidget(video_container)

        analysis_container = QWidget()
        analysis_layout = QVBoxLayout(analysis_container)
        analysis_layout.setContentsMargins(0, 0, 0, 0)

        self.toggle_ai_button = QPushButton("Скрыть AI анализ")
        self.toggle_ai_button.setCheckable(True)
        self.toggle_ai_button.setChecked(True)
        self.toggle_ai_button.toggled.connect(self._on_toggle_ai_panel)
        analysis_layout.addWidget(self.toggle_ai_button)

        self.analysis_label = QLabel("AI анализ (GPT-4o):")
        analysis_layout.addWidget(self.analysis_label)

        self._analysis_md_buffer = ""
        self.analysis_text = QTextBrowser()
        self.analysis_text.setOpenExternalLinks(False)
        analysis_layout.addWidget(self.analysis_text)

        self.splitter.addWidget(analysis_container)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Готово к работе")

    def _create_control_panel(self) -> QWidget:
        """Создает панель управления.

        @returns {QWidget} Виджет панели управления
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)

        source_group = self._create_source_selection()
        layout.addWidget(source_group)

        settings_group = self._create_settings_panel()
        layout.addWidget(settings_group)

        buttons_layout = QHBoxLayout()
        self.start_button = QPushButton("Старт")
        self.stop_button = QPushButton("Стоп")
        self.stop_button.setEnabled(False)
        self.save_button = QPushButton("Сохранить тепловую карту")
        self.save_button.setEnabled(False)

        buttons_layout.addWidget(self.start_button)
        buttons_layout.addWidget(self.stop_button)
        buttons_layout.addWidget(self.save_button)
        layout.addLayout(buttons_layout)

        return panel

    def _create_source_selection(self) -> QWidget:
        """Создает группу выбора источника видео.

        @returns {QWidget} Виджет выбора источника
        """
        group = QWidget()
        layout = QVBoxLayout(group)

        layout.addWidget(QLabel("Источник видео:"))

        self.file_radio = QRadioButton("Видеофайл")
        self.file_radio.setChecked(True)
        file_layout = QHBoxLayout()
        file_layout.addWidget(self.file_radio)
        self.file_path_label = QLabel("Не выбран")
        self.file_button = QPushButton("Выбрать файл")
        file_layout.addWidget(self.file_path_label)
        file_layout.addWidget(self.file_button)
        layout.addLayout(file_layout)

        self.webcam_radio = QRadioButton("Webcam")
        webcam_layout = QHBoxLayout()
        webcam_layout.addWidget(self.webcam_radio)
        webcam_layout.addWidget(QLabel("Индекс:"))
        self.webcam_index = QSpinBox()
        self.webcam_index.setMinimum(0)
        self.webcam_index.setMaximum(10)
        self.webcam_index.setValue(0)
        webcam_layout.addWidget(self.webcam_index)
        webcam_layout.addStretch()
        layout.addLayout(webcam_layout)

        return group

    def _create_settings_panel(self) -> QWidget:
        """Создает панель настроек.

        @returns {QWidget} Виджет панели настроек
        """
        group = QWidget()
        layout = QVBoxLayout(group)

        layout.addWidget(QLabel("Настройки:"))

        detection_target_layout = QVBoxLayout()
        detection_target_layout.addWidget(QLabel("Объект детекции:"))
        self.detection_target_group = QButtonGroup()
        self.person_radio = QRadioButton("Люди (Person)")
        self.person_radio.setChecked(True)
        self.vehicles_radio = QRadioButton("Транспорт (Vehicles)")
        self.detection_target_group.addButton(self.person_radio, 0)
        self.detection_target_group.addButton(self.vehicles_radio, 1)
        detection_target_layout.addWidget(self.person_radio)
        detection_target_layout.addWidget(self.vehicles_radio)
        layout.addLayout(detection_target_layout)

        self.detection_method_container = QWidget()
        detection_method_layout = QVBoxLayout(self.detection_method_container)
        detection_method_layout.addWidget(QLabel("Метод детекции:"))
        self.detection_method_group = QButtonGroup()
        self.full_body_radio = QRadioButton("Полный силуэт (Full Body)")
        self.full_body_radio.setChecked(True)
        self.foot_radio = QRadioButton("Нижняя часть (Foot)")
        self.detection_method_group.addButton(self.full_body_radio, 0)
        self.detection_method_group.addButton(self.foot_radio, 1)
        detection_method_layout.addWidget(self.full_body_radio)
        detection_method_layout.addWidget(self.foot_radio)
        layout.addWidget(self.detection_method_container)

        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("Confidence:"))
        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setMinimum(0)
        self.confidence_slider.setMaximum(100)
        self.confidence_slider.setValue(50)
        self.confidence_value = QLabel("0.50")
        confidence_layout.addWidget(self.confidence_slider)
        confidence_layout.addWidget(self.confidence_value)
        layout.addLayout(confidence_layout)

        blur_layout = QHBoxLayout()
        blur_layout.addWidget(QLabel("Blur Radius:"))
        self.blur_spinbox = QSpinBox()
        self.blur_spinbox.setMinimum(1)
        self.blur_spinbox.setMaximum(100)
        self.blur_spinbox.setValue(25)
        blur_layout.addWidget(self.blur_spinbox)
        layout.addLayout(blur_layout)

        alpha_layout = QHBoxLayout()
        alpha_layout.addWidget(QLabel("Alpha:"))
        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setMinimum(0)
        self.alpha_slider.setMaximum(100)
        self.alpha_slider.setValue(50)
        self.alpha_value = QLabel("0.50")
        alpha_layout.addWidget(self.alpha_slider)
        alpha_layout.addWidget(self.alpha_value)
        layout.addLayout(alpha_layout)

        frame_skip_layout = QHBoxLayout()
        frame_skip_layout.addWidget(QLabel("Frame Skip:"))
        self.frame_skip_spinbox = QSpinBox()
        self.frame_skip_spinbox.setMinimum(1)
        self.frame_skip_spinbox.setMaximum(30)
        self.frame_skip_spinbox.setValue(1)
        frame_skip_layout.addWidget(self.frame_skip_spinbox)
        layout.addLayout(frame_skip_layout)

        return group

    def _connect_signals(self) -> None:
        """Подключает сигналы и слоты."""
        self.start_button.clicked.connect(self._on_start_clicked)
        self.stop_button.clicked.connect(self._on_stop_clicked)
        self.save_button.clicked.connect(self._on_save_clicked)
        self.file_button.clicked.connect(self._on_select_file)

        self.confidence_slider.valueChanged.connect(self._on_confidence_changed)
        self.alpha_slider.valueChanged.connect(self._on_alpha_changed)

        self.person_radio.toggled.connect(self._on_detection_target_changed)
        self.vehicles_radio.toggled.connect(self._on_detection_target_changed)
        self._on_detection_target_changed()

        self.controller.processing_started.connect(self._on_processing_started)
        self.controller.processing_stopped.connect(self._on_processing_stopped)
        self.controller.processing_finished.connect(self._on_processing_finished)

    def _on_detection_target_changed(self) -> None:
        """Обновляет видимость метода детекции для выбранной цели."""
        is_person = self.person_radio.isChecked()
        self.detection_method_container.setVisible(is_person)
        if not is_person:
            self.full_body_radio.setChecked(True)

    def _get_video_source(self) -> str:
        """Получает выбранный источник видео.

        @returns {str} Источник видео
        """
        if self.file_radio.isChecked():
            path = self.file_path_label.text()
            if path == "Не выбран":
                raise ValueError("Выберите видеофайл")
            return path
        else:
            index = self.webcam_index.value()
            return "webcam" if index == 0 else f"webcam:{index}"

    @pyqtSlot()
    def _on_start_clicked(self) -> None:
        """Обработчик нажатия кнопки Старт."""
        try:
            source = self._get_video_source()
            self._update_config_from_ui()
            self.controller.start_processing(source)

            if self.controller.worker:
                self.controller.worker.frame_ready.connect(self._on_frame_ready)
                self.controller.worker.heatmap_ready.connect(self._on_heatmap_ready)
                self.controller.worker.analysis_ready.connect(self._on_analysis_ready)
                self.controller.worker.progress_updated.connect(self._on_progress_updated)
                self.controller.worker.status_message.connect(self._on_status_message)
                self.controller.worker.finished_signal.connect(self._on_processing_finished)
                self.controller.worker.error_occurred.connect(self._on_error)

            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
        except ValueError as e:
            QMessageBox.warning(self, "Ошибка", str(e))
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось запустить обработку: {e}")

    @pyqtSlot()
    def _on_stop_clicked(self) -> None:
        """Обработчик нажатия кнопки Стоп."""
        self.controller.stop_processing()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setVisible(False)

    @pyqtSlot()
    def _on_save_clicked(self) -> None:
        """Обработчик нажатия кнопки Сохранить."""
        if self.current_heatmap is None:
            QMessageBox.warning(self, "Ошибка", "Нет данных для сохранения")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить тепловую карту",
            "",
            "PNG Images (*.png);;JPEG Images (*.jpg *.jpeg);;All Files (*)",
        )

        if file_path:
            try:
                self.controller.save_heatmap(file_path, self.current_heatmap)
                QMessageBox.information(self, "Успех", f"Тепловая карта сохранена: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить: {e}")

    @pyqtSlot()
    def _on_select_file(self) -> None:
        """Обработчик выбора файла."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выбрать видеофайл", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if file_path:
            self.file_path_label.setText(file_path)

    @pyqtSlot(int)
    def _on_confidence_changed(self, value: int) -> None:
        """Обработчик изменения confidence.

        @param value - Значение слайдера (0-100)
        """
        confidence = value / 100.0
        self.confidence_value.setText(f"{confidence:.2f}")

    @pyqtSlot(int)
    def _on_alpha_changed(self, value: int) -> None:
        """Обработчик изменения alpha.

        @param value - Значение слайдера (0-100)
        """
        alpha = value / 100.0
        self.alpha_value.setText(f"{alpha:.2f}")

    def _update_config_from_ui(self) -> None:
        """Обновляет конфигурацию из UI."""
        confidence = self.confidence_slider.value() / 100.0
        blur_radius = self.blur_spinbox.value()
        alpha = self.alpha_slider.value() / 100.0
        frame_skip = self.frame_skip_spinbox.value()
        detection_method = "foot" if self.foot_radio.isChecked() else "full_body"
        target = "vehicles" if self.vehicles_radio.isChecked() else "person"

        self.controller.update_config(
            confidence=confidence,
            blur_radius=blur_radius,
            alpha=alpha,
            frame_skip=frame_skip,
            detection_method=detection_method,
            target=target,
        )

    @pyqtSlot(np.ndarray)
    def _on_frame_ready(self, frame: np.ndarray) -> None:
        """Обработчик готовности кадра.

        @param frame - Кадр для отображения
        """
        self.video_widget.display_frame(frame)

    @pyqtSlot(np.ndarray)
    def _on_heatmap_ready(self, heatmap: np.ndarray) -> None:
        """Обработчик готовности тепловой карты.

        @param heatmap - Тепловая карта для отображения
        """
        self.current_heatmap = heatmap
        self.heatmap_widget.display_frame(heatmap)
        self.save_button.setEnabled(True)

    @pyqtSlot(str)
    def _on_analysis_ready(self, text: str) -> None:
        """Обработчик результата AI анализа.

        @param text - Текстовый отчёт анализа
        """
        if self._analysis_md_buffer:
            self._analysis_md_buffer += "\n\n---\n\n"
        self._analysis_md_buffer += text
        self.analysis_text.setMarkdown(self._analysis_md_buffer)

    def _on_toggle_ai_panel(self, visible: bool) -> None:
        """Скрывает/показывает панель AI анализа.

        @param visible - True если панель должна быть показана
        """
        self.analysis_label.setVisible(visible)
        self.analysis_text.setVisible(visible)
        self.toggle_ai_button.setText("Скрыть AI анализ" if visible else "Показать AI анализ")

    @pyqtSlot(int, int)
    def _on_progress_updated(self, frames: int, detections: int) -> None:
        """Обработчик обновления прогресса.

        @param frames - Количество обработанных кадров
        @param detections - Количество детекций
        """
        self.progress_bar.setValue(frames % 100)

    @pyqtSlot(str)
    def _on_status_message(self, message: str) -> None:
        """Обработчик сообщения статуса.

        @param message - Сообщение статуса
        """
        self.status_bar.showMessage(message)

    @pyqtSlot()
    def _on_processing_started(self) -> None:
        """Обработчик начала обработки."""
        pass

    @pyqtSlot()
    def _on_processing_stopped(self) -> None:
        """Обработчик остановки обработки."""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setVisible(False)

    @pyqtSlot()
    def _on_processing_finished(self) -> None:
        """Обработчик завершения обработки."""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage("Обработка завершена")

    @pyqtSlot(str)
    def _on_error(self, error_message: str) -> None:
        """Обработчик ошибки.

        @param error_message - Сообщение об ошибке
        """
        QMessageBox.critical(self, "Ошибка", error_message)
        self._on_processing_finished()

