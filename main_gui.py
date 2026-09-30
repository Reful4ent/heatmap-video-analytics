"""Точка входа для GUI версии приложения."""
import torch  # noqa: F401  (должен импортироваться раньше PyQt6 на Windows)

import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=True)

from PyQt6.QtWidgets import QApplication

from src.gui.main_window import MainWindow

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Главная функция GUI приложения."""
    app = QApplication(sys.argv)
    app.setApplicationName("Тепловые карты для ретейлинга")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

