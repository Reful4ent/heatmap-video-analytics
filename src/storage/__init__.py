"""Модуль работы с данными прогонов: сохранение и чтение артефактов."""

from src.storage.models import AnalysisRecord, RunMetadata, RunSummary
from src.storage.run_storage import RunStorage

__all__ = ["RunStorage", "RunMetadata", "RunSummary", "AnalysisRecord"]
