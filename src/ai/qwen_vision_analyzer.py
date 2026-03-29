"""Сервис мультимодального анализа (кадр + тепловая карта) через Qwen2.5-VL."""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image
import requests

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QwenVisionAnalyzerConfig:
    """Конфигурация анализатора Qwen2.5-VL."""

    # CometAPI guide suggests using qwen-max
    model_id: str = "gpt-4o"
    max_new_tokens: int = 512
    api_base_url: str = "https://api.cometapi.com"


class QwenVisionAnalyzer:
    """Анализирует кадр и тепловую карту и возвращает текстовый отчёт."""

    def __init__(self, config: QwenVisionAnalyzerConfig) -> None:
        """Инициализирует анализатор.

        @param config - Конфигурация анализатора
        """
        self.config = config
        self._model = None
        self._processor = None

    def analyze(
        self,
        target: str,
        frame_bgr: np.ndarray,
        heatmap_bgr: np.ndarray,
    ) -> str:
        """Готовит промпт, отправляет 2 изображения в Qwen и возвращает отчёт.

        @param target - Цель детекции (person/vehicles)
        @param frame_bgr - Оригинальный кадр (BGR)
        @param heatmap_bgr - Тепловая карта/overlay (BGR)
        @returns {str} Текстовый отчёт
        """
        prompt = self._build_prompt(target=target)
        api_key = os.getenv("COMET_API_KEY")

        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
            frame_img = Image.fromarray(frame_rgb)
            heatmap_img = Image.fromarray(heatmap_rgb)
        except Exception as e:
            logger.error(f"Не удалось подготовить изображения для Qwen: {e}", exc_info=True)
            return f"[AI] Ошибка подготовки изображений: {e}"

        if not api_key:
            return (
                "[AI] Не задан COMET_API_KEY. "
                "Установите переменную окружения COMET_API_KEY для обращения к CometAPI."
            )

        try:
            return self._run_cometapi(prompt=prompt, frame=frame_img, heatmap=heatmap_img, api_key=api_key)
        except Exception as e:
            logger.error(f"Ошибка запроса CometAPI: {e}", exc_info=True)
            return f"[AI] Ошибка запроса CometAPI: {e}"

    def _run_cometapi(self, prompt: str, frame: Image.Image, heatmap: Image.Image, api_key: str) -> str:
        """Вызывает CometAPI чат completion (по гайду CometAPI).

        CometAPI в гайде использует `model=qwen-max` и endpoint вида `/v1/chat/completions`.
        Для изображений используем OpenAI-compatible формат `image_url` с data URI.

        @param prompt - Итоговый промпт (с delta-first)
        @param frame - PIL RGB изображение кадра
        @param heatmap - PIL RGB изображение тепловой карты
        @param api_key - API ключ CometAPI
        @returns {str} Текст ответа модели
        """
        url = f"{self.config.api_base_url.rstrip('/')}/v1/chat/completions"
        api_key_clean = api_key.strip()
        try:
            api_key_clean.encode("ascii")
        except UnicodeEncodeError as e:
            raise RuntimeError(
                "COMET_API_KEY содержит не-ASCII символы. Пересоздайте ключ и вставьте его без лишних пробелов/символов."
            ) from e

        headers = {
            "Authorization": f"Bearer {api_key_clean}",
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
        }

        frame_uri = self._pil_to_data_uri(frame)
        heatmap_uri = self._pil_to_data_uri(heatmap)

        payload = {
            "model": self.config.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": frame_uri}},
                        {"type": "image_url", "image_url": {"url": heatmap_uri}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "stream": False,
            "max_tokens": int(self.config.max_new_tokens),
        }

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        started_at = time.monotonic()
        response = requests.post(url, headers=headers, data=body, timeout=(10, 120))
        elapsed_sec = time.monotonic() - started_at
        logger.info(f"CometAPI response: {response.status_code} in {elapsed_sec:.2f}s")
        if response.status_code >= 400:
            raise RuntimeError(f"CometAPI HTTP {response.status_code}: {response.text}")

        data = response.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise RuntimeError(f"Неожиданный формат ответа CometAPI: {data}") from e

    def _pil_to_data_uri(self, image: Image.Image) -> str:
        """Конвертирует PIL изображение в data URI (PNG base64).

        @param image - PIL изображение
        @returns {str} data:image/png;base64,...
        """
        import io

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def _build_prompt(self, target: str) -> str:
        """Формирует промпт для Qwen.

        @param target - Цель детекции (person/vehicles)
        @returns {str} Промпт
        """
        header = (
            "Сделай полный анализ по текущим изображениям.\n"
            "ВАЖНО: не выдумывай фактов и не ссылайся на несуществующие предыдущие отчёты.\n"
        )

        colors = (
            "Цвета интерпретируем так: 🔴 красный/оранжевый — высокая плотность "
            "🟡 жёлтый — умеренное замедление 🔵/зелёный — свободное движение\n"
        )

        if target == "vehicles":
            task = (
                "Необходимо провести анализ тепловой карты транспортного потока.\n"
                "ВАЖНО: не выдумывай перекрёстки/точки вида A-B-C. "
                "Если нет явных названий улиц/перекрёстков на изображении — используй только описания по ориентирам кадра:\n"
                "- направление: левое/правое (относительно кадра)\n"
                "- расположение: ближе к камере/в середине/у горизонта\n"
                "- полосы: левая/средняя/правая (и если разделитель — укажи отдельно по каждой стороне)\n"
                "1) Где пробки и по каким полосам (для обоих направлений).\n"
                "2) Где есть свободные полосы.\n"
                "3) Рекомендации, что можно предпринять.\n"
            )
        else:
            task = (
                "Необходимо провести анализ тепловой карты человеческого потока.\n"
                "Сначала опиши, что изображено на главном изображении (место: магазин/парк/квартира/метро и т.д.).\n"
                "Далее опиши тепловую карту и изображение одновременно: сначала горячие места (красные), затем жёлтые, затем зелёно-синие.\n"
                "Дай совет/рекомендации по планировке/расположению зон.\n"
            )

        return f"{header}\n{colors}\n{task}"

