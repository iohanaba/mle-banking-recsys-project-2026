FROM python:3.10-slim

WORKDIR /app

# 1. Сначала копируем только requirements и устанавливаем зависимости
# Это позволяет кэшировать этот слой при изменениях кода
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 2. Копируем только необходимый код
COPY src/ ./src/
COPY models/ ./models/
COPY scripts/ ./scripts/

# 3. Создаём не-рут пользователя для безопасности (опционально)
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Запуск через python -m для надёжности импортов
CMD ["python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]