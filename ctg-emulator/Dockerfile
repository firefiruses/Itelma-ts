# Используем официальный Python образ
FROM python:3.9-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл зависимостей
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код
COPY app/ ./app/
COPY data/ ./data/

# Создаем папку для логов (если нужно)
RUN mkdir -p logs

# Открываем порт для WebSocket
EXPOSE 8765

# Запускаем приложение
CMD ["python", "-m", "app.main"]
