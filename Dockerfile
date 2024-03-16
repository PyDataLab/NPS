# Используем официальный образ Python 3.11
FROM python:3.11

# Устанавливаем рабочую директорию в контейнере
WORKDIR /usr/src/app

# Копируем файлы проекта в контейнер
COPY . .

# Устанавливаем необходимые пакеты с помощью pip
RUN pip install --no-cache-dir -r requirements.txt

# Запускаем скрипт nps_prediction.py при старте контейнера
CMD [ "python", "./nps_prediction.py" ]
