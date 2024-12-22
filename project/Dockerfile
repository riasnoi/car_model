# Выкачиваем из dockerhub образ с python версии 3.9
FROM python:3.12
# Устанавливаем рабочую директорию для проекта в контейнере
WORKDIR /project
# Скачиваем/обновляем необходимые библиотеки для проекта 
COPY requirements.txt /project
RUN ls
RUN python3 -m pip install -r requirements.txt
COPY . /project
EXPOSE 80