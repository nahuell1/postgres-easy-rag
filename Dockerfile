FROM python:3.8 AS base

WORKDIR /app

COPY . /app/

RUN pip install torch
RUN pip install -r requirements.txt

CMD ["fastapi", "dev", "main.py"]
