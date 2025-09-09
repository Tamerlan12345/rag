FROM mcr.microsoft.com/devcontainers/python:3.11

# Устанавливаем Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Копируем файл с зависимостями в контейнер
COPY requirements.txt /tmp/pip-tmp/
