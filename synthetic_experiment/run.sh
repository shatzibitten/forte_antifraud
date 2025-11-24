#!/bin/bash
# Автоматический запуск main.py с правильным Python из venv

# Переходим в директорию проекта
cd "$(dirname "$0")"

# Активируем виртуальное окружение
source ../venv/bin/activate
python main.pyЗапускаем скрипт
python main.py "$@"
