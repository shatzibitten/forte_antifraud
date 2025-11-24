#!/bin/bash
# Автоматический запуск main.py с правильным Python из venv

# Переходим в директорию проекта
cd "$(dirname "$0")"

# Активируем виртуальное окружение
source ../venv/bin/activate
python adversarial_validation.py

# Запускаем скрипт
python main.py "$@"
