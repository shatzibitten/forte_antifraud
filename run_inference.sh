#!/bin/bash
# Автоматический запуск inference.py с правильным Python из venv

# Переходим в директорию проекта
cd "$(dirname "$0")"

# Активируем виртуальное окружение
source venv/bin/activate

# Запускаем inference
python inference.py "$@"
