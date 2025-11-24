# Решение проблемы Segmentation Fault

## Проблема
Segmentation fault возникает при использовании **системного Python 3.9.6** (`/usr/bin/python3`) на macOS с Apple Silicon (M1/M2/M3). Библиотеки C++ (SHAP, PyTorch, NumPy) конфликтуют с системными библиотеками.

## Решение
Использовать виртуальное окружение с Homebrew Python 3.13.

---

## Шаги для активации виртуального окружения

### 1. Активируйте виртуальное окружение
```bash
source venv/bin/activate
```

После активации в начале строки терминала появится `(venv)`.

### 2. Обновите pip
```bash
pip install --upgrade pip
```

### 3. Установите все необходимые библиотеки
```bash
pip install pandas numpy scikit-learn catboost lightgbm shap matplotlib networkx torch sdv
```

### 4. Проверьте версию Python
```bash
python --version  # Должно быть Python 3.13.x, НЕ 3.9.6
which python      # Должно быть .../venv/bin/python
```

### 5. Запустите скрипт
```bash
python main.py
```

---

## Как использовать в будущем

**Каждый раз при открытии нового терминала:**
```bash
cd /Users/apolorotov/Desktop/Agents/ForteContest
source venv/bin/activate
python main.py
```

**Для деактивации окружения:**
```bash
deactivate
```

---

## Проверка правильности установки

Запустите эти команды **после активации venv**:
```bash
# Должно показать путь в venv, а НЕ /usr/bin/python3
which python

# Должно быть 3.13.x
python --version

# Должны быть установленные библиотеки
pip list | grep -E "(shap|catboost|torch)"
```

---

## Почему это исправляет segfault?

1. **Современный Python**: 3.13 вместо 3.9.6
2. **Изолированное окружение**: библиотеки не конфликтуют с системными
3. **ARM64-оптимизация**: Homebrew компилирует пакеты для Apple Silicon
4. **Чистая установка**: нет старых версий библиотек
