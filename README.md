# EcoVision: Waste Detection and Cleanup Assistant

EcoVision — приложение компьютерного зрения для автоматического обнаружения мусора на изображениях. Пользователь загружает фотографию городской или природной среды, после чего система находит объекты мусора, выделяет их bounding boxes, определяет тип отхода и формирует краткий визуальный отчет.

Проект решает прикладную задачу экологического мониторинга: быстрая оценка загрязненности территории по изображению и получение статистики по типам найденных отходов.

## Installation and Usage

1. Clone the repository

```bash
git clone https://github.com/chemonss/EcoVision
cd EcoVision
```

2. Create a virtual environment

- PowerShell
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

- CMD
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

- Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Download TACO images

```bash
python scripts/download_taco.py --dataset_path data/raw/annotations.json
```

The images will be downloaded into the same dataset directory according to the file paths specified in `annotations.json`.

5. Prepare the dataset

```bash
python scripts/prepare_dataset.py --overwrite
```

This command converts the original TACO annotations into the YOLO-format dataset used by the project.

## Project Goal

Разработать end-to-end CV-приложение:

```text
→ image upload
→ waste object detection
→ bounding boxes + class labels + confidence scores
→ waste statistics
→ cleanup recommendation
→ visual report
```

В качестве базового датасета используется **TACO (Trash Annotations in Context)** — открытый датасет изображений мусора в реальных условиях. Задача формулируется как **object detection**.

## Main Features

* загрузка изображения через web-интерфейс;
* обнаружение объектов мусора на изображении;
* визуализация bounding boxes;
* вывод класса и confidence score для каждого объекта;
* таблица найденных объектов;
* статистика по категориям мусора;
* автоматическая оценка уровня загрязнения;
* генерация короткой рекомендации по уборке/сортировке.

## Target Classes

Для повышения устойчивости модели исходные категории TACO объединяются в укрупненные классы:

```text
rigid_plastic
soft_plastic
paper/cardboard
metal
glass
other
```

Такой формат упрощает обучение, снижает влияние дисбаланса классов и делает результат более понятным для пользователя.

## Architecture

```text
TACO dataset
   ↓
class remapping
   ↓
COCO → YOLO conversion
   ↓
YOLO fine-tuning
   ↓
model evaluation
   ↓
inference pipeline
   ↓
Streamlit web application
   ↓
visual report
```

## System Components

### 1. Dataset Preparation

Подготовка данных включает:

* загрузку TACO;
* анализ исходных категорий;
* объединение категорий в coarse classes;
* конвертацию аннотаций из COCO-формата в YOLO-формат;
* разделение данных на train/validation;
* создание `dataset.yaml`.

### 2. Model Training

Модель обучается как object detector.

Базовый вариант:

```text
YOLOv8n / YOLOv11n
```

Модель должна возвращать:

```text
bounding boxes
class labels
confidence scores
```

Основные метрики:

```text
mAP@50
precision
recall
confusion matrix
training/validation loss curves
examples of successful predictions
examples of failure cases
```

### 3. Inference Pipeline

Инференс-модуль принимает изображение и возвращает результат в едином формате:

```python
{
    "objects": [
        {
            "class_name": "plastic",
            "confidence": 0.87,
            "box": [120, 45, 310, 260]
        }
    ],
    "summary": {
        "total": 1,
        "by_class": {
            "plastic": 1
        }
    },
    "pollution_level": "low",
    "recommendation": "Detected a small amount of waste. Local cleanup is recommended."
}
```

### 4. Web Application

Интерфейс реализуется на Streamlit.

Приложение должно показывать:

* исходное изображение;
* изображение с предсказаниями модели;
* таблицу найденных объектов;
* статистику по классам;
* итоговый текстовый отчет.

## Repository Structure

```text
ecovision/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   └── dataset.yaml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── best.pt
│
├── notebooks/
│   ├── dataset_preparation.ipynb
│   ├── training.ipynb
│   └── evaluation.ipynb
│
├── src/
│   ├── inference.py
│   ├── visualization.py
│   ├── report.py
│   ├── class_mapping.py
│   └── utils.py
│
├── demo_images/
│
├── results/
│
└── presentation/
    ├── report.pdf
    └── slides.pdf
```

## MVP Requirements

Минимальный жизнеспособный продукт должен включать в себя:

* обученную модель обнаружения объектов;
* рабочий вывод на основе загруженных изображений;
* визуализацию ограничивающих рамок;
* метки классов объектов;
* показатели достоверности;
* сводную таблицу;
* оценку уровня загрязнения;
* рекомендации по очистке;
* Демонстрацию Streamlit.

## Possible Extensions

* сегментация экземпляров вместо ограничивающих рамок;
* вывод данных с видео или веб-камеры;
* Карта загрязнения на основе GPS;
* загружаемый отчет в формате PDF / CSV;
* активное обучение для улучшения прогнозов;
* удобный интерфейс для мобильных устройств.

## Expected Result

К концу проекта мы ожидаем получить работающий прототип системы визуального обнаружения отходов. Окончательное приложение должно продемонстрировать весь процесс от ввода необработанных изображений до обнаружения объектов, визуальных аннотаций, статистики отходов и кратких рекомендаций по очистке.


