# Final Task Version

Реализовать inference pipeline для EcoVision, который принимает одно изображение и обученную YOLO-модель, находит объекты мусора, фильтрует предсказания по confidence threshold и приводит результат к единому формату проекта.

Финальный результат должен включать:

- загрузку обученной модели из `models/best.pt` или указанного пути;
- инференс для одного изображения;
- structured detections: `class_id`, `class_name`, `confidence`, `box`;
- фильтрацию по confidence threshold;
- отрисовку bounding boxes и labels на изображении;
- таблицу найденных объектов;
- статистику по классам;
- pollution level: `clean`, `low`, `medium`, `high`;
- cleanup recommendation;
- обработку edge cases: нет объектов, низкая уверенность, некорректный файл;
- простое Streamlit-приложение для загрузки изображения и просмотра результата.

Основные файлы:

- `src/inference.py`
- `src/visualization.py`
- `src/report.py`
- `src/utils.py`
- `app.py`
