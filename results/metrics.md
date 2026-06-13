# Model Metrics

device: `0`

| metric | value |
|---|---:|
| mAP@50 | 0.1696 |
| mAP@50-95 | 0.1251 |
| precision | 0.2669 |
| recall | 0.2141 |

## Error Analysis

- checked_images: 80
- false_positive: 99
- missed_objects: 198
- missed_small_objects: 144
- class_confusion: 60

Artifacts: `models/best.pt`, `results/confusion_matrix.png`, `results/plots/`, `results/error_examples/`, `results/new_image_tests/`.