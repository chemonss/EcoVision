# Model Metrics

device: `0`

| metric | value |
|---|---:|
| mAP@50 | 0.4061 |
| mAP@50-95 | 0.2703 |
| precision | 0.5679 |
| recall | 0.3853 |

## Error Analysis

- checked_images: 80
- false_positive: 106
- missed_objects: 186
- missed_small_objects: 135
- class_confusion: 0

Artifacts: `models/best.pt`, `results/confusion_matrix.png`, `results/plots/`, `results/error_examples/`, `results/new_image_tests/`.