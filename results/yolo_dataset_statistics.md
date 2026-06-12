# YOLO Dataset Statistics

## Split

- Train images: 1200
- Validation images: 300
- Total images: 1500

## Classes

| class_id | class_name | train objects | val objects | total objects |
|---:|---|---:|---:|---:|
| 0 | rigid_plastic | 992 | 403 | 1395 |
| 1 | soft_plastic | 639 | 218 | 857 |
| 2 | paper/cardboard | 382 | 115 | 497 |
| 3 | metal | 421 | 129 | 550 |
| 4 | glass | 196 | 58 | 254 |
| 5 | other | 942 | 289 | 1231 |

## Conversion Notes

- Original TACO annotations were converted from COCO-like format to YOLO detection format.
- Original TACO categories were mapped to six EcoVision coarse classes using `src/class_mapping.py`.
- Bounding boxes were converted from `[x_min, y_min, width, height]` to normalized YOLO `[x_center, y_center, width, height]` format.
- The train/validation split was performed at image level using a class-aware strategy.