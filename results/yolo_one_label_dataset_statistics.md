# One-Label YOLO Dataset Statistics

- Dataset: `data/processed_one_label`
- Class 0: `waste`
- Images are symlinked from `data/processed`.

## Split

| split | images | objects |
|---|---:|---:|
| train | 1200 | 3572 |
| val | 300 | 1212 |
| total | 1500 | 4784 |

## Original Class Counts Before Collapse

| split | original class id | objects |
|---|---:|---:|
| train | 0 | 992 |
| train | 1 | 639 |
| train | 2 | 382 |
| train | 3 | 421 |
| train | 4 | 196 |
| train | 5 | 942 |
| val | 0 | 403 |
| val | 1 | 218 |
| val | 2 | 115 |
| val | 3 | 129 |
| val | 4 | 58 |
| val | 5 | 289 |

## Conversion Notes

- The train/validation split is inherited from `data/processed`.
- All YOLO label class IDs are rewritten to `0`; bounding boxes are unchanged.
- This dataset tests whether the detector can localize waste without fine-grained waste type classification.