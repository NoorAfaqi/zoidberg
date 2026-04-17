# Chest X-Ray Pneumonia Classification – Preprocessing

This repo contains the data preparation and preprocessing pipeline for a 3-class chest X-ray pneumonia classification task using PyTorch.

## Dataset

The [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset is expected at `datasets/chest_Xray/` with the following structure after class separation:

```
datasets/chest_Xray/
├── train/
│   ├── NORMAL/
│   ├── BACTERIAL_PNEUMONIA/
│   └── VIRAL_PNEUMONIA/
├── val/
│   └── ...
└── test/
    └── ...
```

| Split | Samples |
|-------|---------|
| Train | 5,216   |
| Val   | 24      |
| Test  | 625     |

## Notebooks

### `Class_Separation.ipynb`
The original dataset groups all pneumonia cases into a single `PNEUMONIA/` folder. This notebook splits them into `BACTERIAL_PNEUMONIA/` and `VIRAL_PNEUMONIA/` subfolders by inspecting filenames (files containing `"bacteria"` or `"virus"`).

Run this **before** the preprocessing pipeline.

### `Preprocessing_Pipeline.ipynb`
Builds a PyTorch `DataLoader` pipeline with the following steps applied to every image:

| Step | Detail |
|------|--------|
| Grayscale → RGB | Duplicates single-channel images to 3 channels |
| CLAHE | Contrast Limited Adaptive Histogram Equalization (`clipLimit=2.0`, `tileGridSize=8×8`) |
| Gaussian Blur | 3×3 kernel for noise reduction |
| Resize | 224×224 |
| Normalize | `mean=[0.485, 0.485, 0.485]`, `std=[0.229, 0.229, 0.229]` |

Training additionally applies random horizontal flip, ±10° rotation, and random resized crop (scale 0.9–1.1).

## Dependencies

```bash
pip install pillow opencv-python torch torchvision
```
