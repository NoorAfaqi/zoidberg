# Chest X-Ray Pneumonia Classification: Project Overview

## 1. What We Built

This project is an end-to-end 3-class chest X-ray classifier distinguishing **NORMAL**, **BACTERIAL_PNEUMONIA**, and **VIRAL_PNEUMONIA** using PyTorch transfer learning. Three models were trained and compared: ResNet50, EfficientNet-B4, and ConvNeXt-Base.

---

## 2. Model Comparison Summary

| Model | Test Accuracy | Weighted F1 | Hardware |
|---|---|---|---|
| **ResNet50** | **81.3%** | **0.814** | CPU |
| ConvNeXt-Base | 71.9% | 0.718 | GPU |
| EfficientNet-B4 | 63.6% | 0.616 | GPU |

ResNet50 is the clear winner and is the model used in the production Streamlit app.

---

## 3. Three Questions to Answer

### Q1: Why does the preprocessing convert images to grayscale first before converting back to RGB, and what problem does this create?

The pipeline runs `GrayscaleToRGBTransform` → `CLAHETransform`. The CLAHE step *also* converts to grayscale internally (`cv2.COLOR_RGB2GRAY`) before applying the contrast enhancement. This means even if an input image is already RGB, it is collapsed to single-channel twice in the pipeline.

**The effect:** the final RGB image produced by CLAHE is not a true 3-channel image: all three channels are identical copies of the enhanced grayscale. This is intentional for chest X-rays (which are inherently grayscale radiographs), but the normalization stats used are the standard ImageNet values `[0.485, 0.456, 0.406]` and `[0.229, 0.224, 0.225]`, which assume *actual* RGB variation across channels. Since all three channels are equal, only the first set of mean/std values are meaningfully applied. The model adapts to this during fine-tuning, but there is a small mismatch with the ImageNet pretraining distribution. Using the same value for all three channels (e.g., `mean=[0.485, 0.485, 0.485]`, `std=[0.229, 0.229, 0.229]`) would be a more accurate normalization for this data.

### Q2: What does the autoencoder-based cleaning step actually remove, and does it risk removing valid but rare pathology patterns?

The `clean_dataset_with_autoencoder` function trains a convolutional autoencoder **per class** for 3 epochs (train) or 2 epochs (val/test), then computes pixel-level MSE reconstruction error for each image. The top 5% highest-error images in train and top 3% in val/test are discarded.

**What gets removed:** images whose pixel patterns the autoencoder found difficult to reconstruct: typically mislabeled images, corrupted scans, very unusual orientations, or genuinely atypical X-rays.

**The risk:** VIRAL_PNEUMONIA is the smallest class (~143 test samples vs ~234 bacterial). Removing 5% of an already-imbalanced minority class makes class imbalance worse, and it may drop genuine hard cases that are actually important for the model to learn. The autoencoder is only trained for 3 epochs on a small per-class subset, so its reconstruction error threshold is noisy. A cleaner approach would be to only apply this filtering to the training set and at a lower percentile (e.g., 97–99%) to avoid aggressive removal.

### Q3: Why do the training augmentations use `RandomResizedCrop` *after* `Resize`, and what is the behavioral consequence?

In `train_transforms`, the sequence is:

```
Resize((224, 224)) → RandomHorizontalFlip → RandomRotation(10) → RandomResizedCrop(224, scale=(0.8, 1.0))
```

`RandomResizedCrop` crops a random sub-region of the image and then resizes it back to 224×224. Because `Resize` already ran, the input to the crop is already 224×224, so the crop effectively zooms in on a random 80–100% portion of the already-resized image.

**Consequence:** this is not wrong, but it is redundant with `Resize`. The standard pattern is to use `RandomResizedCrop` as the *first* spatial transform (applied to the original resolution image), which gives more diversity because it samples from the full image before any downscaling. Applying it after `Resize(224)` means you lose the benefit of sampling from higher-resolution source pixels. Additionally, the `scale=(0.8, 1.0)` in `preprocessing.py` differs from `(0.9, 1.1)` stated in the README; the code value is what actually runs, so the crop is slightly more aggressive than documented. The upper bound of 1.0 in torchvision is clamped anyway (you cannot crop a region larger than the image), so anything above 1.0 is inert.

---

## 4. Why ResNet50 Outperforms EfficientNet-B4 and ConvNeXt Here

### Problem with EfficientNet-B4 on this data

EfficientNet-B4 is designed for natural RGB images with complex texture, color, and spatial scale variation. On grayscale chest X-rays (where all three channels are identical), the compound scaling that makes EfficientNet-B4 efficient for natural images becomes unnecessary depth and width. Its recall on VIRAL_PNEUMONIA was only 0.29: it missed 71% of viral cases entirely, collapsing predictions toward the majority NORMAL class (recall 0.94 for NORMAL). The model never learned to differentiate the minority classes reliably, suggesting the fine-tuning regime (3 freeze epochs + partial unfreeze at `lr=1e-5`) was insufficient for this distribution shift.

### Problem with ConvNeXt-Base on this data

ConvNeXt-Base is a modernized CNN with large 7×7 depthwise convolutions and LayerNorm designed to match Vision Transformer performance on large-scale natural image benchmarks. It has ~89M parameters versus ResNet50's ~25M. For a small-scale medical dataset, this means:

- Higher risk of overfitting to the training distribution
- The large receptive fields may not provide an advantage when distinguishing subtle radiological features (bacterial vs viral infiltrates differ in fine local texture more than global structure)
- The model is significantly larger, requiring more data to fine-tune effectively

ConvNeXt-Base scored 72% accuracy, better than EfficientNet but 9 points below ResNet50.

### Why ResNet50 works well here

1. **Residual connections solve the vanishing gradient problem.** With 50 layers, ResNet can learn hierarchical features (edges → lung boundaries → consolidation patterns) without gradients decaying during backpropagation. This is important because the discriminative features in pneumonia X-rays are subtle and require depth to detect.

2. **Appropriate model capacity.** At ~25M parameters, ResNet50 is large enough to learn meaningful features but small enough to generalize from a few thousand training examples without aggressive regularization.

3. **Layer-specific fine-tuning matched the task.** The ResNet50 training unfroze `layer3` and `layer4` specifically (the deepest semantic feature layers), while keeping `layer1` and `layer2` (low-level edge detectors) frozen. This was better matched to the task than the generic "last 4 blocks" strategy used for EfficientNet and ConvNeXt, which unfreezes a different proportion of the network.

4. **ImageNet pretraining transfers well.** Even though chest X-rays are not natural images, the low-level features ResNet learned from ImageNet (texture gradients, intensity contrasts) transfer usefully to radiograph analysis. ResNet50's simpler, well-studied architecture has a well-understood inductive bias that behaves predictably under transfer learning.

5. **Class-specific performance.** ResNet50 achieved 0.86 F1 for BACTERIAL_PNEUMONIA and 0.75 for VIRAL_PNEUMONIA. The viral class is the hardest (fewest samples, most visually similar to normal): ResNet still managed 0.79 recall there vs. 0.29 for EfficientNet. This shows ResNet's fine-tuning captured minority-class features that the other models missed.

---

## 6. Known Issues and Discrepancies

- **README normalization stats** show `mean=[0.485, 0.485, 0.485]` but the actual code uses ImageNet standard `[0.485, 0.456, 0.406]`. The code is correct; the README is a documentation error.
- **README crop scale** says `(0.9, 1.1)` but `preprocessing.py` uses `(0.8, 1.0)`. The code is what runs.
- **ResNet50 was trained on CPU** while the other two models used CUDA. This means the ResNet training was slower but results are directly comparable since hardware does not affect model output.
- **`app.py` description** says it shows "NORMAL or PNEUMONIA" in the UI text but the model actually outputs 3 classes. The `st.write` message is stale from a 2-class version.
