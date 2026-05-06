import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

# ---------------------- CONSTANTS ----------------------
IMG_SIZE = 224

# ---------------------- CLAHE ----------------------
def apply_clahe(img):
    img = np.array(img)

    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # Convert back to RGB
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(img)


class CLAHETransform:
    def __call__(self, img):
        return apply_clahe(img)


# ---------------------- NOISE REDUCTION ----------------------
def apply_noise_reduction(img, method="gaussian"):
    img_np = np.array(img)

    if method == "gaussian":
        img_np = cv2.GaussianBlur(img_np, (3, 3), sigmaX=0)

    elif method == "median":
        if len(img_np.shape) == 3:
            img_np = np.stack(
                [cv2.medianBlur(img_np[:, :, c], 3) for c in range(img_np.shape[2])],
                axis=2
            )
        else:
            img_np = cv2.medianBlur(img_np, 3)

    return Image.fromarray(img_np)


class NoiseReductionTransform:
    def __init__(self, method="gaussian"):
        self.method = method

    def __call__(self, img):
        return apply_noise_reduction(img, method=self.method)


# ---------------------- GRAYSCALE HANDLING ----------------------
class GrayscaleToRGBTransform:
    def __call__(self, img):
        img = np.array(img)

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        elif len(img.shape) == 3 and img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)

        return Image.fromarray(img)


# ---------------------- TRANSFORMS ----------------------

# ⚠️ Use this ONLY for training (NOT in Streamlit)
train_transforms = transforms.Compose([
    GrayscaleToRGBTransform(),
    CLAHETransform(),
    NoiseReductionTransform(method="gaussian"),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ✅ Use this for validation, testing, and Streamlit inference
val_test_transforms = transforms.Compose([
    GrayscaleToRGBTransform(),
    CLAHETransform(),
    NoiseReductionTransform(method="gaussian"),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])