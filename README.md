# Fruit Deep Learning Classifier 🍎🍌🥕

A deep learning classifier that tells apart **healthy vs rotten** produce across 14 fruits and vegetables — **28 classes** in total — built with **PyTorch** using transfer learning on ImageNet-pretrained backbones.

Two backbones are compared side by side in their own subfolders:

- [`EfficientNetV2/`](./EfficientNetV2) — [EfficientNet-V2-S](https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_v2_s.html)
- [`MobileNetV2/`](./MobileNetV2) — [MobileNet-V2](https://pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v2.html) (lightweight, mobile-friendly)

## Classes

Apple · Banana · Bellpepper · Carrot · Cucumber · Grape · Guava · Jujube · Mango · Orange · Pomegranate · Potato · Strawberry · Tomato — each in **Healthy** and **Rotten** variants.

## Approach

1. Load an ImageNet-pretrained backbone from `torchvision.models`.
2. Freeze the feature extractor's weights.
3. Replace the final classifier head with a fresh `nn.Linear(..., 28)`.
4. Train only the new head with `Adam` (`lr = 0.001`) and `CrossEntropyLoss`.
5. Save the trained weights to `best_model.pth`.

Standard ImageNet preprocessing: resize → center-crop to 224×224 → normalize with `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.

## Repository layout

```
FruitDeepLearning-Classifier/
├── EfficientNetV2/
│   ├── main.py         # transfer-learning training loop
│   └── predict.py      # single-image inference with best_model.pth
└── MobileNetV2/
    ├── main.py         # transfer-learning training loop
    ├── predict.py      # single-image inference
    └── reduce_dataset.py  # subsamples each class folder to N images
```

## Dataset

Expected layout — one subfolder per class (compatible with `torchvision.datasets.ImageFolder`):

```
data/
├── Apple__Healthy/
├── Apple__Rotten/
├── Banana__Healthy/
├── Banana__Rotten/
└── ...
```

The [Fruit and Vegetable Disease (Healthy vs Rotten)](https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten) Kaggle dataset matches this structure directly.

`reduce_dataset.py` (in `MobileNetV2/`) subsamples the full dataset down to a fixed number of images per class (default 500) so training stays quick on modest hardware.

## Getting started

```bash
pip install torch torchvision pillow
```

**Train:**

```bash
cd EfficientNetV2         # or MobileNetV2
python main.py            # trains and saves best_model.pth
```

**Predict on a single image:**

```bash
python predict.py         # loads best_model.pth and classifies the test image
```

Point the `image_path` in `predict.py` at your own image before running.

> The training scripts assume a CUDA-capable GPU when available — they fall back to CPU automatically, though training on CPU will be slow.
