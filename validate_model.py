"""
CNN Compatibility Checker
=========================

Use this script to verify that your custom CNN model is compatible with the
workshop training pipeline before plugging it in.

Requirements
------------
Your model must satisfy ALL of the following:

1. Input:  a float32 tensor of shape (N, 3, H, H)
           N = batch size, 3 = RGB channels, H = spatial size (square)
           H must be >= 144 (the pipeline uses 144 by default, but your model
           must accept any square size >= 144)

2. Output: a float32 tensor of shape (N, 2)
           2 = number of classes (female, male)

3. Logits: the output must be RAW and UNNORMALIZED — do NOT apply softmax,
           sigmoid, or any other activation on the final layer. The training
           loop applies CrossEntropyLoss, which expects raw logits.

Usage
-----
    python validate_model.py

Replace the `make_model` factory below with your own model class and run the
script. A clear PASS / FAIL message will tell you whether the model is
compatible for each tested size.
"""

import torch

# ── Replace this with your model ──────────────────────────────────────────────
# If your model takes img_size as a constructor argument, use it below.
# If it does not (e.g. fully convolutional), you can ignore img_size.

# ▼▼▼ EDIT THIS FUNCTION — everything else can stay as-is ▼▼▼
#
# make_model receives the input image size (e.g. 144) and must return
# your model as a torch.nn.Module.
#
# Two common cases:
#
#   Case A — your model takes img_size as a constructor argument:
#
#       from my_cnn import MyCNN
#       def make_model(img_size):
#           return MyCNN(img_size=img_size)
#
#   Case B — your model is fully convolutional (no hard-coded size):
#
#       from my_cnn import MyCNN
#       def make_model(img_size):
#           return MyCNN()          # img_size not needed, just ignore it
#
class SimpleCNN(torch.nn.Module):
    """Minimal example: two conv layers + a linear classifier."""
    def __init__(self, img_size: int, num_classes: int = 2):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
        )
        with torch.no_grad():
            n_feat = self.features(torch.zeros(1, 3, img_size, img_size)).flatten(1).shape[1]
        self.classifier = torch.nn.Linear(n_feat, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


def make_model(img_size: int) -> torch.nn.Module:
    return SimpleCNN(img_size=img_size)  # ← replace with your own model

# ──────────────────────────────────────────────────────────────────────────────

BATCH_SIZE  = 32
IN_CHANNELS = 3
NUM_CLASSES = 2

# Each entry is a square image size (int). Add or remove sizes as needed.
# Your model must pass ALL of them to be considered compatible.
TEST_SIZES = [144, 224]

all_errors = {}

for img_size in TEST_SIZES:
    images = torch.randn(BATCH_SIZE, IN_CHANNELS, img_size, img_size, dtype=torch.float32)
    errors = []
    try:
        model = make_model(img_size)
        model.eval()
        with torch.no_grad():
            logits = model(images)
        if logits.shape != torch.Size([BATCH_SIZE, NUM_CLASSES]):
            errors.append(f"  output shape: expected [{BATCH_SIZE}, {NUM_CLASSES}], got {list(logits.shape)}")
        if logits.dtype != torch.float32:
            errors.append(f"  output dtype: expected float32, got {logits.dtype}")
    except Exception as e:
        errors.append(f"  forward pass failed: {e}")
    all_errors[img_size] = errors

if any(all_errors.values()):
    print("FAIL — model is NOT compatible:\n")
    for img_size, errors in all_errors.items():
        if errors:
            print(f"  size {img_size}x{img_size}:")
            for e in errors:
                print(e)
else:
    print("PASS — model is compatible with the training pipeline.")
    for img_size in TEST_SIZES:
        print(f"  input [{BATCH_SIZE}, {IN_CHANNELS}, {img_size}, {img_size}] float32  →  output [{BATCH_SIZE}, {NUM_CLASSES}] float32  OK")
