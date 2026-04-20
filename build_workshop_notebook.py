"""
Builder script for cvlab_workshop.ipynb.
Run:  python3 build_workshop_notebook.py
"""
import json, uuid

def uid():
    return str(uuid.uuid4())[:8]

def md(source):
    return {"cell_type": "markdown", "id": uid(), "metadata": {}, "source": source}

def code(source):
    return {"cell_type": "code", "id": uid(), "metadata": {}, "source": source,
            "outputs": [], "execution_count": None}

cells = []

# ═══════════════════════════════════════════════════════
# SECTION 0 — Welcome & Setup
# ═══════════════════════════════════════════════════════

cells.append(md(
"""# Workshop: Face Image Classification with CNNs

> **Important note about the labels**
> This dataset contains two face-image classes provided by the dataset creators. These labels are a simplification of a much more complex real-world concept. In this workshop, we treat the task as a **binary image classification exercise**, not as a reliable system for inferring gender identity from appearance.

## By the end of this workshop, you will be able to:
1. Load an image dataset and understand what a model actually "sees" (tensors, channels, pixel values)
2. Train a pretrained CNN classifier quickly using transfer learning
3. Build a small CNN from scratch and understand each component
4. Interpret model decisions visually using Grad-CAM

## Workshop agenda

| # | Section | Who | Time |
|---|---------|-----|------|
| 0 | Setup | run once | 3 min |
| 1 | The task — first look at the data | 🟢 everyone | 8 min |
| 2 | Transfer learning — quick win | 🟢🟡 everyone | 15 min |
| 3 | Build a CNN from scratch | 🟡 everyone | 20 min |
| 4 | Compare & interpret | 🟢🟡 everyone | 12 min |
| 5 | Extensions (optional) | 🔴 advanced | any |

**Look for 🟢 Beginner · 🟡 Intermediate · 🔴 Advanced labels on each task. The default path is 🟢 + 🟡 only.**

---

## How to use this notebook

> **Predict → Run → Reflect**
>
> Before every interactive cell:
> 1. **Predict** — write down what you expect
> 2. **Run** — execute the cell
> 3. **Reflect** — was your prediction right? Why or why not?
"""
))

cells.append(md(
"""## 0. Setup

Run the next two cells once. If your environment already has everything installed, they will finish quickly.

> **⚠️ GPU required** — This notebook runs much faster on a GPU.
>
> In **Google Colab**: go to **Runtime → Change runtime type → T4 GPU → Save**, then re-run from the top.
"""
))

cells.append(code(
"""%pip install -q gitpython opencv-python grad-cam ipywidgets tensorboard tqdm pandas"""
))

cells.append(code(
"""import io
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import git
import ipywidgets as widgets
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from IPython.display import display
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.models import (
    mobilenet_v2, MobileNet_V2_Weights,
    resnet50, ResNet50_Weights,
    vit_b_16, ViT_B_16_Weights,
)
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore")

try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running in Colab:", IN_COLAB)
print("Using device:", device)

if device == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
"""
))

cells.append(md(
"""## 1. Workshop control panel

All tunable parameters live here. **You should not need to change any other cell** to run a different experiment — just update these values and re-run from this cell downwards.

> After changing any setting: **Runtime → Run after** (or Shift+Enter through each cell below).
"""
))

cells.append(code(
"""# =========================
# Workshop control panel
# =========================

# --- Data ---
SEED               = 42
N_IMAGES_PER_CLASS = 500     # images randomly selected from each class (None = all)
VALIDATION_SPLIT   = 0.20    # fraction of training data held out for validation

# --- Shared training ---
BATCH_SIZE         = 32      # images per mini-batch (try 16, 64)
NUM_WORKERS        = 0       # keep 0 in Jupyter/Colab (avoids multiprocessing issues)

# --- Section 2: Transfer learning ---
IMG_SIZE_PRETRAINED = 144    # use 224 when MODEL_NAME = "vit_b_16"
EPOCHS_PRETRAINED   = 4
LR_PRETRAINED       = 1e-4
FREEZE_BACKBONE     = True
UNFREEZE_LAST_BLOCK = False
MODEL_NAME          = "mobilenet_v2"   # "mobilenet_v2" | "resnet50" | "vit_b_16"
DROPOUT_PRETRAINED  = 0.0
USE_DATA_AUGMENTATION = False

# --- Section 3: Handcrafted CNN ---
IMG_SIZE_CNN  = 96
EPOCHS_CNN    = 20
LR_CNN        = 1e-3
NUM_CHANNELS  = 32
DROPOUT_CNN   = 0.0
USE_BATCHNORM = True

# --- Environment ---
MOUNT_DRIVE     = True
USE_INTERACTIVE = True    # set False to skip interactive widgets

# Seed everything
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_PATH = (
    Path('/content/gdrive/MyDrive/cvlab_workshop') if IN_COLAB
    else Path.cwd() / 'data'
)
print("Data will be saved to:", DATA_PATH)

if IN_COLAB and MOUNT_DRIVE:
    from google.colab import drive
    drive.mount('/content/gdrive/')
"""
))

# ═══════════════════════════════════════════════════════
# SECTION 1 — The Task
# ═══════════════════════════════════════════════════════

cells.append(md(
"""---
# Part 1 · The Task — What Are We Building?

We will train a model that looks at a face photo and predicts one of two labels.

First, let's get the data and take a look at it.
"""
))

cells.append(code(
"""DATA_PATH.mkdir(parents=True, exist_ok=True)

FACES_PATH     = DATA_PATH / 'faces'
FEMALE_PATH    = FACES_PATH / 'female'
MALE_PATH      = FACES_PATH / 'male'
BENCHMARK_PATH = FACES_PATH / 'benchmark'

FORCE_RECLONE = True   # set False after first successful download

if FORCE_RECLONE and FACES_PATH.exists():
    shutil.rmtree(FACES_PATH)

if not FACES_PATH.exists():
    print("Cloning dataset (first run only)...")
    repo = git.Repo.clone_from('https://github.com/susuter/faces_red.git', FACES_PATH)
    print("Done.")
else:
    print("Dataset already available at:", FACES_PATH)

assert FEMALE_PATH.exists(),    f"Expected {FEMALE_PATH}"
assert MALE_PATH.exists(),      f"Expected {MALE_PATH}"
assert BENCHMARK_PATH.exists(), f"Expected {BENCHMARK_PATH}"
print("Female:", len(list(FEMALE_PATH.glob('*.jpg'))), "images")
print("Male:  ", len(list(MALE_PATH.glob('*.jpg'))), "images")
print("Benchmark female:", len(list((BENCHMARK_PATH/'female').glob('*.jpg'))), "images")
print("Benchmark male:  ", len(list((BENCHMARK_PATH/'male').glob('*.jpg'))), "images")
"""
))

cells.append(md(
"""### 🟢 Your first look

Browse the images below. Before you scroll, write down:
- Do all images look like clean face photos?
- Can you already guess which label would be harder to classify?
"""
))

cells.append(code(
"""def scroll_face_images(root_folder):
    root_folder = Path(root_folder)
    image_paths, labels = [], []
    for label_dir in sorted(root_folder.iterdir()):
        if not label_dir.is_dir():
            continue
        for fpath in sorted(label_dir.iterdir()):
            if fpath.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                image_paths.append(fpath)
                labels.append(label_dir.name)

    if not image_paths:
        print("No images found in", root_folder)
        return

    max_idx  = len(image_paths) - 1
    slider   = widgets.IntSlider(value=0, min=0, max=max_idx, step=1,
                                 description="Image", continuous_update=False,
                                 layout=widgets.Layout(width="500px"))
    prev_btn = widgets.Button(description="◀ Previous")
    next_btn = widgets.Button(description="Next ▶")
    out      = widgets.Output()

    def render(i):
        with out:
            out.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(Image.open(image_paths[i]).convert("RGB"))
            ax.set_title(f"{labels[i]}  |  {image_paths[i].name}")
            ax.axis("off")
            plt.show()

    prev_btn.on_click(lambda _: setattr(slider, 'value', max(0, slider.value - 1)))
    next_btn.on_click(lambda _: setattr(slider, 'value', min(max_idx, slider.value + 1)))
    slider.observe(lambda c: render(c["new"]) if c["name"] == "value" else None, names="value")
    display(widgets.HBox([prev_btn, next_btn, slider]), out)
    render(0)

if USE_INTERACTIVE:
    scroll_face_images(FACES_PATH)
else:
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for ax, fpath in zip(axes.flat, list(FEMALE_PATH.glob('*.jpg'))[:4] + list(MALE_PATH.glob('*.jpg'))[:4]):
        ax.imshow(Image.open(fpath).convert("RGB"))
        ax.set_title(fpath.parent.name)
        ax.axis("off")
    plt.tight_layout()
    plt.show()
"""
))

cells.append(md(
"""### 🟢 What is an image, really?

Before we feed images to a model, it helps to understand how the computer *sees* them.

A colour image is just a 3D array of numbers: **Height × Width × 3 channels (R, G, B)**.

The cell below shows one image decomposed into its three colour channels.
"""
))

cells.append(code(
"""def show_image_channels(image_path):
    img = np.array(Image.open(image_path).convert("RGB"))
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

    axes[0].imshow(img)
    axes[0].set_title(f"Original\\n{img.shape[1]}×{img.shape[0]} px, 3 channels")

    channel_names = ["Red channel", "Green channel", "Blue channel"]
    cmaps         = ["Reds",        "Greens",         "Blues"]
    for i, (name, cmap) in enumerate(zip(channel_names, cmaps)):
        axes[i+1].imshow(img[:, :, i], cmap=cmap, vmin=0, vmax=255)
        axes[i+1].set_title(f"{name}\\nvalues 0–255")

    for ax in axes:
        ax.axis("off")
    plt.suptitle("An image is a 3D array: Height × Width × Channels", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.show()
    print(f"Array shape: {img.shape}  |  dtype: {img.dtype}  |  "
          f"value range: [{img.min()}, {img.max()}]")

sample_path = next(FEMALE_PATH.glob('*.jpg'))
show_image_channels(sample_path)
"""
))

cells.append(md(
"""### 🟢 Train / Validation / Benchmark — why three splits?

<div style="padding:14px 18px;border-radius:6px;margin:10px 0;background:rgba(33,150,243,0.08);border-left:5px solid #1976d2">

The dataset is divided into three non-overlapping groups, each serving a different purpose:

- **Train** — the model sees these images and updates its weights. This is where learning happens.
- **Validation** — used to monitor progress during training. Weights are **never** updated based on validation data.
- **Benchmark** — opened only once, at the very end. Never touched during development. Think of it as the exam.

</div>
"""
))

cells.append(code(
"""def plot_data_splits(n_train=800, n_val=200, n_bench=200):
    fig, ax = plt.subplots(figsize=(10, 2.2))
    ax.set_xlim(0, n_train + n_val + n_bench + 60)
    ax.set_ylim(0, 1)
    ax.axis("off")

    configs = [
        (0,             n_train, "#4caf50", "Train\n(updates model)"),
        (n_train + 20,  n_val,   "#2196f3", "Validation\n(monitors, no updates)"),
        (n_train + n_val + 40, n_bench, "#ff9800", "Benchmark\n(opened once at the end)"),
    ]
    for x, w, color, label in configs:
        ax.barh(0.5, w, left=x, height=0.5, color=color, edgecolor="white", linewidth=2)
        ax.text(x + w/2, 0.5, label, ha="center", va="center",
                fontsize=10, fontweight="bold", color="white")

    ax.set_title("How the data is split — each group serves a different purpose", fontsize=12)
    plt.tight_layout()
    plt.show()

plot_data_splits()
"""
))

# ═══════════════════════════════════════════════════════
# SECTION 2 — Transfer Learning
# ═══════════════════════════════════════════════════════

cells.append(md(
"""---
# Part 2 · Transfer Learning — Quick Win 🚀

Pretrained models have already learned to recognise edges, textures, and shapes from millions of images.
We *borrow* that knowledge and add a small trainable layer on top for our specific task.

This is called **transfer learning** — and it's why we can get strong results in just a few minutes.
"""
))

cells.append(md(
"""### 🟢 Backbone vs. classifier head

<div style="padding:14px 18px;border-radius:6px;margin:10px 0;background:rgba(33,150,243,0.08);border-left:5px solid #1976d2">

A pretrained CNN has two parts:

- **Backbone** (frozen 🧊) — the deep feature extractor, trained on 1.2 M ImageNet images. We don't change it.
- **Head** (trainable 🔥) — a tiny layer we attach on top. This is the only part we train.

We borrow the knowledge, add a small learnable layer on top for our specific task.

</div>
"""
))

cells.append(code(
"""def plot_backbone_head_diagram(model_name="mobilenet_v2", freeze_backbone=True):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    backbone_color = "#90caf9" if freeze_backbone else "#a5d6a7"
    head_color     = "#ff8a65"

    # Backbone block
    bb = mpatches.FancyBboxPatch((0.3, 0.5), 5.5, 2.8,
                                  boxstyle="round,pad=0.15",
                                  facecolor=backbone_color, edgecolor="#1565c0", linewidth=2)
    ax.add_patch(bb)
    ax.text(3.05, 1.9, "BACKBONE", ha="center", va="center", fontsize=13, fontweight="bold", color="#1565c0")
    backbone_label = f"{model_name}\\n{'❄️ frozen' if freeze_backbone else '🔥 fine-tuning'}"
    ax.text(3.05, 1.25, backbone_label, ha="center", va="center", fontsize=9, color="#1565c0")

    # Arrow
    ax.annotate("", xy=(7.3, 1.9), xytext=(5.8, 1.9),
                 arrowprops=dict(arrowstyle="->", color="#555", lw=2))
    ax.text(6.55, 2.25, "features", ha="center", va="bottom", fontsize=8, color="#555")

    # Head block
    hd = mpatches.FancyBboxPatch((7.3, 0.9), 2.2, 2.0,
                                  boxstyle="round,pad=0.15",
                                  facecolor=head_color, edgecolor="#bf360c", linewidth=2)
    ax.add_patch(hd)
    ax.text(8.4, 2.05, "HEAD", ha="center", va="center", fontsize=13, fontweight="bold", color="#bf360c")
    ax.text(8.4, 1.55, "🔥 trained\\nby us", ha="center", va="center", fontsize=9, color="#bf360c")

    ax.text(5.0, 3.7, "Transfer Learning: borrow the backbone, train the head",
            ha="center", va="center", fontsize=11, fontstyle="italic")
    plt.tight_layout()
    plt.show()

plot_backbone_head_diagram(MODEL_NAME, FREEZE_BACKBONE)
"""
))

cells.append(md(
"""### Data loading for transfer learning

The pretrained models expect images normalised with ImageNet statistics.
We use two separate transform pipelines: one for display, one for the model.
"""
))

cells.append(code(
"""imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

display_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE_PRETRAINED, IMG_SIZE_PRETRAINED)),
    transforms.ToTensor(),
])

if USE_DATA_AUGMENTATION:
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE_PRETRAINED, IMG_SIZE_PRETRAINED)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])
else:
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE_PRETRAINED, IMG_SIZE_PRETRAINED)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE_PRETRAINED, IMG_SIZE_PRETRAINED)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

# Build TRAIN folder expected by ImageFolder (female/, male/ subdirs)
TRAIN_PATH = FACES_PATH / "train"
if not TRAIN_PATH.exists():
    TRAIN_PATH.mkdir(parents=True)
    for cls in ["female", "male"]:
        dst = TRAIN_PATH / cls
        dst.mkdir(exist_ok=True)
        src = FACES_PATH / cls
        if src.exists():
            for f in src.glob("*.jpg"):
                (dst / f.name).symlink_to(f.resolve())
        else:
            print(f"WARNING: {src} not found — run the data clone cell first.")

full_train_display = datasets.ImageFolder(TRAIN_PATH, transform=display_transform)
full_train_model   = datasets.ImageFolder(TRAIN_PATH, transform=train_transform)
full_train_eval    = datasets.ImageFolder(TRAIN_PATH, transform=eval_transform)
benchmark_display  = datasets.ImageFolder(BENCHMARK_PATH, transform=display_transform)
benchmark_model    = datasets.ImageFolder(BENCHMARK_PATH, transform=eval_transform)

pretrained_class_names = full_train_model.classes
num_classes = len(pretrained_class_names)
print("Classes:", pretrained_class_names)
print("Train images:", len(full_train_model))
print("Benchmark images:", len(benchmark_model))
"""
))

cells.append(code(
"""# Person-aware split — same person cannot appear in both train and validation
samples    = full_train_model.samples
persons    = np.array(['_'.join(Path(p).stem.split('_')[:-1]) for p, _ in samples])
labels_arr = np.array([lbl for _, lbl in samples])

rng = np.random.default_rng(SEED)
train_idx, val_idx = [], []

for label in np.unique(labels_arr):
    indices        = np.where(labels_arr == label)[0]
    unique_persons = np.unique(persons[indices])
    rng.shuffle(unique_persons)
    n_val = max(1, round(len(unique_persons) * VALIDATION_SPLIT))
    val_persons = set(unique_persons[:n_val])
    for i in indices:
        (val_idx if persons[i] in val_persons else train_idx).append(i)

train_idx = np.array(train_idx)
val_idx   = np.array(val_idx)

# Optional: subsample to N_IMAGES_PER_CLASS
if N_IMAGES_PER_CLASS is not None:
    _sub = []
    for _lbl in np.unique(labels_arr[train_idx]):
        _cls_idx = train_idx[labels_arr[train_idx] == _lbl]
        _n = min(N_IMAGES_PER_CLASS, len(_cls_idx))
        _sub.append(rng.choice(_cls_idx, size=_n, replace=False))
    train_idx = np.concatenate(_sub)
    rng.shuffle(train_idx)

pretrained_train_ds = Subset(full_train_model, train_idx)
pretrained_val_ds   = Subset(full_train_eval,  val_idx)
pretrained_train_disp = Subset(full_train_display, train_idx)

pretrained_train_loader = DataLoader(pretrained_train_ds, batch_size=BATCH_SIZE,
                                      shuffle=True, num_workers=NUM_WORKERS)
pretrained_val_loader   = DataLoader(pretrained_val_ds,   batch_size=BATCH_SIZE,
                                      shuffle=False, num_workers=NUM_WORKERS)
pretrained_bench_loader = DataLoader(benchmark_model, batch_size=BATCH_SIZE,
                                      shuffle=False, num_workers=NUM_WORKERS)

print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Benchmark: {len(benchmark_model)}")
"""
))

cells.append(md(
"""### 🟢 What does a batch look like?

Each training step processes `BATCH_SIZE` images at once. The grid below shows exactly one batch — this is what the model sees in a single forward pass.
"""
))

cells.append(code(
"""def show_batch(dataset, batch_size, class_names, title="One training batch"):
    n   = min(batch_size, len(dataset), 32)
    cols = min(n, 8)
    rows = int(np.ceil(n / cols))
    rng_ = np.random.default_rng(SEED)
    idxs = rng_.choice(len(dataset), size=n, replace=False)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 1.8))
    axes = np.array(axes).flatten()
    for i, ax in enumerate(axes):
        if i < n:
            img, label = dataset[int(idxs[i])]
            if isinstance(img, torch.Tensor):
                img = img.permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.set_title(class_names[label], fontsize=7)
        ax.axis("off")
    plt.suptitle(f"{title}  ({n} images)", fontsize=11)
    plt.tight_layout()
    plt.show()

show_batch(pretrained_train_disp, BATCH_SIZE, pretrained_class_names,
           title=f"One batch  (BATCH_SIZE = {BATCH_SIZE})")
"""
))

cells.append(md(
"""### Build the pretrained model
"""
))

cells.append(code(
"""def build_pretrained_model(model_name=MODEL_NAME, num_classes=2, dropout=0.0,
                            freeze_backbone=True, unfreeze_last_block=False,
                            img_size=IMG_SIZE_PRETRAINED):
    if model_name == "mobilenet_v2":
        base = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
        if freeze_backbone:
            for p in base.features.parameters(): p.requires_grad = False
        if unfreeze_last_block:
            for p in base.features[-1].parameters(): p.requires_grad = True
        in_f = base.classifier[1].in_features
        base.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_f, num_classes))

    elif model_name == "resnet50":
        base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        if freeze_backbone:
            for name, p in base.named_parameters():
                if not name.startswith("fc"): p.requires_grad = False
        if unfreeze_last_block:
            for p in base.layer4.parameters(): p.requires_grad = True
        in_f = base.fc.in_features
        base.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_f, num_classes))

    elif model_name == "vit_b_16":
        base = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for name, p in base.named_parameters():
                if not name.startswith("heads"): p.requires_grad = False
        if unfreeze_last_block:
            for p in base.encoder.layers[-1].parameters(): p.requires_grad = True
        in_f = base.heads.head.in_features
        base.heads.head = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_f, num_classes))
    else:
        raise ValueError(f"Unknown MODEL_NAME {model_name!r}")
    return base

pretrained_model = build_pretrained_model(
    model_name=MODEL_NAME, num_classes=num_classes,
    dropout=DROPOUT_PRETRAINED, freeze_backbone=FREEZE_BACKBONE,
    unfreeze_last_block=UNFREEZE_LAST_BLOCK,
).to(device)

total_p     = sum(p.numel() for p in pretrained_model.parameters())
trainable_p = sum(p.numel() for p in pretrained_model.parameters() if p.requires_grad)
print(f"Backbone:         {MODEL_NAME}  (frozen={FREEZE_BACKBONE})")
print(f"Total params:     {total_p:,}")
print(f"Trainable params: {trainable_p:,}  ({100*trainable_p/total_p:.1f}%)")
"""
))

cells.append(md(
"""### 🟢 Checkpoint

Look at the numbers above.
- How many parameters are we actually training?
- Why is training so fast compared to training from scratch?

🟡 Try: set `UNFREEZE_LAST_BLOCK = True` in the control panel and re-run. How do the trainable param counts change?
"""
))

cells.append(code(
"""# Training utilities
_scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

def train_one_epoch_pretrained(model, dataloader, criterion, optimizer):
    model.train()
    running_loss, running_correct, running_total = 0.0, 0, 0
    use_amp = device == "cuda"
    for images, labels in tqdm(dataloader, leave=False, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device, enabled=use_amp):
            logits = model(images)
            loss   = criterion(logits, labels)
        if use_amp:
            _scaler.scale(loss).backward(); _scaler.step(optimizer); _scaler.update()
        else:
            loss.backward(); optimizer.step()
        running_loss    += loss.item() * images.size(0)
        running_correct += (logits.argmax(1) == labels).sum().item()
        running_total   += labels.size(0)
    return {"loss": running_loss / running_total, "accuracy": running_correct / running_total}


@torch.no_grad()
def evaluate_pretrained(model, dataloader, criterion):
    from sklearn.metrics import accuracy_score, f1_score
    model.eval()
    running_loss, running_total = 0.0, 0
    all_labels, all_preds = [], []
    use_amp = device == "cuda"
    for images, labels in tqdm(dataloader, leave=False, desc="Evaluating"):
        images, labels = images.to(device), labels.to(device)
        with torch.autocast(device_type=device, enabled=use_amp):
            logits = model(images); loss = criterion(logits, labels)
        preds = logits.argmax(1)
        running_loss  += loss.item() * images.size(0)
        running_total += labels.size(0)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
    ln, pn = np.array(all_labels), np.array(all_preds)
    return {"loss": running_loss / running_total,
            "accuracy": accuracy_score(ln, pn),
            "f1": f1_score(ln, pn, average="weighted", zero_division=0),
            "labels": ln, "preds": pn}


def fit_pretrained(model, train_loader, val_loader, criterion, optimizer, epochs):
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    model.eval()
    with torch.no_grad():
        imgs, _ = next(iter(train_loader))
        model(imgs[:1].to(device))  # warmup

    for epoch in tqdm(range(1, epochs + 1), desc="Training", unit="epoch"):
        tm = train_one_epoch_pretrained(model, train_loader, criterion, optimizer)
        vm = evaluate_pretrained(model, val_loader, criterion)
        history["train_loss"].append(tm["loss"]); history["train_acc"].append(tm["accuracy"])
        history["val_loss"].append(vm["loss"]);   history["val_acc"].append(vm["accuracy"])
        history["val_f1"].append(vm["f1"])
        print(f"Epoch {epoch:02d}/{epochs} | train_loss={tm['loss']:.4f} "
              f"val_loss={vm['loss']:.4f} val_acc={vm['accuracy']:.4f} val_f1={vm['f1']:.4f}")
    return history


def plot_history(history, title="Learning curves"):
    if len(history.get("train_loss", [])) < 2:
        print("Need at least 2 epochs to plot curves. Increase EPOCHS and re-run.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"],   label="validation")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"],   label="validation")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    plt.suptitle(title); plt.tight_layout(); plt.show()
"""
))

cells.append(code(
"""# Train the pretrained model
pretrained_loss_fn  = nn.CrossEntropyLoss()
pretrained_optimizer = Adam(
    [p for p in pretrained_model.parameters() if p.requires_grad],
    lr=LR_PRETRAINED
)

pretrained_history = fit_pretrained(
    pretrained_model, pretrained_train_loader, pretrained_val_loader,
    pretrained_loss_fn, pretrained_optimizer, EPOCHS_PRETRAINED
)
"""
))

cells.append(code(
"""plot_history(pretrained_history, title=f"Transfer learning — {MODEL_NAME}")
"""
))

cells.append(md(
"""### 🟢 Evaluate on benchmark

The benchmark set was never touched during training. Let's see how the model actually performs.

🟡 **Predict first**: will benchmark accuracy be higher, lower, or the same as validation accuracy?
"""
))

cells.append(code(
"""pretrained_val_metrics   = evaluate_pretrained(pretrained_model, pretrained_val_loader, pretrained_loss_fn)
pretrained_bench_metrics = evaluate_pretrained(pretrained_model, pretrained_bench_loader, pretrained_loss_fn)

print(f"Validation  — accuracy: {pretrained_val_metrics['accuracy']:.4f}  F1: {pretrained_val_metrics['f1']:.4f}")
print(f"Benchmark   — accuracy: {pretrained_bench_metrics['accuracy']:.4f}  F1: {pretrained_bench_metrics['f1']:.4f}")

ConfusionMatrixDisplay.from_predictions(
    pretrained_bench_metrics["labels"], pretrained_bench_metrics["preds"],
    display_labels=pretrained_class_names
)
plt.title(f"Benchmark confusion matrix — {MODEL_NAME}")
plt.show()
"""
))

cells.append(md(
"""### 🟡 Quick experiments — try one change

Change a value in the control panel, then re-run from the experiment config cell.

| # | Change | What to watch |
|---|--------|---------------|
| 1 | `EPOCHS_PRETRAINED = 8` | Do val curves keep rising? |
| 2 | `USE_DATA_AUGMENTATION = True` | Any difference in overfitting? |
| 3 | `FREEZE_BACKBONE = False` | How many more params are trained? |
| 4 | `MODEL_NAME = "resnet50"` | Better or worse than MobileNetV2? |

🔴 **Expert track:** `FREEZE_BACKBONE = False` + `LR_PRETRAINED = 1e-5` — this is full fine-tuning.
"""
))

# ═══════════════════════════════════════════════════════
# SECTION 3 — Handcrafted CNN
# ═══════════════════════════════════════════════════════

cells.append(md(
"""---
# Part 3 · Build a CNN from Scratch 🔧

Now we build the neural network ourselves — every layer, by hand.

This takes longer to train and usually performs worse than transfer learning.
But understanding how it works helps you understand *why* pretrained models are so powerful.
"""
))

cells.append(md(
"""### 🟢 How does a convolution work?

A convolution slides a small **kernel** (filter) across the image. At each position it multiplies the kernel values by the image values underneath and sums them. This detects local patterns like edges or corners.

The cell below shows this for a tiny example.
"""
))

cells.append(code(
"""def plot_convolution_intuition():
    np.random.seed(0)
    img_patch = np.array([
        [10, 10, 10, 10, 10],
        [10, 10, 10, 10, 10],
        [80, 80, 80, 80, 80],
        [80, 80, 80, 80, 80],
        [80, 80, 80, 80, 80],
    ], dtype=float)

    kernel = np.array([
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1],
    ], dtype=float)

    # Apply convolution (manual, center position only for demo)
    r, c = 1, 1   # top-left of kernel placement
    patch_under = img_patch[r:r+3, c:c+3]
    result_val  = float(np.sum(patch_under * kernel))

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

    # Image patch
    im0 = axes[0].imshow(img_patch, cmap="gray", vmin=0, vmax=100)
    axes[0].set_title("Image patch\\n(5×5 pixels)", fontsize=11)
    for i in range(5):
        for j in range(5):
            axes[0].text(j, i, int(img_patch[i, j]), ha="center", va="center",
                         fontsize=9, color="red" if (r<=i<=r+2 and c<=j<=c+2) else "white")
    rect = plt.Rectangle((c-0.5, r-0.5), 3, 3, edgecolor="red", facecolor="none", lw=2)
    axes[0].add_patch(rect)
    axes[0].axis("off")

    # Kernel
    im1 = axes[1].imshow(kernel, cmap="RdBu", vmin=-2, vmax=2)
    axes[1].set_title("Kernel (3×3)\\nedge detector", fontsize=11)
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, int(kernel[i, j]), ha="center", va="center",
                         fontsize=12, fontweight="bold")
    axes[1].axis("off")

    # Output value
    axes[2].axis("off")
    axes[2].text(0.5, 0.6, "Output value:", ha="center", va="center",
                 fontsize=11, transform=axes[2].transAxes)
    axes[2].text(0.5, 0.35, f"{result_val:.0f}", ha="center", va="center",
                 fontsize=36, fontweight="bold", color="#e53935",
                 transform=axes[2].transAxes)
    axes[2].text(0.5, 0.1, "sum(patch × kernel)\\n= strong edge detected",
                 ha="center", va="center", fontsize=9, color="#555",
                 transform=axes[2].transAxes)

    plt.suptitle("A convolution multiplies a kernel across the image — detecting local patterns",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    plt.show()

plot_convolution_intuition()
"""
))

cells.append(md(
"""### Preprocess images for the handcrafted CNN

The handcrafted model uses OpenCV for image loading and a simpler normalisation (values in [0, 1]).
"""
))

cells.append(code(
"""# Select a balanced subset
FEMALE_PATH_SEL = DATA_PATH / 'female_selected'
MALE_PATH_SEL   = DATA_PATH / 'male_selected'

for p in [FEMALE_PATH_SEL, MALE_PATH_SEL]:
    if p.exists(): shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

def randomly_select_n_images(in_path, out_path, n):
    files = sorted(list(in_path.glob('*.jpg')))
    assert n <= len(files), f"Only {len(files)} images in {in_path}"
    for src in random.sample(files, n):
        shutil.copy(src, out_path / src.name)

n_per_class = N_IMAGES_PER_CLASS or len(list(FEMALE_PATH.glob('*.jpg')))
randomly_select_n_images(FEMALE_PATH, FEMALE_PATH_SEL, n_per_class)
randomly_select_n_images(MALE_PATH,   MALE_PATH_SEL,   n_per_class)

print(f"Selected {n_per_class} images per class")
"""
))

cells.append(code(
"""LABEL_FEMALE = 0
LABEL_MALE   = 1

def img_preprocessing(image_list, label, img_size=IMG_SIZE_CNN):
    preprocessed, labels = [], []
    for p in image_list:
        img = cv2.imread(str(p))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        preprocessed.append(img.astype(np.float32) / 255.0)
        labels.append(label)
    return preprocessed, labels

print("Preprocessing female images...")
imgs_f, lbls_f = img_preprocessing(sorted(FEMALE_PATH_SEL.glob('*.jpg')), LABEL_FEMALE)
print("Preprocessing male images...")
imgs_m, lbls_m = img_preprocessing(sorted(MALE_PATH_SEL.glob('*.jpg')), LABEL_MALE)

X = np.array(imgs_f + imgs_m, dtype=np.float32)
y = np.array(lbls_f + lbls_m, dtype=np.int64)
print(f"X shape: {X.shape}  y shape: {y.shape}")
"""
))

cells.append(code(
"""# Person-aware train/val split
cnn_class_names = np.array(["female", "male"])
all_files_cnn   = (sorted(FEMALE_PATH_SEL.glob('*.jpg')) +
                   sorted(MALE_PATH_SEL.glob('*.jpg')))
persons_cnn     = np.array(['_'.join(p.stem.split('_')[:-1]) for p in all_files_cnn])

rng2 = np.random.default_rng(SEED)
cnn_train_idx, cnn_val_idx = [], []

for lbl in np.unique(y):
    idxs = np.where(y == lbl)[0]
    upers = np.unique(persons_cnn[idxs])
    rng2.shuffle(upers)
    n_val = max(1, round(len(upers) * VALIDATION_SPLIT))
    val_p = set(upers[:n_val])
    for i in idxs:
        (cnn_val_idx if persons_cnn[i] in val_p else cnn_train_idx).append(i)

cnn_train_idx = np.array(cnn_train_idx)
cnn_val_idx   = np.array(cnn_val_idx)
rng2.shuffle(cnn_train_idx); rng2.shuffle(cnn_val_idx)

X_train, y_train = X[cnn_train_idx], y[cnn_train_idx]
X_val,   y_val   = X[cnn_val_idx],   y[cnn_val_idx]
print(f"Train: {X_train.shape}  Val: {X_val.shape}")
"""
))

cells.append(code(
"""class NumpyClassificationDataset(Dataset):
    def __init__(self, X, y):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return (torch.from_numpy(np.transpose(self.X[idx], (2, 0, 1))).float(),
                torch.tensor(self.y[idx]).long())

cnn_train_ds  = NumpyClassificationDataset(X_train, y_train)
cnn_val_ds    = NumpyClassificationDataset(X_val, y_val)
cnn_train_loader = DataLoader(cnn_train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
cnn_val_loader   = DataLoader(cnn_val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Tiny subset for overfit check
overfit_ds = NumpyClassificationDataset(X_train[:10], y_train[:10])
print(f"train_loader: {len(cnn_train_ds)} samples | val_loader: {len(cnn_val_ds)} samples")
"""
))

cells.append(md(
"""### Define the CNN architecture
"""
))

cells.append(code(
"""class ConvModule(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1,
                 dropout_ratio=0.0, use_batchnorm=False):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding)]
        if use_batchnorm: layers.append(nn.BatchNorm2d(out_ch))
        layers += [nn.ReLU(), nn.Dropout2d(dropout_ratio), nn.MaxPool2d(2)]
        self.block = nn.Sequential(*layers)
    def forward(self, x): return self.block(x)


class HandcraftedCNN(nn.Module):
    def __init__(self, num_classes=2, num_channels=32,
                 dropout_ratio=0.0, use_batchnorm=True, img_size=IMG_SIZE_CNN):
        super().__init__()
        self.model_parameters = dict(num_classes=num_classes,
                                     num_channels=num_channels,
                                     dropout_ratio=dropout_ratio,
                                     use_batchnorm=use_batchnorm)
        self.features = nn.Sequential(
            ConvModule(3,             num_channels,     dropout_ratio=dropout_ratio, use_batchnorm=use_batchnorm),
            ConvModule(num_channels,  num_channels * 2, dropout_ratio=dropout_ratio, use_batchnorm=use_batchnorm),
            ConvModule(num_channels*2, num_channels * 4, dropout_ratio=dropout_ratio, use_batchnorm=use_batchnorm),
        )
        with torch.no_grad():
            n_feat = self.features(torch.zeros(1, 3, img_size, img_size)).flatten(1).shape[1]
        self.classifier = nn.Sequential(
            nn.Linear(n_feat, 128), nn.ReLU(),
            nn.Dropout(dropout_ratio), nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))
    def save_weights(self, path): torch.save(self.state_dict(), path)
"""
))

cells.append(md(
"""### 🟢 Architecture diagram — what happens to the spatial size?

Each `ConvModule` applies a MaxPool that halves the spatial dimensions. The number of channels grows.
"""
))

cells.append(code(
"""def plot_cnn_architecture(img_size=IMG_SIZE_CNN, num_channels=NUM_CHANNELS):
    stages = [
        (img_size,    img_size,    3,             "Input image"),
        (img_size//2, img_size//2, num_channels,  f"After Block 1\\n(3→{num_channels} ch)"),
        (img_size//4, img_size//4, num_channels*2,f"After Block 2\\n({num_channels}→{num_channels*2} ch)"),
        (img_size//8, img_size//8, num_channels*4,f"After Block 3\\n({num_channels*2}→{num_channels*4} ch)"),
    ]
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.set_xlim(0, 13); ax.set_ylim(0, 4); ax.axis("off")

    max_h = img_size
    colors = ["#bbdefb", "#90caf9", "#64b5f6", "#42a5f5"]
    x_positions = [0.3, 3.5, 6.7, 9.9]

    for (h, w, ch, label), color, xp in zip(stages, colors, x_positions):
        bar_h = 2.8 * (h / max_h)
        bar_w = 0.8 + 1.0 * (ch / (num_channels * 4))
        y0 = 0.5 + (2.8 - bar_h) / 2

        rect = mpatches.FancyBboxPatch((xp, y0), bar_w, bar_h,
                                        boxstyle="round,pad=0.05",
                                        facecolor=color, edgecolor="#1565c0", lw=1.5)
        ax.add_patch(rect)
        ax.text(xp + bar_w/2, y0 + bar_h + 0.15, f"{h}×{w}\\n{ch} ch",
                ha="center", va="bottom", fontsize=8, color="#1565c0")
        ax.text(xp + bar_w/2, y0 - 0.25, label,
                ha="center", va="top", fontsize=8)

        if xp < x_positions[-1]:
            ax.annotate("", xy=(xp + bar_w + 0.85, 1.9), xytext=(xp + bar_w + 0.1, 1.9),
                         arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))
            ax.text(xp + bar_w + 0.5, 2.15, "MaxPool\\n÷2", ha="center", va="bottom",
                    fontsize=7, color="#555")

    ax.set_title("CNN spatial dimension shrinks, channel depth grows", fontsize=11)
    plt.tight_layout()
    plt.show()

plot_cnn_architecture(IMG_SIZE_CNN, NUM_CHANNELS)
"""
))

cells.append(md(
"""### 🟢 Sanity check: can the model overfit 10 images?

Before full training, we check that the model *can* learn at all. We train on just 10 images — the model should reach 100% training accuracy quickly. If it can't fit 10 images, something is wrong.
"""
))

cells.append(code(
"""tiny_model   = HandcraftedCNN(num_channels=32, dropout_ratio=0.0, use_batchnorm=True).to(device)
tiny_loss_fn = nn.CrossEntropyLoss()
tiny_optim   = Adam(tiny_model.parameters(), lr=1e-3)

tiny_history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
tiny_loader  = DataLoader(overfit_ds, batch_size=5, shuffle=True)

for epoch in range(20):
    tiny_model.train()
    ep_loss, correct, total = 0.0, 0, 0
    for xb, yb in tiny_loader:
        xb, yb = xb.to(device), yb.to(device)
        tiny_optim.zero_grad()
        logits = tiny_model(xb); loss = tiny_loss_fn(logits, yb)
        loss.backward(); tiny_optim.step()
        ep_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item(); total += xb.size(0)
    tiny_history["train_loss"].append(ep_loss / total)
    tiny_history["train_acc"].append(correct / total)
    tiny_history["val_loss"].append(ep_loss / total)
    tiny_history["val_acc"].append(correct / total)

plot_history(tiny_history, title="Overfit check — 10 training images")
print(f"Final train accuracy on 10 samples: {tiny_history['train_acc'][-1]:.1%}")
"""
))

cells.append(md(
"""### 🟡 Train the full handcrafted CNN

Now train on the full dataset. This will take longer than the pretrained model.

**Before running: write down your prediction.**
- Will it overfit? (train accuracy >> val accuracy)
- Will it be better or worse than the pretrained model?
"""
))

cells.append(code(
"""# Training utilities for handcrafted CNN
def evaluate_cnn(model, dataset, batch_size=BATCH_SIZE, device=device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    y_true, y_pred, total_loss = [], [], 0.0
    loss_fn_ = nn.CrossEntropyLoss()
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            total_loss += loss_fn_(logits, yb).item() * xb.size(0)
            y_true.extend(yb.cpu().numpy())
            y_pred.extend(logits.argmax(1).cpu().numpy())
    yn, pn = np.array(y_true), np.array(y_pred)
    return {"loss": total_loss / len(dataset),
            "accuracy": metrics.accuracy_score(yn, pn),
            "f1": metrics.f1_score(yn, pn, average="weighted", zero_division=0),
            "y_true": yn, "y_pred": pn}
"""
))

cells.append(code(
"""# ╔══════════════════════════════════════════════════╗
# ║  ✏️  Experiment zone — change and re-run         ║
# ╚══════════════════════════════════════════════════╝
EXP_CNN_EPOCHS    = EPOCHS_CNN
EXP_CNN_LR        = LR_CNN
EXP_NUM_CHANNELS  = NUM_CHANNELS
EXP_DROPOUT_CNN   = DROPOUT_CNN
EXP_USE_BATCHNORM = USE_BATCHNORM

# Rebuild model (ensures a fresh start every run)
cnn_model = HandcraftedCNN(
    num_channels=EXP_NUM_CHANNELS, dropout_ratio=EXP_DROPOUT_CNN,
    use_batchnorm=EXP_USE_BATCHNORM
).to(device)
cnn_loss_fn  = nn.CrossEntropyLoss()
cnn_optim    = Adam(cnn_model.parameters(), lr=EXP_CNN_LR)

total_cnn, trainable_cnn = (sum(p.numel() for p in cnn_model.parameters()),
                             sum(p.numel() for p in cnn_model.parameters() if p.requires_grad))
print(f"Model ready | params: {total_cnn:,} | LR: {EXP_CNN_LR} | epochs: {EXP_CNN_EPOCHS}")
"""
))

cells.append(code(
"""# Training loop
cnn_history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

for epoch in tqdm(range(1, EXP_CNN_EPOCHS + 1), desc="Training CNN", unit="epoch"):
    cnn_model.train()
    ep_loss, correct, total = 0.0, 0, 0
    for xb, yb in cnn_train_loader:
        xb, yb = xb.to(device), yb.to(device)
        cnn_optim.zero_grad()
        logits = cnn_model(xb); loss = cnn_loss_fn(logits, yb)
        loss.backward(); cnn_optim.step()
        ep_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item(); total += xb.size(0)
    tm_loss = ep_loss / total; tm_acc = correct / total
    vm = evaluate_cnn(cnn_model, cnn_val_ds)
    cnn_history["train_loss"].append(tm_loss); cnn_history["train_acc"].append(tm_acc)
    cnn_history["val_loss"].append(vm["loss"]); cnn_history["val_acc"].append(vm["accuracy"])
    print(f"Epoch {epoch:02d}/{EXP_CNN_EPOCHS} | train_loss={tm_loss:.4f} "
          f"val_loss={vm['loss']:.4f} val_acc={vm['accuracy']:.4f} val_f1={vm['f1']:.4f}")
"""
))

cells.append(code(
"""plot_history(cnn_history, title="Handcrafted CNN learning curves")
"""
))

cells.append(md(
"""#### 🟢 Reflection

Look at the curves:
- Is the model still improving at the end, or has it plateaued?
- Is the gap between train and validation large? What does that mean?
- How do these curves compare to the pretrained model's curves?

🔴 **Advanced experiments:**

| Change | What to watch |
|--------|---------------|
| `DROPOUT_CNN = 0.3` | Does the val curve become smoother? |
| `NUM_CHANNELS = 64` | Does accuracy improve? What about training time? |
| `USE_BATCHNORM = False` | How does training stability change? |
"""
))

# ═══════════════════════════════════════════════════════
# SECTION 4 — Compare & Interpret
# ═══════════════════════════════════════════════════════

cells.append(md(
"""---
# Part 4 · Compare & Interpret

Let's load the benchmark set for the handcrafted model and do a side-by-side comparison.
"""
))

cells.append(code(
"""# Load benchmark images for handcrafted model
imgs_bf, lbls_bf = img_preprocessing(sorted((BENCHMARK_PATH/'female').glob('*.jpg')), LABEL_FEMALE)
imgs_bm, lbls_bm = img_preprocessing(sorted((BENCHMARK_PATH/'male').glob('*.jpg')),   LABEL_MALE)
X_bench = np.array(imgs_bf + imgs_bm, dtype=np.float32)
y_bench = np.array(lbls_bf + lbls_bm, dtype=np.int64)
cnn_bench_ds = NumpyClassificationDataset(X_bench, y_bench)
print("Benchmark:", X_bench.shape)
"""
))

cells.append(code(
"""# Side-by-side comparison
cnn_bench_m  = evaluate_cnn(cnn_model, cnn_bench_ds)
cnn_val_m    = evaluate_cnn(cnn_model, cnn_val_ds)

comparison_df = pd.DataFrame([
    {"model": f"Transfer ({MODEL_NAME})", "split": "validation",
     "accuracy": pretrained_val_metrics["accuracy"], "F1": pretrained_val_metrics["f1"]},
    {"model": f"Transfer ({MODEL_NAME})", "split": "benchmark",
     "accuracy": pretrained_bench_metrics["accuracy"], "F1": pretrained_bench_metrics["f1"]},
    {"model": "Handcrafted CNN",          "split": "validation",
     "accuracy": cnn_val_m["accuracy"],   "F1": cnn_val_m["f1"]},
    {"model": "Handcrafted CNN",          "split": "benchmark",
     "accuracy": cnn_bench_m["accuracy"], "F1": cnn_bench_m["f1"]},
]).round(4)

print(comparison_df.to_string(index=False))
"""
))

cells.append(md(
"""### 🟢 Checkpoint — discuss with a neighbour

- Which model performs better on the benchmark? By how much?
- Why can the pretrained model do more with fewer training steps?
- Which model's confusion matrix shows a clearer pattern?
"""
))

cells.append(code(
"""fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ConfusionMatrixDisplay.from_predictions(
    pretrained_bench_metrics["labels"], pretrained_bench_metrics["preds"],
    display_labels=pretrained_class_names, ax=axes[0]
)
axes[0].set_title(f"Transfer learning ({MODEL_NAME})")

ConfusionMatrixDisplay.from_predictions(
    cnn_bench_m["y_true"], cnn_bench_m["y_pred"],
    display_labels=cnn_class_names, ax=axes[1]
)
axes[1].set_title("Handcrafted CNN")
plt.tight_layout()
plt.show()
"""
))

cells.append(md(
"""### 🟡 Browse predictions — find interesting cases

Use the browser below to find:
- A **highly confident correct** prediction
- A **highly confident mistake**
- An **uncertain** prediction (probability close to 0.5)

Which cases are most interesting to discuss?
"""
))

cells.append(code(
"""# Get CNN predictions on benchmark
cnn_model.eval()
with torch.no_grad():
    xb_all    = torch.from_numpy(np.transpose(X_bench, (0, 3, 1, 2))).float().to(device)
    cnn_logits = cnn_model(xb_all)
    cnn_probs  = torch.softmax(cnn_logits, dim=1).cpu().numpy()
    cnn_preds  = np.argmax(cnn_probs, axis=1)


def show_prediction(X_, y_true, probs_, class_names_, i, title=""):
    pred = int(np.argmax(probs_[i]))
    correct = pred == y_true[i]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].imshow(X_[i])
    color = "green" if correct else "red"
    axes[0].set_title(f"True: {class_names_[y_true[i]]}\\n"
                      f"Pred: {class_names_[pred]} {'✓' if correct else '✗'}",
                      color=color)
    axes[0].axis("off")
    bar_colors = [color if i == pred else "#90caf9" for i in range(len(class_names_))]
    axes[1].bar(class_names_, probs_[i], color=bar_colors)
    axes[1].set_ylim(0, 1); axes[1].set_title(f"Confidence  {title}")
    plt.tight_layout(); plt.show()


if USE_INTERACTIVE:
    max_idx  = len(X_bench) - 1
    slider   = widgets.IntSlider(value=0, min=0, max=max_idx, step=1,
                                 description="sample", continuous_update=False,
                                 layout=widgets.Layout(width="500px"))
    prev_btn = widgets.Button(description="◀ Previous")
    next_btn = widgets.Button(description="Next ▶")
    out      = widgets.Output()

    def render_pred(i):
        with out:
            out.clear_output(wait=True)
            show_prediction(X_bench, y_bench, cnn_probs, cnn_class_names, i,
                            title="(Handcrafted CNN)")

    prev_btn.on_click(lambda _: setattr(slider, 'value', max(0, slider.value-1)))
    next_btn.on_click(lambda _: setattr(slider, 'value', min(max_idx, slider.value+1)))
    slider.observe(lambda c: render_pred(c["new"]) if c["name"] == "value" else None, names="value")
    display(widgets.HBox([prev_btn, next_btn, slider]), out)
    render_pred(0)
else:
    show_prediction(X_bench, y_bench, cnn_probs, cnn_class_names, 0, "Handcrafted CNN")
"""
))

cells.append(md(
"""### 🟡 Error analysis — misclassified images

Look at the images the model got wrong with high confidence. These are the most interesting failures.
- Is the mistake understandable (bad lighting, unusual pose)?
- Do the errors follow a pattern?
"""
))

cells.append(code(
"""misclassified_idx = [i for i in range(len(y_bench)) if cnn_preds[i] != y_bench[i]]
print(f"Misclassified: {len(misclassified_idx)} / {len(y_bench)}")

# Sort by confidence (most confident mistakes first)
conf_wrong = [(i, cnn_probs[i, cnn_preds[i]]) for i in misclassified_idx]
conf_wrong.sort(key=lambda x: -x[1])

if USE_INTERACTIVE and misclassified_idx:
    max_idx  = len(conf_wrong) - 1
    slider   = widgets.IntSlider(value=0, min=0, max=max_idx, step=1,
                                 description="mistake", continuous_update=False,
                                 layout=widgets.Layout(width="500px"))
    prev_btn = widgets.Button(description="◀ Previous")
    next_btn = widgets.Button(description="Next ▶")
    out      = widgets.Output()

    def render_wrong(pos):
        with out:
            out.clear_output(wait=True)
            idx = conf_wrong[pos][0]
            show_prediction(X_bench, y_bench, cnn_probs, cnn_class_names, idx, "← WRONG")

    prev_btn.on_click(lambda _: setattr(slider, 'value', max(0, slider.value-1)))
    next_btn.on_click(lambda _: setattr(slider, 'value', min(max_idx, slider.value+1)))
    slider.observe(lambda c: render_wrong(c["new"]) if c["name"] == "value" else None, names="value")
    display(widgets.HBox([prev_btn, next_btn, slider]), out)
    render_wrong(0)
elif misclassified_idx:
    show_prediction(X_bench, y_bench, cnn_probs, cnn_class_names, conf_wrong[0][0])
"""
))

cells.append(md(
"""### 🟡 Grad-CAM — what is the model actually looking at?

Grad-CAM produces a heatmap showing which image regions influenced the model's prediction.

The cell below shows both models side by side for the same image.
"""
))

cells.append(code(
"""# Grad-CAM for handcrafted CNN
def _last_conv(model):
    for m in reversed(list(model.modules())):
        if isinstance(m, nn.Conv2d): return m
    raise RuntimeError("No Conv2d found")

cnn_target_layer = _last_conv(cnn_model)
for p in cnn_target_layer.parameters():
    p.requires_grad = True

# Grad-CAM for pretrained model
if MODEL_NAME == "mobilenet_v2":
    pretrained_target_layer = pretrained_model.features[-1]
    pretrained_reshape = None
elif MODEL_NAME == "resnet50":
    pretrained_target_layer = pretrained_model.layer4[-1]
    pretrained_reshape = None
elif MODEL_NAME == "vit_b_16":
    pretrained_target_layer = pretrained_model.encoder.layers[-1].ln_1
    grid = int(pretrained_model.image_size // pretrained_model.patch_size)
    def pretrained_reshape(tensor, height=grid, width=grid):
        return tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2)).permute(0, 3, 1, 2)

for p in pretrained_target_layer.parameters():
    p.requires_grad = True

print("Grad-CAM target layers ready")
"""
))

cells.append(code(
"""def browse_gradcam_comparison(X_bench_cnn, y_bench_cnn, benchmark_display_ds, benchmark_eval_ds,
                               cnn_model_, pretrained_model_):
    n_bench = min(len(X_bench_cnn), len(benchmark_display_ds))

    max_idx  = n_bench - 1
    slider   = widgets.IntSlider(value=0, min=0, max=max_idx, step=1,
                                 description="sample", continuous_update=False,
                                 layout=widgets.Layout(width="500px"))
    prev_btn = widgets.Button(description="◀ Previous")
    next_btn = widgets.Button(description="Next ▶")
    out      = widgets.Output()

    def render(i):
        with out:
            out.clear_output(wait=True)
            # CNN side
            img_np = X_bench_cnn[i]
            cnn_tensor = torch.from_numpy(np.transpose(img_np, (2, 0, 1))).unsqueeze(0).float().to(device)
            with GradCAM(model=cnn_model_, target_layers=[cnn_target_layer]) as cam:
                cnn_cam = cam(input_tensor=cnn_tensor, targets=None)[0]
            cnn_viz = show_cam_on_image(img_np, cnn_cam, use_rgb=True)

            # Pretrained side
            img_disp, true_label = benchmark_display_ds[i]
            img_eval, _          = benchmark_eval_ds[i]
            img_np_pt = img_disp.permute(1, 2, 0).numpy().astype(np.float32)
            if img_np_pt.max() > 1.0: img_np_pt /= 255.0
            pt_tensor = img_eval.unsqueeze(0).to(device)
            with GradCAM(model=pretrained_model_, target_layers=[pretrained_target_layer],
                         reshape_transform=pretrained_reshape) as cam:
                pt_cam = cam(input_tensor=pt_tensor, targets=None)[0]
            pt_viz = show_cam_on_image(img_np_pt, pt_cam, use_rgb=True)

            # Predictions
            with torch.no_grad():
                cnn_prob = torch.softmax(cnn_model_(cnn_tensor), dim=1)[0].cpu().numpy()
                pt_prob  = torch.softmax(pretrained_model_(pt_tensor), dim=1)[0].cpu().numpy()
            cnn_pred = int(cnn_prob.argmax())
            pt_pred  = int(pt_prob.argmax())

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img_np)
            axes[0].set_title(f"Original\\nTrue: {cnn_class_names[y_bench_cnn[i]]}")
            axes[0].axis("off")

            cnn_color = "green" if cnn_pred == y_bench_cnn[i] else "red"
            axes[1].imshow(cnn_viz)
            axes[1].set_title(f"Handcrafted CNN\\nPred: {cnn_class_names[cnn_pred]} ({cnn_prob[cnn_pred]:.0%})",
                              color=cnn_color)
            axes[1].axis("off")

            pt_color = "green" if pt_pred == true_label else "red"
            axes[2].imshow(pt_viz)
            axes[2].set_title(f"Transfer ({MODEL_NAME})\\nPred: {pretrained_class_names[pt_pred]} ({pt_prob[pt_pred]:.0%})",
                              color=pt_color)
            axes[2].axis("off")
            plt.tight_layout(); plt.show()

    prev_btn.on_click(lambda _: setattr(slider, 'value', max(0, slider.value-1)))
    next_btn.on_click(lambda _: setattr(slider, 'value', min(max_idx, slider.value+1)))
    slider.observe(lambda c: render(c["new"]) if c["name"] == "value" else None, names="value")
    display(widgets.HBox([prev_btn, next_btn, slider]), out)
    render(0)


if USE_INTERACTIVE:
    browse_gradcam_comparison(X_bench, y_bench, benchmark_display, benchmark_model,
                               cnn_model, pretrained_model)
else:
    pass  # run browse_gradcam_comparison manually
"""
))

cells.append(md(
"""### 🟢 Final discussion

- Is each model looking at the face, or something else (hair, background, accessories)?
- Do the two models focus on the same regions?
- Can a high-confidence prediction with a strange attention map increase or decrease your trust?
- What would you need to change to make the model more reliable?

---

> **Takeaway:** Transfer learning lets us borrow powerful feature detectors trained on millions of images. Building from scratch gives us more control but requires more data and training time. Grad-CAM helps us check *where* the model looks — though it cannot prove *why* the decision is valid.
"""
))

# ═══════════════════════════════════════════════════════
# SECTION 5 — Extensions (Optional)
# ═══════════════════════════════════════════════════════

cells.append(md(
"""---
# Part 5 · Extensions (Optional) 🔴 Advanced

These sections go beyond the main workshop. Work through them if you finish early or want to explore further.
"""
))

cells.append(md(
"""## 5.1 Upload your own photo

Try the model on a face photo of your choice.
"""
))

cells.append(code(
"""IMAGE_PATH = None   # e.g. "my_photo.jpg" or "/content/drive/MyDrive/photo.jpg"
SKIP       = False

import io as _io
from PIL import Image as _PIL_Image

def predict_image_both_models(img_bytes):
    img = _PIL_Image.open(_io.BytesIO(img_bytes)).convert("RGB")

    # CNN prediction
    img_cnn = np.array(img.resize((IMG_SIZE_CNN, IMG_SIZE_CNN))).astype(np.float32) / 255.0
    t_cnn   = torch.from_numpy(np.transpose(img_cnn, (2, 0, 1))).unsqueeze(0).float().to(device)
    cnn_model.eval()
    with torch.no_grad():
        cnn_p = torch.softmax(cnn_model(t_cnn), dim=1)[0].cpu().numpy()

    # Pretrained prediction
    t_pt = eval_transform(img).unsqueeze(0).to(device)
    pretrained_model.eval()
    with torch.no_grad():
        pt_p = torch.softmax(pretrained_model(t_pt), dim=1)[0].cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img); axes[0].set_title("Uploaded"); axes[0].axis("off")
    axes[1].bar(cnn_class_names, cnn_p)
    axes[1].set_ylim(0, 1)
    axes[1].set_title(f"Handcrafted CNN\\n{cnn_class_names[cnn_p.argmax()]} ({cnn_p.max():.0%})")
    axes[2].bar(pretrained_class_names, pt_p)
    axes[2].set_ylim(0, 1)
    axes[2].set_title(f"Transfer ({MODEL_NAME})\\n{pretrained_class_names[pt_p.argmax()]} ({pt_p.max():.0%})")
    plt.tight_layout(); plt.show()

if not SKIP:
    if IMAGE_PATH:
        with open(IMAGE_PATH, "rb") as f:
            predict_image_both_models(f.read())
    elif IN_COLAB:
        from google.colab import files as _cf
        print("Click 'Choose Files' to upload a face image.")
        _up = _cf.upload()
        if _up: predict_image_both_models(next(iter(_up.values())))
    else:
        print("Set IMAGE_PATH = 'your_image.jpg' above and re-run.")
"""
))

cells.append(md(
"""## 5.2 Hyperparameter tuning with TensorBoard 🔴

Run multiple experiments and compare them visually.
"""
))

cells.append(code(
"""from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from itertools import product

RUNS_DIR = DATA_PATH / "tensorboard_runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

TUNING_GRID = {
    "model_name":      ["mobilenet_v2", "resnet50"],
    "lr":              [1e-4, 5e-4],
    "dropout":         [0.0, 0.2],
    "freeze_backbone": [True],
}
TUNING_EPOCHS = 3
RUN_TUNING    = False   # set True to start

print(f"RUN_TUNING = {RUN_TUNING}")
print("TensorBoard logs →", RUNS_DIR)
"""
))

cells.append(code(
"""if RUN_TUNING:
    keys = list(TUNING_GRID.keys())
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []

    for vals in product(*[TUNING_GRID[k] for k in keys]):
        cfg      = dict(zip(keys, vals))
        run_name = f"{ts}_{cfg['model_name']}_lr{cfg['lr']}_drop{cfg['dropout']}"
        print(f"\\n=== {run_name} ===")

        _m = build_pretrained_model(
            model_name=cfg["model_name"], num_classes=num_classes,
            dropout=cfg["dropout"], freeze_backbone=cfg["freeze_backbone"],
            img_size=IMG_SIZE_PRETRAINED
        ).to(device)
        _opt = Adam([p for p in _m.parameters() if p.requires_grad], lr=cfg["lr"])
        _crit = nn.CrossEntropyLoss()
        _writer = SummaryWriter(log_dir=str(RUNS_DIR / run_name))
        _writer.add_text("config", str(cfg))

        _h = fit_pretrained(_m, pretrained_train_loader, pretrained_val_loader,
                             _crit, _opt, TUNING_EPOCHS)
        _bm = evaluate_pretrained(_m, pretrained_bench_loader, _crit)
        for epoch, (tl, vl, va, vf) in enumerate(zip(_h["train_loss"], _h["val_loss"],
                                                       _h["val_acc"], _h["val_f1"]), 1):
            _writer.add_scalars("loss",     {"train": tl, "val": vl}, epoch)
            _writer.add_scalars("accuracy", {"val": va}, epoch)
            _writer.add_scalar("val/f1", vf, epoch)
        _writer.add_scalar("benchmark/accuracy", _bm["accuracy"], TUNING_EPOCHS)
        _writer.add_scalar("benchmark/f1",       _bm["f1"],       TUNING_EPOCHS)
        _writer.close()
        results.append({**cfg, "val_f1": _h["val_f1"][-1], "benchmark_f1": _bm["f1"]})

    pd.DataFrame(results).sort_values("benchmark_f1", ascending=False)
"""
))

cells.append(md(
"""### Launch TensorBoard

#### In Google Colab (web only)
```python
%load_ext tensorboard
%tensorboard --logdir {RUNS_DIR}
```

#### In a terminal
```bash
tensorboard --logdir path/to/tensorboard_runs
```
"""
))

cells.append(md(
"""## 5.3 DINOv3 Foundation Model 🔴

A state-of-the-art vision transformer trained via self-supervised learning. We extract features and train a tiny MLP on top.

See the original `cvlab_cnns_handcrafted.ipynb` notebook, Extra section, for the full implementation.
"""
))

cells.append(md(
"""---
## Workshop wrap-up

You have now completed a full deep-learning workflow:

1. **Loaded** and explored a real image dataset
2. **Trained** a pretrained CNN with transfer learning — fast and effective
3. **Built** a small CNN from scratch — understand every component
4. **Compared** both models on a held-out benchmark
5. **Interpreted** predictions with Grad-CAM

The key insight: pretrained models borrow powerful knowledge from millions of images.
Building from scratch gives you intuition about what is happening inside.

> **What did the pretrained model buy us? What did it cost?**
"""
))

# ═══════════════════════════════════════════════════════
# Write the notebook
# ═══════════════════════════════════════════════════════

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "colab": {"provenance": []}
    },
    "cells": cells
}

output_path = "cvlab_workshop.ipynb"
with open(output_path, "w") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

n_code = sum(1 for c in cells if c["cell_type"] == "code")
n_md   = sum(1 for c in cells if c["cell_type"] == "markdown")
print(f"Written: {output_path}")
print(f"Total cells: {len(cells)}  ({n_code} code, {n_md} markdown)")
