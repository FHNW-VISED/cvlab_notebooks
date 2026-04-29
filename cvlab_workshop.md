# cvlab_workshop.ipynb — Structure & Content

**Topic:** Face Image Classification with CNNs — merged workshop notebook  
**Replaces:** `cvlab_cnn_pretrained.ipynb` + `cvlab_cnns_handcrafted.ipynb`

---

## Cell Map

| # | Type | Section | Content |
|---|------|---------|---------|
| 00 | Markdown | Title | Workshop title, ethics disclaimer, 4 learning outcomes, agenda table with 🟢🟡🔴 levels |
| 01 | Markdown | 0. Setup | GPU warning (Colab T4), instructions |
| 02 | Code | 0. Setup | `pip install` (opencv-python, grad-cam, ipywidgets, tensorboard, tqdm, pandas, requests) |
| 03 | Code | 0. Setup | All imports (torch, torchvision, cv2, sklearn, matplotlib, ipywidgets, etc.) + IN_COLAB/device detection |
| 04 | Markdown | 1. Control panel | Explanation of unified control panel |
| 05 | Code | 1. Control panel | All hyperparameters: SEED, N_IMAGES_PER_CLASS, VALIDATION_SPLIT, BATCH_SIZE, NUM_WORKERS, IMG_SIZE_PRETRAINED, IMG_SIZE_CNN, EPOCHS_PRETRAINED, EPOCHS_CNN, LR_PRETRAINED, LR_CNN, MODEL_NAME, FREEZE_BACKBONE, UNFREEZE_LAST_BLOCK, DROPOUT_PRETRAINED, USE_DATA_AUGMENTATION, NUM_CHANNELS, DROPOUT_CNN, USE_BATCHNORM, MOUNT_DRIVE, USE_INTERACTIVE |
| 06 | Markdown | Part 1 intro | Task framing — binary face classification |
| 07 | Code | 1. Data | Clone dataset; define FACES_PATH, FEMALE_PATH, MALE_PATH, BENCHMARK_PATH |
| 08 | Markdown | 1. Browse | 🟢 "Your first look" prompt |
| 09 | Code | 1. Browse | `scroll_face_images()` interactive widget + static fallback — **collapsed** |
| 10 | Markdown | 1. Channels | 🟢 "What is an image?" explanation |
| 11 | Code | 1. Channels | `show_image_channels()` — RGB channel decomposition figure |
| 12 | Markdown | 1. Splits | 🟢 Train/Val/Benchmark explanation with colored div + image |
| 13 | Markdown | Part 2 intro | Transfer learning intro |
| 14 | Markdown | 2. Backbone | 🟢 Backbone/head explanation with styled div + image |
| 15 | Markdown | 2. Transforms | Normalization explanation + 🟡 Q6 (intermediate, not beginner) |
| 16 | Code | 2. Transforms | imagenet_mean/std, display/train/eval transforms, ImageFolder datasets |
| 17 | Code | 2. Split | Person-aware train/val split for pretrained model; optional N_IMAGES_PER_CLASS subsample |
| 18 | Markdown | 2. Batch viz | 🟢 Batch size explanation + batch image |
| 19 | Code | 2. Batch viz | `show_batch()` — grid of BATCH_SIZE images |
| 20 | Markdown | 2. Data qty | 🟢 "Try it yourself — how much data do you need?" prompt with N_IMAGES_PER_CLASS table |
| 21 | Markdown | 2. Model | Build model header |
| 22 | Code | 2. Model | `build_pretrained_model()` — MobileNetV2/ResNet50/ViT-B/16, freeze logic, custom head |
| 23 | Markdown | 2. Checkpoint | 🟢 trainable param count + 🟡 UNFREEZE_LAST_BLOCK tip |
| 24 | Code | 2. Train utils | `train_one_epoch_pretrained()`, `evaluate_pretrained()`, `fit_pretrained()`, `plot_history()` |
| 25 | Code | 2. Results store | Shared `results` dict initialized once |
| 26 | Code | 2. Train | Train pretrained model |
| 27 | Code | 2. Curves | `plot_history(pretrained_history)` |
| 28 | Markdown | 2. Evaluate | 🟢 benchmark evaluation prompt with predict-first |
| 29 | Code | 2. Evaluate | `evaluate_pretrained()` on val + benchmark; confusion matrix |
| 30 | Markdown | 2. Experiments | 🟡 Quick experiment table (epochs, aug, freeze, model) + 🔴 expert track |
| 31 | Markdown | Part 3 intro | Handcrafted CNN intro |
| 32 | Markdown | 3. Convolution | 🟢 Convolution explanation |
| 33 | Code | 3. Convolution | `plot_convolution_intuition()` — kernel sliding figure |
| 34 | Markdown | 3. Visualisers | 🔗 CNN Playground / CNN Explainer / 3D Neural Network Vis links |
| 35 | Markdown | 3. Preprocess | OpenCV preprocessing explanation |
| 36 | Code | 3. Preprocess | `cnn_class_names` alias; loads pretrained datasets |
| 37 | Markdown | 3. Architecture | Architecture header |
| 38 | Code | 3. Architecture | `ConvModule`, `HandcraftedCNN` (3 conv blocks + classifier head) |
| 39 | Markdown | 3. Arch diagram | 🟢 Spatial diagram explanation |
| 40 | Markdown | 3. Overfit | 🟢 Overfit sanity check explanation |
| 41 | Code | 3. Overfit | Train tiny_model on 10 images; `plot_history()` |
| 42 | Markdown | 3. Train | 🟡 Full training prompt with prediction |
| 43 | Code | 3. Experiment | Experiment zone: EXP_CNN_EPOCHS, EXP_CNN_LR, EXP_NUM_CHANNELS, etc.; model rebuilt every run |
| 44 | Code | 3. Train | Training loop with tqdm; cnn_history |
| 45 | Code | 3. Curves | `plot_history(cnn_history)` |
| 46 | Markdown | 3. Reflection | 🟢 Reflection + 🔴 advanced experiment table |
| 47 | Markdown | Part 4 intro | Compare & Interpret intro |
| 48 | Code | 4. Benchmark | Load benchmark display arrays (reuses pretrained benchmark loader) |
| 49 | Code | 4. Comparison | Side-by-side metrics table (pretrained vs CNN, val + benchmark) |
| 50 | Code | 4. History | Cross-run comparison table from shared `results` dict |
| 51 | Markdown | 4. Checkpoint | 🟢 discuss-with-neighbour questions |
| 52 | Code | 4. Confusion | Side-by-side confusion matrices |
| 53 | Markdown | 4. Browse | 🟡 prediction browser prompt |
| 54 | Code | 4. Browse | CNN inference; `show_prediction()` widget |
| 55 | Markdown | 4. Errors | 🟡 error analysis prompt |
| 56 | Code | 4. Errors | Misclassified browser sorted by confidence |
| 57 | Markdown | 4. Grad-CAM | 🟡 Grad-CAM explanation |
| 58 | Code | 4. Grad-CAM | `_last_conv()`, target layer setup; `browse_gradcam_comparison()` |
| 59 | Code | 4. Grad-CAM | `browse_gradcam_comparison()` call — side-by-side for CNN + pretrained |
| 60 | Markdown | 4. Discussion | 🟢 final discussion prompts + takeaway |
| 61 | Markdown | Part 5 intro | Extensions header |
| 62 | Markdown | 5.2 TensorBoard | 🔴 TensorBoard hyperparameter tuning intro |
| 63 | Code | 5.2 TensorBoard | TUNING_GRID, RUN_TUNING=False guard |
| 64 | Code | 5.2 TensorBoard | Tuning loop with SummaryWriter |
| 65 | Markdown | 5.2 TensorBoard | How to launch TensorBoard (Colab + terminal) |
| 66 | Markdown | 5.3 DINOv3 | 🔴 DINOv3 pointer to Extra section |
| 67 | Markdown | Wrap-up | 5-point summary + closing question |
| 68 | Markdown | Extra intro | Foundation Model Adaptation with DINOv3 intro |
| 69 | Code | Extra setup | `pip install umap-learn transformers` |
| 70 | Code | Extra setup | Import umap, transformers |
| 71 | Markdown | E.1 | Load DINOv3 backbone — HuggingFace token instructions |
| 72 | Code | E.1 | Read `.access_token_hf` |
| 73 | Code | E.1 | Load `facebook/dinov3-vits16-pretrain-lvd1689m`, freeze weights |
| 74 | Markdown | E.2 | Extract features — CLS token (N,384) and patch tokens (N,196,384) |
| 75 | Code | E.2 | DINOv3 feature extraction loop → `cls_train`, `patches_train`, `cls_bench` |
| 76 | Markdown | E.3 | Feature visualisation header; E.3.1 PCA of patch tokens |
| 77 | Code | E.3.1 | PCA(3) on patch tokens → semantic RGB coloring |
| 78 | Markdown | E.3.2 | UMAP of CLS tokens explanation |
| 79 | Code | E.3.2 | UMAP(2) on CLS tokens → 2-D class separation plot |
| 80 | Markdown | E.3.3 | Self-attention maps — 6 heads in last transformer block |
| 81 | Code | E.3.3 | Discover self-attention module path |
| 82 | Code | E.3.3 | `_find_last_self_attn()`, attention hook registration |
| 83 | Code | E.3.3 | `show_attention()` — 6-head overlay visualisation |
| 84 | Markdown | E.4 | MLP classifier on frozen features |
| 85 | Code | E.4 | `FeatureDataset`, `DinoMLP` (Linear 384→128→2) |
| 86 | Code | E.4 | Train DINOv3 MLP; `dino_history` |
| 87 | Code | E.4 | Save DINOv3 model checkpoint |
| 88 | Markdown | E.5 | Benchmark evaluation & comparison |
| 89 | Code | E.5 | Evaluate DINOv3 MLP on benchmark; final 3-way comparison |
| 90 | Code | E.5 | (empty / trailing) |

---

## Key Variables

| Variable | Set in Cell | Purpose |
|----------|-------------|---------|
| `SEED` | 05 | Reproducibility |
| `N_IMAGES_PER_CLASS` | 05 | Subset size per class (None = all) |
| `VALIDATION_SPLIT` | 05 | Train/val fraction |
| `BATCH_SIZE` | 05 | DataLoader batch size |
| `IMG_SIZE_PRETRAINED` | 05 | Input size for pretrained/CNN models (144 default, 224 for ViT) |
| `IMG_SIZE_CNN` | 05 | Input size for handcrafted CNN (144) |
| `MODEL_NAME` | 05 | Pretrained backbone: "mobilenet_v2" / "resnet50" / "vit_b_16" |
| `FREEZE_BACKBONE` | 05 | Freeze pretrained backbone weights |
| `UNFREEZE_LAST_BLOCK` | 05 | Unfreeze last conv block for fine-tuning |
| `USE_DATA_AUGMENTATION` | 05 | Enable random flips/crops in training transforms |
| `NUM_CHANNELS` | 05 | Base conv filter count for handcrafted CNN |
| `DROPOUT_CNN` | 05 | Dropout rate in CNN conv layers |
| `USE_BATCHNORM` | 05 | BatchNorm in ConvModules |
| `MOUNT_DRIVE` | 05 | Mount Google Drive (Colab only) |
| `USE_INTERACTIVE` | 05 | Enable ipywidgets (set False for static fallbacks) |
| `pretrained_model` | 22 | Transfer learning model |
| `cnn_model` | 43 | Handcrafted CNN model (rebuilt each experiment run) |
| `pretrained_history` | 26 | Loss/accuracy curves for pretrained model |
| `cnn_history` | 44 | Loss/accuracy curves for handcrafted CNN |
| `pretrained_bench_metrics` | 29 | Pretrained benchmark evaluation results |
| `cnn_bench_m` | 49 | CNN benchmark evaluation results |
| `results` | 25 | Shared store for all runs — populated by `fit_pretrained()` |
| `cls_train`, `cls_bench` | 75 | DINOv3 CLS tokens (N, 384) |
| `patches_train` | 75 | DINOv3 patch tokens (N, 196, 384) |

---

## Visualisations

| Cell | Function | What it shows |
|------|----------|---------------|
| 11 | `show_image_channels()` | RGB channel decomposition of one face image |
| 19 | `show_batch()` | Grid of BATCH_SIZE images = one training step |
| 27 | `plot_history()` | Pretrained model loss + accuracy curves |
| 33 | `plot_convolution_intuition()` | 5×5 patch × 3×3 kernel → output value |
| 45 | `plot_history()` | Handcrafted CNN loss + accuracy curves |
| 52 | `ConfusionMatrixDisplay` | Side-by-side confusion matrices (CNN vs pretrained) |
| 59 | `browse_gradcam_comparison()` | Side-by-side Grad-CAM: CNN vs pretrained |
| 77 | PCA patch RGB | Semantic patch coloring without labels |
| 79 | UMAP CLS scatter | 2-D class separation of frozen DINOv3 features |
| 83 | `show_attention()` | 6 self-attention head overlays per image |

---

## Questions by Difficulty

> **How to read these:**
> - Every question has a `🔍 Find it:` line — a one-liner you can run in a new cell, or the exact variable/cell to look at.
> - 🟢 questions are answered by running a cell and reading output. No writing required.
> - 🟡 questions need you to change **one value** and re-run. The change is always shown explicitly.
> - 🔴 questions need a small code snippet or multi-parameter experiment. Template code is provided.

---

### 🟢 Beginner — run, observe, answer

**Part 1 · Cell 09 — First look at the data**

Q1. Do both classes have the same number of images in the widget?  
🔍 Find it: scroll the widget — notice the label shown. Or paste into a new cell and run: `print(Counter(p.parent.name for p in FACES_PATH.rglob("*.jpg")))` (Counter is already imported).

Q2. Can you spot an image that looks noisy, blurry, or oddly framed?  
🔍 Find it: scroll slowly through both classes. Write down the index of one unusual image. What might cause it?

Q3. Before training anything: which class do you think will be *harder* to classify, and why?  
🔍 Find it: just observe — no code needed. Write your gut answer first, then revisit after training.

---

**Part 1 · Cell 11 — What is an image?**

Q4. What is the shape (height × width × channels) of one image?  
🔍 Find it: run in a new cell:
```python
import numpy as np
from PIL import Image
img = np.array(Image.open(next(FEMALE_PATH.iterdir())).convert("RGB"))
print(img.shape)       # (H, W, 3)
print(img[0, 0, :])    # RGB values of top-left pixel
```

Q5. Which channel image looks brightest in skin regions: Red, Green, or Blue? Does that match your intuition?  
🔍 Find it: look at the figure produced by cell 11. Skin is warm-toned — which channel shows it lightest?

---

**Part 2 · Cell 15 — Data loading (normalization)**

Q6 🟡 (Intermediate). After normalization, pixel values are no longer in the 0–255 range. What range do you expect?  
🔍 Find it: paste into a new cell after cell 16 runs:
```python
img_tensor, _ = pretrained_train_ds[0]
print(f"min={img_tensor.min():.2f}  max={img_tensor.max():.2f}")
```

---

**Part 1 · Cell 12 — Data splits**

Q7. How many images are in the train and validation sets?  
🔍 Find it: run after cell 17:
```python
print("Train images:", len(train_idx))
print("Val images  :", len(val_idx))
```

Q8. Why can't we look at the benchmark results during training to decide when to stop?  
🔍 Find it: re-read the colored box in cell 12. The answer is in the definition of "Benchmark".

---

**Part 2 · Cell 18 — What does a batch look like?**

Q9. How many images does the model process in one training step?  
🔍 Find it: count the images in the grid, or run: `print(BATCH_SIZE)`

Q10 🟡 (Intermediate). The grid shows images already resized to the same dimensions. Why does a neural network require fixed-size inputs?  
🔍 Find it: look at the `Linear` layer in cell 22 — its input size is fixed. Change `IMG_SIZE_PRETRAINED` and re-run cell 22 to see what changes.

---

**Part 2 · Cell 23 — Trainable parameters**

Q11. How many parameters does MobileNetV2 have in total? How many are we actually training?  
🔍 Find it: run after cell 22:
```python
total     = sum(p.numel() for p in pretrained_model.parameters())
trainable = sum(p.numel() for p in pretrained_model.parameters() if p.requires_grad)
print(f"Total: {total:,}   Trainable: {trainable:,}   Frozen: {total-trainable:,}")
```

Q12. The model trains in a few minutes despite having millions of parameters. Why?  
🔍 Find it: compare `trainable` vs `total` from Q11. Gradient computation only runs for `requires_grad=True` parameters.

---

**Part 2 · Cell 28 — Benchmark evaluation**

Q13. Is benchmark accuracy higher or lower than validation accuracy? By how much?  
🔍 Find it: the two lines printed by cell 29 show both. Or: `print(pretrained_bench_metrics['accuracy'] - pretrained_val_metrics['accuracy'])`

Q14. What does a large gap between val and benchmark accuracy tell you about your model?  
🔍 Find it: no code — think about which images each set contains and when they were collected.

---

**Part 2 · Cell 20 — How much data do you need?**

Q15. Change `N_IMAGES_PER_CLASS = 50` in the control panel, re-run everything, note benchmark accuracy. Then set it back to `500`. How much did accuracy drop?  
🔍 Find it: after each run, read the benchmark line printed by cell 29.

---

**Part 3 · Cell 40 — Sanity check**

Q16. After 20 epochs on 10 images, what training accuracy does the tiny model reach?  
🔍 Find it: look at the rightmost point on the training accuracy curve (cell 41 output). It should approach 100%.

Q17. If the model *couldn't* overfit 10 images, what would that tell you about the architecture?  
🔍 Find it: no code — a model that can't overfit 10 samples has either wrong dimensions or a bug. Check `HandcraftedCNN` in cell 38.

---

**Part 3 · Cell 46 — Learning curves**

Q18. At the last epoch, is training accuracy still rising or has it plateaued?  
🔍 Find it: look at the curve from cell 45. Check `cnn_history['train_acc'][-3:]` for the last three values.

Q19. How large is the gap between training and validation accuracy at the end?  
🔍 Find it: `print(cnn_history['train_acc'][-1] - cnn_history['val_acc'][-1])`

---

**Part 4 · Cell 51 — Compare models**

Q20. Which model has higher benchmark accuracy, and by how much?  
🔍 Find it: look at the table printed by cell 49, or: `print(comparison_df.to_string())`

Q21. Compare the two models: which one achieves higher benchmark accuracy? Does more training time necessarily mean better accuracy? Why or why not?  
🔍 Find it: re-read the backbone explanation in cell 14. Think about what ImageNet pre-training provides versus training from scratch.

---

**Part 4 · Cell 60 — Grad-CAM final discussion**

Q22. For each model: is the heatmap centred on the face, or spread over hair/background?  
🔍 Find it: cell 59 browser. Browse at least 5 images. Note patterns.

Q23. A model makes a confident *correct* prediction, but the heatmap highlights the background. Should you trust this model in production?  
🔍 Find it: find such a case in the browser. The answer is not in the code — it's in what "correct for the wrong reason" means.

---

### 🟡 Intermediate — change one value, predict, then check

> Rule: **write your prediction before you run**. One sentence is enough.

---

**Part 2 · Cell 23 — Unfreeze last block**

Q24. Predict: if you unfreeze the last conv block of MobileNetV2, how will the number of trainable parameters change?  
🔍 Do it: in cell 05, set `UNFREEZE_LAST_BLOCK = True`, re-run cell 22, then run Q11's snippet again. How many extra parameters became trainable?

Q25. Does unfreezing the last block improve benchmark accuracy after re-training?  
🔍 Do it: re-run cells 26–29 with `UNFREEZE_LAST_BLOCK = True`. Compare `pretrained_bench_metrics['accuracy']` to the frozen run.

---

**Part 2 · Cell 30 — Pick one experiment**

Q26. `EPOCHS_PRETRAINED = 8` — predict: will val accuracy keep rising after epoch 4, or plateau?  
🔍 Do it: change the value in cell 05, re-run cells 26–27. Check the val accuracy curve.

Q27. `USE_DATA_AUGMENTATION = True` — predict: will the gap between train and val accuracy shrink?  
🔍 Do it: set in cell 05, re-run cells 26–29. Run: `print(pretrained_history['train_acc'][-1] - pretrained_history['val_acc'][-1])` before and after.

Q28. `MODEL_NAME = "resnet50"` — predict: is ResNet50 more accurate or faster than MobileNetV2?  
🔍 Do it: change in cell 05, re-run. Compare benchmark accuracy and wall-clock time in the tqdm progress bar.

---

**Part 3 · Cell 42 — Predict CNN behaviour before running**

Q29. Before running cells 43–44: predict whether `train_acc` or `val_acc` will be higher at epoch 20. Write it down.  
🔍 Check: after training, run: `print(f"Train: {cnn_history['train_acc'][-1]:.3f}  Val: {cnn_history['val_acc'][-1]:.3f}")`

Q30. Predict: will the CNN beat the pretrained model on benchmark accuracy?  
🔍 Check: `print(cnn_bench_m['accuracy'], pretrained_bench_metrics['accuracy'])`

---

**Part 3 · Cell 43 — CNN experiment zone**

Q31. Set `EXP_DROPOUT_CNN = 0.3` in cell 43 and re-run 43–45. Does the train-val gap shrink compared to `0.0`?  
🔍 Find it: `print(cnn_history['train_acc'][-1] - cnn_history['val_acc'][-1])` — run once with each value and compare.

Q32. Set `EXP_NUM_CHANNELS = 64` and re-run. Does benchmark accuracy improve? By how much does training take longer?  
🔍 Find it: read the tqdm time estimate and compare `cnn_bench_m['accuracy']`.

Q33. Set `EXP_USE_BATCHNORM = True` and re-run. Do the training curves become smoother?  
🔍 Find it: compare the loss curves visually. Or: `import numpy as np; print(np.std(cnn_history['val_loss']))`

---

**Part 4 · Cell 53 — Browse predictions**

Q34. Find the prediction with the **highest confidence** in the browser. What is its probability value?  
🔍 Find it: paste into a new cell after 54:
```python
max_i = int(np.argmax(cnn_probs.max(axis=1)))
print(f"Index {max_i}: pred={cnn_class_names[cnn_preds[max_i]]}, conf={cnn_probs[max_i].max():.3f}, true={cnn_class_names[y_bench[max_i]]}")
```

Q35. Find the **most confident mistake** in the error browser (cell 56 is sorted this way). What confidence does it have?  
🔍 Find it: the first image shown in the cell 56 browser is the most confident wrong prediction. The confidence is printed in the title.

Q36. Find a prediction close to 0.5 probability. Why is the model uncertain on this image?  
🔍 Find it:
```python
uncertain_i = int(np.argmin(np.abs(cnn_probs.max(axis=1) - 0.5)))
print(f"Index {uncertain_i}: conf={cnn_probs[uncertain_i].max():.3f}")
```
Then navigate to that index in the cell 53/54 browser.

---

**Part 4 · Cell 55 — Error analysis**

Q37. How many images were misclassified in total?  
🔍 Find it: the print at the top of the error browser shows the count.

Q38. Do the misclassified images share a pattern (pose, lighting, accessories, occlusion)?  
🔍 Find it: browse the first 10 mistakes. Write down one common feature you notice.

Q39. Which class does the CNN confuse more — female→male or male→female?  
🔍 Find it: paste into a new cell after the confusion matrix code:
```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(cnn_bench_m['labels'], cnn_bench_m['preds'])
print(f"female→male errors: {cm[0,1]}   male→female errors: {cm[1,0]}")
```

---

**Part 4 · Cell 57 — Grad-CAM**

Q40. For the same image, does the CNN heatmap or the pretrained model heatmap cover a tighter region of the face?  
🔍 Find it: cell 59 browser. Try at least 3 images. The pretrained model usually shows a more focused heatmap — does yours?

Q41. Find one image where the pretrained model focuses on something *other than the face* (hair, glasses, background). Does it still predict correctly?  
🔍 Find it: browse cell 59 and look for heatmaps that are spread outside the face area.

---

### 🔴 Advanced — write code, design experiments, reason from evidence

---

**Part 2 · Cell 30 — Full fine-tuning**

Q42. Set `FREEZE_BACKBONE = False` and `LR_PRETRAINED = 1e-5`. Train. Does benchmark accuracy improve over the frozen baseline?  
🔍 Do it: change both in cell 05, re-run 26–29. Why use `1e-5` and not the default `1e-4`? What happens if you use `1e-4` instead? Try it.

Q43. Write a snippet to count how many gradient updates the backbone receives per epoch and compare it to the head-only case:
```python
backbone_params = sum(p.numel() for name, p in pretrained_model.named_parameters()
                      if p.requires_grad and 'classifier' not in name)
head_params     = sum(p.numel() for name, p in pretrained_model.named_parameters()
                      if p.requires_grad and 'classifier' in name)
print(f"Backbone trainable: {backbone_params:,}   Head trainable: {head_params:,}")
```
What fraction of total parameters are updated during full fine-tuning?

---

**Part 3 · Cell 43 — Architecture search**

Q44. Run three configurations of the CNN and record benchmark accuracy for each. Fill in the table:

| NUM_CHANNELS | DROPOUT_CNN | USE_BATCHNORM | benchmark_acc |
|---|---|---|---|
| 32 | 0.0 | False | ? |
| 32 | 0.3 | True | ? |
| 64 | 0.3 | True | ? |

🔍 Do it: change values in cell 43, re-run 43–49. Read `cnn_bench_m['accuracy']` each time.  
Which single change had the largest effect? Can you beat the pretrained model's benchmark accuracy?

Q45. Write a loop that trains the CNN with three different learning rates and prints the final val accuracy for each:
```python
for lr in [1e-2, 1e-3, 1e-4]:
    model = HandcraftedCNN(num_channels=32, dropout_ratio=0.0,
                           use_batchnorm=True, img_size=IMG_SIZE_PRETRAINED).to(device)
    opt   = Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(5):   # quick 5-epoch test
        train_one_epoch_pretrained(model, pretrained_train_loader, loss_fn, opt)
    m = evaluate_pretrained(model, pretrained_val_loader, loss_fn)
    print(f"LR={lr:.0e}  val_acc={m['accuracy']:.3f}")
```
Which learning rate works best for 5 epochs? Does the ranking hold at 20 epochs?

---

**Part 4 · Cell 52 — Confusion matrix analysis**

Q46. From the confusion matrices, which class does each model confuse more: female→male or male→female?  
🔍 Find it:
```python
from sklearn.metrics import confusion_matrix
cm_pretrained = confusion_matrix(pretrained_bench_metrics['labels'], pretrained_bench_metrics['preds'])
cm_cnn        = confusion_matrix(cnn_bench_m['y_true'], cnn_bench_m['y_pred'])
print("Pretrained FP/FN:", cm_pretrained[0,1], "/", cm_pretrained[1,0])
print("CNN        FP/FN:", cm_cnn[0,1],        "/", cm_cnn[1,0])
```
Which direction of error is more common for each model? Do they make the same types of mistakes?

---

**Part 5 · Cell 62 — TensorBoard sweep**

Q47. Set `RUN_TUNING = True` and run cells 63–64. After TensorBoard opens, which learning rate reaches the best val accuracy in the fewest epochs?  
🔍 Find it: TensorBoard → Scalars tab → group by `lr`. The run name encodes the hyperparameters.

Q48. What is the best `(lr, batch_size)` combination from the sweep? Does a larger batch size always help?  
🔍 Find it: read the final val accuracy from the TensorBoard table. Think about the trade-off between gradient noise (small batch) and compute efficiency (large batch).

---

**Extra · E.3.1 — PCA of patch tokens**

Q49. The 3-D PCA is mapped to R, G, B with no class labels used. Yet hair, skin, and background appear in different colours. What does this mean about what DINOv3 has learned?  
🔍 Find it: run cell 77 on several images. Check whether the colour assignment is consistent across different people.

Q50. Run PCA with 2 components instead of 3 and plot as a scatter coloured by class:
```python
from sklearn.decomposition import PCA
pca2 = PCA(n_components=2).fit(patches_train.reshape(-1, 384))
proj = pca2.transform(cls_train)   # project CLS tokens instead
plt.scatter(proj[:,0], proj[:,1], c=y_train, cmap='bwr', alpha=0.5, s=10)
plt.title("CLS tokens — PCA(2)"); plt.colorbar(); plt.show()
```
How much variance do the first 2 components explain? Does the scatter show class separation?

---

**Extra · E.3.2 — UMAP of CLS tokens**

Q51. Do the two classes form distinct clusters in the UMAP plot, or do they overlap?  
🔍 Find it: cell 79 output. Look for clear boundaries between colours.

Q52. If the UMAP shows clean separation, can you predict DINOv3 MLP benchmark accuracy before running E.4–E.5? Make the prediction, then check.  
🔍 Check: `print(dino_bench_metrics['accuracy'])` after cell 89.

---

**Extra · E.5 — Three-way benchmark comparison**

Q53. Rank all three approaches (handcrafted CNN, transfer learning, DINOv3 MLP) by benchmark accuracy and training time. Fill in:

| Approach | Benchmark acc | Training time | Backbone params trained |
|---|---|---|---|
| Handcrafted CNN | ? | ? | 0 (trained from scratch) |
| Transfer learning (MobileNetV2) | ? | ? | 0 (frozen) or all |
| DINOv3 MLP | ? | ? | 0 (frozen) |

🔍 Find it: cell 89 prints all three. Training time: check tqdm output for each section.  
Why does DINOv3 win despite training only 2 linear layers? What does that tell you about the value of pre-training at scale?

---

## Difficulty Labels Used

- 🟢 **Beginner** — run cell, observe output, answer with what you see
- 🟡 **Intermediate** — change one parameter, predict outcome, explain result
- 🔴 **Advanced** — multi-parameter experiments, open-ended design questions

Default path (🟢 + 🟡): Parts 1–4. Optional: Part 5 + Extra.

---

## Dependencies

- `torch`, `torchvision`
- `opencv-python` (`cv2`)
- `pytorch-grad-cam`
- `scikit-learn`, `numpy`, `pandas`, `matplotlib`
- `ipywidgets`, `tqdm`
- `requests`
- `tensorboard` (Part 5.2)
- `transformers`, `umap-learn` (Extra DINOv3 — not installed by default)

---

## Technical Notes

| Issue | Fix |
|-------|-----|
| `epoch = 1` plot crash | `plot_history()` guards on `len < 2` |
| Model not reset between runs | Experiment zone cell rebuilds model every run |
| Symlink fallback | `try symlink / except OSError: shutil.copy` |
| Person-aware split | Same person cannot appear in both train and val |
| `IMG_SIZE_CNN` now matches `IMG_SIZE_PRETRAINED` | Both default to 144; same DataLoaders shared across Parts 2–3 |
