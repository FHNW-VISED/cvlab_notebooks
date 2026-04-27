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
| 09 | Code | 1. Browse | `scroll_face_images()` interactive widget + static fallback |
| 10 | Markdown | 1. Channels | 🟢 "What is an image?" explanation |
| 11 | Code | 1. Channels | `show_image_channels()` — RGB channel decomposition figure |
| 12 | Markdown | 1. Splits | 🟢 Train/Val/Benchmark explanation with colored div + image |
| 13 | Markdown | Part 2 intro | Transfer learning intro |
| 14 | Markdown | 2. Backbone | 🟢 Backbone/head explanation with styled div + image |
| 15 | Markdown | 2. Transforms | Normalization explanation |
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

### 🟢 Beginner — run, observe, answer

These questions have **concrete, visible answers** — run the cell and look at what appears.

**Cell 08 — First look at the data**
- Do all images have the same size and framing?
- Can you spot any image that looks noisy or unusual? What might cause that?
- Before training anything: if you had to guess by eye, which class looks harder to classify and why?

**Cell 10 — What is an image?**
- What are the three numbers printed for one pixel? What do they represent?
- Which channel looks brightest in skin regions — Red, Green, or Blue?
- If you set all Blue channel values to 0, what colour shift would you expect?

**Cell 12 — Data splits**
- Why can't we use the benchmark set to decide when to stop training?
- What would go wrong if the same person appeared in both train and validation?
- The split keeps people in one set only ("person-aware"). Why is that important here?

**Cell 18 — What does a batch look like?**
- Count the images in the grid. Does it match BATCH_SIZE?
- Are the images already resized to the same shape? Why does the model require this?

**Cell 20 — How much data do you need?**
- Set `N_IMAGES_PER_CLASS = 50`, train, note the benchmark accuracy. Then try `N_IMAGES_PER_CLASS = 500`. By how much did accuracy change?
- Why does more data usually help?

**Cell 23 — Checkpoint: trainable parameters**
- Look at the parameter counts printed above. How many parameters are trainable vs frozen?
- If training time is proportional to trainable parameters, why is transfer learning so fast?

**Cell 28 — Evaluate on benchmark**
- Write your prediction first: will benchmark accuracy be higher, lower, or the same as validation?
- Run the cell. Was your prediction right? What might cause a gap between val and benchmark?

**Cell 40 — Sanity check: overfit 10 images**
- The model trains on only 10 images. What do you expect training accuracy to reach?
- Why would it be a problem if it *couldn't* overfit 10 images?

**Cell 46 — Reflection on CNN curves**
- Is training accuracy still rising at the last epoch, or has it flattened?
- Is there a large gap between train and val accuracy? What does that gap mean?
- Compare these curves to the pretrained model. Which trained faster?

**Cell 51 — Checkpoint: compare models**
- Which model has higher benchmark accuracy? By what margin?
- The pretrained model trained for fewer epochs yet performs better. Why?

**Cell 60 — Final discussion**
- Do the Grad-CAM heatmaps focus on the face, or somewhere else (hair, background)?
- If a model makes a confident correct prediction but the heatmap highlights the background, should you trust it?

---

### 🟡 Intermediate — change one thing, predict the outcome

For each experiment below: **write your prediction before running**, then check.

**Cell 23 — UNFREEZE_LAST_BLOCK**
- Set `UNFREEZE_LAST_BLOCK = True`, re-run from the control panel. How many new parameters become trainable?
- Does accuracy change? Does training time change? Why?

**Cell 28 — Benchmark vs validation gap**
- If benchmark accuracy is much lower than validation, what does that tell you about the training set?
- What would you change first to close that gap?

**Cell 30 — Quick experiments (pick one)**
1. `EPOCHS_PRETRAINED = 8` — do val curves keep rising, or plateau before epoch 8?
2. `USE_DATA_AUGMENTATION = True` — does the gap between train and val accuracy shrink? Why would augmentation reduce overfitting?
3. `FREEZE_BACKBONE = False` — how many more parameters are now trainable? Does the model improve?
4. `MODEL_NAME = "resnet50"` — is ResNet50 better or worse than MobileNetV2? How long does it take to train?

**Cell 42 — Predict CNN behaviour**
- Before running: will the handcrafted CNN overfit (train >> val)? Why or why not?
- Will it beat the pretrained model? If not, why not?

**Cell 43 — CNN experiment zone**
- Try `DROPOUT_CNN = 0.3`. Does val accuracy increase? Does train accuracy drop?
- Try `NUM_CHANNELS = 64`. Does accuracy improve? What about training time?
- Explain why dropout helps generalisation but hurts final training accuracy.

**Cell 53 — Browse predictions**
- Find one image where the model is highly confident AND correct. What visual feature do you think made it easy?
- Find one image where the model is highly confident AND wrong. Why might the model have been fooled?
- Find a prediction close to 0.5 probability. Why is the model uncertain here?

**Cell 55 — Error analysis**
- Do the misclassified images share any common visual features (pose, lighting, occlusion)?
- Which model makes more systematic errors — the CNN or the pretrained model?

**Cell 57 — Grad-CAM**
- Compare the Grad-CAM for the pretrained model vs the handcrafted CNN on the same image. Which focuses more on the face?
- A heatmap that covers the whole image uniformly suggests the model hasn't learned useful features. Do you see this in either model?

---

### 🔴 Advanced — multi-step experiments, open questions

**Cell 30 — Full fine-tuning**
- Set `FREEZE_BACKBONE = False` and `LR_PRETRAINED = 1e-5`. Train. Compare to the frozen version.
- Why use a much lower learning rate when fine-tuning the backbone? What risk does a high LR create?
- At what point does fine-tuning hurt rather than help?

**Cell 46 — CNN architecture search**
- Try three combinations of (`NUM_CHANNELS`, `DROPOUT_CNN`, `USE_BATCHNORM`). Record benchmark accuracy for each.
- Which hyperparameter has the largest effect? Can you reach the pretrained model's benchmark accuracy?
- Why is it hard to beat transfer learning even with a well-tuned handcrafted CNN?

**Cell 62 — TensorBoard hyperparameter sweep**
- Set `RUN_TUNING = True` and run all three cells. In TensorBoard, which run converges fastest?
- What does the learning rate vs. final accuracy surface look like? Is there a clear optimum?

**Extra E.3.1 — PCA of patch tokens**
- The 3-D PCA is mapped to RGB without any label supervision. Yet similar regions get similar colours. What does this tell you about DINOv3's representations?
- Does PCA separate skin, hair, and background into distinct colours? Is this consistent across images?

**Extra E.3.2 — UMAP of CLS tokens**
- Do the two classes separate cleanly in the 2-D UMAP projection? What would a clean separation mean?
- If the classes overlap, does that mean the model will fail? Or can the MLP still learn to separate them?

**Extra E.3.3 — Self-attention heads**
- Different heads attend to different regions. Can you identify what each head focuses on (edges, face, background)?
- One head sometimes attends globally (flat map) and another sharply (one region). Why might a foundation model benefit from diverse attention?

**Extra E.4–E.5 — DINOv3 vs CNN vs Transfer learning**
- Compare all three approaches on benchmark accuracy. Rank them.
- DINOv3 MLP trains for 30 epochs on frozen features in seconds, yet beats the handcrafted CNN. What does this tell you about the role of the backbone in performance?
- What is the practical trade-off between the three approaches (compute, data, interpretability)?

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
