# cvlab_workshop.ipynb — Structure & Content

**Topic:** Face Image Classification with CNNs — merged workshop notebook  
**Replaces:** `cvlab_cnn_pretrained.ipynb` + `cvlab_cnns_handcrafted.ipynb`

---

## Cell Map

| # | Type | Section | Content |
|---|------|---------|---------|
| 00 | Markdown | Title | Workshop title, 3 learning outcomes, agenda table with 🟢🟡🔴 levels, Predict→Run→Reflect instruction |
| 01 | Markdown | 0. Setup | GPU warning, instructions |
| 02 | Code | 0. Setup | `pip install` (gitpython, opencv-python, grad-cam, ipywidgets, tensorboard, tqdm, pandas) |
| 03 | Code | 0. Setup | All imports (torch, torchvision, cv2, sklearn, matplotlib, ipywidgets, etc.) + IN_COLAB/device detection |
| 04 | Markdown | 1. Control panel | Explanation of unified control panel |
| 05 | Code | 1. Control panel | All hyperparameters: SEED, N_IMAGES_PER_CLASS, VALIDATION_SPLIT, BATCH_SIZE, IMG_SIZE_PRETRAINED, IMG_SIZE_CNN, EPOCHS_PRETRAINED, EPOCHS_CNN, LR_PRETRAINED, LR_CNN, MODEL_NAME, FREEZE_BACKBONE, NUM_CHANNELS, DROPOUT_CNN, USE_BATCHNORM, MOUNT_DRIVE, USE_INTERACTIVE |
| 06 | Markdown | Part 1 intro | Task framing |
| 07 | Code | 1. Data | Clone dataset (`FORCE_RECLONE = True`); define FACES_PATH, FEMALE_PATH, MALE_PATH, BENCHMARK_PATH |
| 08 | Markdown | 1. Browse | 🟢 "Your first look" prompt |
| 09 | Code | 1. Browse | `scroll_face_images()` interactive widget + static fallback |
| 10 | Markdown | 1. Channels | 🟢 "What is an image?" explanation |
| 11 | Code | 1. Channels | `show_image_channels()` — RGB channel decomposition figure |
| 12 | Markdown | 1. Splits | 🟢 Train/Val/Benchmark explanation with colored div |
| 13 | Code | 1. Splits | `plot_data_splits()` — proportional bar diagram |
| 14 | Markdown | Part 2 intro | Transfer learning intro |
| 15 | Markdown | 2. Backbone | 🟢 Backbone/head explanation with styled div |
| 16 | Code | 2. Backbone | `plot_backbone_head_diagram()` — matplotlib figure with FancyBboxPatch |
| 17 | Markdown | 2. Transforms | Normalization explanation |
| 18 | Code | 2. Transforms | imagenet_mean/std, display/train/eval transforms, TRAIN_PATH creation, ImageFolder datasets |
| 19 | Code | 2. Split | Person-aware train/val split for pretrained model; optional N_IMAGES_PER_CLASS subsample |
| 20 | Markdown | 2. Batch viz | 🟢 Batch size explanation |
| 21 | Code | 2. Batch viz | `show_batch()` — grid of BATCH_SIZE images |
| 22 | Markdown | 2. Model | Build model header |
| 23 | Code | 2. Model | `build_pretrained_model()` — MobileNetV2/ResNet50/ViT-B/16, freeze logic, custom head |
| 24 | Markdown | 2. Checkpoint | 🟢 checkpoint + 🟡 UNFREEZE_LAST_BLOCK tip |
| 25 | Code | 2. Train utils | `train_one_epoch_pretrained()`, `evaluate_pretrained()`, `fit_pretrained()`, `plot_history()` |
| 26 | Code | 2. Train | Train pretrained model |
| 27 | Code | 2. Curves | `plot_history()` call |
| 28 | Markdown | 2. Evaluate | 🟢 benchmark evaluation prompt |
| 29 | Code | 2. Evaluate | `evaluate_pretrained()` on val + benchmark; confusion matrix |
| 30 | Markdown | 2. Experiments | 🟡 Quick experiment table (epochs, aug, freeze, model) |
| 31 | Markdown | Part 3 intro | Handcrafted CNN intro |
| 32 | Markdown | 3. Convolution | 🟢 Convolution explanation |
| 33 | Code | 3. Convolution | `plot_convolution_intuition()` — kernel sliding figure |
| 34 | Markdown | 3. Preprocess | OpenCV preprocessing explanation |
| 35 | Code | 3. Preprocess | `randomly_select_n_images()`, select subset |
| 36 | Code | 3. Preprocess | `img_preprocessing()`, build X/y arrays |
| 37 | Code | 3. Split | Person-aware train/val split for CNN; X_train, X_val |
| 38 | Code | 3. Dataset | `NumpyClassificationDataset`, DataLoaders, overfit_ds |
| 39 | Markdown | 3. Architecture | Architecture header |
| 40 | Code | 3. Architecture | `ConvModule`, `HandcraftedCNN` (3 conv blocks + classifier head) |
| 41 | Markdown | 3. Arch diagram | 🟢 Spatial diagram explanation |
| 42 | Code | 3. Arch diagram | `plot_cnn_architecture()` — shrinking spatial dims figure |
| 43 | Markdown | 3. Overfit | 🟢 Overfit sanity check explanation |
| 44 | Code | 3. Overfit | Train tiny_model on 10 images; `plot_history()` |
| 45 | Markdown | 3. Train | 🟡 Full training prompt with prediction |
| 46 | Code | 3. Train utils | `evaluate_cnn()` |
| 47 | Code | 3. Experiment | Experiment zone: EXP_CNN_EPOCHS, EXP_CNN_LR, EXP_NUM_CHANNELS, etc.; model rebuilt every run |
| 48 | Code | 3. Train | Training loop with tqdm; cnn_history |
| 49 | Code | 3. Curves | `plot_history(cnn_history)` |
| 50 | Markdown | 3. Reflection | 🟢 Reflection + 🔴 advanced experiment table |
| 51 | Markdown | Part 4 intro | Compare & Interpret intro |
| 52 | Code | 4. Benchmark | Load benchmark images for CNN; X_bench, y_bench, cnn_bench_ds |
| 53 | Code | 4. Comparison | Side-by-side metrics table (pretrained vs CNN, val + benchmark) |
| 54 | Markdown | 4. Checkpoint | 🟢 discussion questions |
| 55 | Code | 4. Confusion | Side-by-side confusion matrices |
| 56 | Markdown | 4. Browse | 🟡 prediction browser prompt |
| 57 | Code | 4. Browse | `show_prediction()`, interactive benchmark browser for CNN |
| 58 | Markdown | 4. Errors | 🟡 error analysis prompt |
| 59 | Code | 4. Errors | Misclassified browser sorted by confidence |
| 60 | Markdown | 4. Grad-CAM | 🟡 Grad-CAM explanation |
| 61 | Code | 4. Grad-CAM | `_last_conv()`, target layer setup for both models |
| 62 | Code | 4. Grad-CAM | `browse_gradcam_comparison()` — side-by-side Grad-CAM for CNN + pretrained |
| 63 | Markdown | 4. Discussion | 🟢 final discussion prompts + takeaway |
| 64 | Markdown | Part 5 intro | Extensions header |
| 65 | Markdown | 5.1 Upload | Upload your own photo |
| 66 | Code | 5.1 Upload | `predict_image_both_models()` — Colab upload or IMAGE_PATH |
| 67 | Markdown | 5.2 TensorBoard | 🔴 TensorBoard tuning intro |
| 68 | Code | 5.2 TensorBoard | TUNING_GRID, RUN_TUNING=False guard |
| 69 | Code | 5.2 TensorBoard | Tuning loop with SummaryWriter |
| 70 | Markdown | 5.2 TensorBoard | How to launch TensorBoard (Colab + terminal) |
| 71 | Markdown | 5.3 DINOv3 | 🔴 DINOv3 pointer to handcrafted notebook |
| 72 | Markdown | Wrap-up | 5-point summary, closing question |

---

## Key Variables

| Variable | Set in Cell | Purpose |
|----------|-------------|---------|
| `SEED` | Control panel | Reproducibility |
| `N_IMAGES_PER_CLASS` | Control panel | Subset size per class |
| `VALIDATION_SPLIT` | Control panel | Train/val fraction |
| `BATCH_SIZE` | Control panel | DataLoader batch size |
| `IMG_SIZE_PRETRAINED` | Control panel | Input size for pretrained model (144 default, 224 for ViT) |
| `IMG_SIZE_CNN` | Control panel | Input size for handcrafted CNN (96) |
| `MODEL_NAME` | Control panel | Pretrained backbone: "mobilenet_v2" / "resnet50" / "vit_b_16" |
| `FREEZE_BACKBONE` | Control panel | Freeze pretrained backbone weights |
| `NUM_CHANNELS` | Control panel | Base conv filter count for CNN |
| `DROPOUT_CNN` | Control panel | Dropout in CNN layers |
| `USE_BATCHNORM` | Control panel | BatchNorm in ConvModules |
| `pretrained_model` | Cell 23 | Transfer learning model |
| `cnn_model` | Cell 47 | Handcrafted CNN model |
| `X`, `y` | Cell 36 | Numpy arrays: preprocessed images + labels |
| `X_bench`, `y_bench` | Cell 52 | Benchmark set for CNN |
| `pretrained_bench_metrics` | Cell 29 | Pretrained benchmark evaluation results |
| `cnn_bench_m` | Cell 53 | CNN benchmark evaluation results |

---

## New Visualizations (compared to original notebooks)

| Cell | Function | What it shows |
|------|----------|---------------|
| 11 | `show_image_channels()` | RGB channel decomposition of one face image |
| 13 | `plot_data_splits()` | Proportional bar: Train / Val / Benchmark with labels |
| 16 | `plot_backbone_head_diagram()` | Frozen backbone + trainable head with color coding |
| 21 | `show_batch()` | Grid of BATCH_SIZE images = one training step |
| 33 | `plot_convolution_intuition()` | 5×5 patch × 3×3 kernel → output value |
| 42 | `plot_cnn_architecture()` | Spatial dimensions shrinking, channels growing |
| 62 | `browse_gradcam_comparison()` | Side-by-side Grad-CAM: CNN vs pretrained |

---

## Difficulty Labels Used

- 🟢 **Beginner** — observation, reading output, running cells
- 🟡 **Intermediate** — changing one parameter, predicting outcome
- 🔴 **Advanced** — extensions, multi-parameter experiments, optional sections

Default path (🟢 + 🟡): Parts 1–4. Optional: Part 5.

---

## Technical Fixes Applied (vs original notebooks)

| Issue | Fix |
|-------|-----|
| `FORCE_RECLONE = False` default | Changed to `True` |
| `epoch = 1` plot crash | `plot_history()` guards on `len < 2` |
| Model not reset between runs | Experiment zone cell rebuilds model every run |
| Symlink fallback | `try symlink / except OSError: shutil.copy` |
| Local Colab setup removed | Colab-web only instructions |

---

## Dependencies

- `torch`, `torchvision`
- `opencv-python` (`cv2`)
- `pytorch-grad-cam`
- `scikit-learn`, `numpy`, `pandas`, `matplotlib`
- `ipywidgets`, `tqdm`
- `gitpython`
- `tensorboard` (Extensions §5.2)
- `transformers`, `umap-learn` (Extensions §5.3, not installed by default)
