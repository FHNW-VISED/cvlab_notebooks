# cvlab_cnns_handcrafted.ipynb — Structure & Content

**Topic:** Hand-crafted CNNs for Face Image Classification (train from scratch)

---

## Cell Map

| # | Type | Section | Content |
| --- | ------ | --------- | --------- |
| 1 | Markdown | Title | Workshop title and intro to building a small CNN |
| 2 | Markdown | 0. Setup | Package install instructions with package-group table; blue checkpoint question |
| 3 | Markdown | How to use | "Predict → Run → Reflect" boxed instruction — shown to participants after roadmap |
| 4 | Code | 0. Setup | `pip install gitpython opencv-python grad-cam tqdm pandas` |
| 4 | Code | 0. Setup | Imports: `json`, `random`, `shutil`, `Path`, `cv2`, `matplotlib`, `numpy`, `pandas`, `torch`, `sklearn`, `tqdm`, `ipywidgets`, etc.; IN_COLAB detection; device |
| 5 | Markdown | 1. Control Panel | Control panel explanation; blue "predict first" prompt before running |
| 6 | Code | 1. Control Panel | `SEED`, `IMG_SIZE`, `N_IMAGES_PER_CLASS`, `BATCH_SIZE`, `VAL_FRACTION`, `EPOCHS`, `LEARNING_RATE`, `NUM_CHANNELS`, `DROPOUT_RATIO`, `USE_BATCHNORM`, `MOUNT_DRIVE`, `USE_INTERACTIVE`, `RANDOM_EXAMPLE_INDEX`, `DATA_PATH` |
| 7 | Code | 1. Control Panel | Drive mount: `if IN_COLAB and MOUNT_DRIVE: drive.mount(...)` |
| 8 | Markdown | 2. Data | Dataset folder structure; explanation of why benchmark stays untouched; blue "Think before you run" prompt |
| 9 | Code | 2. Data | Defines `FACES_PATH`, `FEMALE_PATH`, `MALE_PATH`, `BENCHMARK_PATH`; `FORCE_RECLONE` flag (default `False`) — set `True` to delete and re-clone; idempotent git clone otherwise |
| 10 | Markdown | 2.1 Explore | Folder inspection rationale; blue "Your Turn" with class-balance and image-quality questions |
| 11 | Code | 2.1 Explore | `count_jpgs_in_directory` function; prints class counts |
| 12 | Code | 2.1 Explore | `scroll_face_images` with ipywidgets slider; call guarded by `USE_INTERACTIVE` |
| 13 | Markdown | 3. Subset | Balanced subset rationale; blue "Your Turn" asking to predict effect of different `N_IMAGES_PER_CLASS` |
| 14 | Code | 3. Subset | `randomly_select_n_images` helper; creates `female_selected/` and `male_selected/` |
| 15 | Code | 3. Subset | Calls `randomly_select_n_images` with `N_IMAGES_PER_CLASS` from control panel |
| 16 | Markdown | 4. Preprocess | 4-step preprocessing explanation (load, BGR→RGB, resize, scale); label convention |
| 17 | Code | 4. Preprocess | `LABEL_FEMALE/MALE`; `img_preprocessing`; loads `X`, `y` |
| 18 | Markdown | 4.1 Inspect | Blue checkpoint with 4 questions about `X`/`y` shape and widget inspection |
| 19 | Code | 4.1 Inspect | `show_img` + `show_label`; call guarded by `USE_INTERACTIVE` |
| 20 | Code | 4.1 Inspect | Class distribution bar plot |
| 21 | Markdown | 5. Split | Train/val split explanation; `stratify=y`; blue "Your Turn" on `VAL_FRACTION` extremes |
| 22 | Code | 5. Split | Person-aware split: groups images by person identity (filename stem minus trailing `_NNNN`), splits unique persons per class into train/val at `VAL_FRACTION`, then collects image indices — same person cannot appear in both sets |
| 23 | Markdown | 6. Dataset | "Dataset vs DataLoader" explanation with ASCII diagram; `pin_memory` note |
| 24 | Code | 6. Dataset | `NumpyClassificationDataset(Dataset)` class; creates `train_dataset`, `val_dataset`, `train_loader`, `val_loader` (using `BATCH_SIZE`) |
| 25 | Code | 6. Dataset | Sanity check — batch shape and label print using `train_loader` |
| 26 | Markdown | 7. Model | ASCII architecture diagram; BN and MaxPool explanations; blue checkpoint on spatial size |
| 27 | Code | 7. Model | `ConvModule(nn.Module)`: conv → BN → ReLU → dropout → MaxPool; `HandcraftedCNN(nn.Module)`: stack of ConvModules + classifier head |
| 28 | Markdown | 7.1 Sanity | Sanity check rationale (output shape, forward pass, param count); blue "Your Turn" on output units and parameter comparison |
| 29 | Code | 7.1 Sanity | `count_parameters`; forward pass shape check |
| 30 | Markdown | 8. Utilities | Function-table for `evaluate_model`, `train_with_history`, `plot_training_performance`; 4-step training cycle table |
| 31 | Code | 8. Utilities | `evaluate_model`, `train_with_history` (with `tqdm` progress bar, prints `val_f1`), `plot_training_performance` |
| 32 | Markdown | 9. Overfit | Overfit-10-samples rationale; blue "Your Turn — Predict first, then run" |
| 33 | Code | 9. Overfit | Train `tiny_model` for 20 steps; check loss goes to ~0 |
| 34 | Markdown | 10. Train | Training guidance; expected from-scratch behaviour; blue "Your Turn — Guided experiments" table |
| 35 | Code | 10. Train | Instantiate `HandcraftedCNN` using control panel params; training loop; stores `history` |
| 36 | Code | 10. Train | `plot_training_performance(history)` |
| 37 | Markdown | Reflection | Blue reflection prompt on curves, overfitting, and comparison with pretrained |
| 38 | Code | 10. Train | `evaluate_model` on validation set; print metrics + confusion matrix |
| 39 | Markdown | 11. Save | Explains saving `.pth` + `.json` config pattern |
| 40 | Code | 11. Save | Save model weights + JSON config to `MODELS_PATH / "handcrafted"` |
| 41 | Markdown | 12. Benchmark | Held-out evaluation explanation; metrics table with random baselines; val/benchmark gap guidance |
| 42 | Code | 12. Benchmark | Load benchmark images into `X_bench`, `y_bench` |
| 43 | Code | 12. Benchmark | `evaluate_model` on benchmark; print metrics + confusion matrix |
| 44 | Markdown | Reflection | Blue reflection prompt comparing val vs benchmark; cross-group comparison |
| 45 | Markdown | 13. Predictions | Individual predictions intro; blue "Your Turn" on confident vs uncertain examples |
| 46 | Code | 13. Predictions | Get probs; `show_example_prediction`; interactive slider guarded by `USE_INTERACTIVE` |
| 47 | Markdown | 13.1 Misclassified | Misclassified examples rationale; blue "Your Turn" with pattern-detection questions |
| 48 | Code | 13.1 Misclassified | `show_misclassified` with interactive slider guarded by `USE_INTERACTIVE` |
| 49 | Markdown | 13.2 Correct | Correct examples; easy vs hard; blue "Your Turn" on high/low confidence correct examples |
| 50 | Code | 13.2 Correct | `show_correctly_classified` with interactive slider guarded by `USE_INTERACTIVE` |
| 51 | Markdown | 14. Grad-CAM | Grad-CAM explanation (gradient intuition, colour meaning, caveats); blue checkpoint on background focus |
| 52 | Code | 14. Grad-CAM | `_last_conv`; `show_example_prediction_xai`; interactive slider guarded by `USE_INTERACTIVE` |
| 53 | Markdown | 15. Extensions | Four blue-headed exercises (LR, Dropout, Width, Interpretation) |
| 54 | Markdown | 16. Wrap-up | Workflow summary; comparison with transfer learning; blue final reflection with 5 questions |
| 55 | Markdown | Extra: DINOv3 | Foundation model adaptation intro |
| 56 | Code | Extra: DINOv3 | `pip install umap-learn transformers` |
| 57 | Code | Extra: DINOv3 | Import `umap`; feature extraction setup |
| 58 | Markdown | E.1 Load | Load DINOv3 ViT-S/16 backbone via HuggingFace |
| 59 | Code | E.1 Load | Read `.access_token_hf` for `HF_TOKEN` |
| 60 | Code | E.1 Load | `AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")` |
| 61 | Markdown | E.2 Features | Feature extraction pipeline description |
| 62 | Code | E.2 Features | ImageNet normalisation; extract CLS tokens and patch tokens for train/benchmark |
| 63 | Markdown | E.3 Visualise | PCA of patch tokens intro |
| 64 | Code | E.3.1 PCA | Sample 200 images; PCA on patch tokens |
| 65 | Markdown | E.3.2 UMAP | UMAP of CLS tokens intro |
| 66 | Code | E.3.2 UMAP | Combine train+benchmark; UMAP 2-D projection coloured by class |
| 67 | Markdown | E.3.3 Attention | Self-attention maps intro |
| 68 | Code | E.3.3 Attention | Discover self-attention module path in model |
| 69 | Code | E.3.3 Attention | `_find_last_self_attn(model)` helper |
| 70 | Code | E.3.3 Attention | `show_attention(i)` — visualise attention heads per image |
| 71 | Markdown | E.4 MLP | MLP classifier on frozen DINOv3 features |
| 72 | Code | E.4 MLP | `FeatureDataset(Dataset)` wrapping extracted CLS features; `DinoMLP(nn.Module)` |
| 73 | Code | E.4 MLP | Train `dino_model`; store `dino_history` |
| 74 | Code | E.4 MLP | Save DINOv3 model to `DINO_MODELS_PATH` |
| 75 | Markdown | E.5 Benchmark | DINOv3 vs handcrafted CNN comparison |
| 76 | Code | E.5 Benchmark | `evaluate_model` on both; print comparison metrics |

---

## Key Variables

| Variable | Set in Cell | Purpose |
| -------- | ----------- | ------- |
| `IMG_SIZE` | Control panel | Input image resolution (96×96) |
| `N_IMAGES_PER_CLASS` | Control panel | Number of images selected per class |
| `BATCH_SIZE` | Control panel | DataLoader batch size |
| `VAL_FRACTION` | Control panel | Fraction held out for validation |
| `EPOCHS` | Control panel | Training epochs |
| `LEARNING_RATE` | Control panel | Adam optimizer LR |
| `NUM_CHANNELS` | Control panel | Base conv filter count |
| `DROPOUT_RATIO` | Control panel | Dropout probability |
| `USE_BATCHNORM` | Control panel | Enables BatchNorm in conv blocks |
| `MOUNT_DRIVE` | Control panel | Whether to mount Google Drive in Colab |
| `USE_INTERACTIVE` | Control panel | Whether to show interactive widgets |
| `X`, `y` | Preprocess cell | Numpy arrays of images and labels |
| `X_train`, `X_validation` | Split cell | Train/val split arrays |
| `train_loader`, `val_loader` | Dataset cell | PyTorch DataLoaders |
| `model` | Training cell | `HandcraftedCNN` instance |
| `X_bench`, `y_bench` | Benchmark cell | Benchmark numpy arrays |

---

## Dependencies

- `torch`, `torchvision`
- `opencv-python` (`cv2`)
- `pytorch-grad-cam`
- `scikit-learn`, `numpy`, `pandas`, `matplotlib`
- `ipywidgets`
- `tqdm`
- `transformers`, `umap-learn` (Extra section only)
