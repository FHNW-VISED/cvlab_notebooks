# cvlab_cnn_pretrained.ipynb — Structure & Content

**Topic:** Transfer Learning for Face Image Classification (pretrained CNN backbones)

---

## Cell Map

| # | Type | Section | Content |
|---|------|---------|---------|
| 1 | Markdown | Title | Workshop title and introduction to Transfer Learning for Face Image Classification |
| 2 | Markdown | Roadmap | Part A (Core lab) and Part B (Extensions) overview |
| 3 | Markdown | Setup | Required packages instructions |
| 4 | Code | Setup | `pip install gitpython grad-cam ipywidgets tensorboard` (optional) |
| 5 | Code | Imports | `Path`, `matplotlib`, `numpy`, `pandas`, `torch`, `torchvision` models (MobileNetV2, ResNet50, ViT-B/16), `SummaryWriter`; sets `device` |
| 6 | Markdown | Control Panel | Instructions for editing baseline parameters + "re-run from here" note |
| 7 | Code | Control Panel | `IMG_SIZE`, `BATCH_SIZE`, `MODEL_NAME`, `SEED`, `SHOW_DATASET_EXAMPLES`, `EPOCHS`, `LR`, `DROPOUT`, `FREEZE_BACKBONE`, `UNFREEZE_LAST_BLOCK` |
| 8 | Markdown | Checkpoint | Questions about expected settings impact |
| 9 | Markdown | Data | Instructions and expected dataset folder structure |
| 10 | Code | Data | Git repo setup and `DATA_PATH` / `BENCHMARK_PATH` configuration |
| 11 | Markdown | Transforms | Explanation of display vs model transform pipelines |
| 12 | Code | Transforms | ImageNet mean/std, `display_transform` and `model_transform` definitions |
| 13 | Markdown | Dataset | How `ImageFolder` is loaded with both transform types |
| 14 | Code | Dataset | Load `ImageFolder` for `full_train_display`, `full_train_dataset`, `benchmark_display`, `benchmark_dataset`; defines `class_names` |
| 15 | Markdown | Inspection | "### 5.1 Single image" intro + 2 questions about tensor shape and pixel value range |
| 16 | Code | Inspection | Load `full_train_display[0]` and `full_train_model[0]`; print shapes; side-by-side `plt.imshow` of display vs normalized tensor |
| 17 | Markdown | Inspection | Class balance section intro |
| 18 | Code | Inspection | `Counter`-based class distribution analysis |
| 19 | Code | Inspection | Bar plot of class balance |
| 20 | Markdown | Checkpoint | Questions about class balance |
| 21 | Markdown | Browser | Interactive widget intro |
| 22 | Code | Browser | `show_example_grid` function; imports `ipywidgets`, `display` |
| 23 | Code | Browser | `browse_dataset` function with `IntSlider`; calls `browse_dataset(full_train_display, class_names)` |
| 22 | Markdown | Your Turn | Discussion of image variations |
| 23 | Markdown | Sanity | Checks before training section |
| 24 | Code | Sanity | Print batch shape and labels; run dataloader iteration |
| 25 | Code | Sanity | `denormalize` function; visualize a denormalized batch |
| 26 | Markdown | Model | Pretrained backbone options explanation |
| 27 | Code | Model | `build_model(model_name, num_classes, dropout, freeze_backbone, unfreeze_last_block)` supporting `mobilenet_v2`, `resnet50`, `vit_b_16`; instantiates `model` |
| 28 | Markdown | Model | ASCII diagram + prose explaining backbone (frozen feature extractor) vs classifier head (trained Linear layer); note on freezing and `UNFREEZE_LAST_BLOCK` |
| 29 | Markdown | Checkpoint | Questions about freezing backbone and dropout |
| 29 | Code | Sanity | Forward pass sanity check with `torch.no_grad()` |
| 30 | Markdown | Training | Training utilities section intro |
| 31 | Code | Training | Import `tqdm`, `sklearn.metrics` (`confusion_matrix`, `precision_score`, `recall_score`, `f1_score`) |
| 32 | Markdown | Training | Transfer learning expected behaviour description |
| 33 | Code | Training | `CrossEntropyLoss`, `Adam` optimizer; `train_one_epoch` and `evaluate` helpers; training loop; stores `history` |
| 34 | Code | Training | `plot_history(history)` call to visualize learning curves |
| 35 | Markdown | Reflection | Prompt about learning curve interpretation |
| 36 | Markdown | Evaluation | Validation and benchmark sets explanation |
| 37 | Code | Evaluation | `evaluate_model` function; runs on val and benchmark sets |
| 38 | Code | Evaluation | Metrics DataFrame comparing split results |
| 39 | Code | Evaluation | Confusion matrix plot for benchmark |
| 40 | Markdown | Checkpoint | Questions about separate benchmark set |
| 41 | Markdown | Predictions | Interactive predictions section intro |
| 42 | Code | Predictions | `predict_single_image(index)` with model predictions and per-class confidences |
| 43 | Markdown | Your Turn | Browse confident/uncertain predictions |
| 44 | Markdown | Error Analysis | Error analysis section intro |
| 45 | Code | Error Analysis | `collect_predictions` function gathering all predictions with metrics |
| 46 | Code | Error Analysis | Query misclassified examples sorted by confidence |
| 47 | Code | Error Analysis | `show_prediction_rows` function to display example rows |
| 48 | Markdown | Reflection | Prompt on prediction mistakes |
| 49 | Markdown | **Grad-CAM** | Section 13 intro — target layer selection table per model |
| 50 | Code | **Grad-CAM** | Import `GradCAM`, `ClassifierOutputTarget`, `show_cam_on_image`; select `target_layer`; define `gradcam_for_index(index, target_class)` |
| 51 | Code | **Grad-CAM** | `browse_gradcam(dataset_display, dataset_model, class_names)` — interactive slider showing original + GradCAM overlay side-by-side with predicted label and confidence; calls `browse_gradcam(benchmark_display, benchmark_dataset, class_names)` |
| 52 | Markdown | Grad-CAM | Final discussion prompts (model focus, trust, limitations) |
| 53 | Markdown | TensorBoard | Section 14 — hyperparameter tuning with TensorBoard |
| 54 | Code | TensorBoard | Import `datetime`, `itertools`; define `RUNS_DIR` and `TUNING_GRID` |
| 55 | Markdown | TensorBoard | Instructions for launching TensorBoard |
| 56 | Code | TensorBoard | Tuning loop over `iter_experiments(TUNING_GRID)` |
| 57 | Markdown | Reflection | Final reflection on hyperparameter impact |
| 58 | Markdown | Wrap-up | Workshop learning objectives summary |
| 59–61 | Markdown | — | Empty/continuation cells |

---

## Key Variables

| Variable | Set in Cell | Purpose |
|----------|-------------|---------|
| `MODEL_NAME` | 7 | Selects backbone (`mobilenet_v2` / `resnet50` / `vit_b_16`) |
| `IMG_SIZE` | 7 | Input resolution |
| `BATCH_SIZE` | 7 | Dataloader batch size |
| `FREEZE_BACKBONE` | 7 | Freeze pretrained weights |
| `UNFREEZE_LAST_BLOCK` | 7 | Fine-tune last backbone block |
| `benchmark_display` | 14 | Benchmark set with display transforms |
| `benchmark_dataset` | 14 | Benchmark set with model transforms |
| `model` | 27 | Instantiated pretrained model |
| `target_layer` | 50 | Layer used for Grad-CAM hooks |

---

## Dependencies

- `torch`, `torchvision`
- `pytorch-grad-cam`
- `ipywidgets`
- `tensorboard`
- `scikit-learn`, `pandas`, `matplotlib`, `numpy`
