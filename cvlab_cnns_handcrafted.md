# cvlab_cnns_handcrafted.ipynb — Structure & Content

**Topic:** Hand-crafted CNNs for Face Image Classification (train from scratch)

---

## Cell Map

| # | Type | Section | Content |
|---|------|---------|---------|
| 1 | Markdown | Title | Workshop title and intro to building a small CNN |
| 2 | Markdown | 1. Setup | Package install instructions; checkpoint question about dependencies |
| 3 | Code | 1. Setup | `pip install gitpython opencv-python grad-cam` |
| 4 | Code | 1. Setup | Imports: `json`, `random`, `shutil`, `Path`, `cv2`, `matplotlib`, `numpy`, `torch`, `sklearn`, `ipywidgets`, etc. |
| 5 | Code | 1. Setup | Colab/local detection; `DATA_PATH` configuration |
| 6 | Markdown | 2. Data | Dataset description and folder structure (`female/`, `male/`) |
| 7 | Code | 2. Data | `DATA_PATH`, `BENCHMARK_PATH`; git repo clone/pull |
| 8 | Markdown | 2.1 Explore | Folder inspection intro |
| 9 | Code | 2.1 Explore | `count_jpgs_in_directory` function; prints class counts |
| 10 | Code | 2.1 Explore | `scroll_face_images` with ipywidgets slider; displays images |
| 11 | Markdown | 3. Subset | Build a balanced training subset rationale |
| 12 | Code | 3. Subset | `randomly_select_n_images` helper; creates `female_selected/` and `male_selected/` |
| 13 | Code | 3. Subset | `n_f = 500`, `n_m = 500`; calls selection function |
| 14 | Markdown | 4. Preprocess | Resize, normalize, load into numpy arrays |
| 15 | Code | 4. Preprocess | `IMG_SIZE = 96`; loads images into `X` (float32) and `y` arrays |
| 16 | Markdown | 4.1 Inspect | Checkpoint questions about `X` and `y` shapes |
| 17 | Code | 4.1 Inspect | `show_img` function; displays a single processed image |
| 18 | Code | 4.1 Inspect | Class distribution bar plot |
| 19 | Markdown | 5. Split | Train/validation split explanation |
| 20 | Code | 5. Split | Shuffle; `VAL_FRACTION = 0.15`; `train_test_split` into `X_train`, `X_validation`, `y_train`, `y_validation` |
| 21 | Markdown | 6. Dataset | Wrap numpy arrays in PyTorch `Dataset` and `DataLoader` |
| 22 | Code | 6. Dataset | `NumpyClassificationDataset(Dataset)` class |
| 23 | Code | 6. Dataset | Sanity check — batch shape and label print |
| 24 | Markdown | 7. Model | Define hand-crafted CNN; building block description |
| 25 | Code | 7. Model | `ConvModule(nn.Module)`: conv → BN → ReLU → dropout; `HandcraftedCNN(nn.Module)`: stack of ConvModules + classifier head |
| 26 | Markdown | 7.1 Sanity | Model sanity check rationale |
| 27 | Code | 7.1 Sanity | `count_parameters`; forward pass shape check |
| 28 | Markdown | 8. Utilities | Training helpers intro |
| 29 | Code | 8. Utilities | `evaluate_model`, `train_one_epoch`, `plot_training_performance` |
| 30 | Markdown | 9. Overfit | Overfit tiny dataset debugging trick |
| 31 | Code | 9. Overfit | Train `tiny_model` for a few steps; check loss goes to ~0 |
| 32 | Markdown | 10. Train | Full model training; suggested starting config |
| 33 | Code | 10. Train | `lr = 1e-3`, `batch_size = 32`, `epochs = 20`; instantiate `HandcraftedCNN`; training loop; stores `history` |
| 34 | Code | 10. Train | `plot_training_performance(history)` |
| 35 | Markdown | Reflection | Prompt on learning curves (still learning? overfitting?) |
| 36 | Code | 10. Train | `evaluate_model` on validation set; print metrics |
| 37 | Markdown | 11. Save | Model save rationale |
| 38 | Code | 11. Save | Save model weights + JSON config to `MODELS_PATH / "handcrafted"` |
| 39 | Markdown | 12. Benchmark | Held-out benchmark evaluation explanation |
| 40 | Code | 12. Benchmark | Load benchmark images into `X_bench`, `y_bench` |
| 41 | Code | 12. Benchmark | `evaluate_model` on benchmark; print metrics |
| 42 | Markdown | Reflection | Compare validation vs benchmark performance |
| 43 | Markdown | 13. Predictions | Individual predictions intro |
| 44 | Code | 13. Predictions | Get probabilities on benchmark set with `torch.no_grad()` |
| 45 | Markdown | 13.1 Misclassified | Misclassified examples rationale |
| 46 | Code | 13.1 Misclassified | `show_misclassified(X, y_true, probs_)` |
| 47 | Markdown | 13.2 Correct | Correctly classified examples |
| 48 | Code | 13.2 Correct | `show_correctly_classified(X, y_true, probs_)` |
| 49 | Markdown | 14. Grad-CAM | Grad-CAM section intro and explanation |
| 50 | Code | 14. Grad-CAM | `_last_conv(model)` to find last Conv2d; `compute_gradcam` helper; display heatmaps on sample images |
| 51 | Markdown | 15. Extensions | Optional exercises A–D (architecture changes, augmentation, etc.) |
| 52 | Markdown | 16. Wrap-up | Full workflow summary |
| 53 | Markdown | Extra: DINOv3 | Foundation model adaptation intro |
| 54 | Code | Extra: DINOv3 | `pip install umap-learn transformers` |
| 55 | Code | Extra: DINOv3 | Import `umap`; feature extraction setup |
| 56 | Markdown | E.1 Load | Load DINOv3 ViT-S/16 backbone via HuggingFace |
| 57 | Code | E.1 Load | Read `.access_token_hf` for `HF_TOKEN` |
| 58 | Code | E.1 Load | `AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")` |
| 59 | Markdown | E.2 Features | Feature extraction pipeline description |
| 60 | Code | E.2 Features | ImageNet normalisation; extract CLS tokens and patch tokens for train/benchmark |
| 61 | Markdown | E.3 Visualise | PCA of patch tokens intro |
| 62 | Code | E.3.1 PCA | Sample 200 images; PCA on patch tokens |
| 63 | Markdown | E.3.2 UMAP | UMAP of CLS tokens intro |
| 64 | Code | E.3.2 UMAP | Combine train+benchmark; UMAP 2-D projection coloured by class |
| 65 | Markdown | E.3.3 Attention | Self-attention maps intro |
| 66 | Code | E.3.3 Attention | Discover self-attention module path in model |
| 67 | Code | E.3.3 Attention | `_find_last_self_attn(model)` helper |
| 68 | Code | E.3.3 Attention | `show_attention(i)` — visualise attention heads per image |
| 69 | Markdown | E.4 MLP | MLP classifier on frozen DINOv3 features |
| 70 | Code | E.4 MLP | `FeatureDataset(Dataset)` wrapping extracted CLS features; `DinoMLP(nn.Module)` |
| 71 | Code | E.4 MLP | Train `dino_model`; store `dino_history` |
| 72 | Code | E.4 MLP | Save DINOv3 model to `DINO_MODELS_PATH` |
| 73 | Markdown | E.5 Benchmark | DINOv3 vs handcrafted CNN comparison |
| 74 | Code | E.5 Benchmark | `evaluate_model` on both; print comparison metrics |

---

## Key Variables

| Variable | Set in Cell | Purpose |
|----------|-------------|---------|
| `IMG_SIZE` | 15 | Input image resolution (96×96) |
| `n_f`, `n_m` | 13 | Number of female/male images selected |
| `X`, `y` | 15 | Numpy arrays of images and labels |
| `X_train`, `X_validation` | 20 | Train/val split arrays |
| `model` | 33 | `HandcraftedCNN` instance |
| `X_bench`, `y_bench` | 40 | Benchmark numpy arrays |

---

## Dependencies

- `torch`, `torchvision`
- `opencv-python` (`cv2`)
- `pytorch-grad-cam`
- `scikit-learn`, `numpy`, `matplotlib`, `pandas`
- `ipywidgets`
- `transformers`, `umap-learn` (Extra section only)
