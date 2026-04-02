# cnn_features.ipynb — Structure & Content

**Topic:** Hierarchical Feature Extraction from a Pretrained CNN (ResNet-50 / VGG-16)

---

## Cell Map

| # | Type | Section | Content |
|---|------|---------|---------|
| 1 | Markdown | Title | Title, author (`susanne.suter@fhnw.ch`), learning objectives |
| 2 | Markdown | Setup | Setup section header |
| 3 | Code | Setup | Optional `pip install torch torchvision pillow scikit-learn matplotlib` |
| 4 | Markdown | Helper Functions | Section header |
| 5 | Code | Helper Functions | `pil_loader(path)` — load image as RGB PIL; `list_image_paths(path)` — glob images in dir or single file |
| 6 | Markdown | Model | Task: choose model and define Early/Mid/Late taps |
| 7 | Code | Model | `model_name = "resnet50"`; `get_resnet50_feature_extractor(device)` returning `extractor, preprocess`; analogous VGG-16 version |
| 8 | Markdown | Data | Load data section header |
| 9 | Code | Data | Colab drive mount or local path setup; `work_dir_path` |
| 10 | Code | Data | Download test image with `gdown` |
| 11 | Markdown | Data | Load sample image from Susanne instruction |
| 12 | Code | Data | `img_path = work_dir_path / "test.jpg"`; `paths = list_image_paths(img_path)` |
| 13 | Markdown | Data | Optional task: upload your own image |
| 14 | Code | Data | Colab file upload block; loads uploaded image as PIL |
| 15 | Markdown | Features | Run feature extractor section header |
| 16 | Code | Features | `extract_features(image_paths, extractor, preprocess, device, batch_size)` — returns dict `{path: {level: tensor}}` |
| 17 | Code | Features | Call `extract_features`; print summary |
| 18 | Markdown | Inspect | Inspect feature shapes (Early / Mid / Late) |
| 19 | Code | Inspect | Print shape of each level for one example key |
| 20 | Markdown | Visualise | Visualise channels from each level; explanation of feature maps |
| 21 | Code | Visualise | `show_feature_grid(t, max_ch, title, save_dir)` — grid of up to `max_ch` channels as grayscale images |
| 22 | Code | Visualise | Loop over `["early", "mid", "late"]`; calls `show_feature_grid` for the example image |
| 23 | Markdown | Conv Block | Convolutional block section header |
| 24 | Code | Conv Block | `plot_layer_channels(t, layer_name, max_ch, cmap, save_dir)` — colourmap version of feature channel grid |
| 25 | Code | Conv Block | `input_path = work_dir_path / "test.jpg"`; `out_dir = Path("resnet50_rows")`; calls `plot_layer_channels` per level |
| 26 | Markdown | Filters | Visualise filters (weights) section header |
| 27 | Code | Filters | `kernels_to_rgb_grid(weight, max_filters, title, save_path)` — visualise conv1 RGB kernels as a grid |
| 28 | Code | Filters | Calls `kernels_to_rgb_grid(model.conv1.weight, max_filters=64, title="ResNet-50 conv1 (RGB kernels)")` |

---

## Key Variables

| Variable | Set in Cell | Purpose |
|----------|-------------|---------|
| `model_name` | 7 | `"resnet50"` or `"vgg16"` |
| `extractor` | 7 | Hook-based feature extractor module |
| `preprocess` | 7 | ImageNet preprocessing transform |
| `work_dir_path` | 9 | Root directory for data/output |
| `paths` | 12 | List of image paths to process |
| `features` | 17 | `{path: {"early": tensor, "mid": tensor, "late": tensor}}` |

---

## Dependencies

- `torch`, `torchvision`
- `Pillow`
- `scikit-learn`
- `matplotlib`, `numpy`
- `gdown` (for downloading test image)
