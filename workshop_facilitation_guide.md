# Workshop Facilitation Guide — cvlab CNN Notebooks

## Core principle

**Predict → Run → Reflect.** Before every interactive cell, ask them to write down what they expect. Makes results land harder.

---

## Notebook 1 — Pretrained CNN (`cvlab_cnn_pretrained.ipynb`)

**Run this first.** Pretrained models train fast and produce strong results immediately — good hook. Handcrafted CNN then shows what it costs to build from scratch.

### You present

| Section | What to say | Time |
| --- | --- | --- |
| Setup + imports | "Run it, don't read it." | 2 min |
| Control Panel | Point at `MODEL_NAME`, `FREEZE_BACKBONE`. Ask: *"Why would freezing help?"* | 3 min |
| Transforms | "Two pipelines: one for your eyes, one for the model. ImageNet stats, not 0–1." | 2 min |
| Backbone/head diagram | Frozen feature extractor → tiny trainable head. Draw it. | 4 min |

### They explore

| Section | Task to give them | Time |
| --- | --- | --- |
| Dataset inspection (§5.1 + class balance) | "Count images per class. Is it balanced? Look at 10 images — anything surprising?" | 8 min |
| Checkpoint: freezing + params (cell 29) | "Count trainable params. How few are they training?" | 5 min |
| Quick experiment config → train (§8–9) | Change one variable. Predict curve shape. Run. | 12 min |
| Evaluation table (§10) | "Val vs benchmark gap — what does a big gap mean?" | 5 min |
| Grad-CAM (§13) | "Is the model looking at the face or the background?" | 8 min |

### Debrief (5 min)

Ask: *"Strong result — but what is actually happening inside? Could you build this yourself?"* Bridge to notebook 2.

---

## Notebook 2 — Handcrafted CNN (`cvlab_cnns_handcrafted.ipynb`)

**Key message to set upfront:** "Same task, same data — now we build the brain ourselves."

### You present

| Section | What to say | Time |
| --- | --- | --- |
| Control Panel | Walk through each knob. Ask: *"If you double N_IMAGES_PER_CLASS, what happens to training time?"* | 4 min |
| Preprocess pipeline | Show one image before/after. Explain `float32 / 255`. | 3 min |
| Architecture diagram | Draw ConvModule stack on whiteboard. Point at spatial shrinkage after each MaxPool. | 5 min |

### They explore

| Section | Task to give them | Time |
| --- | --- | --- |
| Dataset inspection (§2.1) | "Same dataset. Anything look different from notebook 1?" | 5 min |
| Subset selection (§3) | Change `N_IMAGES_PER_CLASS`. Predict effect on val F1. Then run. | 5 min |
| Overfit sanity check (§9) | "Why do we want training loss → 0 on 10 samples? What does it prove?" | 5 min |
| Full training + curves (§10) | Run, watch curves. Predict before: *"Will it overfit?"* | 10–12 min |
| Error analysis (§13) | Find one misclassified image. Compare to notebook 1 — same mistake? | 8 min |
| Grad-CAM (§14) | Compare attention maps to notebook 1. Same image, different focus? | 8 min |

### Debrief (5 min)

Side-by-side: notebook 1 F1 vs notebook 2 F1. Ask: *"What did the pretrained model buy us? What did it cost?"*

---

## Timing budget (both notebooks, ~2 h total)

```
0:00  Notebook 1 intro + transforms      11 min  (you)
0:11  Dataset inspection                  8 min  (them)
0:19  Param count + train experiment     17 min  (them)
0:36  Evaluation + Grad-CAM             13 min  (them)
0:49  Debrief + bridge to notebook 2     5 min  (you)
0:54  Notebook 2 intro + architecture   12 min  (you)
1:06  Dataset + subset exploration       10 min  (them)
1:16  Overfit check + train + curves    17 min  (them)
1:33  Error analysis + Grad-CAM         16 min  (them)
1:49  Final debrief                      5 min  (you)
1:54  Buffer / extensions                6 min
```

**Skip entirely:** TensorBoard tuning grid (notebook 1 §14) and DINOv3 extra — assign as take-home.

---

## Common failure modes to avoid

- Don't live-code from scratch — cells are already there, your job is to *frame*, not type
- Don't wait for everyone to finish before moving — set a timer, debrief with whoever ran it
- If training is slow: run the overfit cell (10 samples) while waiting — it finishes in seconds and teaches the same concept
