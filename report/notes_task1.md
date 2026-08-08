# Task 1 Notes — Inner Workings of ResNet-152
(Sources: MIT 6.S191 CNN lecture, Stanford CS231n Lecture 6 — Training CNNs & CNN Architectures)

## 1. Baseline Setup

  ### Baseline Setup — concrete numbers (from our own run)
- ResNet-152 pretrained on ImageNet has 60,192,808 total parameters.
- After freezing the backbone and replacing the 1000-class head with a fresh
  10-class head (nn.Linear(2048, 10)) for CIFAR-10, only 20,490 parameters
  remain trainable (2048*10 weights + 10 biases) — a ~0.03% of the original
  model's parameter count.
- This is a concrete illustration of *why* freezing is efficient: we get to
  reuse all ~58 million parameters' worth of learned ImageNet features for
  free, and only need to learn a small linear mapping (20,490 params) from
  those features to our 10 target classes.

  ### Baseline Setup — training results (5 epochs, frozen backbone + linear head)
| Epoch | Train Loss | Train Acc | Val Acc |
|-------|-----------|-----------|---------|
| 1     | 0.696     | 78.7%     | 83.2%   |
| 2     | 0.491     | 83.5%     | 84.5%   |
| 3     | 0.448     | 84.9%     | 84.2%   |
| 4     | 0.425     | 85.8%     | 84.1%   |
| 5     | 0.408     | 86.2%     | 85.3%   |

- Val accuracy plateaus around 84-85% after epoch 2, while train accuracy keeps
  climbing slowly — the ~20K-parameter linear head quickly extracts what it can
  from the frozen 2048-dim ResNet-152 features, and further epochs give
  diminishing returns.
- Reaching ~85% validation accuracy on CIFAR-10 after just 5 epochs, training
  only 0.03% of the model's parameters, is direct evidence that the ImageNet-
  pretrained backbone's features already generalize well to a different (but
  related) image classification task — supporting the case that training
  ResNet-152 from scratch on CIFAR-10 would be both unnecessary and wasteful.

  ### Residual Connections — training results (3 disabled skip connections in layer3)
| Epoch | Train Loss | Train Acc | Val Acc |
|-------|-----------|-----------|---------|
| 1     | 2.279     | 13.8%     | 14.2%   |
| 5     | 2.253     | 15.4%     | 14.4%   |

- Disabling skip connections in just 3 of layer3's 36 blocks caused near-total
  training failure: accuracy stays around 14-15% (barely above the 10% random-
  guess baseline for CIFAR-10's 10 classes), and loss plateaus near ln(10)≈2.303
  (the theoretical loss of a uniformly random classifier).
- This is a substantially more severe effect than the original ResNet paper's
  degradation problem (which showed elevated error, not near-complete failure
  to learn), likely because our setting differs in a key way: we're not just
  disrupting gradient flow in a plain network trained from scratch, but also
  corrupting the *forward-pass* representations that the remaining pretrained
  layers (rest of layer3, all of layer4) were never trained to interpret without
  their expected identity/residual signal.
- Confirms the qualitative direction predicted by the residual-connection theory:
  removing skip connections severely harms both convergence and final accuracy,
  even when only a small fraction of the network's blocks are affected.

  ### Feature Hierarchies — activation shapes (batch of 64 CIFAR-10 images)
| Layer  | Channels | Spatial size |
|--------|----------|--------------|
| early (layer1)  | 256  | 56x56 |
| middle (layer3) | 1024 | 14x14 |
| late (layer4)   | 2048 | 7x7   |

As depth increases, spatial resolution shrinks (56->14->7) while channel
depth grows (256->1024->2048) -- concrete confirmation of the
downsample-while-deepening pattern: early layers preserve spatial detail
with fewer feature types, late layers compress spatial detail almost
entirely in favor of many more abstract, high-level feature channels.

  ### Feature Hierarchies -- t-SNE visualization (500 CIFAR-10 validation images)
See: task1_resnet/feature_hierarchy_tsne.png

**Note:** first attempt used only 64 images + raw t-SNE and showed no
clustering anywhere. Fixed by using ~500 images (~50/class) and running
PCA (to 50 dims) before t-SNE -- standard practice, needed for a clean result.

- Early layer (layer1): all 10 classes mixed together, no clustering --
  matches early layers detecting generic, class-agnostic features (edges, textures).
- Middle layer (layer3): partial clustering starts forming, but animal
  classes still overlap heavily.
- Late layer (layer4): clean, well-separated clusters for almost every class --
  strong evidence for why a simple linear head (subtask 1) reaches ~85% accuracy.
- Bonus: late-layer plot splits into two semantic halves -- animals vs.
  vehicles -- a distinction the network was never explicitly trained for.

  ### Transfer Learning — 4-way comparison (CIFAR-100, 5,000-image subset, 2 epochs)
Note: dataset reduced to 5,000 images (from 50,000) and capped at 2 epochs due
to GPU time/thermal constraints on the laptop used. Absolute accuracies are far
from each setting's ceiling, but the *relative* comparison (the actual point of
this subtask) is clear and consistent with theory.

| Setting | Final Val Acc | Time for 2 epochs |
|---|---|---|
| Pretrained, last-block-only | 48.6% | ~3.5 min |
| Pretrained, full-backbone | 68.6% | ~61 min |
| Random-init, last-block-only | 3.1% | ~29 min |
| Random-init, full-backbone | 3.4% | ~68 min |

- Pretrained massively outperforms random-init in both fine-tuning modes
  (48-69% vs 3%), confirming that ImageNet-pretrained features transfer well
  even to a new, harder, fine-grained dataset (CIFAR-100, 100 classes).
- Random-init barely improves over 2 epochs (~3%, near the 1% random-guess
  floor for 100 classes) — training from scratch needs far more data/epochs
  to get off the ground, consistent with CS231n's framework.
- Full-backbone fine-tuning beats last-block-only in both conditions, but at
  steep compute cost: ~17x longer for pretrained (+20 accuracy points),
  illustrating the real compute-vs-accuracy tradeoff this subtask asks about.

  ### Why is it unnecessary/impractical to train ResNet-152 from scratch on a small dataset?
- ResNet-152 has tens of millions of parameters, originally trained on ImageNet (~1.2M images, 1000 classes).
- CIFAR-10 has only 50,000 training images across 10 classes — far too little data relative to the
  model's capacity to train all those parameters from random initialization without badly overfitting.
- From CS231n: models trained on huge datasets learn general-purpose low/mid-level features (edges,
  textures, simple shapes) that are common across almost all natural images — relearning these from
  scratch on a small dataset just wastes compute and data.

  ### What does freezing most of the network tell us about transferability of features?
- CS231n's "linear classifier" strategy: freeze the entire pretrained backbone (`requires_grad = False`
  on all backbone params), replace only the final FC layer, and train just that new head.
- This works because the frozen backbone acts as a fixed, general-purpose feature extractor — CS231n
  compares it directly to old-school hand-designed feature extractors (e.g. color histograms), except
  here the features were *learned* on ImageNet rather than hand-designed.
- Evidence given in lecture: feature vectors from the layer just before classification cluster by class
  under L2 distance — i.e. a simple linear/k-NN classifier works well on top of these frozen features —
  which is direct evidence that the learned features are transferable, not ImageNet-specific.
- CS231n's practical framework (their 2x2 chart): with a small dataset similar to ImageNet (like
  CIFAR-10), freezing and training only a linear head is the recommended strategy — matches exactly
  what subtask 1 asks us to do.




## 2. Residual Connections in Practice


## 3. Feature Hierarchies and Representations


## 4. Transfer Learning and Generalization

