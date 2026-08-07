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

### The degradation problem (why ResNet was invented)
- CS231n: empirically, stacking more plain conv layers past a certain depth made networks perform
  *worse* — a 56-layer plain network had higher training error AND higher test error than a 20-layer one.
- Important: this rules out overfitting as the explanation, since overfitting would show lower training
  error but higher test error. Both were worse, meaning the deeper model was harder to *optimize*, not
  higher-capacity-but-overfit.
- Deeper networks are strictly more expressive (a deep net can represent everything a shallow net can,
  by setting extra layers to the identity function) — so in theory more depth should never hurt training
  performance. The fact that it did points to an optimization difficulty, not a representational one.

### How residual/skip connections fix it
- Instead of a block learning the full mapping H(x), it learns the residual F(x) = H(x) - x, and the
  block outputs F(x) + x — the input x is copied forward past the conv layers and added back in.
- This makes learning the identity function trivial (just drive F(x) to ~0), so a deep ResNet can easily
  match a shallower network's performance as a "worst case," then improve from there.
- Gradient flow mechanics (my own addition, not explicitly derived in either video, but consistent with
  what CS231n implies): the shortcut path gives gradients a direct route back through addition
  (derivative of x w.r.t. x = 1), so gradients don't have to survive multiplying through every weight
  layer to reach earlier layers — this is why 100+ layer networks became trainable.

### Expected effect of disabling skip connections (what we should observe experimentally)
- Based on the degradation-problem discussion: removing skip connections in a deep network should
  reproduce the original failure mode ResNet fixed — slower convergence, and likely worse training
  *and* validation performance compared to the version with skip connections intact, especially as we
  ablate deeper/more blocks.

## 3. Feature Hierarchies and Representations

- Both lectures describe the same core idea: early conv layers detect low-level, generic features
  (edges, colors, simple textures); as depth increases, layers detect increasingly complex,
  composite, and class-specific patterns (parts, then whole-object-level concepts).
- CS231n's direct evidence: near the end of the network (just before classification), feature vectors
  for same-class images end up close together in L2 distance — implying that late-layer
  representations are far more class-separable than early-layer ones.
- Expectation for our t-SNE/UMAP visualization: early-layer features should show weak/no class
  clustering (since they encode generic edge/texture info shared across classes), while late-layer
  features should show tight, well-separated clusters per class.

## 4. Transfer Learning and Generalization

- CS231n gives a practical 2x2 framework based on (a) how similar the new dataset is to the
  pretraining dataset, and (b) how much data the new dataset has:
  - Similar dataset, small data → freeze backbone, train linear head only.
  - Similar dataset, large data → fine-tune the whole network (still initialized from pretrained
    weights, not random).
  - Different dataset → riskier; with lots of data, training from scratch becomes more viable, but
    initializing from pretrained weights is still worth testing since there's no guaranteed outcome.
  - Different dataset + small data → hardest case; look for a pretrained model trained on something
    closer to the target domain.
- Bias caveat explicitly mentioned in lecture: a model pretrained on ImageNet will perform best on data
  that "looks like" everyday ImageNet-style images, and worse on very different domains (their example:
  photos of Mars) — this is a limitation to note when interpreting our own transfer results.