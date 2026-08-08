import torch
import torch.nn as nn
import torchvision.models as models
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# same as baseline
model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(in_features=2048, out_features=10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()  # we're only doing forward passes here, no training

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
val_loader = DataLoader(val_set, batch_size=64, shuffle=True)

print("Model and data ready.")


features = {}

def make_capture_function(layer_name):
    def capture(module, input, output):
        features[layer_name] = output.detach().cpu()
    return capture

model.layer1.register_forward_hook(make_capture_function("early"))
model.layer3.register_forward_hook(make_capture_function("middle"))
model.layer4.register_forward_hook(make_capture_function("late"))

# gather ~500 images across multiple batches instead of just one batch of 64 --
# t-SNE needs enough points per class to actually show clustering
all_images, all_labels = [], []
for images, labels in val_loader:
    all_images.append(images)
    all_labels.append(labels)
    if len(all_images) * 64 >= 500:
        break

images = torch.cat(all_images)[:500]
labels = torch.cat(all_labels)[:500]
images = images.to(device)

with torch.no_grad():
    _ = model(images)

for layer_name, feature_map in features.items():
    print(f"{layer_name}: shape {feature_map.shape}")


# feature visualization using t-SNE
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

labels_np = labels.numpy()
class_names = val_set.classes

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, layer_name in zip(axes, ["early", "middle", "late"]):
    feature_map = features[layer_name]                        # shape: [500, channels, H, W]
    flattened = feature_map.reshape(feature_map.size(0), -1)   # shape: [500, channels*H*W]

    # PCA first to cut down the huge flattened dimension to something
    # manageable and less noisy, then t-SNE on that reduced version
    reduced = PCA(n_components=50, random_state=42).fit_transform(flattened.numpy())
    tsne_result = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(reduced)

    scatter = ax.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels_np, cmap="tab10")
    ax.set_title(f"{layer_name} layer features")

handles, _ = scatter.legend_elements()
fig.legend(handles, class_names, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig("task1_resnet/feature_hierarchy_tsne.png", bbox_inches="tight")
print("Saved plot to task1_resnet/feature_hierarchy_tsne.png")