import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import random

compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

cifar100_train = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=img_transform)
cifar100_val = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=img_transform)

# Use a smaller training subset so all four experiments finish in reasonable time
random.seed(42)
subset_indices = random.sample(range(len(cifar100_train)), 5000)  # ~50 images/class
cifar100_train_small = Subset(cifar100_train, subset_indices)

train_loader = DataLoader(cifar100_train_small, batch_size=64, shuffle=True)
val_loader = DataLoader(cifar100_val, batch_size=64, shuffle=False)

print(f"CIFAR-100 (subset): {len(cifar100_train_small)} train, {len(cifar100_val)} val, {len(cifar100_train.classes)} classes")


def build_resnet(start_from_imagenet):
    """Builds a ResNet-152 with a fresh 100-class head.
    start_from_imagenet=True -> load ImageNet weights, False -> random init."""
    weights = models.ResNet152_Weights.IMAGENET1K_V2 if start_from_imagenet else None
    net = models.resnet152(weights=weights)
    net.fc = nn.Linear(2048, 100)
    return net


def set_unfrozen_layers(net, unfreeze_everything):
    """Decides what's allowed to train.
    unfreeze_everything=True -> the whole network trains.
    unfreeze_everything=False -> only layer4 + fc train, rest stays frozen."""
    if unfreeze_everything:
        for p in net.parameters():
            p.requires_grad = True
    else:
        for p in net.parameters():
            p.requires_grad = False
        for p in net.layer4.parameters():
            p.requires_grad = True
        for p in net.fc.parameters():
            p.requires_grad = True


def run_training_and_validate(net, epoch_count, run_label):
    net = net.to(compute_device)
    params_to_learn = [p for p in net.parameters() if p.requires_grad]
    print(f"[{run_label}] trainable parameter count: {sum(p.numel() for p in params_to_learn)}")

    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(params_to_learn, lr=0.0001)

    final_val_accuracy = 0.0

    for e in range(epoch_count):
        epoch_start_time = time.time()

        net.train()
        right, seen = 0, 0
        for batch_imgs, batch_labels in train_loader:
            batch_imgs = batch_imgs.to(compute_device)
            batch_labels = batch_labels.to(compute_device)

            opt.zero_grad()
            predictions = net(batch_imgs)
            loss_value = loss_fn(predictions, batch_labels)
            loss_value.backward()
            opt.step()

            right += (predictions.argmax(dim=1) == batch_labels).sum().item()
            seen += batch_labels.size(0)
        epoch_train_acc = right / seen

        net.eval()
        right, seen = 0, 0
        with torch.no_grad():
            for batch_imgs, batch_labels in val_loader:
                batch_imgs = batch_imgs.to(compute_device)
                batch_labels = batch_labels.to(compute_device)
                predictions = net(batch_imgs)
                right += (predictions.argmax(dim=1) == batch_labels).sum().item()
                seen += batch_labels.size(0)
        epoch_val_acc = right / seen
        final_val_accuracy = epoch_val_acc

        print(f"[{run_label}] epoch {e+1}/{epoch_count} -> train acc {epoch_train_acc:.4f}, val acc {epoch_val_acc:.4f}")
        print(f"  (took {time.time() - epoch_start_time:.1f} seconds)")

    return final_val_accuracy


run_results = {}

net_a = build_resnet(start_from_imagenet=True)
set_unfrozen_layers(net_a, unfreeze_everything=False)
run_results["pretrained_lastblock"] = run_training_and_validate(net_a, epoch_count=2, run_label="pretrained, last block only")

net_b = build_resnet(start_from_imagenet=True)
set_unfrozen_layers(net_b, unfreeze_everything=True)
run_results["pretrained_full"] = run_training_and_validate(net_b, epoch_count=2, run_label="pretrained, full backbone")

net_c = build_resnet(start_from_imagenet=False)
set_unfrozen_layers(net_c, unfreeze_everything=False)
run_results["random_lastblock"] = run_training_and_validate(net_c, epoch_count=2, run_label="random init, last block only")

net_d = build_resnet(start_from_imagenet=False)
set_unfrozen_layers(net_d, unfreeze_everything=True)
run_results["random_full"] = run_training_and_validate(net_d, epoch_count=2, run_label="random init, full backbone")

print("\n=== final validation accuracy, all four runs ===")
for label, acc in run_results.items():
    print(f"{label}: {acc:.4f}")