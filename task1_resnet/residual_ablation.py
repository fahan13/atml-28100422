import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import Bottleneck

class DisabledSkipBottleneck(Bottleneck):
    """Same as Bottleneck, but the skip connection is disabled."""
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        # Skip connection removed: no "out += identity" here
        out = self.relu(out)
        return out


# Load a fresh pretrained ResNet-152, same as baseline
model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)

# Freeze the backbone (same as subtask 1)
for param in model.parameters():
    param.requires_grad = False

# Replace the head for CIFAR-10 (same as subtask 1)
model.fc = nn.Linear(in_features=2048, out_features=10)



blocks_to_break = [0, 1, 2]  # indices of blocks within layer3 to modify

for index in blocks_to_break:
    block = model.layer3[index]
    block.__class__ = DisabledSkipBottleneck  

print("Modified blocks:", blocks_to_break, "in layer3")
print(model.layer3[0])  


import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
val_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        train_correct += (outputs.argmax(1) == labels).sum().item()
        train_total += labels.size(0)
    train_acc = train_correct / train_total

    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += labels.size(0)
    val_acc = val_correct / val_total

    print(f"Epoch {epoch+1}/{num_epochs} | Train loss: {train_loss/train_total:.4f} | Train acc: {train_acc:.4f} | Val acc: {val_acc:.4f}")