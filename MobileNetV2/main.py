import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset_path = r'data'
fruit_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

batch_size = 32
dataloader = DataLoader(fruit_dataset, batch_size=batch_size, shuffle=True)

weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights)

for param in model.features.parameters():
    param.requires_grad = False

num_classes = len(fruit_dataset.classes)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features=in_features, out_features=num_classes)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = total_loss / total
    epoch_acc = correct / total
    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}")

torch.save(model.state_dict(), "best_model.pth")
print("Model state_dict saved as best_model.pth")
