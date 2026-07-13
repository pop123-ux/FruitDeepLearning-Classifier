import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

classes = ['Apple__Healthy', 'Apple__Rotten', 'Banana__Healthy', 'Banana__Rotten',
           'Bellpepper__Healthy', 'Bellpepper__Rotten', 'Carrot__Healthy', 'Carrot__Rotten',
           'Cucumber__Healthy', 'Cucumber__Rotten', 'Grape__Healthy', 'Grape__Rotten',
           'Guava__Healthy', 'Guava__Rotten', 'Jujube__Healthy', 'Jujube__Rotten',
           'Mango__Healthy', 'Mango__Rotten', 'Orange__Healthy', 'Orange__Rotten',
           'Pomegranate__Healthy', 'Pomegranate__Rotten', 'Potato__Healthy', 'Potato__Rotten',
           'Strawberry__Healthy', 'Strawberry__Rotten', 'Tomato__Healthy', 'Tomato__Rotten']

weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights)

num_classes = len(classes)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features=in_features, out_features=num_classes)
model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

image_path = r'testing/healthy_mango_test.jpeg'
if not os.path.exists(image_path):
    print(f"Image not found at {image_path}")
    raise SystemExit(1)

image = Image.open(image_path).convert('RGB')
image = transform(image).unsqueeze(0)

with torch.no_grad():
    outputs = model(image)
    probs = torch.nn.functional.softmax(outputs, dim=1)
    values, indices = torch.max(probs, 1)
    label = classes[indices.item()]
    print(f"Prediction: {label} (confidence: {values.item():.2f})")
