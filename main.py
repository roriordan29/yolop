import torch
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the HUSTVL/YOLOP model
model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)

print(model.state_dict())

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from custom_dataset import CustomDataset  # import your custom dataset class
from model import YOLOP  # import the YOLOP model

# define hyperparameters
batch_size = 8
num_epochs = 10
lr = 0.001

# create a transform to normalize the input images
transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# create train and test datasets
train_dataset = CustomDataset('train', transform=transform)
test_dataset = CustomDataset('test', transform=transform)

# create data loaders for batching
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# create a YOLOP model with pre-trained weights
model = YOLOP(num_classes=2)
pretrained_dict = torch.load('yolop_weights.zip')
model.load_state_dict(pretrained_dict)

# freeze the pre-trained layers
for param in model.parameters():
    param.requires_grad = False
model.fc.requires_grad = True  # unfreeze the last layer

# create a loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)

# train the model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))

    # test the model
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

# save the fine-tuned model
torch.save(model.state_dict(), 'path/to/fine_tuned_model.pth')

