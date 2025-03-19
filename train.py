import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report ,multilabel_confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import copy
import numpy as np
from PIL import Image
from tqdm import tqdm

def train_model(model, criterion, optimizer, train_loader, val_loader, device, num_epochs=20):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Train Loss: {epoch_loss:.4f}")

        model.eval()
        val_preds, val_labels = [], []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Val process"):
                images = images.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)

                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.numpy())

        acc = accuracy_score(val_labels, val_preds)
        print(f"Validation Accuracy: {acc*100:.2f}%")

        if acc > best_acc:
            best_acc = acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Best model saved with accuracy: {best_acc*100:.2f}%")

    model.load_state_dict(best_model_wts)

    model.eval()
    val_preds, val_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.numpy())

    binary_cms = multilabel_confusion_matrix(val_labels, val_preds, labels=[0, 1, 2, 3])

    for i, cm in enumerate(binary_cms):
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'])
        plt.title(f'Binary Confusion Matrix for Class {i} ({val_loader.dataset.classes[i]})')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(f'binary_confusion_matrix_class_{i}.png')

    return model
    
if __name__ == "__main__":
    data_dir = 'cropped_dataset'
    num_classes = 4
    batch_size = 32
    num_epochs = 1
    learning_rate = 1e-4

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset  = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)
    val_dataset  = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4) 
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    trained_model = train_model(model, criterion, optimizer, train_loader, val_loader, device, num_epochs=num_epochs)

    torch.save(trained_model.state_dict(), 'final_model.pth')
    print("Final model saved as final_model.pth")