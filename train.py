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
import argparse

import imgaug.augmenters as iaa
from imgaug import parameters as iap

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, input, target):
        ce_loss = self.ce(input, target)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_factor = self.alpha.gather(0, target)
            focal_loss *= alpha_factor

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ImgAugTransform:
    def __init__(self, is_train=True):
        if is_train:
            self.aug = iaa.Sequential([
                iaa.Resize({"longer-side": 224, "shorter-side": "keep-aspect-ratio"}),
                iaa.Fliplr(0.5), 
                iaa.Affine(rotate=(-25, 25)), 
                iaa.AdditiveGaussianNoise(scale=(0, 30)),  
                iaa.Multiply((0.6, 1.4)),
                iaa.LinearContrast((0.5, 1.5)), 
                iaa.PadToFixedSize(width=224, height= 224, position="center", pad_cval=0)
            ])
        else:
            self.aug = iaa.Sequential([
                iaa.Resize({"longer-side": 224, "shorter-side": "keep-aspect-ratio"}),
                iaa.PadToFixedSize(width=224, height=224, position="center", pad_cval=0)
            ])

    def __call__(self, img):
        img = np.array(img)
        img = self.aug(image=img)
        return img

class CustomDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomDataset, self).__init__(root, transform=transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        sample = np.array(sample)
        sample = np.transpose(sample, (2, 0, 1))  # HWC to CHW
        sample = torch.tensor(sample, dtype=torch.float32) / 255.0
        return sample, target

class Model:
    def __init__(self, data_dir, num_classes, batch_size , num_epochs , learning_rate , device):
        self.data_dir = data_dir
        self.classes = num_classes
        self.batch = batch_size
        self.epochs = num_epochs
        self.lr = learning_rate
        self.device = torch.device(device)

        # Model Setting
        self.model = models.efficientnet_b0(pretrained=True)
        self.model.classifier[1] = nn.Linear( self.model.classifier[1].in_features, 4) 
        self.model = self.model.to(device)
        
    def _setup_train(self):
        train_transform = ImgAugTransform(is_train=True)
        val_transform = ImgAugTransform(is_train=False)

        train_image_folder = os.path.join(self.data_dir, 'train')       
        val_image_folder = os.path.join(self.data_dir, 'val')

        train_dataset = CustomDataset(train_image_folder, transform=train_transform)
        val_dataset = CustomDataset(val_image_folder, transform=val_transform)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch, shuffle=True)
        self.val_loader =  DataLoader(val_dataset, batch_size=self.batch)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

    def _do_train(self):
        self._setup_train()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            for images, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * images.size(0)

            epoch_loss = running_loss / len(self.train_loader.dataset)
            print(f"Train Loss: {epoch_loss:.4f}")

            self.model.eval()
            val_preds, val_labels = [], []

            with torch.no_grad():
                for images, labels in tqdm(self.val_loader, desc="Val process"):
                    images = images.to(self.device)
                    outputs = self.model(images)
                    _, preds = torch.max(outputs, 1)

                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(labels.numpy())

            acc = accuracy_score(val_labels, val_preds)
            print(f"Validation Accuracy: {acc*100:.2f}%")

            if acc > best_acc:
                best_acc = acc
                best_model_wts = copy.deepcopy(self.model.state_dict())
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f"Best model saved with accuracy: {best_acc*100:.2f}%")

        self.model.load_state_dict(best_model_wts)

        self.model.eval()
        val_preds, val_labels = [], []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)

                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.numpy())

        binary_cms = multilabel_confusion_matrix(val_labels, val_preds, labels=[0, 1, 2, 3])

        for i, cm in enumerate(binary_cms):
            plt.figure(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'])
            plt.title(f'Binary Confusion Matrix for Class {i} ({self.val_loader.dataset.classes[i]})')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            plt.savefig(f'binary_confusion_matrix_class_{i}.png')

def parse_args():
    parser = argparse.ArgumentParser(description='Train an EfficientNet model on custom dataset')
    parser.add_argument('--data_dir', type=str, default='cropped_dataset',
                        help='Directory containing train and val folders')
    parser.add_argument('--num_classes', type=int, default=4,
                        help='Number of classes in the dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate for optimizer')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to train on (cuda or cpu)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    model = Model(
        data_dir=args.data_dir,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=args.device
    )
    model._do_train()