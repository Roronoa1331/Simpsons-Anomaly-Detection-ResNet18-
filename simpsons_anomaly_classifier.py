"""
Simpsons Character Classifier - Training Script
Train a ResNet18-based classifier on Simpsons character images
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Configuration
CONFIG = {
    'train_dir': 'dataset/simpsons_dataset',
    'img_size': 128,
    'batch_size': 32,
    'num_epochs': 15,
    'learning_rate': 0.001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 2,
    'train_split': 0.8,
    'model_save_path': 'best_classifier.pth',
}

print("\n" + "="*70)
print("Simpsons Character Classifier - Training")
print("="*70)
print(f"Device: {CONFIG['device']}")
print(f"Epochs: {CONFIG['num_epochs']}")
print(f"Batch size: {CONFIG['batch_size']}")
print(f"Image size: {CONFIG['img_size']}")
print("="*70 + "\n")


class SimpsonsDataset(Dataset):
    """Custom Dataset for loading Simpsons character images"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def load_training_data(train_dir, train_split=0.8):
    """Load training data from directory"""

    print("="*70)
    print("Loading Training Data")
    print("="*70)

    train_paths = []
    train_labels = []

    # Get all character class directories
    class_dirs = [d for d in os.listdir(train_dir)
                 if os.path.isdir(os.path.join(train_dir, d)) and not d.startswith('.')]
    class_dirs = sorted(class_dirs)
    class_to_idx = {cls: idx for idx, cls in enumerate(class_dirs)}

    print(f"\nFound {len(class_dirs)} character classes:")
    for class_name in class_dirs:
        class_dir = os.path.join(train_dir, class_name)
        images = [os.path.join(class_dir, img) for img in os.listdir(class_dir)
                 if img.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))]

        if len(images) == 0:
            continue

        train_paths.extend(images)
        train_labels.extend([class_to_idx[class_name]] * len(images))
        print(f"  {class_name}: {len(images)} images")

    # Split into train and validation
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, train_size=train_split, random_state=42, stratify=train_labels
    )

    print(f"\nData split:")
    print(f"  Training: {len(train_paths)} images")
    print(f"  Validation: {len(val_paths)} images")
    print(f"  Total classes: {len(class_to_idx)}")
    print("="*70 + "\n")

    return train_paths, train_labels, val_paths, val_labels, class_to_idx


class CharacterClassifier(nn.Module):
    """CNN Classifier based on ResNet18"""

    def __init__(self, num_classes):
        super(CharacterClassifier, self).__init__()
        # Use pretrained ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=True)
        # Replace final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_path):
    """Train the classifier model"""

    print("="*70)
    print("Training Classifier")
    print("="*70)

    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Print progress
        if (epoch + 1) % 1 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {save_path}")
    print("="*70 + "\n")

    return model, history


def visualize_training_history(history, save_path='training_history.png'):
    """Visualize training and validation metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history saved to {save_path}")
    plt.close()


def main():
    """Main training function"""

    # Load data
    train_paths, train_labels, val_paths, val_labels, class_to_idx = load_training_data(
        CONFIG['train_dir'], CONFIG['train_split']
    )

    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = SimpsonsDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = SimpsonsDataset(val_paths, val_labels, transform=val_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                             shuffle=True, num_workers=CONFIG['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'],
                           shuffle=False, num_workers=CONFIG['num_workers'])

    # Initialize model
    num_classes = len(class_to_idx)
    model = CharacterClassifier(num_classes).to(CONFIG['device'])

    print(f"Model initialized with {num_classes} output classes\n")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    # Train model
    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        CONFIG['num_epochs'], CONFIG['device'], CONFIG['model_save_path']
    )

    # Visualize training history
    visualize_training_history(history)

    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"\nGenerated files:")
    print(f"  - {CONFIG['model_save_path']}: Trained model weights")
    print(f"  - training_history.png: Training curves")
    print(f"\nNext step:")
    print(f"  Run 'python inference_analysis.py' to perform anomaly detection")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

