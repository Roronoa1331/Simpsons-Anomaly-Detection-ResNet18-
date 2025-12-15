"""
Simpsons Anomaly Detection - Inference and Analysis Script
Load a pre-trained classifier and perform anomaly detection analysis
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# Configuration
CONFIG = {
    'train_dir': 'dataset/simpsons_dataset',
    'test_normal_dir': 'dataset/test/normal',
    'test_anomaly_dir': 'dataset/test/anomaly',
    'img_size': 128,
    'batch_size': 32,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 2,
    'train_split': 0.8,
    'model_path': 'best_classifier.pth',
}

print("\n" + "="*70)
print("Simpsons Anomaly Detection - Inference and Analysis")
print("="*70)
print(f"Device: {CONFIG['device']}")
print(f"Model: {CONFIG['model_path']}")
print("="*70 + "\n")

# Check if model exists
if not os.path.exists(CONFIG['model_path']):
    print("ERROR: Model file not found!")
    print(f"Expected: {CONFIG['model_path']}")
    print("\nPlease train the model first:")
    print("  python simpsons_anomaly_classifier.py")
    print("\n" + "="*70 + "\n")
    sys.exit(1)


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


class CharacterClassifier(nn.Module):
    """ResNet18-based classifier"""

    def __init__(self, num_classes):
        super(CharacterClassifier, self).__init__()
        # Use same structure as training script (backbone instead of resnet)
        self.backbone = models.resnet18(pretrained=False)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def load_dataset(train_dir, test_normal_dir, test_anomaly_dir, train_split=0.8):
    """Load training and test datasets"""

    print("="*70)
    print("Loading Dataset")
    print("="*70)

    # Load training data to get class mapping
    train_paths = []
    train_labels = []

    class_dirs = [d for d in os.listdir(train_dir)
                 if os.path.isdir(os.path.join(train_dir, d)) and not d.startswith('.')]
    class_dirs = sorted(class_dirs)
    class_to_idx = {cls: idx for idx, cls in enumerate(class_dirs)}

    print(f"\nLoading training data from {len(class_dirs)} character classes:")
    for class_name in class_dirs:
        class_dir = os.path.join(train_dir, class_name)
        images = [os.path.join(class_dir, img) for img in os.listdir(class_dir)
                 if img.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))]

        if len(images) == 0:
            continue

        train_paths.extend(images)
        train_labels.extend([class_to_idx[class_name]] * len(images))
        print(f"  {class_name}: {len(images)} images")

    # Split training data
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, train_size=train_split, random_state=42, stratify=train_labels
    )

    print(f"\nTraining data split:")
    print(f"  Training: {len(train_paths)} images")
    print(f"  Validation: {len(val_paths)} images")

    # Load test data
    test_paths = []
    test_labels = []
    is_anomaly = []

    print(f"\nLoading test data:")
    if os.path.exists(test_normal_dir):
        normal_images = [os.path.join(test_normal_dir, img) for img in os.listdir(test_normal_dir)
                        if img.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))]
        test_paths.extend(normal_images)
        test_labels.extend([0] * len(normal_images))
        is_anomaly.extend([0] * len(normal_images))
        print(f"  Normal test images: {len(normal_images)}")
    else:
        print(f"  Warning: Normal test directory '{test_normal_dir}' not found")

    if os.path.exists(test_anomaly_dir):
        anomaly_images = [os.path.join(test_anomaly_dir, img) for img in os.listdir(test_anomaly_dir)
                         if img.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))]
        test_paths.extend(anomaly_images)
        test_labels.extend([-1] * len(anomaly_images))
        is_anomaly.extend([1] * len(anomaly_images))
        print(f"  Anomaly test images: {len(anomaly_images)}")
    else:
        print(f"  Warning: Anomaly test directory '{test_anomaly_dir}' not found")

    print(f"\nTest set composition:")
    print(f"  Normal: {sum(1 for x in is_anomaly if x == 0)} images")
    print(f"  Anomaly: {sum(1 for x in is_anomaly if x == 1)} images")
    print(f"  Total: {len(test_paths)} images")
    print("="*70 + "\n")

    return test_paths, test_labels, is_anomaly, class_to_idx


def get_logits_and_predictions(model, data_loader, device):
    """Get logits and predictions from model"""

    model.eval()
    all_logits = []
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)

            all_logits.append(outputs.cpu().numpy())
            all_predictions.append(outputs.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.numpy())

    logits = np.concatenate(all_logits, axis=0)
    predictions = np.concatenate(all_predictions, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    return logits, predictions, labels


def compute_anomaly_scores(logits):
    """Compute anomaly scores from logits"""

    # Method 1: Max logit (negative, so higher logit = lower anomaly score)
    max_logit = -np.max(logits, axis=1)

    # Method 2: Entropy (higher entropy = more uncertain = more anomalous)
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    entropy = -(probs * np.log(probs + 1e-10)).sum(axis=1)

    # Method 3: Max probability (negative, so lower prob = higher anomaly score)
    max_prob = -np.max(probs, axis=1)

    return {
        'max_logit': max_logit,
        'entropy': entropy,
        'max_prob': max_prob
    }


def visualize_logit_distributions(scores, is_anomaly, save_path='logit_distributions.png'):
    """Visualize logit distributions for normal vs anomaly samples"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    methods = ['max_logit', 'entropy', 'max_prob']
    titles = ['Max Logit (Anomaly Score)', 'Entropy (Uncertainty)', 'Max Probability (Negative)']

    for idx, (method, title) in enumerate(zip(methods, titles)):
        score = scores[method]
        normal_scores = score[np.array(is_anomaly) == 0]
        anomaly_scores = score[np.array(is_anomaly) == 1]

        # Histogram
        ax = axes[0, idx]
        ax.hist(normal_scores, bins=30, alpha=0.6, label='Normal', color='blue', density=True)
        ax.hist(anomaly_scores, bins=30, alpha=0.6, label='Anomaly', color='red', density=True)
        ax.set_xlabel('Score', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Box plot
        ax = axes[1, idx]
        data = [normal_scores, anomaly_scores]
        bp = ax.boxplot(data, labels=['Normal', 'Anomaly'], patch_artist=True)
        bp['boxes'][0].set_facecolor('blue')
        bp['boxes'][1].set_facecolor('red')
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(f'{title} - Box Plot', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Logit distributions saved to {save_path}")
    plt.close()


def visualize_roc_curves(scores, is_anomaly, save_path='roc_curves.png'):
    """Visualize ROC curves for different anomaly detection methods"""

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    methods = {
        'Max Logit': 'max_logit',
        'Entropy': 'entropy',
        'Max Probability': 'max_prob'
    }

    colors = ['blue', 'red', 'green']

    for (name, method), color in zip(methods.items(), colors):
        score = scores[method]
        auc = roc_auc_score(is_anomaly, score)
        fpr, tpr, _ = roc_curve(is_anomaly, score)

        ax.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})',
               linewidth=2, color=color)

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves for Anomaly Detection', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC curves saved to {save_path}")
    plt.close()


def visualize_sample_predictions(model, test_loader, is_anomaly, device,
                                 num_samples=16, focus_on_anomaly=True,
                                 save_path='sample_predictions.png'):
    """Visualize sample predictions"""

    model.eval()

    # Get all samples
    images_list = []
    logits_list = []

    with torch.no_grad():
        for images, labels in test_loader:
            images_list.append(images)
            outputs = model(images.to(device))
            logits_list.append(outputs.cpu())

    images_all = torch.cat(images_list, dim=0)
    logits_all = torch.cat(logits_list, dim=0)

    # Select samples based on focus
    if focus_on_anomaly:
        anomaly_indices = [i for i, label in enumerate(is_anomaly) if label == 1]
        normal_indices = [i for i, label in enumerate(is_anomaly) if label == 0]

        num_anomalies = len(anomaly_indices)
        num_normals = max(0, num_samples - num_anomalies)

        selected_indices = anomaly_indices[:num_anomalies]
        if num_normals > 0:
            selected_indices += normal_indices[:num_normals]

        selected_indices = selected_indices[:num_samples]
    else:
        selected_indices = list(range(min(num_samples, len(images_all))))

    # Create visualization
    rows = 4
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 16))
    axes = axes.flatten()

    for plot_idx, data_idx in enumerate(selected_indices[:rows * cols]):
        img = images_all[data_idx].permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())

        probs = F.softmax(logits_all[data_idx], dim=0).numpy()
        max_prob = probs.max()
        predicted_class = probs.argmax()
        entropy = -(probs * np.log(probs + 1e-10)).sum()

        axes[plot_idx].imshow(img)
        axes[plot_idx].axis('off')

        status = "ANOMALY" if is_anomaly[data_idx] == 1 else "NORMAL"
        color = 'red' if is_anomaly[data_idx] == 1 else 'blue'
        title = f"{status}\nClass: {predicted_class}, Prob: {max_prob:.3f}\nEntropy: {entropy:.3f}"
        axes[plot_idx].set_title(title, fontsize=9, color=color, fontweight='bold')

    # Hide unused subplots
    for plot_idx in range(len(selected_indices), rows * cols):
        axes[plot_idx].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Sample predictions saved to {save_path}")
    plt.close()


def print_anomaly_detection_results(scores, is_anomaly):
    """Print anomaly detection results"""

    print("\n" + "="*70)
    print("Anomaly Detection Results")
    print("="*70)

    methods = {
        'Max Logit': 'max_logit',
        'Entropy': 'entropy',
        'Max Probability': 'max_prob'
    }

    for name, method in methods.items():
        score = scores[method]
        auc = roc_auc_score(is_anomaly, score)

        normal_scores = score[np.array(is_anomaly) == 0]
        anomaly_scores = score[np.array(is_anomaly) == 1]

        print(f"\n{name}:")
        print(f"  AUC Score: {auc:.4f}")
        print(f"  Normal samples - Mean: {normal_scores.mean():.4f}, Std: {normal_scores.std():.4f}")
        print(f"  Anomaly samples - Mean: {anomaly_scores.mean():.4f}, Std: {anomaly_scores.std():.4f}")

    print("="*70 + "\n")


def main():
    """Main inference function"""

    # Load dataset
    test_paths, test_labels, is_anomaly, class_to_idx = load_dataset(
        CONFIG['train_dir'],
        CONFIG['test_normal_dir'],
        CONFIG['test_anomaly_dir'],
        train_split=CONFIG['train_split']
    )

    # Data transform
    test_transform = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dataset and loader
    test_dataset = SimpsonsDataset(test_paths, test_labels, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'],
                            shuffle=False, num_workers=CONFIG['num_workers'])

    # Initialize and load model
    num_classes = len(class_to_idx)
    model = CharacterClassifier(num_classes).to(CONFIG['device'])

    print(f"Loading model from: {CONFIG['model_path']}")
    model.load_state_dict(torch.load(CONFIG['model_path'], map_location=CONFIG['device']))
    model.eval()
    print(f"✓ Model loaded successfully! ({num_classes} classes)\n")

    # Get predictions
    print("="*70)
    print("Running Inference on Test Set")
    print("="*70)
    logits, predictions, true_labels = get_logits_and_predictions(model, test_loader, CONFIG['device'])
    print(f"Processed {len(logits)} test samples")
    print("="*70 + "\n")

    # Compute anomaly scores
    scores = compute_anomaly_scores(logits)

    # Visualizations
    print("Generating visualizations...\n")
    visualize_logit_distributions(scores, is_anomaly)
    visualize_roc_curves(scores, is_anomaly)

    # Visualize samples - focus on anomalies
    visualize_sample_predictions(model, test_loader, is_anomaly, CONFIG['device'],
                                num_samples=16, focus_on_anomaly=True,
                                save_path='sample_predictions_anomaly_focus.png')

    # Also create a mixed visualization
    visualize_sample_predictions(model, test_loader, is_anomaly, CONFIG['device'],
                                num_samples=16, focus_on_anomaly=False,
                                save_path='sample_predictions_mixed.png')

    # Print results
    print_anomaly_detection_results(scores, is_anomaly)

    print("="*70)
    print("Inference and Analysis Complete!")
    print("="*70)
    print(f"\nGenerated files:")
    print(f"  - logit_distributions.png: Distribution analysis")
    print(f"  - roc_curves.png: ROC curves with AUC scores")
    print(f"  - sample_predictions_anomaly_focus.png: Anomaly samples visualization")
    print(f"  - sample_predictions_mixed.png: Mixed samples visualization")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

