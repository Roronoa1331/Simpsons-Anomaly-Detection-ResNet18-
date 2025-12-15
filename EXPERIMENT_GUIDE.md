# Simpsons Character Anomaly Detection - Experiment Guide

## 📖 Overview

This project implements **classifier-based anomaly detection** for Simpsons character images. The system trains a ResNet18 classifier on normal character classes and uses logit distributions to detect anomalous samples at test time.

### Key Concept

- **Training**: Learn to classify 43 Simpsons characters (normal data)
- **Testing**: Detect anomalies using prediction uncertainty metrics
- **Hypothesis**: Anomalous images will produce uncertain/low-confidence predictions

---

## 🗂️ Project Structure

```
anomaly/
├── simpsons_anomaly_classifier.py    # Training script (Step 1)
├── inference_only.py                 # Inference & analysis script (Step 2)
├── check_dataset.py                  # Dataset verification tool
├── WORKFLOW.md                       # Workflow guide
├── EXPERIMENT_GUIDE.md              # This file
├── best_classifier.pth              # Trained model (generated)
├── training_history.png             # Training curves (generated)
├── logit_distributions.png          # Analysis results (generated)
├── roc_curves.png                   # ROC curves (generated)
├── sample_predictions_*.png         # Sample visualizations (generated)
└── dataset/
    ├── simpsons_dataset/            # Training data (43 character classes)
    └── test/
        ├── normal/                  # Normal test images (990 images)
        └── anomaly/                 # Anomaly test images (19 images)
```

---

## 🔬 Experiment Workflow

### Step 0: Environment Setup

**Requirements:**
```bash
pip install torch torchvision matplotlib scikit-learn pillow numpy
```

**Verify Dataset:**
```bash
python check_dataset.py
```

**Expected Output:**
```
TRAINING DATA
Total Characters: 43
Total Training Images: 20933

TEST DATA
Normal Test Images: 990
Anomaly Test Images: 19
```

---

### Step 1: Train the Classifier

**Script:** `simpsons_anomaly_classifier.py`

**Purpose:** Train a ResNet18-based classifier to recognize 43 Simpsons characters.

**Run:**
```bash
python simpsons_anomaly_classifier.py
```

**What Happens:**
1. Loads 20,933 training images from 43 character classes
2. Splits data: 80% training (16,746 images), 20% validation (4,187 images)
3. Trains ResNet18 with pretrained ImageNet weights
4. Saves best model based on validation accuracy
5. Generates training history visualization

**Configuration (edit in script):**
```python
CONFIG = {
    'train_dir': 'dataset/simpsons_dataset',
    'img_size': 128,           # Image resolution
    'batch_size': 32,          # Batch size
    'num_epochs': 5,          # Training epochs
    'learning_rate': 0.001,    # Learning rate
    'model_save_path': 'best_classifier.pth',
}
```

**Training Time:**
- CPU: 30-60 minutes (30 epochs)
- GPU: 10-20 minutes (30 epochs)

**Quick Test (5 epochs):**
```python
'num_epochs': 5,  # Change from 30 to 5
```

**Output Files:**
- `best_classifier.pth` - Trained model weights (~43 MB)
- `training_history.png` - Training/validation curves

**Expected Results:**
- Validation accuracy: >85%
- Training and validation loss should converge
- No significant overfitting (train/val curves should be close)

---

### Step 2: Perform Anomaly Detection

**Script:** `inference_only.py`

**Purpose:** Load trained model and detect anomalies using logit-based metrics.

**Run:**
```bash
python inference_only.py
```

**What Happens:**
1. Loads pre-trained model (`best_classifier.pth`)
2. Loads test data: 990 normal + 19 anomaly images
3. Computes predictions and logits for all test samples
4. Calculates 3 anomaly scores for each sample
5. Generates visualizations and evaluation metrics

**Anomaly Detection Methods:**

| Method | Formula | Interpretation |
|--------|---------|----------------|
| **Max Logit** | `-max(logits)` | Higher logit = more confident = normal |
| **Entropy** | `-Σ(p·log(p))` | Higher entropy = more uncertain = anomaly |
| **Max Probability** | `-max(softmax(logits))` | Lower probability = anomaly |

**Output Files:**
1. `logit_distributions.png` - Distribution analysis (6 subplots)
   - Top row: Histograms (normal vs anomaly)
   - Bottom row: Box plots
   
2. `roc_curves.png` - ROC curves with AUC scores
   - Compares all 3 detection methods
   
3. `sample_predictions_anomaly_focus.png` - **All 19 anomaly samples** ⭐
   - Shows predicted class, probability, entropy
   
4. `sample_predictions_mixed.png` - Mixed normal/anomaly samples

**Inference Time:**
- CPU: 2-5 minutes
- GPU: 1-2 minutes

---

## 📊 Understanding the Results

### 1. Training History (`training_history.png`)

**Left Plot - Loss:**
- Should decrease over epochs
- Training and validation curves should be close
- Large gap = overfitting

**Right Plot - Accuracy:**
- Should increase over epochs
- Target: >85% validation accuracy
- Plateau indicates convergence

### 2. Logit Distributions (`logit_distributions.png`)

**6 Subplots:**

**Row 1 - Histograms:**
- Blue = Normal samples
- Red = Anomaly samples
- **Good separation** = effective detection
- Anomalies should have different distribution

**Row 2 - Box Plots:**
- Shows median, quartiles, outliers
- Anomalies should have distinct statistical properties

**What to Look For:**
- Clear separation between blue and red
- Minimal overlap
- Anomalies should have:
  - Lower max logit
  - Higher entropy
  - Lower max probability

### 3. ROC Curves (`roc_curves.png`)

**AUC Score Interpretation:**
- **AUC > 0.95**: Excellent ⭐⭐⭐
- **AUC 0.90-0.95**: Very Good ⭐⭐
- **AUC 0.80-0.90**: Good ⭐
- **AUC 0.70-0.80**: Fair
- **AUC < 0.70**: Poor

**Expected Results:**
```
Max Logit:        AUC ≈ 0.9755
Entropy:          AUC ≈ 0.9780  (Best)
Max Probability:  AUC ≈ 0.9699
```

**Interpretation:**
- All methods achieve excellent performance (>0.96)
- Entropy performs slightly better
- Random classifier would have AUC = 0.5

### 4. Sample Predictions

**`sample_predictions_anomaly_focus.png`** ⭐ **Most Important**

**For Each Sample:**
- Image visualization
- Status: ANOMALY (red) or NORMAL (blue)
- Predicted class index
- Max probability (0-1)
- Entropy (uncertainty measure)

**What to Look For:**
- **Anomalies** should have:
  - High entropy (>1.5)
  - Low probability (<0.5)
  - Uncertain predictions

- **Normal samples** should have:
  - Low entropy (<0.5)
  - High probability (>0.8)
  - Confident predictions

---

## 🔍 Detailed Analysis

### Anomaly Score Comparison

| Method | Type | Range | Best for Anomaly | Advantages | Disadvantages |
|--------|------|-------|------------------|------------|---------------|
| **Max Logit** | Raw output | (-∞, +∞) | Low values | Sensitive to extremes | Not normalized |
| **Entropy** | Uncertainty | [0, log(C)] | High values | Captures full distribution | Computationally expensive |
| **Max Probability** | Confidence | [0, 1] | Low values | Intuitive interpretation | Compressed by softmax |

**Key Differences:**

1. **Max Logit vs Max Probability:**
   - Logit: Linear, unbounded, raw model output
   - Probability: Non-linear, bounded [0,1], after softmax
   - Softmax compresses extreme values → probabilities are smoother
   - In practice: Max Logit slightly better (AUC 0.9755 vs 0.9699)

2. **Entropy:**
   - Considers entire probability distribution, not just maximum
   - High entropy = uniform distribution = model is confused
   - Best overall performance (AUC 0.9780)

### Statistical Analysis

**Expected Statistics (from results):**

| Metric | Normal Mean | Normal Std | Anomaly Mean | Anomaly Std |
|--------|-------------|------------|--------------|-------------|
| Max Logit | -11.97 | 5.06 | -2.82 | 1.03 |
| Entropy | 0.19 | 0.47 | 2.08 | 0.61 |
| Max Prob | -0.94 | 0.14 | -0.43 | 0.18 |

**Interpretation:**
- Clear separation in means (>2 standard deviations)
- Anomalies have higher variance in some metrics
- Entropy shows strongest separation

---

## 🧪 Experimental Variations

### Experiment 1: Effect of Training Epochs

**Hypothesis:** More training improves anomaly detection.

**Procedure:**
```bash
# Edit simpsons_anomaly_classifier.py
'num_epochs': 5   # Test with 5, 10, 20, 30, 50
python simpsons_anomaly_classifier.py
python inference_only.py
```

**Compare:**
- Validation accuracy
- AUC scores
- Training time

### Experiment 2: Effect of Image Resolution

**Hypothesis:** Higher resolution improves detection.

**Procedure:**
```bash
# Edit both scripts
'img_size': 64   # Test with 64, 128, 224
```

**Compare:**
- Model accuracy
- AUC scores
- Training/inference time
- Memory usage

### Experiment 3: Different Architectures

**Hypothesis:** Deeper networks improve detection.

**Procedure:**
```python
# In CharacterClassifier class
self.backbone = models.resnet34(pretrained=True)  # Try resnet34, resnet50
```

**Compare:**
- Model size
- Training time
- Detection performance

### Experiment 4: Data Augmentation

**Hypothesis:** More augmentation improves generalization.

**Procedure:**
```python
# Modify train_transform
transforms.RandomRotation(20),  # Increase from 10
transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Add
transforms.RandomGrayscale(p=0.1),  # Add
```

**Compare:**
- Validation accuracy
- Overfitting (train/val gap)
- Anomaly detection AUC

---

## 📝 Experiment Report Template

### 1. Experiment Setup

**Date:** [Date]

**Objective:** [What are you testing?]

**Configuration:**
```
Epochs: [number]
Batch Size: [number]
Image Size: [number]
Learning Rate: [number]
Architecture: [ResNet18/34/50]
```

### 2. Training Results

**Training Time:** [minutes]

**Final Metrics:**
- Training Loss: [value]
- Training Accuracy: [value]%
- Validation Loss: [value]
- Validation Accuracy: [value]%

**Observations:**
- [Convergence behavior]
- [Overfitting signs]
- [Any issues]

### 3. Anomaly Detection Results

**Inference Time:** [minutes]

**AUC Scores:**
- Max Logit: [value]
- Entropy: [value]
- Max Probability: [value]

**Best Method:** [method name]

**Statistical Separation:**
- [Describe distribution overlap]
- [Mean differences]

### 4. Visual Analysis

**Training History:**
- [Describe loss/accuracy curves]

**Logit Distributions:**
- [Describe separation quality]
- [Overlap analysis]

**Sample Predictions:**
- [Number of correctly detected anomalies]
- [False positives/negatives]
- [Interesting cases]

### 5. Conclusions

**Key Findings:**
- [Main results]
- [Performance summary]

**Limitations:**
- [What didn't work well]
- [Edge cases]

**Future Work:**
- [Potential improvements]
- [Next experiments]

---

## 🎯 Troubleshooting

### Issue 1: Model file not found

**Error:**
```
ERROR: Model file not found!
Expected: best_classifier.pth
```

**Solution:**
```bash
# Train the model first
python simpsons_anomaly_classifier.py
```

### Issue 2: Low validation accuracy (<70%)

**Possible Causes:**
- Too few epochs
- Learning rate too high/low
- Data augmentation too aggressive

**Solutions:**
- Increase epochs to 30-50
- Try learning rate: 0.0001 or 0.01
- Reduce augmentation strength

### Issue 3: Overfitting (train acc >> val acc)

**Symptoms:**
- Training accuracy >95%, validation <80%
- Large gap between train/val loss

**Solutions:**
- Add more data augmentation
- Reduce model complexity
- Add dropout layers
- Early stopping

### Issue 4: Poor anomaly detection (AUC <0.8)

**Possible Causes:**
- Model not trained well
- Anomalies too similar to normal data
- Wrong anomaly score metric

**Solutions:**
- Improve classifier accuracy first
- Try different anomaly score methods
- Ensemble multiple methods
- Adjust detection threshold

### Issue 5: Out of memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```python
'batch_size': 16,  # Reduce from 32
'img_size': 64,    # Reduce from 128
'num_workers': 0,  # Reduce from 2
```

---

## 📚 Technical Background

### Why Classifier-Based Anomaly Detection?

**Advantages:**
1. **Supervised Learning:** Leverages labeled normal data
2. **Interpretable:** Uses prediction confidence
3. **No Reconstruction:** Unlike autoencoders, no need to reconstruct images
4. **Transfer Learning:** Can use pretrained models

**Disadvantages:**
1. **Requires Labels:** Needs labeled normal classes
2. **Class Imbalance:** Few anomalies vs many normal samples
3. **Threshold Selection:** Need to choose detection threshold

### Model Architecture

**ResNet18:**
- 18 layers deep
- Residual connections (skip connections)
- Pretrained on ImageNet (1000 classes)
- Fine-tuned on 43 Simpsons characters

**Why ResNet?**
- Proven performance on image classification
- Residual connections prevent vanishing gradients
- Pretrained weights provide good initialization
- Relatively lightweight (11M parameters)

### Data Preprocessing

**Training Augmentation:**
```python
RandomHorizontalFlip()      # 50% chance to flip
RandomRotation(10)          # Rotate ±10 degrees
ColorJitter(0.2, 0.2)       # Vary brightness/contrast
Normalize(ImageNet stats)   # Standardize
```

**Why Augmentation?**
- Increases effective dataset size
- Improves generalization
- Reduces overfitting
- Makes model robust to variations

---

## 🔗 References

**Papers:**
1. He et al. (2016) - "Deep Residual Learning for Image Recognition"
2. Hendrycks & Gimpel (2017) - "A Baseline for Detecting Misclassified and Out-of-Distribution Examples"

**Concepts:**
- **Logits:** Raw model outputs before softmax
- **Softmax:** Converts logits to probabilities
- **Entropy:** Measure of uncertainty in probability distribution
- **AUC-ROC:** Area Under Receiver Operating Characteristic curve
- **Transfer Learning:** Using pretrained models on new tasks

---

## ✅ Checklist

### Before Starting:
- [ ] Python environment set up (torch, torchvision, etc.)
- [ ] Dataset in correct location
- [ ] `check_dataset.py` runs successfully
- [ ] Enough disk space (~1GB)

### After Training:
- [ ] `best_classifier.pth` exists (~43 MB)
- [ ] `training_history.png` generated
- [ ] Validation accuracy >80%
- [ ] No severe overfitting

### After Inference:
- [ ] All 4 PNG files generated
- [ ] AUC scores >0.9
- [ ] Anomaly samples visualized
- [ ] Results documented

---

**Last Updated:** 2025-12-14
**Version:** 1.0
**Author:** Anomaly Detection Experiment



