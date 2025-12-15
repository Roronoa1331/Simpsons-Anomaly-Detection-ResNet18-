## 📝 Experiment Report

------

### 1. Experiment Setup

**Date:** 12/15/2025

**Objective:**
 To evaluate whether a **classifier-based anomaly detection approach** can reliably distinguish **out-of-distribution (OOD) images** from normal Simpsons character images by analyzing prediction confidence and uncertainty metrics derived from a trained ResNet18 classifier.

The experiment specifically investigates how **Max Logit**, **Max Softmax Probability**, and **Prediction Entropy** perform as anomaly scores.

**Configuration:**

```
Epochs: 15
Batch Size: 32
Image Size: 128 × 128
Learning Rate: 0.001
Architecture: ResNet18 (ImageNet pretrained)
Optimizer: Adam
Loss Function: Cross-Entropy Loss
```

------

### 2. Training Results

**Training Time:**
 ~30–60 minutes (CPU-based training)

**Final Metrics:**

- **Training Loss:** 0.10 – 0.20
- **Training Accuracy:** ~95%
- **Validation Loss:** 0.20 – 0.30
- **Validation Accuracy:** ~93%

**Observations:**

- The training and validation loss curves show **smooth and stable convergence**, indicating effective optimization.
- Training and validation accuracy curves remain **closely aligned**, suggesting **minimal overfitting**.
- No signs of instability such as exploding gradients or oscillating loss were observed.
- Transfer learning with ImageNet-pretrained ResNet18 significantly ускорated convergence and improved generalization.

------

### 3. Anomaly Detection Results

**Inference Time:**
 ~2–5 minutes on CPU

**AUC Scores:**

- **Max Logit:** **0.9872**
- **Entropy:** **0.9869**
- **Max Probability:** **0.9832**

**Best Method:**
 **Max Logit** (highest AUC, although all three methods perform exceptionally well)

**Statistical Separation:**

| Metric          | Normal Mean | Normal Std | Anomaly Mean | Anomaly Std |
| --------------- | ----------- | ---------- | ------------ | ----------- |
| Max Probability | -0.9794     | 0.0861     | -0.4982      | 0.2363      |
| Max Logit       | -14.9796    | 6.2747     | -2.7859      | 1.2300      |
| Entropy         | 0.0682      | 0.2677     | 1.7292       | 0.7378      |

- All three metrics show **clear separation between normal and anomaly distributions**.
- Anomalies consistently exhibit:
  - **Lower confidence** (higher negative Max Probability)
  - **Much lower maximum logits**
  - **Significantly higher entropy**
- Mean differences exceed **multiple standard deviations**, explaining the near-perfect AUC values.

------

### 4. Visual Analysis

**Training History:**

- Loss curves decrease monotonically and plateau smoothly.
- Accuracy curves increase rapidly in early epochs and stabilize near convergence.
- No divergence between training and validation curves, indicating good generalization.

**Logit Distributions:**

- Histograms show **minimal overlap** between normal and anomaly samples.
- Box plots reveal:
  - Tight distributions for normal samples
  - Wider, shifted distributions for anomalies
- Entropy exhibits the most intuitive separation, while Max Logit shows the strongest numerical separation.

**Sample Predictions:**

- The majority of anomaly samples are correctly identified as uncertain.
- Anomalies typically display:
  - High entropy (>1.5)
  - Low confidence (<0.6 max probability)
- False positives are rare and mostly occur for visually ambiguous Simpsons-like images.
- No systematic failure mode observed.

------

### 5. Conclusions

**Key Findings:**

- A standard image classifier trained **only on normal classes** can serve as a powerful anomaly detector.
- Confidence- and uncertainty-based metrics derived from logits are highly effective for OOD detection.
- All three anomaly scores achieved **excellent performance (AUC > 0.98)**.
- Max Logit slightly outperformed other methods, while Entropy provided the most interpretable uncertainty signal.

**Performance Summary:**

- Classification accuracy: **~93% validation**
- Anomaly detection performance: **Near-perfect separation**
- Method is simple, efficient, and does not require explicit anomaly training data.

**Limitations:**

- The anomaly test set is relatively small (19 images).
- Extremely Simpsons-like anomalies may still confuse the classifier.
- Threshold selection is required for real-world deployment.

**Future Work:**

- Evaluate performance on larger and more diverse anomaly datasets.
- Compare against autoencoder-based and contrastive anomaly detection methods.
- Extend to deeper architectures (ResNet34/50, ViT).
- Explore ensemble-based uncertainty estimation.
# Simpsons-Anomaly-Detection-ResNet18-
