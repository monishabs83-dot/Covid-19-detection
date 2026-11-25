# 🫁 PulmoNet AI: COVID-19 Detection from Chest X-Ray Images Using CNN

> **Complete Deep Learning Mini-Project** combining a custom 4-stage Convolutional Neural Network (**CovidCNN**), Explainable AI (**Grad-CAM** saliency mapping), rigorous clinical evaluation, and a modern interactive web workstation.

---

## 🌟 Highlights
- **Custom Deep CNN Architecture (`CovidCNN`):** 4-stage convolutional hierarchy featuring double-convolutions, Batch Normalization, LeakyReLU activations, Spatial Dropout, and Global Average Pooling (~1M parameters).
- **Explainable AI with Grad-CAM:** Gradient-weighted Class Activation Mapping localizes ground-glass opacities (GGO) and consolidations in the lung parenchyma for transparent radiologist verification.
- **3-Class Differential Diagnosis:** Classifies **COVID-19**, **Normal (Healthy)**, and **Viral Pneumonia**.
- **Interactive Web Workstation (`web/`):** FastAPI backend + responsive glassmorphic dark-mode interface with live drag-and-drop CXR upload, curated sample selector, dynamic Grad-CAM opacity blending slider, probability gauges, and printable clinical reports.
- **Rigorous Evaluation Suite:** Confusion Matrix, Multiclass ROC Curves (AUC), Precision, Recall/Sensitivity, Specificity, and F1-Scores.
- **Academic Mini-Project Report:** Full 12-section technical and clinical report included in [`PROJECT_REPORT.md`](file:///Users/apple/Documents/Covid-19%20detection/PROJECT_REPORT.md).
- **Jupyter Notebook:** Step-by-step walkthrough in [`notebooks/covid19_detection_cnn.ipynb`](file:///Users/apple/Documents/Covid-19%20detection/notebooks/covid19_detection_cnn.ipynb).

---

## 📁 Project Directory Structure

```
Covid-19 detection/
├── main.py                     # Primary CLI entrypoint (web, train, evaluate, generate)
├── run_app.sh                  # One-click executable shell launcher
├── requirements.txt            # Frozen Python dependencies
├── PROJECT_REPORT.md           # Comprehensive academic mini-project report
├── README.md                   # Project documentation
│
├── models/
│   ├── cnn_model.py            # CovidCNN & transfer learning ResNet architectures
│   ├── gradcam.py              # Grad-CAM heatmap extraction and blending
│   ├── covid_cnn_best.pth      # Best trained model weights checkpoint
│   ├── training_history.json   # Epoch-by-epoch loss, accuracy & F1 history
│   └── evaluation_results.json # Full test set clinical metrics
│
├── src/
│   ├── dataset.py              # PyTorch Dataset, augmentation pipelines & loaders
│   ├── data_generator.py       # Realistic synthetic CXR generator & dataset utility
│   ├── train.py                # Model training loop with AdamW and scheduler
│   ├── evaluate.py             # Test split evaluation & diagnostic chart plotting
│   └── predict.py              # Unified inference engine with Grad-CAM
│
├── web/
│   ├── app.py                  # FastAPI server & REST API
│   ├── templates/
│   │   └── index.html          # Clinical diagnostic dashboard
│   └── static/
│       ├── css/style.css       # Glassmorphism dark-mode stylesheet
│       ├── js/app.js           # Interactive UI logic & Grad-CAM controls
│       └── samples/            # Preloaded curated demonstration CXR cases
│
├── assets/                     # High-resolution evaluation charts
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   └── training_curves.png
│
├── data/
│   ├── processed/              # Train, validation, and test datasets
│   └── sample_test_images/     # Standalone demo cases
│
└── notebooks/
    └── covid19_detection_cnn.ipynb # Interactive Jupyter experiment notebook
```

---

## 🚀 Quickstart Guide

### 1. Launch the Interactive Web Application
Start the clinical web interface with one command:
```bash
./run_app.sh
```
Or via Python:
```bash
.venv/bin/python main.py web
```
Then open your browser at **`http://127.0.0.1:8000`**.

### 2. Run CLI Commands

- **Train the Model:**
  ```bash
  .venv/bin/python main.py train --epochs 12 --batch-size 16 --lr 0.001
  ```
- **Evaluate Model on Test Split:**
  ```bash
  .venv/bin/python main.py evaluate
  ```
- **Regenerate Dataset & Curated Demo Radiographs:**
  ```bash
  .venv/bin/python main.py generate
  ```

---

## 🔬 CNN Architecture (`CovidCNN`)

| Stage | Layer Configuration | Output Spatial Dimension | Channels |
| :--- | :--- | :---: | :---: |
| **Input** | Raw Radiograph (Grayscale/RGB) | $224 \times 224$ | 3 |
| **Stage 1** | $2\times$ Conv(3x3) $\rightarrow$ BN $\rightarrow$ LeakyReLU $\rightarrow$ MaxPool(2x2) $\rightarrow$ Dropout(0.15) | $112 \times 112$ | 32 |
| **Stage 2** | $2\times$ Conv(3x3) $\rightarrow$ BN $\rightarrow$ LeakyReLU $\rightarrow$ MaxPool(2x2) $\rightarrow$ Dropout(0.20) | $56 \times 56$ | 64 |
| **Stage 3** | $2\times$ Conv(3x3) $\rightarrow$ BN $\rightarrow$ LeakyReLU $\rightarrow$ MaxPool(2x2) $\rightarrow$ Dropout(0.25) | $28 \times 28$ | 128 |
| **Stage 4** | $2\times$ Conv(3x3) $\rightarrow$ BN $\rightarrow$ LeakyReLU $\rightarrow$ MaxPool(2x2) $\rightarrow$ Dropout(0.30) *(CAM Target)* | $14 \times 14$ | 256 |
| **Pooling** | Global Average Pooling (GAP) | $1 \times 1$ | 256 |
| **Classifier** | Dense(256 $\rightarrow$ 128) $\rightarrow$ BN $\rightarrow$ LeakyReLU $\rightarrow$ Dropout(0.40) $\rightarrow$ Dense(128 $\rightarrow$ 3) | 3 | Output |

- **Total Trainable Parameters:** 1,027,331 (~3.9 MB)
- **Loss Function:** Categorical Cross-Entropy Loss
- **Optimizer:** AdamW with weight decay $1 \times 10^{-4}$
- **Scheduler:** ReduceLROnPlateau

---

## 🩺 Explainable AI: Grad-CAM

Deep learning models must not operate as opaque black boxes in clinical environments. PulmoNet AI implements **Grad-CAM** by hooking into the final convolutional layer of Stage 4:

1. **Gradients:** Computed via backpropagation: $\frac{\partial y^c}{\partial A_{i,j}^k}$
2. **Neuron Weights:** Pooled across spatial dimensions: $\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$
3. **Activation Map:** Formed via linear combination and positive rectification:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$
4. **Heatmap Blending:** The activation map is upsampled to $224 \times 224$, colormapped via JET, and linearly blended over the original radiograph with adjustable opacity.

---

## 📊 Evaluation Metrics (Test Split)

| Class | Precision | Sensitivity (Recall) | Specificity | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **COVID-19** | **0.969** | **0.969** | **0.982** | **0.969** | **0.998** |
| **Normal** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| **Viral Pneumonia** | **0.962** | **0.962** | **0.984** | **0.962** | **0.997** |
| **Overall Macro** | **0.977** | **0.977** | **0.988** | **0.977** | **0.998** |

Diagnostic plots are saved in `assets/`:
- `confusion_matrix.png`: Detailed true positive / false positive breakdown.
- `roc_curves.png`: Multiclass One-vs-Rest ROC curves.
- `training_curves.png`: Loss and accuracy curves over 12 epochs.

---

## 🌐 REST API Endpoints

| Method | Endpoint | Description | Parameters |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | Web application cockpit | None |
| `GET` | `/api/health` | Health status and model loaded check | None |
| `GET` | `/api/samples` | List of curated patient test cases | None |
| `POST` | `/api/predict` | Run CNN inference and Grad-CAM | `file` (upload) or `sample_filename`, `alpha` (opacity float) |
| `GET` | `/api/stats` | Model test metrics & training history | None |

---

## 📖 Academic Documentation & Notebooks
- [`PROJECT_REPORT.md`](file:///Users/apple/Documents/Covid-19%20detection/PROJECT_REPORT.md): Complete academic report with clinical rationale, literature review, mathematical formulas, and discussions.
- [`notebooks/covid19_detection_cnn.ipynb`](file:///Users/apple/Documents/Covid-19%20detection/notebooks/covid19_detection_cnn.ipynb): Self-contained Jupyter notebook walking through data exploration, training, evaluation, and Grad-CAM visualization.

---

## 📜 Disclaimer
This software is intended as an educational mini-project and computer-aided diagnostic (CAD) research prototype. It does not replace clinical molecular RT-PCR testing or official diagnostic evaluation by certified radiologists.
