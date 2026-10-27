# 🩸 Blood Cell Classification & Detection Using Deep Learning

*An explainable deep learning system for automated blood cell classification and object detection.*

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit&logoColor=white)
![YOLO11](https://img.shields.io/badge/YOLO11-Detection%20Pipeline-111F68)
![Grad--CAM](https://img.shields.io/badge/Explainable%20AI-Grad--CAM-purple)
![Classification](https://img.shields.io/badge/Classification-Evaluated-success)
![Object Detection](https://img.shields.io/badge/Object%20Detection-In%20Progress-orange)

---

## 🎯 Overview

Blood cell analysis is an important application of computer vision in healthcare. Manually examining microscopic blood cell images can be time-consuming and requires specialized expertise.

This project applies deep learning to classify blood cell images and interpret model predictions using Explainable AI techniques. It also extends the workflow toward object detection, with the goal of identifying and localizing individual blood cells using bounding boxes.

The current classification pipeline uses a fine-tuned MobileNet model with Grad-CAM visualizations. A YOLO11-based detection pipeline is under development.

## ✨ Features

- 🧠 **Blood Cell Classification** — Classifies input images using a fine-tuned MobileNet model.
- 🔬 **Explainable AI** — Uses Grad-CAM heatmaps to visualize image regions influencing predictions.
- 📊 **Performance Evaluation** — Measures accuracy, precision, recall, and F1-score.
- 🩸 **Eight Blood Cell Categories** — Supports the project's eight expected cell classes.
- 🎯 **Object Detection** — YOLO11-based bounding-box detection pipeline under development.
- 🌐 **Interactive Web Application** — Provides an interface for image-based prediction using Streamlit.
- 🧪 **Model Evaluation Workflow** — Includes evaluation scripts, reports, and experiment outputs.

## 🧬 Blood Cell Categories

| Class ID | Category |
|---:|---|
| 0 | Basophil |
| 1 | Eosinophil |
| 2 | Erythroblast |
| 3 | IG (Immature Granulocytes) |
| 4 | Lymphocyte |
| 5 | Monocyte |
| 6 | Neutrophil |
| 7 | Platelet |

*Verify that the class IDs match the final dataset configuration before using them for detection training.*

## 📈 Model Performance

### MobileNet Classification

The fine-tuned MobileNet model was evaluated on a held-out test set.

| Metric | Result |
|---|---:|
| Test Samples | 2,562 |
| Accuracy | 94.54% |
| Macro Precision | 93.86% |
| Macro Recall | 94.63% |
| Macro F1-score | 94.13% |
| Weighted F1-score | 94.50% |

These metrics describe the classification model only and do not represent object detection performance.

### 🔍 Explainable AI with Grad-CAM

Grad-CAM generates heatmaps highlighting image regions that contribute to the classification model's predictions. This provides a visual way to inspect model behavior beyond the predicted class label.

Grad-CAM is an interpretation aid and does not establish medical correctness.

### 🎯 YOLO11 Object Detection

The detection pipeline aims to:

- Detect individual blood cells in microscopic images.
- Generate bounding boxes around detected objects.
- Assign class labels and confidence scores.
- Evaluate detection performance using precision, recall, and mean Average Precision (mAP).

Detection performance will be reported after the model has been trained and evaluated on valid bounding-box annotations.

## 🛠️ Technology Stack

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO11-111F68)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![CVAT](https://img.shields.io/badge/CVAT-Annotation-4B32C3)
![Git](https://img.shields.io/badge/Git-Version%20Control-F05032?logo=git&logoColor=white)

| Component | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | PyTorch |
| Classification Model | Fine-tuned MobileNet |
| Object Detection | Ultralytics YOLO11 — in progress |
| Explainability | Grad-CAM |
| Data Processing | NumPy, Pandas |
| Web Application | Streamlit |
| Annotation | CVAT |
| Development Environment | VS Code, Python virtual environment |
| GPU Training | Google Colab, when available |
| Version Control | Git and GitHub |

## 📁 Project Structure

The repository includes classification assets, data preparation resources, evaluation reports, and a pilot dataset for detection development. The exact structure may evolve during development.

```text
blood-cell-capstone/
├── data/
│   ├── raw/
│   └── annotations/
│       └── yolo11_pilot/
├── src/
│   ├── evaluation/
│   └── detection/
├── outputs/
│   ├── models/
│   └── reports/
├── app.py
├── tests/
├── docs/
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/vikasmvicky/blood-cell-capstone.git
cd blood-cell-capstone
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Application

If `app.py` is the configured Streamlit entry point:

```bash
streamlit run app.py
```

Make sure the required dependencies, model checkpoint, and dataset paths are configured correctly.

## 🧪 Model Evaluation

The existing MobileNet evaluation workflow has used this command:

```bash
python -m src.evaluation.evaluate_mobilenet
```

The command requires the configured dataset, model checkpoint, and dependencies.

**Classification checkpoint:** `best_mobilenet_finetuned.pt`

Ensure the checkpoint is available at the configured path before running evaluation.

## 🏗️ YOLO11 Detection Workflow

The object detection component is being developed as a separate pipeline.

1. **Annotation:** Create bounding-box annotations for individual blood cells using CVAT.
2. **Dataset Validation:** Validate image paths, label files, class IDs, bounding-box coordinates, and dataset splits.
3. **Training:** Train a YOLO11 detector using a correctly annotated dataset.
4. **Evaluation:** Measure precision, recall, mAP@0.50, and mAP@0.50:0.95.
5. **Inference:** Generate predicted bounding boxes, class labels, and confidence scores.
6. **Integration:** Connect the trained detector to the existing web application.

Google Colab can be used for GPU training when a suitable runtime is available.

> **Important:** Image-level labels alone are insufficient for object detection training. YOLO11 requires valid bounding-box annotations, and the detector should not be described as trained until training has completed successfully.

## 🔮 Future Enhancements

- Complete the YOLO11 training and inference pipeline.
- Expand and validate bounding-box annotations.
- Integrate detection into the Streamlit application.
- Add per-class detection metrics and visualizations.
- Compare different classification and detection architectures.
- Improve inference performance and application usability.
- Evaluate the models on additional independent datasets.

## ⚠️ Limitations

- Classification metrics do not establish object detection performance.
- Model results depend on dataset quality and class representation.
- Grad-CAM visualizations do not guarantee that a prediction is medically correct.
- The detection component requires valid annotations and independent evaluation.
- This project is intended for academic and research purposes, not as a replacement for professional medical diagnosis.

## 🎓 Academic Project

**Domain:** Computer Vision, Deep Learning, Explainable AI  
**Institution:** Global Academy of Technology, Bengaluru  
**Program:** Computer Science and Engineering
