# 🩸 Blood Cell Detection and Counting with YOLO11

*An end-to-end computer vision project for detecting and counting white blood cells, red blood cells, and platelets in microscopic blood-smear images and videos.*

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO11-111F68)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-TXL--PBC-blue)
![Inference](https://img.shields.io/badge/Inference-CPU%20Supported-success)
![Tests](https://img.shields.io/badge/Testing-pytest-0A9EDC?logo=pytest&logoColor=white)

---

## 🚀 Overview

This project implements an end-to-end computer vision pipeline for detecting and counting blood cells in microscopic blood-smear images and videos.

Using a fine-tuned YOLO11n object detection model, the system identifies three categories: white blood cells (WBC), red blood cells (RBC), and platelets. The project includes dataset validation, model training, evaluation, image inference, video processing, and an interactive Streamlit application.

> **Disclaimer:** This is an academic project. It is not a medical device and must not be used for clinical diagnosis.

## ✨ Highlights

- 🎯 **YOLO11n Object Detection** — Detects WBCs, RBCs, and platelets.
- 🔢 **Automated Cell Counting** — Calculates per-class counts from predicted bounding boxes.
- 🖼️ **Image Detection** — Upload blood-smear images and view annotated predictions.
- 🎥 **Video Detection** — Processes video frames and generates annotated video output.
- 📊 **Performance Evaluation** — Reports precision, recall, mAP@50, and mAP@50–95.
- 📁 **CSV Export** — Downloads structured detection results.
- 🧪 **Dataset Validation** — Checks dataset configuration, labels, image pairing, and class IDs.
- 🔁 **Reproducible Training** — Uses configurable training parameters and a fixed random seed.
- 💻 **CPU Inference** — Supports inference without requiring a dedicated GPU.

## 📈 Model Results

The following metrics are from the reported final training run, measured on the **validation split at the best epoch (epoch 29 of 30)**.

| Metric | Score |
|---|---:|
| Precision | 95.0% |
| Recall | 96.6% |
| mAP@50 | 98.5% |
| mAP@50–95 | 84.7% |

![mAP@50](https://img.shields.io/badge/mAP%4050-98.5%25-success)
![mAP@50--95](https://img.shields.io/badge/mAP%4050--95-84.7%25-brightgreen)
![Precision](https://img.shields.io/badge/Precision-95.0%25-blue)
![Recall](https://img.shields.io/badge/Recall-96.6%25-blue)

**Training configuration**

- Model: YOLO11n pretrained weights
- Image size: 640 × 640
- Optimizer: AdamW
- Random seed: 42
- Training duration: 30 epochs
- Reported training environment: CPU
- Approximate training time: 5.3 hours

These are validation metrics, not final held-out test-set results. Run `python eval_test.py` to evaluate on the test split.

## 🧬 Dataset

The project uses the TXL-PBC peripheral blood cell dataset in YOLO object detection format.

| Class ID | Class | Annotated Bounding Boxes |
|---:|---|---:|
| 0 | WBC | 1,298 |
| 1 | RBC | 16,302 |
| 2 | Platelets | 543 |
| **Total** | | **18,143** |

The dataset is strongly imbalanced, with RBCs representing most annotated objects and platelets being comparatively rare. Per-class evaluation is therefore important alongside overall detection metrics.

The dataset itself is not included in this repository.

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![YOLO11](https://img.shields.io/badge/YOLO11-Object%20Detection-111F68)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![pytest](https://img.shields.io/badge/pytest-Testing-0A9EDC?logo=pytest&logoColor=white)
![Git](https://img.shields.io/badge/Git-Version%20Control-F05032?logo=git&logoColor=white)

| Component | Technology |
|---|---|
| Programming Language | Python 3.11 |
| Object Detection | Ultralytics YOLO11n |
| Deep Learning | PyTorch |
| Web Application | Streamlit |
| Image and Video Processing | OpenCV |
| Data Validation | Python scripts |
| Testing | pytest |
| Version Control | Git and GitHub |

## 📁 Project Structure

```text
blood-cell-yolo11-research/
├── app.py
├── detector.py
├── video_processor.py
├── train.py
├── eval_test.py
├── predict.py
├── predict_test_cleanpath.py
├── validate_dataset.py
├── configs/
│   └── data.yaml
├── final_results/
│   ├── best.pt
│   └── results.csv
├── tests/
├── requirements.txt
└── README.md
```

## ⚙️ Installation

**Requirements:** Python 3.11 or compatible environment supported by the project dependencies.

### 1. Clone the Repository

```bash
git clone https://github.com/vikasmvicky/blood-cell-yolo11-research.git
cd blood-cell-yolo11-research
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU training, install a compatible CUDA-enabled PyTorch build according to the official [PyTorch installation guide](https://pytorch.org/get-started/locally/).

## 🗂️ Dataset Configuration

Download and prepare the TXL-PBC dataset in YOLO format.

Update the `path:` entry in `configs/data.yaml` to point to the dataset directory on your machine.

The class mapping must remain consistent:

```yaml
names:
  0: WBC
  1: RBC
  2: Platelets
```

Ensure that the dataset contains valid training, validation, and test splits with matching image and annotation files.

## ▶️ Usage

### 1. Validate the Dataset

```bash
python validate_dataset.py
```

The validator checks:

- Dataset configuration and class names.
- Dataset split directories.
- Image and label pairing.
- YOLO annotation format.
- Bounding-box coordinate ranges.
- Per-class annotated object counts.

### 2. Train the Model

Check the environment and dataset before training:

```bash
python train.py --check-only
```

Train the default YOLO11n model:

```bash
python train.py
```

Train a larger model with custom parameters:

```bash
python train.py --model yolo11s.pt --epochs 50 --batch 8
```

Training options depend on the arguments implemented in `train.py`.

### 3. Evaluate on the Test Split

```bash
python eval_test.py
```

This evaluates the detector on the configured test split and reports precision, recall, mAP@50, mAP@50–95, and per-class results. Evaluation plots are saved to the configured output directory.

### 4. Run Predictions

Use the prediction script to generate annotated outputs:

```bash
python predict.py
```

Refer to the script's command-line options if you need to specify an input image, directory, or output path.

### 5. Launch the Web Application

```bash
streamlit run app.py
```

**Image mode**

- Upload a JPG or PNG blood-smear image.
- View predicted bounding boxes and class labels.
- Inspect per-class cell counts.
- Download detection results as CSV.
- Adjust the confidence threshold.

**Video mode**

- Upload an MP4, AVI, or MOV video.
- Configure the frame-processing interval.
- Process frames and generate annotated video output.
- Download the resulting video.

The application includes output encoding checks and a fallback path when MP4 encoding is unavailable.

### 6. Run Tests

Install pytest if required:

```bash
python -m pip install pytest
```

Run the test suite:

```bash
python -m pytest -q
```

Tests that require model weights may be skipped automatically if the checkpoint is unavailable.

## 🏗️ Design Notes

- **Shared inference logic:** Image and video processing use the same detector logic to maintain consistent class mappings and confidence settings.
- **Memory-efficient video processing:** Frames are processed individually instead of loading an entire video into memory.
- **Configurable training:** Training parameters can be adjusted through the training CLI.
- **Input validation:** Invalid images and out-of-range confidence thresholds are handled with appropriate errors.
- **Structured outputs:** Detection results can be exported for further analysis.

## ⚠️ Limitations

- The model has been trained and validated on a single public dataset; performance on other staining techniques, microscopes, and imaging conditions has not been established.
- The dataset is imbalanced, particularly for platelets and WBCs compared with RBCs.
- Overlapping or touching RBCs may be incorrectly counted in dense regions.
- Validation metrics are not a substitute for held-out test-set evaluation.
- CPU inference is supported, but processing speed depends on the hardware and input size.

## 🔮 Roadmap

- [ ] Add per-class precision and recall tables.
- [ ] Include a confusion matrix and detailed evaluation plots.
- [ ] Compare YOLO11n with larger model variants.
- [ ] Experiment with data augmentation and class-imbalance strategies.
- [ ] Containerize the application using Docker.
- [ ] Deploy a hosted demonstration.

## 🙏 Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- TXL-PBC dataset authors — add the official dataset reference when available.
  

## 👨‍💻 Author

**Vikas M**

[![GitHub](https://img.shields.io/badge/GitHub-vikasmvicky-181717?logo=github&logoColor=white)](https://github.com/vikasmvicky)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Vikas%20M-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/vikas-m-6615062a2/)

