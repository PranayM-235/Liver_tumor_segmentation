# 🧠 Liver Tumor Segmentation using U-Net

An AI-powered medical image segmentation system that detects liver tumor regions from CT scan images using a U-Net deep learning architecture built with PyTorch and deployed using Streamlit.

---

# 🚀 Features

- Liver tumor segmentation using U-Net
- PyTorch deep learning pipeline
- Dice Score evaluation
- Tumor probability heatmap
- Tumor overlay visualization
- Streamlit web application
- Confidence-based medical alert system

---

# 🧠 Model Architecture

This project uses a custom U-Net architecture for semantic segmentation.

### Pipeline

```text
CT Scan → Preprocessing → U-Net Model → Tumor Segmentation → Visualization
```

---

# 📊 Model Performance

| Metric | Score |
|--------|--------|
| Dice Score | ~0.73 |
| Framework | PyTorch |
| UI | Streamlit |

---

# 📁 Project Structure

```text
liver-tumor-segmentation/
│
├── app.py
├── model.py
├── predict.py
├── train.py
├── evaluate.py
├── requirements.txt
├── README.md
├── UNet_final_model.pth
│
├── dataset/
├── notebooks/
└── outputs/
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone (https://github.com/PranayM-235/Liver_tumor_segmentation)
cd liver-tumor-segmentation
```

---

## 2️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

---

## 3️⃣ Run Streamlit App

```bash
streamlit run app.py
```

---

# 🖼️ Application Preview

## Input CT Scan
- Upload CT scan image

## Prediction
- Tumor segmentation mask
- Probability heatmap
- Overlay visualization

---

# 🧪 Training

Model trained using:

- BCEWithLogitsLoss
- Adam Optimizer
- Learning Rate Scheduling
- Threshold-based segmentation

---

# 📌 Future Improvements

- Dice Loss integration
- Multi-class segmentation
- Attention U-Net
- 3D CT segmentation
- Cloud deployment

---

# 👨‍💻 Author

Pranay  Masurkar

