# 🦅 Aerial Object Classification & Detection

> A deep learning solution to classify and detect aerial objects — distinguishing **Birds** from **Drones** — for applications in security surveillance, wildlife protection, and airspace safety.

---

## 📌 Problem Statement

This project builds a deep learning pipeline that classifies aerial images into two categories — **Bird** or **Drone** — and optionally performs object detection to locate and label these objects in real-world scenes.

Accurate identification between drones and birds is critical for security surveillance, wildlife protection, and airspace safety.

---

## 🌍 Real-World Use Cases

| Domain | Description |
|---|---|
| 🦅 Wildlife Protection | Detect birds near wind farms or airports to prevent accidents |
| 🛡️ Security & Defense | Identify drones in restricted airspace for timely alerts |
| ✈️ Airport Safety | Monitor runway zones for bird activity to prevent bird strikes |
| 🔬 Environmental Research | Track bird populations via aerial footage without misclassification |

---

## 📁 Project Structure

```
aerial-object-classification/
│
├── classification_dataset/
│   ├── train/
│   │   ├── bird/          # 1,414 images
│   │   └── drone/         # 1,248 images
│   ├── valid/
│   │   ├── bird/          # 217 images
│   │   └── drone/         # 225 images
│   └── test/
│       ├── bird/          # 121 images
│       └── drone/         # 94 images
│
├── object_detection_dataset/
│   ├── images/
│   │   ├── train/         # 2,662 images
│   │   ├── val/           # 442 images
│   │   └── test/          # 215 images
│   ├── labels/            # YOLOv8 .txt annotations
│   └── data.yaml
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_custom_cnn.ipynb
│   ├── 03_transfer_learning.ipynb
│   └── 04_yolov8_detection.ipynb
│
├── models/                # Saved .h5 / .pt model files
├── app.py                 # Streamlit deployment app
├── requirements.txt
└── README.md
```

---

## 📊 Datasets

### Classification Dataset
- **Task:** Binary Image Classification (Bird / Drone)
- **Format:** `.jpg` RGB images, resized to `224×224`

| Split | Bird | Drone |
|---|---|---|
| Train | 1,414 | 1,248 |
| Validation | 217 | 225 |
| Test | 121 | 94 |

### Object Detection Dataset (YOLOv8 Format)
- **Total Images:** 3,319
- **Annotation format:** `<class_id> <x_center> <y_center> <width> <height>`

| Split | Images |
|---|---|
| Train | 2,662 |
| Validation | 442 |
| Test | 215 |

---

## 🛠️ Tech Stack

- **Language:** Python 3.x
- **Deep Learning:** TensorFlow / Keras, PyTorch, torchvision
- **Object Detection:** YOLOv8 (ultralytics)
- **Image Processing:** OpenCV, PIL
- **Evaluation:** scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Deployment:** Streamlit
- **Utilities:** NumPy, Pandas


---

## 🔄 Project Workflow

```
1. Dataset Inspection
   └── Explore structure, check class balance, visualize samples

2. Data Preprocessing
   └── Normalize to [0,1], resize to 224×224
   └── TF: preprocess_input | PyTorch: ImageNet Normalize(mean, std)

3. Data Augmentation
   └── Rotation, flip, zoom, brightness, random crop

4. Model Building
   ├── Custom CNN: Conv → Pool → BatchNorm → Dropout → Dense
   └── Transfer Learning: ResNet50 / MobileNet / EfficientNetB0

5. Model Training
   └── EarlyStopping + ModelCheckpoint
   └── Metrics: Accuracy, Precision, Recall, F1-score

6. Evaluation
   └── Confusion matrix, classification report, loss/accuracy plots

7. Model Comparison
   └── Accuracy, training time, generalization → save best model

8. (Optional) YOLOv8 Object Detection
   └── Prepare data.yaml → Train → Validate → Inference

9. Streamlit Deployment
   └── Upload image → Predict (Bird / Drone) + confidence score
   └── Optional: YOLOv8 bounding box overlay
```



