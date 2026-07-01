# 🎨 Draw2Development

> **An AI-powered Sketch-to-Code platform that transforms hand-drawn UI sketches into production-ready frontend code using Computer Vision and Generative AI.**

---

## 📖 Overview

Draw2Development is an AI-powered web application developed as part of our **B.Tech Semester VII Project** at **Dharmsinh Desai University**.

The project automates the conversion of **hand-drawn UI sketches** into production-ready frontend code through an end-to-end AI pipeline. It leverages a custom-trained **YOLOv8m** model for UI component detection, a layout parsing module for structured JSON generation, and **Google Gemini 1.5 Flash** for multi-framework frontend code generation.

The objective of this project is to bridge the gap between UI design and frontend development by reducing manual coding effort and accelerating rapid prototyping.

---

## ✨ Features

* 🎨 Upload hand-drawn UI sketches
* 🤖 Detect UI components using a custom-trained YOLOv8m model
* 📐 Automatic layout parsing
* 📄 Generate structured JSON representation
* 💻 Generate frontend code using Google Gemini
* ⚛️ Support for multiple frontend frameworks
* ⚡ Fast end-to-end processing
* 🌐 Simple and responsive web interface

---

# 🚀 Supported Frameworks

* React
* HTML + CSS
* Vue.js
* Angular
* Flutter

---

# 🏗️ System Workflow


Hand-Drawn UI Sketch
          │
          ▼
Image Preprocessing
          │
          ▼
YOLOv8m UI Component Detection
          │
          ▼
Layout Parsing
          │
          ▼
Structured JSON Generation
          │
          ▼
Gemini 1.5 Flash
          │
          ▼
Production-Ready Frontend Code


---

# 🛠️ Technology Stack

### Frontend

* React.js
* Tailwind CSS

### Backend

* FastAPI
* Python

### AI / Machine Learning

* YOLOv8m
* Google Gemini 1.5 Flash
* OpenCV
* NumPy
* PyTorch

### Tools

* Google Colab
* LabelImg

---

# 📚 Custom Dataset

To address the lack of publicly available annotated datasets for hand-drawn UI sketches, we created our own dataset.

### Dataset Statistics

| Attribute                 | Value     |
| ------------------------- | --------- |
| Hand-drawn UI Sketches    | **200**   |
| Bounding Box Annotations  | **3,847** |
| UI Component Classes      | **14**    |
| Annotation Tool           | LabelImg  |
| Train / Validation Split  | 80 / 20   |
| Augmented Training Images | ~480      |

The sketches were collected from students, professionals, and faculty members and manually annotated for training the YOLOv8m model.

---

# 🎯 Supported UI Components

* Button
* Text Field
* Label
* Icon
* Image
* Card
* Navigation Bar
* Dropdown
* Table
* Checkbox (Checked)
* Checkbox (Unchecked)
* Radio Button (Checked)
* Radio Button (Unchecked)
* Container / Panel

---

# 📊 Model Performance

| Metric         | Score       |
| -------------- | ----------- |
| mAP@0.5        | **85.22%**  |
| mAP@0.5:0.95   | **62.20%**  |
| Precision      | **89.93%**  |
| Recall         | **76.88%**  |
| Inference Time | **9.93 ms** |

---

# ⚙️ Project Workflow

1. Upload a hand-drawn UI sketch.
2. Select the target frontend framework.
3. Preprocess the uploaded image.
4. Detect UI components using the trained YOLOv8m model.
5. Convert detections into structured JSON.
6. Generate frontend code using Google Gemini.
7. Display the generated code for the selected framework.


# 🌟 Key Contributions

* Developed an end-to-end Sketch-to-Code pipeline.
* Created a custom dataset of **200 hand-drawn UI sketches**.
* Manually annotated **3,847** UI components across **14** classes.
* Trained a custom **YOLOv8m** model for sketch-based UI detection.
* Implemented automatic layout parsing and JSON generation.
* Integrated **Google Gemini 1.5 Flash** for multi-framework code generation.
* Developed a web application for automated sketch-to-code conversion.


# 👥 Team

This project was collaboratively developed as part of the **B.Tech Semester VII Project** at **Dharmsinh Desai University**.

| Contributor         | Responsibilities                                                                                                                                 |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Dhruvi Patel**    | Backend development, FastAPI APIs, database integration, Gemini API integration, backend workflow, system integration, testing and deployment    |
| **Bhargav Makwana** | React frontend development, custom YOLOv8m model training, dataset preparation, annotation, UI component detection pipeline and model evaluation |

Both contributors collaborated in project planning, system design, testing, debugging, documentation, and overall project development.


## 👨‍💻 Authors

### Dhruvi Patel

* GitHub: https://github.com/Dhruviptl26

### Bhargav Makwana

* GitHub: https://github.com/MakwanaBhargav026

---

⭐ If you found this project interesting, consider giving it a star on GitHub.
