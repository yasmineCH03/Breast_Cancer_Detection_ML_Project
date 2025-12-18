<div align="center">

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=600&size=40&pause=1000&color=FF1493&center=true&vCenter=true&width=600&lines=ONCOGUARD+AI;INTELLIGENT+CANCER+DETECTION;CLINICAL+DECISION+SUPPORT" alt="Typing SVG" />

<p align="center">
    <img src="https://img.shields.io/badge/OncoGuard-AI-FF1493?style=for-the-badge&logo=none" />
    <img src="https://img.shields.io/badge/Accuracy-99.8%25-2ea44f?style=for-the-badge&logo=none" />
    <img src="https://img.shields.io/badge/Powered%20By-MLOps-000000?style=for-the-badge&logo=githubactions&logoColor=white" />
</p>

<h3 align="center" style="color: #999; font-weight: 400;">Advanced Machine Learning for Breast Cancer Diagnosis & Prognosis</h3>

</div>

---

## 🏥 Project Overview

**OncoGuard AI** is a robust artificial intelligence solution designed to assist in the clinical diagnosis and prognosis of breast cancer. Leveraging the **CRISP-DM** methodology, this project optimizes advanced machine learning models to provide accurate tumor classification, rapid patient triage, and personalized follow-up strategies.

> **Mission:** To ensure precise tumor classification, facilitating rapid triage and personalized patient monitoring using evidence-based AI.

### 👥 The Team
*   **Yasmine Chebbi**
*   **Maram Chebbi**
*   **Malek Kammoun**
*   **Marwen Jnen**

---

## 🧠 Machine Learning Engines

This platform integrates two powerful predictive engines optimized for different clinical stages:

### 1. Cellular Analysis (Diagnosis)
*   **Dataset:** Wisconsin Breast Cancer Diagnosis (WBCD)
*   **Model:** **SGD-SVM (Stochastic Gradient Descent - Support Vector Machine)**
*   **Performance:**
    *   🏆 **Accuracy:** 99.81%
    *   **AUC Score:** 0.9981
*   **Functionality:**
    *   Real-time benign vs. malignant classification.
    *   Probabilistic risk estimation.
    *   Key feature contribution analysis (Radius, Texture, Perimeter, Area, etc.).

### 2. Clinical Prognosis (Metabric)
*   **Dataset:** METABRIC (Molecular Taxonomy of Breast Cancer International Consortium)
*   **Model:** **Gradient Boosting Regressor**
*   **Functionality:**
    *   **Aggressiveness Score:** Predicts the potential severity of the tumor.
    *   **Growth Rate Prediction:** Estimates the speed of tumor progression.
    *   **Evolution Tracking:** Forecasts tumor size changes over a 6-month horizon.

---

## ⚙️ MLOps & Architecture

We have implemented a production-grade MLOps pipeline to ensure reliability, reproducibility, and scalability.

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Data Versioning** | ![DVC](https://img.shields.io/badge/DVC-purple?style=flat-square&logo=dvc) | Tracks dataset versions and lineage, ensuring reproducibility of every experiment. |
| **Experiment Tracking** | ![MLflow](https://img.shields.io/badge/MLflow-blue?style=flat-square&logo=mlflow) | Logs parameters, metrics, and artifacts for model comparison and selection. |
| **API Serving** | ![FastAPI](https://img.shields.io/badge/FastAPI-teal?style=flat-square&logo=fastapi) | High-performance, asynchronous API for serving model inference in real-time. |
| **CI/CD** | ![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-black?style=flat-square&logo=githubactions) | Automated testing (Pytest), linting (Ruff), and deployment workflows. |

---

## 🎨 Web Application (Octobre Rose Edition)

The user interface has been completely redesigned with a focus on empathy and usability, featuring a **Glassmorphism** aesthetic and **Responsive Design**.

*   **Dashboard:** Centralized view of patient analytics.
*   **Interactive Triage:** Forms for inputting cellular and clinical data.
*   **Visual Analytics:** Dynamic charts for risk assessment and prognosis curves.
*   **Inspirational UI:** Animated "Octobre Rose" quotes to support patient well-being.

---

## 🚀 Quick Start

To run the full stack (Frontend, Backend, and MLflow) locally:

### 1. Start the Backend (API)
```bash
cd Backend
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### 2. Start the MLflow Server
```bash
mlflow ui
```

### 3. Start the Frontend
```bash
cd Frontend
python -m http.server 3000
```

Access the application at: `http://localhost:3000/oncoai_signin.html`

---

<div align="center">
    <p style="color: #888; font-size: 12px;">© 2025 OncoGuard AI • Advanced ML Module • ESPRIT</p>
</div>
