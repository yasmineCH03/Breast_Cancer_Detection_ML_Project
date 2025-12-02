<div align="center">

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=600&size=40&pause=1000&color=FF1493&center=true&vCenter=true&width=600&lines=ONCOGUARD+AI;CLINICAL+TRIAGE+SYSTEM" alt="Typing SVG" />

<p align="center">
    <img src="https://img.shields.io/badge/Accuracy-99.8%25-2ea44f?style=for-the-badge&logo=none" />
    <img src="https://img.shields.io/badge/Model-SGD--SVM-ff1493?style=for-the-badge&logo=scikitlearn&logoColor=white" />
    <img src="https://img.shields.io/badge/Status-Production-000000?style=for-the-badge&border=white" />
</p>

<h3 align="center" style="color: #999; font-weight: 400;">From Raw Data to Diagnosis</h3>

</div>

---

## 🧬 1. The Data Pipeline
> *Translating biological complexity into mathematical structure.*

<table>
  <tr>
    <td width="33%" align="center" style="background-color: #0d1117;">
        <h3 style="color: #ff1493;">01. MAP</h3>
        <p><b>Target Encoding</b></p>
        <code>M → 1</code><br>
        <code>B → 0</code>
    </td>
    <td width="33%" align="center" style="background-color: #0d1117;">
        <h3 style="color: #79c0ff;">02. CLEAN</h3>
        <p><b>Noise Reduction</b></p>
        Dropped <code>ID</code><br>
        Dropped <code>Unnamed</code>
    </td>
    <td width="33%" align="center" style="background-color: #0d1117;">
        <h3 style="color: #d2a8ff;">03. SCALE</h3>
        <p><b>Standardization</b></p>
        Mean = 0<br>
        Variance = 1
    </td>
  </tr>
</table>

<br>

## 🧠 2. The Model Arena
> *We pitted three algorithms against each other. Simplicity won.*

<table>
  <thead>
    <tr>
      <th align="left">Model</th>
      <th align="left">Mechanism</th>
      <th align="center">AUC Score</th>
      <th align="center">Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>SGD-SVM</b></td>
      <td>Linear / Hinge Loss</td>
      <td align="center"><b>0.9981</b></td>
      <td align="center">🏆 <b>SELECTED</b></td>
    </tr>
    <tr>
      <td>Logistic Regression</td>
      <td>Probabilistic</td>
      <td align="center">0.9950</td>
      <td align="center">🥈 Backup</td>
    </tr>
    <tr>
      <td>XGBoost</td>
      <td>Gradient Boosting</td>
      <td align="center">0.9910</td>
      <td align="center">🥉 Overkill</td>
    </tr>
  </tbody>
</table>

<br>

## 🎯 3. Success Metrics
> *Understanding the "Cost" of errors in a medical context.*

<table>
  <tr>
    <td width="50%" style="border: 1px solid #ff4757;">
       <h4 style="color: #ff4757; margin: 0;">🚨 False Negatives (Danger)</h4>
       <p><b>Goal: 0</b> <br> We sent a sick patient home. <br> <i>(Model said Safe, was Cancer)</i></p>
    </td>
    <td width="50%" style="border: 1px solid #ffa502;">
       <h4 style="color: #ffa502; margin: 0;">😟 False Positives (Anxiety)</h4>
       <p><b>Goal: Low</b> <br> Unnecessary biopsy stress. <br> <i>(Model said Cancer, was Safe)</i></p>
    </td>
  </tr>
  <tr>
    <td width="50%" style="border: 1px solid #2ed573;">
       <h4 style="color: #2ed573; margin: 0;">🕸️ Sensitivity (Safety)</h4>
       <p><b>Target: >99%</b> <br> Did we catch all the sick people?</p>
    </td>
    <td width="50%" style="border: 1px solid #1e90ff;">
       <h4 style="color: #1e90ff; margin: 0;">🎯 Precision (Trust)</h4>
       <p><b>Target: High</b> <br> When we warn, are we right?</p>
    </td>
  </tr>
</table>

<br>

## 💼 4. Business Intelligence
> *Delivering actionable value beyond just "predictions".*

| ID | Objective | Value Add |
| :--- | :--- | :--- |
| **DSO 3** | **Explainability** | We interpret the "Why" using Feature Importance. |
| **DSO 1** | **Calibration** | We provide exact probability scores, not just labels. |
| **DSO 2** | **Stratification** | We categorize patients to allocate hospital resources efficiently. |

<br>

## ⚡ Quick Start

**1. Installation**
```bash
git clone https://github.com/your-username/oncoguard.git
cd oncoguard
make install
```

**2. Run API**
```bash
make run
# Serving on localhost:8000
```

---

<div align="center">

<p style="font-size: 12px; color: #555;">
    <b>ONCOGUARD AI</b> • 2025 Clinical Support Systems • <i>"Production Ready"</i>
</p>

</div>
