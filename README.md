<div align="center">

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=600&size=40&pause=1000&color=FF1493&center=true&vCenter=true&width=600&lines=ONCOGUARD+AI;OBJECTIVE+2:+RISK+STRATIFICATION" alt="Typing SVG" />

<p align="center">
    <img src="https://img.shields.io/badge/Module-Objective%202-FF1493?style=for-the-badge&logo=none" />
    <img src="https://img.shields.io/badge/Accuracy-99.8%25-2ea44f?style=for-the-badge&logo=none" />
    <img src="https://img.shields.io/badge/Model-SGD--SVM-000000?style=for-the-badge&logo=scikitlearn&logoColor=white" />
</p>

<h3 align="center" style="color: #999; font-weight: 400;">Intelligent Triage & Predictive Modeling</h3>

</div>

---

## 👥 Project Scope & Responsibilities
> *This documentation details the development of **Objective 2**. Objectives 1 & 3 are handled by external modules.*

<table>
  <thead>
    <tr>
      <th align="center" width="30%">Objective 1 (Teammate A)</th>
      <th align="center" width="40%" style="background-color: #161b22; border-bottom: 3px solid #ff1493;">✨ Objective 2 (My Focus)</th>
      <th align="center" width="30%">Objective 3 (Teammate B)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">Data Ingestion & Infrastructure</td>
      <td align="center" style="background-color: #0d1117;"><b>Predictive Modeling & Risk Engine</b></td>
      <td align="center">Dashboarding & Deployment</td>
    </tr>
    <tr>
      <td align="center" style="color: #777; font-size: 12px;">SQL / Warehouse Setup</td>
      <td align="center" style="color: #ccc; font-size: 12px;">
        • Data Preprocessing<br>
        • Algorithm Selection (SGD-SVM)<br>
        • Clinical Metric Optimization
      </td>
      <td align="center" style="color: #777; font-size: 12px;">Streamlit / API Integration</td>
    </tr>
  </tbody>
</table>

<br>

## 🧬 1. The Data Pipeline (Obj 2 Prep)
> *Translating raw clinical inputs into model-ready tensors.*

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
> *We pitted three algorithms against each other to find the best "Brain" for triage.*

<table>
  <thead>
    <tr>
      <th align="left">Contender</th>
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

## 🎯 3. Clinical Metrics
> *Defining "Success" in the context of patient safety (Objective 2 Key Deliverable).*

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

## ⚡ Quick Start

**1. Installation**
```bash
git clone https://github.com/your-username/oncoguard.git
cd oncoguard
make install
```

**2. Run Objective 2 Module**
```bash
make run
# API endpoint for Risk Stratification active at localhost:8000
```

---

<div align="center">

<p style="font-size: 12px; color: #555;">
    <b>ONCOGUARD AI</b> • Objective 2 Module • <i>"Production Ready"</i>
</p>

</div>