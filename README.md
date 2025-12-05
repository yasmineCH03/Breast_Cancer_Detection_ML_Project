<div align="center">

# 🩺 ONCOGUARD <span style="color: #ff1493;">AI</span>
### Intelligent Breast Cancer Triage System

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit%20Learn-orange?style=for-the-badge)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success?style=for-the-badge)

<p style="font-size: 14px; color: #999; max-width: 600px; margin: 0 auto;">
"An automated MLOps pipeline designed to assist clinical decision-making by predicting tumor malignancy with 99% accuracy."
</p>

</div>

---

## 🗺️ Project Roadmap

<div style="width: 100%; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color: #e8e8e8;">

<!-- PHASE 1: DATA PREP -->
<div style="padding: 20px 0; border-bottom: 1px solid #333333; margin-bottom: 30px;">
    <div style="font-size: 12px; text-transform: uppercase; letter-spacing: 2px; color: #ff1493; margin-bottom: 5px;">Phase 1: The Foundation</div>
    <h2 style="font-size: 2.2em; color: #ffffff; margin: 0; font-weight: 800; letter-spacing: -1px;"> Data <span style="color: #ff1493;">Preparation Pipeline</span></h2>
    <p style="font-size: 14px; color: #999; margin-top: 8px; max-width: 800px;">"Before the AI can learn, the raw clinical data must be translated into a clean, mathematical language."</p>
</div>

<div style="width: 100%; padding: 20px; background-color: #0f0f0f; border-radius: 8px; font-family: 'Segoe UI', sans-serif; color: #e8e8e8;">
    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px;">
        <!-- CARD 1 -->
        <div style="background-color: #1a1a1a; padding: 20px; border-radius: 6px; border-top: 2px solid #ff1493;">
            <div style="font-size: 0.8em; font-weight: bold; color: #ff1493; margin-bottom: 5px;">STEP 01</div>
            <h4 style="margin: 0 0 10px; font-size: 1.1em; color: #fff;">Target Mapping</h4>
            <p style="margin: 0; font-size: 0.85em; color: #bbb; line-height: 1.5;">
                Converted text labels to binary integers.<br>
                <code style="background:#333; color:#fff; padding:2px;">M -> 1</code> &nbsp; <code style="background:#333; color:#fff; padding:2px;">B -> 0</code>
            </p>
        </div>
        <!-- CARD 2 -->
        <div style="background-color: #1a1a1a; padding: 20px; border-radius: 6px; border-top: 2px solid #777;">
            <div style="font-size: 0.8em; font-weight: bold; color: #777; margin-bottom: 5px;">STEP 02</div>
            <h4 style="margin: 0 0 10px; font-size: 1.1em; color: #fff;">Cleaning</h4>
            <p style="margin: 0; font-size: 0.85em; color: #bbb; line-height: 1.5;">
                Dropped non-predictive columns to prevent noise.<br>
                <span style="color: #666;">Removed:</span> <i>id, Unnamed:32</i>
            </p>
        </div>
        <!-- CARD 3 -->
        <div style="background-color: #1a1a1a; padding: 20px; border-radius: 6px; border-top: 2px solid #fff;">
            <div style="font-size: 0.8em; font-weight: bold; color: #fff; margin-bottom: 5px;">STEP 03</div>
            <h4 style="margin: 0 0 10px; font-size: 1.1em; color: #fff;">Scaling</h4>
            <p style="margin: 0; font-size: 0.85em; color: #bbb; line-height: 1.5;">
                Standardized features to Zero Mean & Unit Variance.<br>
                <span style="color: #666;">Why:</span> Critical for SVM distances.
            </p>
        </div>
    </div>
</div>

<br>

<!-- PHASE 2: MODELING -->
<div style="padding: 20px 0; border-bottom: 1px solid #333333; margin-bottom: 30px;">
    <div style="font-size: 12px; text-transform: uppercase; letter-spacing: 2px; color: #ff1493; margin-bottom: 5px;">Phase 2: The Engine</div>
    <h2 style="font-size: 2.2em; color: #ffffff; margin: 0; font-weight: 800; letter-spacing: -1px;"> Modeling & <span style="color: #ff1493;">Comparison</span></h2>
    <p style="font-size: 14px; color: #999; margin-top: 8px; max-width: 800px;">"We trained three distinct algorithms to find the most robust 'Brain' for our triage system."</p>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px;">
    <!-- CARD 1: SGD-SVM -->
    <div style="background-color: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 20px;">
        <h4 style="margin: 0 0 10px 0; color: #fff; font-size: 16px; border-bottom: 1px solid #333; padding-bottom: 10px;">
            1. SGD-SVM <span style="font-size: 10px; color: #777; font-weight: normal; display: block; margin-top: 2px;">The Efficient Linear</span>
        </h4>
        <p style="font-size: 13px; color: #aaa; line-height: 1.5;">
            <strong style="color: #ff1493;">Mechanism:</strong> Uses Stochastic Gradient Descent to find the widest margin.<br>
            <strong style="color: #fff;">Result:</strong> Winner (AUC 0.9981).
        </p>
    </div>
    <!-- CARD 2: LOGISTIC REGRESSION -->
    <div style="background-color: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 20px;">
        <h4 style="margin: 0 0 10px 0; color: #fff; font-size: 16px; border-bottom: 1px solid #333; padding-bottom: 10px;">
            2. Logistic Regression <span style="font-size: 10px; color: #777; font-weight: normal; display: block; margin-top: 2px;">The Medical Standard</span>
        </h4>
        <p style="font-size: 13px; color: #aaa; line-height: 1.5;">
            <strong style="color: #ff1493;">Mechanism:</strong> Probabilistic linear classifier.<br>
            <strong style="color: #fff;">Result:</strong> Strong baseline, slightly higher FP rate.
        </p>
    </div>
    <!-- CARD 3: XGBOOST -->
    <div style="background-color: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 20px;">
        <h4 style="margin: 0 0 10px 0; color: #fff; font-size: 16px; border-bottom: 1px solid #333; padding-bottom: 10px;">
            3. XGBoost <span style="font-size: 10px; color: #777; font-weight: normal; display: block; margin-top: 2px;">The Non-Linear Powerhouse</span>
        </h4>
        <p style="font-size: 13px; color: #aaa; line-height: 1.5;">
            <strong style="color: #ff1493;">Mechanism:</strong> Gradient Boosting ensemble.<br>
            <strong style="color: #fff;">Result:</strong> Overkill for this dataset; prone to overfitting.
        </p>
    </div>
</div>

<!-- METRICS -->
<div style="padding: 20px 0; border-bottom: 1px solid #333333; margin-bottom: 20px;">
    <h2 style="font-size: 1.8em; color: #ffffff; margin: 0; font-weight: 800;">
        Understanding the <span style="color: #ff1493;">Metrics</span>
    </h2>
</div>
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
    <div style="background-color: #1a1a1a; border-radius: 6px; padding: 15px; border-left: 4px solid #ff4757;">
        <h4 style="margin: 0; color: #ff4757; font-size: 14px;">False Negatives (Danger) 🚨</h4>
        <p style="font-size: 12px; color: #ccc; margin-bottom: 0;">Model said "Safe", reality was Cancer.</p>
    </div>
    <div style="background-color: #1a1a1a; border-radius: 6px; padding: 15px; border-left: 4px solid #ffa502;">
        <h4 style="margin: 0; color: #ffa502; font-size: 14px;">False Positives (Anxiety) 😟</h4>
        <p style="font-size: 12px; color: #ccc; margin-bottom: 0;">Model said "Cancer", reality was Safe.</p>
    </div>
</div>

<br>

<!-- PHASE 3: BUSINESS INTELLIGENCE -->
<div style="padding: 20px 0; border-bottom: 1px solid #333333; margin-bottom: 30px;">
    <div style="font-size: 12px; text-transform: uppercase; letter-spacing: 2px; color: #ff1493; margin-bottom: 5px;">Phase 3: The Business Intelligence</div>
    <h2 style="font-size: 2.2em; color: #ffffff; margin: 0; font-weight: 800; letter-spacing: -1px;"> The <span style="color: #ff1493;">Value Add</span></h2>
</div>

<div style="overflow-x: auto;">
    <table style="width: 100%; border-collapse: separate; border-spacing: 0; background-color: #1a1a1a; border-radius: 8px; overflow: hidden; font-size: 14px;">
        <thead>
            <tr style="background-color: #0f0f0f;">
                <th style="text-align: left; padding: 15px 20px; color: #777; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; border-bottom: 2px solid #ff1493; width: 15%;">ID</th>
                <th style="text-align: left; padding: 15px 20px; color: #777; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; border-bottom: 2px solid #ff1493; width: 50%;">Objective Name</th>
                <th style="text-align: right; padding: 15px 20px; color: #777; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; border-bottom: 2px solid #ff1493; width: 35%;">Business Goal</th>
            </tr>
        </thead>
        <tbody>
            <tr style="border-bottom: 1px solid #333;">
                <td style="padding: 15px 20px; color: #ff1493; font-weight: bold; border-bottom: 1px solid #2a2a2a;">DSO 3</td>
                <td style="padding: 15px 20px; color: #fff; font-weight: 600; border-bottom: 1px solid #2a2a2a;">Feature Importance Analysis</td>
                <td style="padding: 15px 20px; text-align: right; border-bottom: 1px solid #2a2a2a;"><span style="background-color: #333; padding: 4px 10px; border-radius: 4px; font-size: 11px; color: #ccc;">Explainability</span></td>
            </tr>
            <tr style="border-bottom: 1px solid #333;">
                <td style="padding: 15px 20px; color: #ff1493; font-weight: bold; border-bottom: 1px solid #2a2a2a;">DSO 1</td>
                <td style="padding: 15px 20px; color: #fff; font-weight: 600; border-bottom: 1px solid #2a2a2a;">Probability Scoring</td>
                <td style="padding: 15px 20px; text-align: right; border-bottom: 1px solid #2a2a2a;"><span style="background-color: #333; padding: 4px 10px; border-radius: 4px; font-size: 11px; color: #ccc;">Calibration</span></td>
            </tr>
            <tr>
                <td style="padding: 15px 20px; color: #ff1493; font-weight: bold;">DSO 2</td>
                <td style="padding: 15px 20px; color: #fff; font-weight: 600;">Risk Stratification</td>
                <td style="padding: 15px 20px; text-align: right;"><span style="background-color: #333; padding: 4px 10px; border-radius: 4px; font-size: 11px; color: #ccc;">Resource Allocation</span></td>
            </tr>
        </tbody>
    </table>
</div>

</div>

---

## 🚀 Installation & Usage


### 1. Clone & Setup
```bash
git clone https://github.com/your-username/oncoguard.git
cd oncoguard
make install
```

### 2. Run the API
```bash
make run
# Access at http://localhost:8000
```

### 3. Docker Deployment
```bash
make docker-build
make docker-run
```

<div style="width: 100%; margin-top: 50px; padding: 30px 0; border-top: 1px solid #333; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; display: flex; justify-content: space-between; align-items: center;">
    <div>
        <div style="color: #e8e8e8; font-weight: 800; letter-spacing: 1px; font-size: 16px;">
            ONCOGUARD <span style="color: #ff1493;">AI</span>
        </div>
        <div style="font-size: 11px; color: #777; margin-top: 5px;">
            &copy; 2025 Clinical Support Systems
        </div>
    </div>
    <div style="text-align: right; font-size: 11px; line-height: 1.6;">
        <div style="color: #999;">
            Status: <span style="color: #2ed573; font-weight: bold;">● Production Ready</span>
        </div>
        <div style="color: #999;">
            Engine: <strong style="color: #e8e8e8;">SGD-SVM</strong>
        </div>
    </div>
</div>

