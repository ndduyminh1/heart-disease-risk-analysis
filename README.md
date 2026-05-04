# Heart Disease Risk Prediction — ISOM 835 Final Project

[![Open In Colab](https://colab.research.google.com/drive/1Rfo1n1ApraoKflVp8kGf3SgkFRo5LyRv?usp=sharing)

**Author:** Doan Duy Minh Nguyen
**Course:** ISOM 835 — Predictive Analytics, Suffolk University (Sawyer Business School)

---

## Project Overview

This project builds and evaluates predictive models that flag patients at elevated risk of heart disease, using clinical and demographic data routinely captured at intake. The goal is to support primary-care triage: prioritize who should receive the more expensive cardiology workup first, while keeping false negatives (missed cases) low.

The analysis follows the predictive-analytics workflow taught in ISOM 835 — problem framing, data pre-processing, exploratory analysis, cross-validated baseline modeling, hyperparameter tuning, multi-metric evaluation, and threshold analysis tied to clinical cost.

## Business Problem

Cardiovascular disease is the leading cause of death globally. Early identification of high-risk patients enables preventive interventions (lifestyle counseling, statin therapy, imaging) that reduce both mortality and downstream cost of care.

- **Decision-maker:** primary-care clinician or care-management team
- **Prediction target:** binary indicator of heart disease (1 = disease present, 0 = not)
- **Cost asymmetry:** a missed case (false negative) is far costlier than an unnecessary follow-up test (false positive) — model selection and threshold choice reflect this

## Dataset

- **Source:** UCI Heart Disease — combined Cleveland, Hungary, Switzerland, and VA Long Beach
- **Observations:** 920 patients
- **Features:** 11 retained after preprocessing (age, sex, chest pain type, resting BP, cholesterol, fasting blood sugar, resting ECG, max heart rate, exercise-induced angina, ST depression, plus encoded categoricals)
- **Target:** original 0–4 severity collapsed to 0/1 (any disease vs. none)

## Methodology

### Data Pre-processing
- Dropped columns with excessive missingness (`ca`, `thal`, `slope`)
- Median imputation for numeric features, mode imputation for categorical
- Standard scaling for numeric features (required for logistic regression)
- One-hot encoding for categorical features
- All preprocessing wrapped inside a scikit-learn `Pipeline` to prevent data leakage during cross-validation

### Exploratory Data Analysis
Key findings:
- Older patients show substantially higher disease prevalence
- Asymptomatic chest pain is associated with the highest disease rate (mirrors published clinical literature)
- Lower maximum heart rate (`thalch`) and higher ST depression (`oldpeak`) both track strongly with disease
- Male patients have noticeably higher prevalence than female in this sample

### Modeling
Four candidate models compared with **5-fold stratified cross-validation** (the resampling technique covered in ISOM 835):

| Family | Model | Role |
|---|---|---|
| Linear | Logistic Regression | Interpretable baseline |
| Tree (single) | Decision Tree | Highly interpretable, prone to overfit |
| Tree (bagging) | Random Forest | Variance reduction via bootstrap aggregating |
| Tree (boosting) | Gradient Boosting | Sequential error correction |

### Hyperparameter Tuning
GridSearchCV with 5-fold CV, optimizing **ROC-AUC**:
- **Logistic Regression** — regularization strength `C`
- **Random Forest** — `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`
- **Gradient Boosting** — `learning_rate`, `n_estimators`, `max_depth`

### Evaluation
On the held-out 20% test set:
- Accuracy, Precision, Recall, F1, ROC-AUC
- Confusion matrices, ROC curves, Precision–Recall curves
- Threshold analysis to satisfy clinical recall target ≥ 0.90

## Results

### Test-Set Performance (tuned models)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (tuned) | 0.826 | 0.812 | 0.892 | 0.850 | 0.904 |
| Random Forest (tuned) | 0.826 | 0.818 | 0.882 | 0.849 | **0.924** |
| Gradient Boosting (tuned) | 0.832 | 0.845 | 0.853 | 0.849 | 0.905 |

**Best model: Random Forest (tuned)**, selected on **ROC-AUC = 0.924**. ROC-AUC is the right selection metric here because it is threshold-independent and rewards good ranking of patients by risk — which is what a triage system actually does. All three tuned models exceed 0.90 ROC-AUC, indicating consistent and reliable performance across model families.

### Top Predictors
Random Forest feature importance and Logistic Regression coefficients agree on the top drivers of risk:
1. ST depression (`oldpeak`)
2. Maximum heart rate achieved (`thalch`) — inverse relationship
3. Chest pain type (`cp`) — particularly asymptomatic
4. Exercise-induced angina (`exang`)
5. Age

These are the variables a cardiologist would expect to matter, which builds confidence that the model has learned clinically meaningful structure rather than spurious noise.

### Threshold Analysis

Lowering the decision threshold from the default 0.50 to **0.48** raises recall to **0.902** while precision stays at **0.793** (F1 = 0.844, accuracy = 0.815). This is the appropriate operating point given the clinical cost asymmetry — a small reduction in precision is the right price for catching more true cases.

## Business Insights

1. **Predictive performance is strong enough to be useful at triage.** A ROC-AUC of 0.924 on a held-out test set means the model meaningfully out-performs simple rules-based screening (e.g., age + sex alone), particularly for the borderline patients hardest to triage by intuition.

2. **The most informative variables are clinically expected.** Every importance ranking is dominated by `oldpeak`, `thalch`, `exang`, `cp`, and age — exactly the variables flagged by published cardiology literature.

3. **The operating point should reflect cost asymmetry.** At the recall-optimized threshold of 0.48, the system flags 90% of true heart-disease cases for follow-up, accepting a small amount of over-triage as the cost of safety.

4. **Recommended deployment.** Wrap the tuned Random Forest pipeline as a scoring service that returns (a) the predicted probability, (b) the top three contributing features, and (c) an explicit low/medium/high risk band tied to threshold ranges. Track recall on a rolling basis to detect drift, and report subgroup performance (sex, age band) at every monitoring cycle.

## Limitations & Ethics

- **Sample size and source.** 920 patients across four hospitals is small by modern standards; performance on a different population may degrade.
- **Demographic skew.** Male patients are over-represented; the model likely generalizes worse to female patients. Subgroup performance must be reported in any deployment.
- **Outcome label is binary.** The original 0–4 severity gradation contains information that is discarded when collapsed to 0/1.
- **No causal claims.** Predictive associations are not causal mechanisms — the model identifies risk, not cause.
- **Privacy.** Heart-disease risk is sensitive health information. Any production system must comply with HIPAA, log all accesses, and limit disclosure to the care relationship.

## Repository Contents

- `Final_2_Doan_Duy_Minh_Nguyen_ISOM_835.ipynb` — full analysis notebook
- `Final_1_Doan_Duy_Minh_Nguyen_ISOM_835.ipynb` — initial baseline notebook (kept for history)
- `heart_disease_uci.csv` — dataset (UCI, via Kaggle mirror)
- `README.md` — this file

## Tools & Technologies

Python 3 · pandas · NumPy · scikit-learn (Pipelines, GridSearchCV, StratifiedKFold) · Matplotlib · Seaborn · Google Colab

## How to Reproduce

1. Open the Colab badge above (or upload the notebook to Colab manually)
2. Upload `heart_disease_uci.csv` to the Colab session storage
3. **Runtime → Run all**
4. Total runtime ≈ 3–5 minutes (most of it is GridSearchCV)
