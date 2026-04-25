# Heart Disease Prediction using Machine Learning

## Project Overview
This project aims to predict the presence of heart disease using clinical and demographic data. By applying machine learning techniques, the goal is to assist healthcare providers in identifying high-risk patients and improving early diagnosis.


## Business Problem
Heart disease is one of the leading causes of death worldwide. Early detection can significantly improve patient outcomes and reduce healthcare costs.

This project builds predictive models to classify whether a patient has heart disease based on medical attributes.


## Dataset
- Source: UCI Heart Disease Dataset
- Observations: 920 patients
- Features: Clinical and demographic variables
- Target Variable:
  - 0 = No Heart Disease
  - 1 = Heart Disease


## Data Preprocessing
- Removed columns with excessive missing values (`ca`, `thal`, `slope`)
- Handled missing values:
  - Numerical → Median imputation
  - Categorical → Mode imputation
- Converted target variable into binary classification
- Applied one-hot encoding for categorical variables
- Split data into training and testing sets (80/20)


## Exploratory Data Analysis
Key insights from EDA:
- Older patients have a higher risk of heart disease
- Males are more likely to develop heart disease
- Asymptomatic chest pain is strongly associated with disease
- Lower maximum heart rate indicates higher risk
- Exercise-induced angina is a key predictor


## Models Used
- Logistic Regression
- Decision Tree
- Random Forest


## Model Performance

| Model                | Accuracy |
|---------------------|----------|
| Logistic Regression | 80%      |
| Decision Tree       | 75%      |
| Random Forest       | **83%**  |

The Random Forest model achieved the best performance with the highest accuracy and balanced precision/recall.


## Model Comparison
Random Forest outperformed the other models due to its ensemble learning approach, which improves generalization and reduces overfitting.

Logistic Regression provided a strong baseline, while Decision Tree showed lower performance due to weaker generalization.


## Business Insights
- Age, maximum heart rate, and exercise-induced angina are key predictors
- High-risk patients can be identified earlier using machine learning
- Healthcare providers can use this model to support clinical decision-making
- Preventive strategies can be targeted toward high-risk groups


## Ethics & Considerations
- Potential bias in dataset (e.g., gender imbalance)
- Importance of patient privacy and data security
- Need for model transparency in healthcare applications


## Tools & Technologies
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Google Colab
