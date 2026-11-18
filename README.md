# Customer Transaction Prediction (PRCP-1003)

## Project Overview
This project predicts whether a banking customer will make
a future transaction using machine learning.

## Dataset
- Records: 200,000
- Numerical features: 200
- Target: 0 (No Transaction), 1 (Transaction)
- Missing values: 0
- Duplicate records: 0
- Target imbalance: 89.95% vs 10.05%

## Methodology
1. Dataset inspection and EDA
2. Data quality assessment
3. Stratified 80:20 train-validation split
4. Model development and comparison
5. Hyperparameter and threshold optimization
6. Model evaluation and export

## Model Comparison

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Dummy Classifier | 0.5000 | 0.1005 |
| SGD Logistic Regression | 0.8453 | 0.4716 |
| Gaussian Naive Bayes | 0.8822 | 0.5613 |
| Histogram Gradient Boosting (60k) | 0.8263 | 0.4154 |
| Optimized Gaussian NB | 0.8882 | 0.5763 |

## Selected Model
Optimized Gaussian Naive Bayes

- var_smoothing: 1e-6
- Decision threshold: 0.264069
- Accuracy: 90.46%
- Precision: 52.36%
- Recall: 56.82%
- F1-score: 0.5450
- ROC-AUC: 0.8882
- PR-AUC (Average Precision): 0.5763

## Evaluation Results

### Confusion Matrix
![Confusion Matrix](customer_transaction_outputs/confusion_matrix.png)

### ROC Curve
![ROC Curve](customer_transaction_outputs/roc_curve.png)

### Precision-Recall Curve
![Precision-Recall Curve](customer_transaction_outputs/precision_recall_curve.png)

## Installation
Install dependencies:

    pip install -r requirements.txt

Place Train.csv beside the notebook and run the notebook
cells in order.

## Project Outputs
The customer_transaction_outputs folder contains the saved
model, metadata, model comparison CSV and evaluation figures.

## Challenges
- Imbalanced target classes
- Anonymized numerical features
- Precision-recall trade-off
- Computational efficiency

## Limitations
Results are based on development-validation data.
The model requires independent testing, probability
calibration and business-cost assessment before deployment.
