# Depression Risk Estimator

A small educational machine learning project that estimates a person's **relative depression risk** from a few demographic, financial and food-security questions, using logistic regression.

> **Disclaimer:** This is a learning project, **not a diagnostic tool**. The model is weak (cross-validated AUC ≈ 0.57) and was trained on data from a specific population. Do not use it to make decisions about anyone's health. If you or someone you know is struggling, please talk to a qualified health professional.

## How it works

1. Loads `train.csv` and selects 9 features.
2. Trains a scikit-learn pipeline:
   - `SimpleImputer` (median, with missing-value indicators)
   - `StandardScaler`
   - `LogisticRegression` (`C=0.05`, strong regularization because the dataset is small)
3. Computes out-of-fold predictions with 5-fold stratified cross-validation, so a new user's score can be compared against honest (unseen-data) predictions.
4. Asks the user 9 questions in the terminal and reports where their estimated risk falls relative to people in the training data.

The result shows both an estimated probability and a relative ranking (percentile) compared to the training data. Because the model is weak, treat both with caution.

## Features used

| Column | Meaning | Question asked |
|---|---|---|
| `age` | Age | Your age |
| `married` | Married (0/1) | Are you married? |
| `edu` | Years of education | Years of education (1-19) |
| `children` | Number of children | Number of children |
| `asset_savings` | Savings | Your savings |
| `asset_durable` | Value of durable assets | Value of your durable assets |
| `fs_meat` | Meat consumption | Times you ate meat in the past week |
| `fs_enoughtom` | Enough food for tomorrow (0/1) | Do you have enough food for tomorrow? |
| `fs_sleephun` | Went to sleep hungry (0/1) | Have you gone to sleep hungry recently? |


The target column is `depressed` (0/1). Column meanings are inferred from their names; please check the dataset's data dictionary for exact definitions and units.

## Installation

```bash
git clone https://github.com/ATG55/depression_risk_estimator.git
cd depression_risk_estimator
pip install -r requirements.txt
```

## Usage

Place `train.csv` in the same folder as the script, then run:

```bash
python main.py
```

Example session:

```
=== Depression Risk Estimate (for educational use) ===

Your age: 30
Are you married? (y/n): y
Years of education (1-19): 9
Number of children: 3
Your savings (0 = none): 0
Value of your durable assets (0 = none): 100
How many times did you eat meat in the past week? 2
Do you have enough food for tomorrow? (y/n): n
Have you gone to sleep hungry recently? (y/n): y
Estimated probability: 14.5%
Higher than 40% of people in the training data
```

Inputs are validated (ranges are taken from the training data), and invalid entries are asked again.

## Results

| Model | 5-fold CV ROC-AUC |
|---|---|
| Logistic regression, 9 features, no log transform | ~0.57 |
| Logistic regression, 9 features, log-transformed money columns | ~0.59 |
| Random Forest, all columns | ~0.59 |

An AUC of 0.5 is random guessing, so the signal in this data is weak. About 17% of the training sample is labeled as depressed.

## Limitations

- **Small dataset:** 1,143 training rows.
- **Population mismatch:** The data appears to come from a household survey in Kenya (M-Pesa mobile money columns). Results will not generalize to other countries or cultures.
- **Currency/units:** Money columns are in the dataset's own unit; users in other currencies would need conversion.
- **Self-reported, correlational data:** The model finds statistical associations, not causes.
- **Missing values:** Many food-security and medical columns are missing in large blocks (likely "question not asked"), handled with median imputation plus missing indicators.

## Ideas for improvement

- Tune `C` and compare with Random Forest / gradient boosting using cross-validation.
- Try log-transforming or binarizing the savings and asset columns.
- Aggregate the food-security questions into a single score.
- Report recall and F1 in addition to AUC, since classes are imbalanced.
- Build a simple web interface (Streamlit, Flask).

## Data

The data comes from a household survey about finances, food security and mental health (columns include assets, consumption, and mobile money usage). The target column is `depressed` (0/1).

**Source:** [b_depression on Kaggle](https://www.kaggle.com/datasets/francispython/b-depression)

The dataset is not included in this repository. Download `train.csv` from the link above and place it in the project folder. Only 9 of its 75 columns are used by the model.
