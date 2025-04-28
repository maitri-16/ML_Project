#Project: **Multiclass Classification under Distribution Shift**

---

## Project Objective

The goal of this project is to **build a machine learning model** that can classify customer feedback into **28 different categories**, even under challenging conditions like:
- **Class imbalance**,
- **Distribution shift**.
---

## Approach and Methodology

1. **Data Loading and Exploratory Data Analysis (EDA)**
   - Loaded training feature data (`X_train.csv`) and labels (`y_train.csv`).
   - Checked for missing values, and the findings were that no missing values.
   - Visualized **class distribution** showing strong class imbalance.
   - Analyzed feature statistics and computed **mutual information scores** to find important features.

2. **Preprocessing**
   - **Standardized** all input features using `StandardScaler` to ensure features have mean 0 and variance 1, which  can improves my           model's performance.

3. **Model Training**
     - Initially tried all these models individually, but they didn't work, so I decided to use these models for Stacking Ensemble.
     -  Trained multiple base models:
     - **XGBoost Classifier** 
     - **Random Forest** 
     - **Logistic Regression** 
     - **Multi-Layer Perceptron (MLP)**
   - **Calibrated** Random Forest and Logistic Regression to produce better probability estimates.

4. **Stacking Ensemble**
   - Build a **stacked model** that combines predictions from all base models.
   - Used another **XGBoost model** as a meta-learner to make final predictions.
   - **Passthrough** option was enabled to let the meta-learner use original features and base model outputs.

5. **Model Calibration**
   - Calibrated the stacked ensemble using **isotonic regression** on validation data to ensure better probabilistic outputs.

6. **Validation and Evaluation**
   - Split `X_train` into an 80/20 training/validation set for tuning and calibration.
   - Evaluated performance using:
     - **Accuracy**
     - **Weighted F1 score** 
     - **Log loss** 

7. **Testing on New Data**
   - Loaded `X_test_1.csv` and `X_test_2.csv`
   - Predicted **probabilities** 
   - For `X_test_2`, model predictions were compared against the provided true labels (`y_test_2_reduced.csv`) for the first 202 samples.
---

## How to Run the Project

1. **Install required libraries:**
   ```bash
   pip install numpy pandas scikit-learn xgboost matplotlib seaborn
   ```

2. **Prepare Data Files:**
   - Make sure these files are present in your working directory:
     - `X_train.csv`
     - `y_train.csv`
     - `X_test_1.csv`
     - `X_test_2.csv`
     - `y_test_2_reduced.csv`

3. **Run the Code:**
   - Train base models individually.
   - Build the stacked ensemble.
   - Calibrate the ensemble model.
   - Evaluate on validation data.
   - Predict on X_test_1 and X_test_2.

4. **Evaluate on y_test_2_reduced:**
   - Compare predictions on the first 202 samples of X_test_2 to see model generalization performance.

---

## Important Metrics You Focused On

| Metric       | Purpose |
|--------------|---------|
| **Accuracy** | General correctness |
| **Weighted F1 Score** | Balance between precision and recall under class imbalance |
| **Log Loss** | Penalize confident wrong predictions |

---

## Final Results

| Metric | Training  | Testing|
|-------------------|---------|
| Accuracy  | 0.809 | 0.604 |
| Weighted F1 Score| 0.787 | 0.6075 |
| Log Loss | 0.629 | 1.9509 |
---
