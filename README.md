# Machine Learning Assignment - 2

## a. Problem statement
Implement multiple classification models on a public classification dataset, evaluate them using Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC), and deploy an interactive Streamlit application that allows the evaluator to upload test data and compare model performance.

## b. Dataset description
**Dataset:** Breast Cancer Wisconsin (Diagnostic)

**Source:** UCI Machine Learning Repository

**Official dataset page:** https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

The dataset is a binary classification problem with **569 instances** and **30 real-valued features**. The target is `Diagnosis`, with `B` = benign and `M` = malignant. The `ID` column is retained for traceability but is not used as a model feature. The 30 model features describe characteristics computed from a digitized image of a fine needle aspirate of a breast mass.

The dataset therefore satisfies the assignment minimum of 12 features and 500 instances.

## c. GitHub Repository Link

`https://github.com/atc05696/ML-Assignments/tree/main`

## d. Models used

The implementation below follows the five model names listed in the PDF and the five model rows shown in its comparison table.

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (kNN)
4. Gaussian Naive Bayes
5. Random Forest (Ensemble)

### Evaluation metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9211 | 0.9448 | 0.9459 | 0.8333 | 0.8861 | 0.8299 |
| kNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Random Forest (Ensemble) | 0.9737 | 0.9944 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

**Train/test setup:** 80/20 stratified split with `random_state=42`.

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong linear baseline. It achieved very high AUC and strong precision/recall, showing that the standardized feature space is highly informative for this dataset. |
| Decision Tree | The tree performed reasonably well but was the weakest model among the five on this split. Its lower recall and MCC indicate more errors in identifying the positive class than the other models. |
| kNN | kNN performed strongly after feature standardization. Its accuracy, AUC, F1 and MCC were all high, but it was slightly below Logistic Regression and Random Forest. |
| Naive Bayes | Gaussian Naive Bayes achieved perfect precision on this test split and very high AUC, but its recall was lower than Logistic Regression and Random Forest. |
| Random Forest (Ensemble) | Random Forest produced the highest accuracy, F1 and MCC on the chosen split while maintaining very high AUC and perfect precision. It is therefore the best overall performer for this dataset/split. |
| **Overall Winner for your dataset?** | **Random Forest (Ensemble)** based on the strongest overall combination of Accuracy, F1 and MCC. |



## Important links
- GitHub Repository: https://github.com/atc05696/ML-Assignments/tree/main
- Live Streamlit App: https://ml-assignments-dtbiujgnlfpbwwgeiqehci.streamlit.app/
