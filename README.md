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
**To be updated after creating the repository:**

`https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPOSITORY_NAME>`

## d. Models used

The assignment PDF explicitly names five classification models (although one line says "all 6 ML models"). The implementation below follows the five model names listed in the PDF and the five model rows shown in its comparison table.

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

## Project structure

```text
project-folder/
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── model/
│   ├── train_models.py
│   ├── metadata.json
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   └── random_forest_ensemble.joblib
└── model_metrics.csv
```

## How to run on BITS Virtual Lab

### 1. Open the terminal in BITS Virtual Lab
Use the terminal provided by the BITS Virtual Lab and create/enter the project folder.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train and evaluate the models

```bash
python model/train_models.py
```

This downloads the official UCI dataset, creates the 80/20 train/test split, trains all five required models, calculates the six required metrics, saves the model files, and creates `test_data.csv`.

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

Open the URL printed by Streamlit in the BITS Virtual Lab browser.

### 5. App usage

1. Upload `test_data.csv` in the sidebar.
2. Select a model from the model dropdown.
3. Review the all-model comparison table.
4. Review the selected model's Accuracy, AUC, Precision, Recall, F1 and MCC.
5. Review the confusion matrix and classification report.
6. Review the prediction results on the uploaded test data.

## Streamlit Community Cloud deployment

1. Push this complete project to GitHub.
2. Confirm `requirements.txt`, `README.md`, `app.py`, `test_data.csv`, and the saved model files are committed.
3. Open Streamlit Community Cloud.
4. Sign in using GitHub.
5. Create a new app and select the repository, branch and `app.py`.
6. Deploy and test the public app URL.

## Deployment notes

- The Streamlit app loads saved model files; it does not retrain models during upload.
- Only test data should be uploaded in the app, matching the free-tier capacity requirement in the assignment.
- The uploaded test CSV must contain the `Diagnosis` column and all 30 feature columns used during training.

## Academic integrity

The assignment brief states that AI tools may be used for learning support and not for direct copy-paste submissions. Understand the code, customise the UI/content, maintain your own Git history, and make your own final submission.

## Links to fill before submission

- GitHub Repository: **PASTE YOUR FINAL GITHUB URL HERE**
- Live Streamlit App: **PASTE YOUR FINAL STREAMLIT URL HERE**
- BITS Virtual Lab screenshot: **INSERT YOUR SCREENSHOT IN THE SUBMISSION PDF**
