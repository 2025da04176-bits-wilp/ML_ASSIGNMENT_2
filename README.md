# ML Assignment 2 – Handwritten Digits Classification (0–9)

**Submitted by:** SRAVAN S KUMAR

**BITS ID:** 2025da04176

## a. Problem Statement

The objective of this project is to classify handwritten digits (0–9) using multiple machine learning classification algorithms on the Digits dataset. The performance of each model is evaluated using standard classification metrics to identify the most effective model.

## b. Dataset Description

The **Digits** dataset, available through the **scikit-learn** library, is a multiclass classification dataset containing **1,797 handwritten digit samples** represented by **64 numerical pixel-intensity features**. Each sample belongs to one of **10 target classes (0–9)**, and the dataset contains no missing values, making it suitable for comparing different machine learning classification models.

### Dataset Summary

| Attribute | Value |
|-----------|-------|
| Dataset | Digits |
| Source | scikit-learn |
| Samples | 1,797 |
| Features | 64 |
| Classes | 10 (0–9) |
| Missing Values | None |

## c. Project Links

- **GitHub Repository:** `https://github.com/2025da04176-bits-wilp/ML_ASSIGNMENT_2`
- **Streamlit App:** `https://mlassignment2-bwjomdafq4623ytft5ttxx.streamlit.app/`

## d. Models Used

The following machine learning models were implemented and evaluated using Accuracy, AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

### Evaluation Metrics

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|-------|---------:|----:|----------:|-------:|---:|----:|
| Logistic Regression | 0.9722 | 0.9991 | 0.9724 | 0.9722 | 0.9722 | 0.9692 |
| Decision Tree | 0.8250 | 0.9028 | 0.8241 | 0.8250 | 0.8237 | 0.8057 |
| k-Nearest Neighbors (kNN) | 0.9639 | 0.9951 | 0.9648 | 0.9639 | 0.9636 | 0.9600 |
| Naive Bayes | 0.8111 | 0.9707 | 0.8480 | 0.8111 | 0.8151 | 0.7940 |
| Random Forest (Ensemble) | 0.9611 | 0.9992 | 0.9620 | 0.9611 | 0.9609 | 0.9569 |

## e. Observations on Model Performance

| ML Model | Observation |
|----------|-------------|
| **Logistic Regression** | Achieved the highest overall accuracy (97.22%) and excellent MCC, making it the best-performing model. |
| **Decision Tree** | Produced the lowest accuracy, indicating weaker generalization than the other models. |
| **kNN** | Delivered strong performance with high precision and recall after feature scaling. |
| **Naive Bayes** | Achieved a high AUC but comparatively lower accuracy and F1 score. |
| **Random Forest (Ensemble)** | Provided balanced performance with high AUC and reliable predictions across all classes. |
| **Overall Winner** | **Logistic Regression** achieved the highest accuracy, AUC, F1 score, and MCC, making it the best-performing model for this dataset. |

## Project Structure

```text
ML_ASSIGNMENT_2/
│── app.py
│── requirements.txt
│── README.md
│── test_data.csv
│── model_metrics.csv
│── model/
│   ├── Logistic_Regression.pkl
│   ├── Decision_Tree.pkl
│   ├── kNN.pkl
│   ├── Naive_Bayes.pkl
│   ├── Random_Forest.pkl
│   └── All_Implemented_Models_With_Model_Save.ipynb
```
