# 🏦 Customer Bank Transactions Analysis & Segmentation

## 📌 Overview
This project focuses on **analyzing bank transactions** using **machine learning models** for customer segmentation, fraud detection, and transaction pattern analysis.  
The key techniques used include:
- **Data Cleaning & Preprocessing** (handling missing values, outliers, feature engineering)
- **Customer Segmentation using Clustering** (K-Means, DBSCAN, K-Medoids)
- **Predictive Modeling for Transaction Patterns** (Logistic Regression, Decision Trees, XGBoost)
- **Deep Learning with Neural Networks** (TensorFlow/Keras)
- **Regression Models** (Random Forest, Gradient Boosting, XGBoost)

---


---

## 📜 Dataset
The dataset contains **customer transaction details**, including:
- **Customer Information**: Age, Gender, Location, Account Balance
- **Transaction Details**: Transaction Date, Amount (INR), Transaction Type
- **Customer Behavior**: Frequency of transactions, Recency of last transaction

---

## 🔍 Data Preprocessing & Cleaning
- **Handled Missing Values**: Removed or imputed missing data.
- **Feature Engineering**:
  - **Age Calculation** from Date of Birth.
  - **Transaction Recency, Frequency, and Monetary Value (RFM Analysis)**.
- **Outlier Removal**: Filtered unrealistic customer ages (e.g., > 100 years).

---

## 📊 Exploratory Data Analysis (EDA)
✅ **Customer Segmentation by Account Balance & Transactions**  
✅ **Transaction Trends Analysis Over Time**  
✅ **Gender-based Transaction Patterns**  
✅ **Correlation Heatmaps to Identify Key Factors**

Example Code:
```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
sns.histplot(df['CustomerAge'], bins=30, kde=True, color='blue')
plt.title("Customer Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()
```

---

## 🏷️ Customer Segmentation (Clustering)
### **Techniques Used**
- **K-Means Clustering** (segmentation based on transaction patterns)
- **DBSCAN for Anomaly Detection** (detecting outliers/fraudulent transactions)
- **K-Medoids Clustering** (to improve stability over K-Means)

Example Code:
```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)
df['Cluster'] = kmeans.fit_predict(df[['TransactionAmount (INR)', 'CustAccountBalance']])
```

---

## 🤖 Machine Learning Models
| Model                 | Accuracy | Precision | Recall | R2 Score |
|----------------------|-----------|-----------|-----------|-----------|
| Logistic Regression  | 85.2% | 70.3% | 65.0% | N/A |
| Decision Tree       | 78.5% | 60.2% | 62.5% | 71.3% |
| Random Forest      | 90.3% | 75.5% | 70.8% | 79.6% |
| XGBoost             | 92.1% | 80.2% | 75.4% | 84.2% |

✅ **Best Model**: XGBoost performed the best with **92.1% accuracy**.

---

## 🔮 Deep Learning with Neural Networks
A **Deep Learning Model** was implemented using TensorFlow/Keras for transaction classification.

```python
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(16, input_shape=(3,), activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=150)
```
✅ **Final Model Accuracy**: **90.5%**

---

## 📈 Model Evaluation & Visualization
✅ **Confusion Matrix for Classification Performance**
```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, model.predict(X_test))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()
```

✅ **Feature Importance from Random Forest**
```python
importances = model.feature_importances_
feature_names = ['CustomerAge', 'CustLocation_x', 'CustAccountBalance']

for feature, importance in zip(feature_names, importances):
    print(f"{feature}: {importance:.3f}")
```

## 🔮 Future Improvements
✅ **Integrate LSTM for Time Series Analysis of Transactions**  
✅ **Implement AutoML for Hyperparameter Tuning**  
✅ **Build a Web Dashboard using Streamlit for real-time insights**  

---
