# 🌸 Iris Flower Classification System

A complete end-to-end Machine Learning project to classify Iris flower species (**Setosa**, **Versicolor**, and **Virginica**) using physical flower measurements.

---

## 🚀 Live Localhost Web Application

The interactive web application is running and accessible at:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

### Features:
- **Real-Time Classification**: Adjust sliders for Sepal Length, Sepal Width, Petal Length, and Petal Width to classify the flower instantly.
- **Multi-Model Support**: Switch between **Support Vector Machine (SVM)**, **Logistic Regression**, **K-Nearest Neighbors (KNN)**, **Random Forest**, and **Decision Tree**.
- **Probability Breakdown**: View exact percentage confidence for all 3 flower species.
- **Interactive Evaluation**: Inspect accuracy comparisons, confusion matrix grids, and feature importance rankings.
- **ML Classification Guide**: Educational walkthrough of foundational concepts (Supervised Learning, Train/Test Split, Scaling, Metrics, Overfitting).
- **Dataset Explorer**: Inspect dataset samples and summary statistics.

---

## 📊 Dataset Summary

The dataset was extracted from the provided archive (`Iris.csv`):
- **Total Samples**: 150 flowers
- **Classes (Balanced)**:
  - `Iris Setosa` (50 samples)
  - `Iris Versicolor` (50 samples)
  - `Iris Virginica` (50 samples)
- **Input Features (Measurements in cm)**:
  1. `SepalLengthCm` (Range: 4.3 - 7.9 cm)
  2. `SepalWidthCm` (Range: 2.0 - 4.4 cm)
  3. `PetalLengthCm` (Range: 1.0 - 6.9 cm)
  4. `PetalWidthCm` (Range: 0.1 - 2.5 cm)

---

## 🏆 Model Evaluation & Performance

All models were evaluated on an **unseen held-out stratified test set (30 flowers, 10 per species)**:

| Algorithm | Test Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Support Vector Machine (SVM)** | **96.67%** | **96.97%** | **96.67%** | **96.66%** | 🏆 **Best Model** |
| **Logistic Regression** | **93.33%** | 93.33% | 93.33% | 93.33% | Excellent |
| **K-Nearest Neighbors (KNN)** | **93.33%** | 94.44% | 93.33% | 93.27% | Excellent |
| **Random Forest Classifier** | **90.00%** | 90.24% | 90.00% | 89.97% | High |
| **Decision Tree Classifier** | **90.00%** | 90.24% | 90.00% | 89.97% | High |

### Confusion Matrix (Best Model: SVM)
```
                 Predicted Setosa   Predicted Versicolor   Predicted Virginica
Actual Setosa          10                     0                      0
Actual Versicolor       0                     9                      1  (Borderline)
Actual Virginica        0                     0                     10
```
- **Accuracy**: 29 / 30 correct predictions (**96.67%**).
- `Iris Setosa` is linearly separable and was classified with 100% precision and recall.

### Feature Importance (Random Forest MDI):
- **Petal Width**: **43.72%** (Dominant discriminator)
- **Petal Length**: **43.15%** (Dominant discriminator)
- **Sepal Length**: **11.63%**
- **Sepal Width**: **1.50%**
*(Petal measurements account for ~87% of all predictive power).*

---

## 💡 Machine Learning Classification Concepts Explained

1. **Supervised Learning**: The algorithm learns a predictive mapping from known input features $X$ to labeled target outputs $y$.
2. **Classification vs Regression**:
   - *Regression* predicts continuous quantities (e.g., house prices).
   - *Classification* predicts categorical classes (e.g., flower species: Setosa, Versicolor, Virginica).
3. **Train / Test Split & Stratification**:
   - The data is split into 80% training (120 samples) and 20% testing (30 samples).
   - `stratify=y` guarantees equal representation of all 3 species in both training and test sets.
4. **Feature Standardization (`StandardScaler`)**:
   - Transforms each feature to have $\mu = 0$ and $\sigma = 1$:
     $$z = \frac{x - \mu}{\sigma}$$
   - Prevents features with larger magnitudes from biasing distance-based algorithms like SVM and KNN.
5. **Evaluation Metrics**:
   - **Accuracy**: $\frac{\text{Correct Predictions}}{\text{Total Predictions}}$
   - **Precision**: $\frac{TP}{TP + FP}$ (Exactness of positive predictions)
   - **Recall**: $\frac{TP}{TP + FN}$ (Completeness of identifying actual positives)
   - **F1-Score**: Harmonic mean of Precision and Recall:
     $$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
   - **Confusion Matrix**: Detailed table revealing where errors occur.

---

## 💻 How to Run

### 1. Retrain the Models:
```bash
python train_model.py
```

### 2. View Console Evaluation Report:
```bash
python evaluate_and_report.py
```

### 3. Launch Localhost Web App:
```bash
python app.py
```
Then open your web browser at:
`http://127.0.0.1:5000`

### 4. Test via REST API:
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
     -H "Content-Type: application/json" \
     -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```
