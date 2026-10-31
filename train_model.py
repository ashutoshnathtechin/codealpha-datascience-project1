import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def train_and_evaluate():
    # 1. Load dataset
    df = pd.read_csv('Iris.csv')
    print(f"Dataset shape: {df.shape}")

    # Drop Id column if present
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])

    # Clean species names (e.g. 'Iris-setosa' -> 'Setosa')
    df['Species'] = df['Species'].apply(
        lambda x: x.replace('Iris-', '').capitalize() if 'Iris-' in str(x) else str(x).capitalize()
    )

    # Features and Target
    feature_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    X = df[feature_cols]
    y = df['Species']

    # Encode target labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    classes = list(le.classes_)

    # 2. Train-test split (80% train, 20% test with stratification)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # 3. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train multiple classification models
    models = {
        'Support Vector Machine (SVM)': SVC(probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=200, random_state=42),
        'K-Nearest Neighbors (KNN)': KNeighborsClassifier(n_neighbors=5),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }

    results = {}
    best_model_name = None
    best_accuracy = -1
    best_model_obj = None

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred).tolist()

        results[name] = {
            'accuracy': round(acc * 100, 2),
            'precision': round(prec * 100, 2),
            'recall': round(rec * 100, 2),
            'f1_score': round(f1 * 100, 2),
            'confusion_matrix': cm
        }

        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            best_model_obj = model

    # Save artifacts
    joblib.dump(best_model_obj, 'best_iris_model.pkl')
    joblib.dump(models, 'all_models.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(le, 'label_encoder.pkl')

    rf_model = models['Random Forest']
    feature_importances = dict(zip(feature_cols, [round(float(v), 4) for v in rf_model.feature_importances_]))

    # Dataset summary statistics
    summary_stats = {
        col: {
            'min': round(float(df[col].min()), 2),
            'max': round(float(df[col].max()), 2),
            'mean': round(float(df[col].mean()), 2),
            'median': round(float(df[col].median()), 2),
            'std': round(float(df[col].std()), 2)
        } for col in feature_cols
    }

    species_summary = {}
    for sp in classes:
        sub_df = df[df['Species'] == sp]
        species_summary[sp] = {
            col: {
                'mean': round(float(sub_df[col].mean()), 2),
                'min': round(float(sub_df[col].min()), 2),
                'max': round(float(sub_df[col].max()), 2)
            } for col in feature_cols
        }

    # Samples preview
    sample_data = df.head(10).to_dict(orient='records')

    metadata = {
        'classes': classes,
        'feature_cols': feature_cols,
        'best_model': best_model_name,
        'models_comparison': results,
        'feature_importances': feature_importances,
        'summary_stats': summary_stats,
        'species_summary': species_summary,
        'sample_data': sample_data,
        'total_samples': len(df),
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }

    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"Training complete! Best model: {best_model_name} ({best_accuracy*100:.2f}%)")

if __name__ == '__main__':
    train_and_evaluate()
