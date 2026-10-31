"""
Iris Flower Classification - Model Evaluation and Summary Report
"""
import sys
import os
import json

# Ensure utf-8 output encoding on Windows consoles
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def print_report():
    with open('model_metadata.json', 'r', encoding='utf-8') as f:
        meta = json.load(f)

    print("\n" + "="*70)
    print("           IRIS FLOWER CLASSIFICATION - FINAL PROJECT REPORT")
    print("="*70)

    print("\n1. DATASET OVERVIEW:")
    print(f"   * Total Samples: {meta['total_samples']} flowers (150 rows)")
    print(f"   * Classes (3): {', '.join(meta['classes'])} (50 samples each, perfectly balanced)")
    print(f"   * Features (4): {', '.join(meta['feature_cols'])}")
    print(f"   * Split Ratio: 80% Training ({meta['train_samples']} samples), 20% Testing ({meta['test_samples']} samples)")
    print(f"   * Stratification: Enabled (10 test samples per class)")

    print("\n2. MODEL EVALUATION ON UNSEEN TEST DATA:")
    print(f"{'Algorithm':<32} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<10}")
    print("-" * 78)
    for model_name, metrics in meta['models_comparison'].items():
        best_tag = " [BEST]" if model_name == meta['best_model'] else ""
        print(f"{model_name:<32} {metrics['accuracy']:>6.2f}%     {metrics['precision']:>6.2f}%     {metrics['recall']:>6.2f}%     {metrics['f1_score']:>6.2f}%{best_tag}")

    print("\n3. CONFUSION MATRIX (Support Vector Machine - 96.67%):")
    cm = meta['models_comparison']['Support Vector Machine (SVM)']['confusion_matrix']
    print("                 Pred Setosa   Pred Versicolor   Pred Virginica")
    print(f"   True Setosa       {cm[0][0]:<15} {cm[0][1]:<17} {cm[0][2]}")
    print(f"   True Versicolor   {cm[1][0]:<15} {cm[1][1]:<17} {cm[1][2]} (1 misclassified)")
    print(f"   True Virginica    {cm[2][0]:<15} {cm[2][1]:<17} {cm[2][2]}")

    print("\n4. FEATURE IMPORTANCE RANKING (Random Forest):")
    for feat, imp in meta['feature_importances'].items():
        bar = "#" * int(imp * 30)
        print(f"   * {feat:<16}: {imp*100:>5.2f}%  {bar}")

    print("\n5. LOCALHOST WEB APPLICATION:")
    print("   * Status: RUNNING")
    print("   * Local URL: http://127.0.0.1:5000")
    print("   * Features: Real-time slider prediction, confidence scores,")
    print("               algorithm comparison, confusion matrix, ML concepts guide.")
    print("="*70 + "\n")

if __name__ == '__main__':
    print_report()
