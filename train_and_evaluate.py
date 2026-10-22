import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def train_and_evaluate():
    print("=" * 60)
    print("DISEASE PREDICTION FROM SYMPTOMS - ML TRAINING & EVALUATION")
    print("=" * 60)
    
    # 1. Load Data
    train_path = 'Training.csv'
    test_path = 'Testing.csv'
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training dataset not found at {train_path}")
        
    df_train = pd.read_csv(train_path)
    if 'Unnamed: 133' in df_train.columns:
        df_train = df_train.drop(columns=['Unnamed: 133'])
        
    print(f"Loaded training data: {df_train.shape[0]} rows, {df_train.shape[1]} columns")
    
    X = df_train.drop('prognosis', axis=1)
    y = df_train['prognosis']
    symptoms = list(X.columns)
    
    print(f"Total symptoms (features): {len(symptoms)}")
    print(f"Total unique diseases (targets): {y.nunique()}")
    
    # Train/Validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Load testing benchmark if present
    has_test_benchmark = os.path.exists(test_path)
    if has_test_benchmark:
        df_test = pd.read_csv(test_path)
        if 'Unnamed: 133' in df_test.columns:
            df_test = df_test.drop(columns=['Unnamed: 133'])
        X_test_benchmark = df_test.drop('prognosis', axis=1)
        y_test_benchmark = df_test['prognosis']
        print(f"Loaded testing benchmark: {df_test.shape[0]} rows")
    else:
        X_test_benchmark, y_test_benchmark = X_val, y_val
        
    # 2. Train Primary Model: Random Forest Classifier
    print("\nTraining Primary Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Evaluate RF on test split and benchmark
    y_val_pred = rf_model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"Random Forest Validation Accuracy: {val_acc * 100:.2f}%")
    
    y_bench_pred = rf_model.predict(X_test_benchmark)
    bench_acc = accuracy_score(y_test_benchmark, y_bench_pred)
    print(f"Random Forest Benchmark Accuracy (Testing.csv): {bench_acc * 100:.2f}%")
    
    # 3. Model Comparison (Random Forest, Logistic Regression, Decision Tree, Naive Bayes, SVM)
    print("\nTraining & Comparing Multiple Classifiers...")
    models = {
        'Random Forest': rf_model,
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Gaussian Naive Bayes': GaussianNB(),
        'Support Vector Machine': SVC(probability=True, random_state=42)
    }
    
    comparison_results = []
    
    for name, clf in models.items():
        if name != 'Random Forest':
            clf.fit(X_train, y_train)
        pred = clf.predict(X_test_benchmark)
        acc = accuracy_score(y_test_benchmark, pred)
        prec = precision_score(y_test_benchmark, pred, average='weighted', zero_division=0)
        rec = recall_score(y_test_benchmark, pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test_benchmark, pred, average='weighted', zero_division=0)
        
        comparison_results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1
        })
        print(f" - {name:22s} | Acc: {acc*100:6.2f}% | F1: {f1*100:6.2f}%")
        
    comp_df = pd.DataFrame(comparison_results)
    
    # 4. Generate & Save Required Graphs
    os.makedirs('static/graphs', exist_ok=True)
    os.makedirs('graphs', exist_ok=True)
    
    # Graph 1: Top 20 Most Predictive Symptoms (Feature Importance)
    print("\nGenerating Graph 1: Feature Importance...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]
    top_features = [symptoms[i].replace('_', ' ').capitalize() for i in indices]
    top_importances = importances[indices]
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, 20))
    bars = plt.barh(range(20), top_importances[::-1], color=colors, edgecolor='none', height=0.65)
    plt.yticks(range(20), top_features[::-1], fontsize=11, fontweight='medium')
    plt.xlabel('Gini Importance (Weight)', fontsize=12, fontweight='bold', labelpad=10)
    plt.title('Top 20 Most Predictive Symptoms (Random Forest)', fontsize=15, fontweight='bold', pad=15)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.0005, bar.get_y() + bar.get_height()/2, f'{width*100:.2f}%', 
                 va='center', ha='left', fontsize=9, color='#333333')
        
    plt.tight_layout()
    plt.savefig('static/graphs/feature_importance.png', dpi=300)
    plt.savefig('graphs/feature_importance.png', dpi=300)
    plt.close()
    
    # Graph 2: Confusion Matrix Heatmap
    print("Generating Graph 2: Confusion Matrix...")
    cm = confusion_matrix(y_test_benchmark, y_bench_pred, labels=rf_model.classes_)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=rf_model.classes_, yticklabels=rf_model.classes_,
                cbar_kws={'label': 'Sample Count'})
    plt.xlabel('Predicted Disease', fontsize=12, fontweight='bold', labelpad=10)
    plt.ylabel('True Disease', fontsize=12, fontweight='bold', labelpad=10)
    plt.title(f'Confusion Matrix on Testing Dataset (Accuracy: {bench_acc*100:.2f}%)', fontsize=16, fontweight='bold', pad=15)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig('static/graphs/confusion_matrix.png', dpi=300)
    plt.savefig('graphs/confusion_matrix.png', dpi=300)
    plt.close()
    
    # Graph 3: Model Performance Comparison Bar Chart
    print("Generating Graph 3: Multi-Model Benchmark Comparison...")
    plt.figure(figsize=(12, 7))
    comp_plot = comp_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
    palette = sns.color_palette("mako", 4)
    ax = sns.barplot(data=comp_plot, x='Model', y='Score', hue='Metric', palette=palette)
    plt.title('Performance Comparison of 5 Machine Learning Algorithms', fontsize=15, fontweight='bold', pad=15)
    plt.ylabel('Score (0.0 to 1.0)', fontsize=12, fontweight='bold')
    plt.xlabel('Machine Learning Model', fontsize=12, fontweight='bold', labelpad=10)
    plt.ylim(0, 1.15)
    plt.legend(title='Metrics', frameon=True, loc='upper right')
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f'{h*100:.1f}%', (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='bottom', fontsize=8, color='#222222', xytext=(0, 2),
                        textcoords='offset points', rotation=0)
    plt.xticks(fontsize=10, fontweight='medium')
    plt.tight_layout()
    plt.savefig('static/graphs/model_comparison.png', dpi=300)
    plt.savefig('graphs/model_comparison.png', dpi=300)
    plt.close()
    
    # Graph 4: Symptom Count Distribution
    print("Generating Graph 4: Active Symptoms Distribution...")
    symptom_counts = X.sum(axis=1)
    plt.figure(figsize=(10, 6))
    sns.histplot(symptom_counts, bins=range(int(symptom_counts.min()), int(symptom_counts.max()) + 2), 
                 kde=True, color='#2b5c8f', edgecolor='white')
    plt.title('Distribution of Active Symptoms per Patient Case', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Number of Symptoms Reported', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Patient Cases', fontsize=12, fontweight='bold')
    plt.axvline(symptom_counts.median(), color='#e63946', linestyle='--', linewidth=2, 
                label=f'Median Symptoms: {symptom_counts.median():.0f}')
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('static/graphs/symptom_distribution.png', dpi=300)
    plt.savefig('graphs/symptom_distribution.png', dpi=300)
    plt.close()
    
    # Graph 5: Disease Class Distribution in Dataset
    print("Generating Graph 5: Disease Class Distribution...")
    plt.figure(figsize=(15, 6))
    disease_counts = df_train['prognosis'].value_counts()
    sns.barplot(x=disease_counts.index, y=disease_counts.values, palette='crest')
    plt.title('Patient Sample Distribution Across 41 Disease Categories (Balanced Dataset)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Disease', fontsize=12, fontweight='bold')
    plt.ylabel('Sample Count', fontsize=12, fontweight='bold')
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()
    plt.savefig('static/graphs/disease_distribution.png', dpi=300)
    plt.savefig('graphs/disease_distribution.png', dpi=300)
    plt.close()
    
    # 5. Save the Primary Model and Metadata
    print("\nSaving Trained Model and Metadata...")
    joblib.dump(rf_model, 'disease_model.pkl')
    print("Successfully saved model to: disease_model.pkl")
    
    # Clean symptom display mapping and category classification
    symptom_categories = categorize_symptoms(symptoms)
    
    metadata = {
        'total_symptoms': len(symptoms),
        'symptoms': symptoms,
        'symptom_categories': symptom_categories,
        'diseases': sorted(list(rf_model.classes_)),
        'total_diseases': len(rf_model.classes_),
        'accuracy_benchmark': float(bench_acc),
        'accuracy_validation': float(val_acc),
        'comparison': comparison_results
    }
    
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("Successfully saved metadata to: model_metadata.json")
    
    print("\nTraining and Graph Generation Completed Successfully!")
    return metadata

def categorize_symptoms(symptoms):
    categories = {
        "Skin & Dermatology": [],
        "Respiratory & ENT": [],
        "Digestive & Abdominal": [],
        "Systemic & General": [],
        "Neurological & Mental": [],
        "Musculoskeletal": [],
        "Urinary & Reproductive": [],
        "Cardiovascular & Others": []
    }
    
    for s in symptoms:
        name = s.lower()
        readable = s.replace('_', ' ').strip().title()
        item = {"id": s, "label": readable}
        
        if any(w in name for w in ['skin', 'rash', 'itching', 'eruption', 'blister', 'peeling', 'nails', 'yellowish_skin', 'bruising', 'redness_of_eyes', 'drying', 'patches']):
            categories["Skin & Dermatology"].append(item)
        elif any(w in name for w in ['cough', 'breathlessness', 'sneezing', 'phlegm', 'throat', 'sinus', 'runny_nose', 'congestion', 'chest_pain', 'mucoid_sputum', 'blood_in_sputum', 'loss_of_smell']):
            categories["Respiratory & ENT"].append(item)
        elif any(w in name for w in ['stomach', 'acidity', 'vomiting', 'nausea', 'abdominal', 'diarrhoea', 'constipation', 'digestion', 'appetite', 'liver', 'passage_of_gases', 'distention_of_abdomen']):
            categories["Digestive & Abdominal"].append(item)
        elif any(w in name for w in ['fever', 'chills', 'fatigue', 'weight', 'sweating', 'shivering', 'lethargy', 'malaise', 'dehydration', 'high_fever', 'mild_fever']):
            categories["Systemic & General"].append(item)
        elif any(w in name for w in ['headache', 'dizziness', 'unsteadiness', 'altered_sensorium', 'slurred_speech', 'depression', 'irritability', 'lack_of_concentration', 'visual_disturbances']):
            categories["Neurological & Mental"].append(item)
        elif any(w in name for w in ['joint_pain', 'muscle', 'back_pain', 'neck_pain', 'knee_pain', 'weakness_in_limbs', 'cramps', 'pain_in_anal_region', 'stiff_neck', 'swelling_joints']):
            categories["Musculoskeletal"].append(item)
        elif any(w in name for w in ['urine', 'urination', 'micturition', 'bladder', 'polyuria', 'yellow_urine']):
            categories["Urinary & Reproductive"].append(item)
        else:
            categories["Cardiovascular & Others"].append(item)
            
    return categories

if __name__ == '__main__':
    train_and_evaluate()
