"""
European Bank Customer Churn Prediction - Model Training & Evaluation Pipeline
================================================================================
This script trains baseline and advanced ensemble machine learning models
(Logistic Regression, Random Forest, XGBoost) on the European Bank dataset
to predict account churn and saves the trained pipeline artifact.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

def load_data():
    """Load and combine partitioned CSV files."""
    df1 = pd.read_csv("European_Bank_part1.csv")
    df2 = pd.read_csv("European_Bank_part2.csv")
    df = pd.concat([df1, df2], ignore_index=True)
    print(f"Dataset successfully loaded. Total rows: {len(df)}, Total columns: {len(df.columns)}")
    return df

def train_and_evaluate():
    df = load_data()
    
    # Feature & Target Selection
    X = df.drop(columns=['Year', 'CustomerId', 'Surname', 'Exited'])
    y = df['Exited']
    
    cat_cols = ['Geography', 'Gender']
    num_cols = [col for col in X.columns if col not in cat_cols]
    
    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
        ]
    )
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    models = {
        'Logistic Regression Baseline': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest Classifier': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
        'XGBoost Classifier': XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, eval_metric='logloss', random_state=42)
    }
    
    best_model = None
    best_auc = 0.0
    
    print("\n" + "="*60)
    print("MODEL EVALUATION BENCHMARK")
    print("="*60)
    
    for name, clf in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, y_prob)
        print(f"\nModel: {name}")
        print(f"ROC-AUC Score: {auc:.4f}")
        print(classification_report(y_test, y_pred, digits=4))
        
        if auc > best_auc:
            best_auc = auc
            best_model = pipeline
            
    # Save best model
    joblib.dump(best_model, 'churn_model.pkl')
    print("="*60)
    print(f"Top Performing Model Saved to 'churn_model.pkl' (ROC-AUC: {best_auc:.4f})")
    print("="*60)

if __name__ == '__main__':
    train_and_evaluate()
