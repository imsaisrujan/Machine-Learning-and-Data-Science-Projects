import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import tree
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

def load_data(path="C://Users//saisr//OneDrive//Desktop//student-por.csv"):
    try:
        df = pd.read_csv(path, sep=';')
        return df
    except FileNotFoundError:
        print(f"Error: File {path} not found.")
        return None

def preprocess_data(df):
    df_processed = df.copy()
    df_processed['pass'] = df_processed['G3'].apply(lambda x: 1 if x >= 10 else 0)

    le = LabelEncoder()
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df_processed[col] = le.fit_transform(df_processed[col])

    features = df_processed.drop(['pass', 'G3'], axis=1)
    target = df_processed['pass']
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42, stratify=target)
    
    numerical_cols = X_train.select_dtypes(include=['int32', 'int64', 'float32', 'float64']).columns
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    return X_train, X_test, y_train, y_test, features.columns

def train_evaluate_model(X_train, X_test, y_train, y_test, feature_names):
    dt_classifier = DecisionTreeClassifier(random_state=42, max_depth=3, min_samples_split=4, min_samples_leaf=3)
    dt_classifier.fit(X_train, y_train)

    y_pred = dt_classifier.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    feature_importance = dict(zip(feature_names, dt_classifier.feature_importances_))
    sorted_importance = {k: v for k, v in sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)}
    
    print("\nFeature Importance:")
    for feature, importance in sorted_importance.items():
        print(f"{feature}: {importance:.4f}")
    
    return dt_classifier, sorted_importance, y_pred

def visualize_results(model, feature_names, y_test, y_pred, sorted_importance):
    plt.figure(figsize=(10, 10))
    plt.title('Decision Tree')
    tree.plot_tree(model, max_depth=3, feature_names=feature_names, class_names=['Fail', 'Pass'], filled=True, fontsize=10)
    plt.savefig('decision_tree.png')
    plt.close()
    plt.figure(figsize=(10, 6))
    top_features = dict(list(sorted_importance.items())[:10])
    plt.barh(list(top_features.keys()), list(top_features.values()))
    plt.title('Top Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.savefig('feature_importance.png')
    plt.close()
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    plt.close()
    plt.figure(figsize=(10, 6))
    comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    comparison_df = comparison_df.head(20)
    comparison_df.index = range(1, 21)
    comparison_df.plot(kind='bar', figsize=(10, 6))
    plt.title('Actual vs Predicted')
    plt.xlabel('Student Index')
    plt.ylabel('Performance (0 = Fail, 1 = Pass)')
    plt.savefig('actual_vs_predicted.png')
    plt.close()

def main():
    print("=== Student Performance Prediction using Decision Tree ===")
    df = load_data()
    if df is not None:
        X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)
        dt_model, sorted_importance, y_pred = train_evaluate_model(X_train, X_test, y_train, y_test, feature_names)
        visualize_results(dt_model, feature_names, y_test, y_pred, sorted_importance)

if __name__ == "__main__":
    main()
