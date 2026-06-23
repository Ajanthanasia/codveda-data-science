import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    classification_report
)


def load_data(path='iris.csv'):
    df = pd.read_csv(path)
    return df


def preprocess(df):
    X = df.drop('species', axis=1)
    y = df['species']

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_encoded, encoder


def train_models(X_train, y_train):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=200, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42)
    }

    for model in models.values():
        model.fit(X_train, y_train)

    return models


def evaluate_model(name, model, X_test, y_test, classes):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    print(f"\n=== {name} ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1 score (macro): {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=classes, zero_division=0))

    if hasattr(model, 'predict_proba'):
        y_score = model.predict_proba(X_test)
    else:
        y_score = model.decision_function(X_test)
        if y_score.ndim == 1:
            y_score = label_binarize(y_score, classes=range(len(classes)))

    return accuracy, precision, recall, f1, y_score


def plot_roc(models, X_test, y_test, classes):
    y_test_bin = label_binarize(y_test, classes=range(len(classes)))
    plt.figure(figsize=(10, 8))

    for name, model in models.items():
        if hasattr(model, 'predict_proba'):
            y_score = model.predict_proba(X_test)
        else:
            y_score = model.decision_function(X_test)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(classes)):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr['macro'], tpr['macro'], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
        roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

        plt.plot(
            fpr['macro'],
            tpr['macro'],
            label=f'{name} (macro AUC = {roc_auc["macro"]:.3f})'
        )

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.title('ROC Curve Comparison (macro-average)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()


def main():
    df = load_data()
    X, y, encoder = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = train_models(X_train, y_train)
    classes = encoder.classes_

    results = []
    for name, model in models.items():
        accuracy, precision, recall, f1, _ = evaluate_model(name, model, X_test, y_test, classes)
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        })

    results_df = pd.DataFrame(results)
    print("\nSummary of model performance:")
    print(results_df)

    plot_roc(models, X_test, y_test, classes)


if __name__ == '__main__':
    main()
