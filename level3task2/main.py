import pandas as pd
import nltk
import re
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


def load_data(filename='spam.csv'):
    df = pd.read_csv(filename)
    return df


def preprocess_text(text, stop_words, lemmatizer):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]
    return ' '.join(words)


def prepare_data(df):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    df['clean_text'] = df['text'].astype(str).apply(lambda x: preprocess_text(x, stop_words, lemmatizer))
    return df


def train_model(df):
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=2)
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)

    return model, vectorizer, X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    report = classification_report(y_test, y_pred, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred)
    return accuracy, precision, recall, f1, report, matrix


def plot_label_distribution(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='label')
    plt.title('Label Distribution')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()


def plot_confusion(matrix, labels):
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()


def show_menu():
    print('\nSelect an option:')
    print('1. Dataset preview')
    print('2. Label distribution')
    print('3. Model evaluation metrics')
    print('4. Confusion matrix')
    print('5. Predict custom text')
    print('0. Exit')


def main():
    df = load_data('spam.csv')
    df = prepare_data(df)
    model, vectorizer, X_train, X_test, y_train, y_test = train_model(df)
    accuracy, precision, recall, f1, report, matrix = evaluate_model(model, X_test, y_test)
    labels = sorted(df['label'].unique())

    print('Training complete.')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

    while True:
        show_menu()
        choice = input('Enter option number: ').strip()
        if choice == '0':
            print('Exiting.')
            break
        elif choice == '1':
            print('\nData preview:')
            print(df[['text', 'clean_text', 'label']].head())
        elif choice == '2':
            plot_label_distribution(df)
        elif choice == '3':
            print('\nClassification Report:')
            print(report)
            print(f'Accuracy: {accuracy:.4f}')
            print(f'Precision: {precision:.4f}')
            print(f'Recall: {recall:.4f}')
            print(f'F1 Score: {f1:.4f}')
        elif choice == '4':
            plot_confusion(matrix, labels)
        elif choice == '5':
            text = input('Enter text to classify: ').strip()
            clean_text = preprocess_text(text, set(stopwords.words('english')), WordNetLemmatizer())
            vector = vectorizer.transform([clean_text])
            prediction = model.predict(vector)[0]
            print(f'Prediction: {prediction}')
        else:
            print('Invalid option. Please choose again.')


if __name__ == '__main__':
    main()
