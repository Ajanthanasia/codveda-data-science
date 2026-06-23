import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    return x_train, y_train, x_test, y_test


def build_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def plot_image(x, y, prediction=None):
    plt.figure(figsize=(4, 4))
    plt.imshow(x, cmap='gray')
    title = f'Label: {y}'
    if prediction is not None:
        title += f' | Predicted: {prediction}'
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_accuracy(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_loss(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def show_menu():
    print('\nChoose an option:')
    print('1. Show a sample MNIST image')
    print('2. Plot training / validation accuracy')
    print('3. Plot training / validation loss')
    print('4. Show a prediction example')
    print('0. Exit')


def main():
    x_train, y_train, x_test, y_test = load_data()
    model = build_model()

    history = model.fit(
        x_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=2
    )

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f'\nTest Loss: {loss:.4f}')
    print(f'Test Accuracy: {accuracy:.4f}')

    predictions = model.predict(x_test)
    predicted_labels = np.argmax(predictions, axis=1)

    sample_index = 0
    plot_image(x_test[sample_index], y_test[sample_index], predicted_labels[sample_index])

    while True:
        show_menu()
        choice = input('Enter option number: ').strip()
        if choice == '0':
            print('Exiting.')
            break
        elif choice == '1':
            idx = input('Enter sample index (0 to {}): '.format(len(x_test)-1)).strip()
            if idx.isdigit():
                idx = int(idx)
                if 0 <= idx < len(x_test):
                    plot_image(x_test[idx], y_test[idx])
                else:
                    print('Index out of range.')
            else:
                print('Please enter a valid number.')
        elif choice == '2':
            plot_accuracy(history)
        elif choice == '3':
            plot_loss(history)
        elif choice == '4':
            idx = input('Enter sample index (0 to {}): '.format(len(x_test)-1)).strip()
            if idx.isdigit():
                idx = int(idx)
                if 0 <= idx < len(x_test):
                    plot_image(x_test[idx], y_test[idx], predicted_labels[idx])
                else:
                    print('Index out of range.')
            else:
                print('Please enter a valid number.')
        else:
            print('Invalid choice. Please try again.')

    model.save('mnist_model.keras')
    print('Model saved to mnist_model.keras')


if __name__ == '__main__':
    main()
