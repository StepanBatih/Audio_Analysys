import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

# Завантаження моделі
model_path = "clear_data.h5"
model = load_model(model_path)

# 1. Візуалізація структури моделі
def plot_model_structure(model):
    plot_model(model, show_shapes=True, show_layer_names=True, to_file="model_structure.png")
    print("Структура моделі збережена як model_structure.png")

# 2. Аналіз вагів
def plot_weights_histogram(model):
    for i, layer in enumerate(model.layers):
        if len(layer.get_weights()) > 0:  # Шар має ваги
            weights, biases = layer.get_weights()
            plt.figure(figsize=(12, 5))

            # Гістограма вагів
            plt.subplot(1, 2, 1)
            plt.hist(weights.flatten(), bins=50, color='blue', edgecolor='black', alpha=0.7)
            plt.title(f'Ваги шару {layer.name}')
            plt.xlabel('Значення вагів')
            plt.ylabel('Частота')

            # Гістограма bias
            plt.subplot(1, 2, 2)
            plt.hist(biases.flatten(), bins=50, color='green', edgecolor='black', alpha=0.7)
            plt.title(f'Зсуви (bias) шару {layer.name}')
            plt.xlabel('Значення зсувів')
            plt.ylabel('Частота')

            plt.tight_layout()
            plt.show()

# 3. Графіки метрик навчання
def plot_training_metrics(history_path):
    with open(history_path, 'rb') as file:
        history = pickle.load(file)  # Завантаження історії навчання

    # Графік втрат (loss)
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Тренувальні втрати', color='blue')
    plt.plot(history['val_loss'], label='Валідаційні втрати', color='orange')
    plt.title('Графік втрат (Loss)')
    plt.xlabel('Епохи')
    plt.ylabel('Втрати')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Графік точності (accuracy)
    plt.figure(figsize=(10, 6))
    plt.plot(history['accuracy'], label='Тренувальна точність', color='green')
    plt.plot(history['val_accuracy'], label='Валідаційна точність', color='red')
    plt.title('Графік точності (Accuracy)')
    plt.xlabel('Епохи')
    plt.ylabel('Точність')
    plt.legend()
    plt.grid(True)
    plt.show()

# 4. Активації шарів
def plot_layer_activations(model, sample_input):
    from tensorflow.keras.models import Model

    for layer in model.layers:
        if 'dense' in layer.name or 'conv' in layer.name:  # Лише "активні" шари
            intermediate_model = Model(inputs=model.input, outputs=layer.output)
            activations = intermediate_model.predict(sample_input)

            plt.figure(figsize=(10, 6))
            plt.hist(activations.flatten(), bins=50, color='purple', edgecolor='black', alpha=0.7)
            plt.title(f'Розподіл активацій шару {layer.name}')
            plt.xlabel('Значення активацій')
            plt.ylabel('Частота')
            plt.show()

# Виконання аналізу
print("Аналіз моделі...")
plot_model_structure(model)  # Візуалізація структури
plot_weights_histogram(model)  # Гістограми вагів

# Завантаження історії навчання (якщо доступно)
history_path = "training_history.pkl"
plot_training_metrics(history_path)

# Розподіл активацій (на прикладі)
# Вставте будь-яке вхідне тестове зображення або сигнал
# sample_input = np.random.rand(1, 200)  # Замініть на реальний приклад
# plot_layer_activations(model, sample_input)
