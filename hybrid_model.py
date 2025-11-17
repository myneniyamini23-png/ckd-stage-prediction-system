import tensorflow as tf
from tensorflow.keras import layers, models

def build_hybrid_model(input_shape, num_classes):
    model = models.Sequential()

    # 1. Convolutional Block (for feature extraction)
    model.add(layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(layers.Conv1D(128, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Dropout(0.3))

    # 2. BiLSTM Block (for sequential feature learning)
    model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
    model.add(layers.Bidirectional(layers.LSTM(32)))

    # 3. Fully Connected Layers
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(32, activation='relu'))

    # 4. Output Layer
    if num_classes == 1:
        model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification
    else:
        model.add(layers.Dense(num_classes, activation='softmax'))  # Multiclass

    return model
