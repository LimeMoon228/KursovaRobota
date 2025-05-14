import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.utils import to_categorical

def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip().replace('[', '').replace(']', '').replace(" ", "")
            items = line.split(',')

            numbers = [int(x.strip()) for x in items if x.strip()]
            data.append(numbers)

    return data

def read_y(file_path):
        with open(file_path, 'r') as file:
            y = [int(line.strip())-1 for line in file if line.strip()]
        return y

data = load_data('ukrainian_analysis.txt')
data = np.array(data)

X = data
y = np.array(read_y('celyova_zminna.txt'))


y_onehot = to_categorical(y, num_classes=8)


model = Sequential([
    Dense(64, activation='relu', input_shape=(8,)),
    BatchNormalization(),
    Dropout(0.1),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.1),
    Dense(16, activation='relu'),
    Dense(8, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X,
                    y_onehot,
                    epochs=300,
                    batch_size=100,
                    verbose=1)

model.save('model.keras')