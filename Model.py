import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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

data = load_data('ukrainian_analysis.txt')

data = np.array(data)

X = data[:, :-1]
y = data[:, -1]


y_onehot = to_categorical(y, num_classes=15)


model = Sequential([
    Dense(20, activation='relu', input_shape=(9,)),
    Dense(10, activation='relu'),
    Dense(15, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy')

history = model.fit(X,
                    y_onehot,
                    epochs=100,
                    batch_size=32,
                    verbose=1)

model.save('model.keras')