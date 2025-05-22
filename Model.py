import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

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


X = np.array(load_data('ukrainian_analysis.txt'))
y = np.array(read_y('celyova_zminna.txt'))

X, X_validation, y, y_validation = train_test_split(X, y, test_size=0.2, random_state=33)

y_onehot = to_categorical(y, num_classes=8)
y_validation_onehot = to_categorical(y_validation, num_classes=8)

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

model.fit(X,
          y_onehot,
          epochs=300,
          batch_size=100,
          verbose=1)

y_pred = model.predict(X_validation)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_validation_onehot, axis=1)

cnt = 0
true4_pred2 = []

for i in range(len(y_true_classes)):
    if y_pred_classes[i]==y_true_classes[i]: cnt += 1
    if y_pred_classes[i] == 2 and y_true_classes[i] == 4: true4_pred2.append(X_validation[i])

matrica = confusion_matrix(y_true_classes, y_pred_classes)

model.fit(X_validation
          ,y_validation_onehot
          ,epochs=50
          ,batch_size=100
          ,verbose=1)

print(cnt/len(y_true_classes))
print(matrica)
for i in range(0, len(true4_pred2), 3):
    print(true4_pred2[i:i+3])
model.save('model.keras')