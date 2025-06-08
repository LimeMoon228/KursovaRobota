import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout,LeakyReLU
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
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

X, X_validation, y, y_validation = train_test_split(X, y, test_size=0.1)

y_onehot = to_categorical(y, num_classes=8)
y_validation_onehot = to_categorical(y_validation, num_classes=8)

model = Sequential([
    Dense(128,kernel_initializer="he_normal",input_shape=(8,)),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    Dropout(0.2),
    Dense(64,kernel_initializer="he_normal",),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    Dropout(0.2),
    Dense(32,kernel_initializer="he_normal",),
    LeakyReLU(alpha=0.1),
    Dense(8,kernel_initializer="glorot_normal", activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

model.fit(X,
          y_onehot,
          epochs=300,
          batch_size=100,
          verbose=1,
          validation_data=(X_validation,y_validation_onehot),
          callbacks=[early_stopping])

y_pred = model.predict(X_validation)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_validation_onehot, axis=1)

cnt = 0
for i in range(len(y_true_classes)):
    if y_pred_classes[i]==y_true_classes[i]: cnt += 1

matrica = confusion_matrix(y_true_classes, y_pred_classes)

print(cnt/len(y_true_classes))
print(matrica)

model.save('model.keras')