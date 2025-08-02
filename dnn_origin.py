import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# === 載入 rwa_labels.csv ===
df = pd.read_csv("train_dataset_origin.csv")

# === 特徵工程 ===
df['src'] = df['demand'].apply(lambda x: int(x.split('->')[0]))
df['dst'] = df['demand'].apply(lambda x: int(x.split('->')[1]))
df['hop'] = df['path'].apply(lambda x: len(eval(x)) - 1)

X = df[['src', 'dst', 'hop']].values
Y = df['wavelength'].values
Y_cat = to_categorical(Y)
num_classes = Y_cat.shape[1]

# === 分割資料 ===
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_cat, test_size=0.2, random_state=42)

# === 建立模型 ===
model = Sequential([
    Input(shape=(3,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === 訓練模型 ===
history = model.fit(X_train, Y_train, epochs=30, batch_size=64, validation_split=0.1)

# === 測試評估 ===
loss, acc = model.evaluate(X_test, Y_test)
print(f"\n✅ 測試準確率：{acc:.4f}")

# === 可選：儲存模型 ===
model.save("dnn_rwa_model.h5")

# === 畫訓練曲線 ===
plt.plot(history.history['accuracy'], label="Train")
plt.plot(history.history['val_accuracy'], label="Val")
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()