import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# === 載入資料 ===
df = pd.read_csv("dataset88.csv")
X = df.iloc[:, :-1].values
Y = df['label'].values
Y_cat = to_categorical(Y)
num_classes = Y_cat.shape[1]

# === Normalize 特徵 ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

import joblib
joblib.dump(scaler, "scaler.pkl") 

# === 分割資料 ===
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_cat, test_size=0.2, random_state=42)

# === 建立更深的 DNN 模型 ===
model = Sequential([
    Input(shape=(X.shape[1],)),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

# === 編譯模型 ===
initial_lr = 0.0005
model.compile(optimizer=Adam(learning_rate=initial_lr),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# === Callbacks ===
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
checkpoint = ModelCheckpoint("best_model.keras", monitor='val_accuracy', save_best_only=True)

# === 訓練模型 ===
history = model.fit(
    X_train, Y_train,
    epochs=200,
    batch_size=64,
    validation_split=0.1,
    callbacks=[lr_scheduler, checkpoint],
    verbose=2
)

# === 評估模型 ===
loss, acc = model.evaluate(X_test, Y_test)
print(f"\n✅ 測試準確率：{acc:.4f}")

# === 儲存最終模型 ===
model.save("final_model_8nodes.keras")

# === 畫圖 ===
plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Val Accuracy")
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("accuracy_curve_deep.png")

plt.figure()
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("loss_curve_deep.png")
