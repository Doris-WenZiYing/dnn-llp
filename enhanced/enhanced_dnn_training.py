import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Input, Dense, Dropout, BatchNormalization, 
                                   Concatenate, Reshape, MultiHeadAttention, GlobalMaxPooling1D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import joblib

# ====== 參數設定 ======
DATASET_FILE = "./enhanced_dataset_all_topologies.csv"
LABEL_MAP_FILE = "./enhanced_label_mappings.pkl"
MODEL_NAME = "topology_aware_dnn_model.keras"
SCALER_NAME = "./enhanced_scaler.pkl"
RESULTS_DIR = "./training_results"

import os
os.makedirs(RESULTS_DIR, exist_ok=True)

# ====== 載入數據和標籤映射 ======
print("📥 載入增強數據集...")
df = pd.read_csv(DATASET_FILE)
with open(LABEL_MAP_FILE, 'rb') as f:
    label_info = pickle.load(f)

X = df.iloc[:, :-1].values
y = df['label'].values

print(f"原始數據形狀: X={X.shape}, y={y.shape}")
print(f"類別數量: {len(np.unique(y))}")
print(f"特徵維度: {X.shape[1]} (64維TM + 1維拓撲)")

# ====== 關鍵：數據平衡策略 ======
label_counts = Counter(y)
print(f"原始類別分佈: min={min(label_counts.values())}, max={max(label_counts.values())}")

# 更寬鬆的過濾條件：每個類別至少3個樣本
min_samples = 3
valid_labels = [label for label, count in label_counts.items() if count >= min_samples]
valid_indices = np.isin(y, valid_labels)

X_filtered = X[valid_indices]
y_filtered = y[valid_indices]

print(f"過濾後數據形狀: X={X_filtered.shape}, y={y_filtered.shape}")
print(f"保留類別數量: {len(np.unique(y_filtered))}")

# ====== 確保拓撲平衡 ======
# 檢查每種拓撲的樣本分佈
topo_encoding = {'full_mesh': 0, 'ring': 1, 'mesh': 2, 'random': 3}
reverse_topo = {v: k for k, v in topo_encoding.items()}

print("\n📊 各拓撲樣本分佈:")
for topo_code, topo_name in reverse_topo.items():
    topo_samples = np.sum(X_filtered[:, -1] == topo_code)
    print(f"  {topo_name.upper()}: {topo_samples} 樣本")

# 重新編碼標籤
le = LabelEncoder()
y_encoded = le.fit_transform(y_filtered)
y_categorical = to_categorical(y_encoded)
num_classes = len(np.unique(y_encoded))

# 計算類別權重以處理不平衡
class_weights = compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
class_weight_dict = dict(zip(np.unique(y_encoded), class_weights))

# 特徵標準化（只標準化TM特徵，保持拓撲編碼）
scaler = StandardScaler()
X_tm_scaled = scaler.fit_transform(X_filtered[:, :-1])  # 只標準化前64維
X_scaled = np.column_stack([X_tm_scaled, X_filtered[:, -1]])  # 重新組合
joblib.dump(scaler, SCALER_NAME)

# ====== 保持原有DNN架構基礎上的最小修改 ======
def create_enhanced_model(input_dim, num_classes):
    """基於原有架構的最小增強"""
    
    # 輸入層
    main_input = Input(shape=(input_dim,), name='main_input')
    
    # 分離TM特徵和拓撲特徵
    tm_features = main_input[:, :-1]  # 前64維：TM特徵
    topo_features = main_input[:, -1:]  # 最後1維：拓撲編碼
    
    # TM特徵處理路徑（保持原有邏輯）
    tm_dense1 = Dense(256, activation='relu', name='tm_dense1')(tm_features)
    tm_dense1 = BatchNormalization()(tm_dense1)
    tm_dense1 = Dropout(0.3)(tm_dense1)
    tm_dense2 = Dense(128, activation='relu', name='tm_dense2')(tm_dense1)
    
    # 拓撲特徵處理路徑（簡單處理）
    topo_dense = Dense(32, activation='relu', name='topo_dense')(topo_features)
    
    # 特徵融合
    merged = Concatenate()([tm_dense2, topo_dense])
    
    # 保持原有的注意力機制
    attention_input = Reshape((1, -1))(merged)
    attention_output = MultiHeadAttention(
        num_heads=4, 
        key_dim=64,
        name='multi_head_attention'
    )(attention_input, attention_input)
    attention_output = GlobalMaxPooling1D()(attention_output)
    
    # 保持原有的深層網路結構
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(attention_output)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # 輸出層
    output = Dense(num_classes, activation='softmax', name='output')(x)
    
    model = Model(inputs=main_input, outputs=output)
    return model

# ====== 建立模型 ======
print("🏗️ 建立拓撲感知DNN模型...")
model = create_enhanced_model(X_scaled.shape[1], num_classes)

# 編譯模型（保持原有設置）
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ====== 智能數據分割（確保拓撲平衡）======
def topology_aware_split(X, y_encoded, test_size=0.2):
    """確保每種拓撲在訓練和測試集中都有代表"""
    train_indices, test_indices = [], []
    
    for topo_code in range(4):  # 0,1,2,3
        topo_indices = np.where(X[:, -1] == topo_code)[0]
        if len(topo_indices) > 5:  # 確保有足夠樣本分割
            topo_train, topo_test = train_test_split(
                topo_indices, test_size=test_size, random_state=42
            )
            train_indices.extend(topo_train)
            test_indices.extend(topo_test)
        else:
            # 樣本太少，全部放入訓練集
            train_indices.extend(topo_indices)
    
    return train_indices, test_indices

train_idx, test_idx = topology_aware_split(X_scaled, y_encoded)
X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
y_train, y_test = y_categorical[train_idx], y_categorical[test_idx]

print(f"訓練集大小: {X_train.shape[0]}")
print(f"測試集大小: {X_test.shape[0]}")

# 檢查測試集拓撲分佈
print("\n📊 測試集拓撲分佈:")
for topo_code, topo_name in reverse_topo.items():
    test_topo_count = np.sum(X_test[:, -1] == topo_code)
    print(f"  {topo_name.upper()}: {test_topo_count} 樣本")

# ====== 訓練設定 ======
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1),
    ModelCheckpoint(MODEL_NAME, monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
]

# ====== 訓練模型 ======
print("🚀 開始訓練拓撲感知DNN...")
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.15,
    callbacks=callbacks,
    class_weight=class_weight_dict,  # 使用類別權重
    verbose=1
)

# ====== 評估模型 ======
print("\n📊 評估模型性能...")

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"整體測試準確率: {test_acc:.4f}")

# 各拓撲準確率分析
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# 按拓撲計算準確率
topo_results = {}
for topo_code, topo_name in reverse_topo.items():
    topo_test_mask = X_test[:, -1] == topo_code
    if np.sum(topo_test_mask) > 0:
        topo_acc = np.mean(y_pred_classes[topo_test_mask] == y_test_classes[topo_test_mask])
        topo_results[topo_name] = {
            'accuracy': topo_acc,
            'samples': np.sum(topo_test_mask)
        }
        print(f"{topo_name.upper()} 拓撲準確率: {topo_acc:.4f} (樣本數: {np.sum(topo_test_mask)})")

# ====== 結果視覺化 ======
def plot_training_results():
    """繪製訓練結果"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 訓練曲線
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.axhline(y=0.8, color='red', linestyle='--', label='80% Target')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    # 各拓撲準確率
    if topo_results:
        topos = list(topo_results.keys())
        accuracies = [topo_results[t]['accuracy'] for t in topos]
        colors = ['green' if acc >= 0.8 else 'orange' if acc >= 0.6 else 'red' for acc in accuracies]
        
        bars = ax3.bar(topos, accuracies, color=colors)
        ax3.set_title('Accuracy by Topology')
        ax3.set_ylabel('Accuracy')
        ax3.set_ylim(0, 1)
        
        # 在柱狀圖上標註數值
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # 添加80%基準線
        ax3.axhline(y=0.8, color='red', linestyle='--', label='80% Target')
        ax3.legend()
    
    # 預測信心度分佈
    confidence_scores = np.max(y_pred, axis=1)
    ax4.hist(confidence_scores, bins=20, alpha=0.7, color='skyblue')
    ax4.axvline(x=0.8, color='red', linestyle='--', alpha=0.7, label='80% Threshold')
    ax4.set_title('Prediction Confidence Distribution')
    ax4.set_xlabel('Confidence Score')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_training_results()

# ====== 儲存最終模型和結果 ======
model.save(f'{RESULTS_DIR}/{MODEL_NAME}')

# 儲存訓練結果
results_summary = {
    'overall_test_accuracy': test_acc,
    'topology_results': topo_results,
    'model_params': {
        'input_dim': X_scaled.shape[1],
        'num_classes': num_classes,
        'feature_dimensions': {'tm_size': 64, 'topo_size': 1}
    },
    'training_samples': len(X_train),
    'test_samples': len(X_test)
}

with open(f'{RESULTS_DIR}/results_summary.pkl', 'wb') as f:
    pickle.dump(results_summary, f)

# ====== 最終報告 ======
print("\n" + "="*60)
print("🎯 拓撲感知DNN訓練完成！")
print("="*60)
print(f"整體測試準確率: {test_acc:.4f}")
print("\n各拓撲準確率:")

all_above_80 = True
for topo, result in topo_results.items():
    accuracy = result['accuracy']
    status = "✅" if accuracy >= 0.8 else "❌"
    print(f"  {status} {topo.upper()}: {accuracy:.4f}")
    if accuracy < 0.8:
        all_above_80 = False

if all_above_80:
    print("\n🎉 恭喜！所有拓撲都達到80%以上準確率！")
else:
    print("\n⚠️ 部分拓撲未達到80%目標，但應該比原來的42%有大幅改善。")

print(f"\n📁 結果已儲存至 {RESULTS_DIR}/")
print(f"📁 模型已儲存至 {RESULTS_DIR}/{MODEL_NAME}")
print(f"📁 特徵標準化器已儲存至 {SCALER_NAME}")

# ====== 關鍵診斷信息 ======
print(f"\n🔧 診斷信息:")
print(f"特徵維度: {X_scaled.shape[1]} (64維TM + 1維拓撲)")
print(f"類別數量: {num_classes}")
print(f"樣本數量: 訓練{len(X_train)}, 測試{len(X_test)}")
high_confidence_preds = np.sum(np.max(y_pred, axis=1) > 0.8)
print(f"高信心預測(>80%): {high_confidence_preds}/{len(y_pred)} ({high_confidence_preds/len(y_pred)*100:.1f}%)")