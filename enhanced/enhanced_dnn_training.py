import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import joblib
import os

# ====== 參數設定 ======
DATASET_FILE = "./enhanced_dataset_all_topologies.csv"
LABEL_MAP_FILE = "./enhanced_label_mappings.pkl"
MODEL_NAME = "topology_aware_dnn_model.keras"
SCALER_NAME = "./enhanced_scaler.pkl"
RESULTS_DIR = "./training_results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ====== 載入簡化後的數據集 ======
print("📥 載入簡化數據集...")
df = pd.read_csv(DATASET_FILE)
with open(LABEL_MAP_FILE, 'rb') as f:
    label_info = pickle.load(f)

X = df.iloc[:, :-1].values  # 前65維：64維TM + 1維拓撲
y = df['label'].values      # 簡化後的標籤：最大波長索引

print(f"原始數據形狀: X={X.shape}, y={y.shape}")
print(f"類別數量: {len(np.unique(y))} (簡化後)")
print(f"特徵維度: {X.shape[1]} (64維TM + 1維拓撲)")

# ====== 數據分佈分析 ======
print(f"\n📊 簡化標籤分佈:")
label_counts = Counter(y)
for label, count in sorted(label_counts.items()):
    percentage = count / len(y) * 100
    print(f"  最大波長 {label}: {count} 樣本 ({percentage:.1f}%)")

# ====== 拓撲分佈分析 ======
topo_encoding = {'full_mesh': 0, 'ring': 1, 'mesh': 2, 'random': 3}
reverse_topo = {v: k for k, v in topo_encoding.items()}

print(f"\n📊 各拓撲樣本分佈:")
for topo_code, topo_name in reverse_topo.items():
    topo_samples = np.sum(X[:, -1] == topo_code)
    percentage = topo_samples / len(X) * 100
    print(f"  {topo_name.upper()}: {topo_samples} 樣本 ({percentage:.1f}%)")

# ====== 修正特徵標準化：只標準化TM特徵 ======
print("\n🔧 進行特徵標準化...")
X_tm = X[:, :-1]    # 前64維：流量矩陣特徵
X_topo = X[:, -1:]  # 最後1維：拓撲編碼

# 只對TM特徵進行標準化
scaler = StandardScaler()
X_tm_scaled = scaler.fit_transform(X_tm)

# 重新組合：標準化的TM + 原始的拓撲編碼
X_scaled = np.concatenate([X_tm_scaled, X_topo], axis=1)
joblib.dump(scaler, SCALER_NAME)

print(f"✅ 特徵標準化完成，標準化器已儲存至 {SCALER_NAME}")

# ====== 標籤重新映射為連續整數 ======
unique_labels = np.unique(y)
num_classes = len(unique_labels)

# 建立標籤映射：原始標籤 -> 連續整數
label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted(unique_labels))}
reverse_label_mapping = {new_label: old_label for old_label, new_label in label_mapping.items()}

print(f"📊 標籤重新映射:")
for old_label, new_label in label_mapping.items():
    count = np.sum(y == old_label)
    print(f"  最大波長 {old_label} -> 類別 {new_label} ({count} 樣本)")

# 將原始標籤映射為連續整數
y_mapped = np.array([label_mapping[label] for label in y])
y_categorical = to_categorical(y_mapped, num_classes=num_classes)

print(f"📊 將標籤轉換為 {num_classes} 類分類問題")

# ====== 確保拓撲平衡的數據切分 ======
print("\n🔀 進行拓撲平衡的數據切分...")

# 使用拓撲編碼作為分層依據，確保每種拓撲在訓練/測試集中比例一致
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_categorical,
    test_size=0.2,
    random_state=42,
    stratify=X_scaled[:, -1]  # 按拓撲ID分層
)

print(f"訓練集大小: {X_train.shape[0]}")
print(f"測試集大小: {X_test.shape[0]}")

# 檢查訓練和測試集的拓撲分佈
print(f"\n📊 訓練集拓撲分佈:")
for topo_code, topo_name in reverse_topo.items():
    train_topo_count = np.sum(X_train[:, -1] == topo_code)
    percentage = train_topo_count / len(X_train) * 100
    print(f"  {topo_name.upper()}: {train_topo_count} 樣本 ({percentage:.1f}%)")

print(f"\n📊 測試集拓撲分佈:")
for topo_code, topo_name in reverse_topo.items():
    test_topo_count = np.sum(X_test[:, -1] == topo_code)
    percentage = test_topo_count / len(X_test) * 100
    print(f"  {topo_name.upper()}: {test_topo_count} 樣本 ({percentage:.1f}%)")

# ====== 建立穩健的MLP模型 ======
def create_robust_mlp_model(input_dim, num_classes):
    """建立穩健的多層感知機模型"""
    model = Sequential([
        # 輸入層
        Dense(512, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.4),
        
        # 隱藏層 1
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        
        # 隱藏層 2
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        
        # 輸出層
        Dense(num_classes, activation='softmax')
    ])
    
    return model

print("\n🏗️ 建立穩健的MLP模型...")
model = create_robust_mlp_model(X_scaled.shape[1], num_classes)

# 編譯模型
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ====== 訓練設定（加入早停機制）======
callbacks = [
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=15,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        f'{RESULTS_DIR}/{MODEL_NAME}',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=25,
        restore_best_weights=True,
        verbose=1
    )
]

# ====== 訓練模型 ======
print("\n🚀 開始訓練穩健MLP模型...")
history = model.fit(
    X_train, y_train,
    epochs=300,              # 增加訓練輪數
    batch_size=32,
    validation_split=0.15,
    callbacks=callbacks,
    verbose=1
)

# ====== 整體模型評估 ======
print("\n📊 評估整體模型性能...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"整體測試準確率: {test_acc:.4f}")

# ====== 關鍵新增：按拓撲分項評估 ======
print("\n📊 各拓撲準確率分析:")
y_pred = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# 將預測的類別索引映射回原始波長索引，用於更好的解釋
y_pred_wavelengths = np.array([reverse_label_mapping[cls] for cls in y_pred_classes])
y_test_wavelengths = np.array([reverse_label_mapping[cls] for cls in y_test_classes])

topo_results = {}
for topo_code, topo_name in reverse_topo.items():
    # 篩選屬於當前拓撲的測試樣本
    topo_test_mask = X_test[:, -1] == topo_code
    topo_sample_count = np.sum(topo_test_mask)
    
    if topo_sample_count > 0:
        # 計算該拓撲的準確率
        topo_accuracy = np.mean(
            y_pred_classes[topo_test_mask] == y_test_classes[topo_test_mask]
        )
        
        # 統計該拓撲的預測波長分佈
        topo_pred_wavelengths = y_pred_wavelengths[topo_test_mask]
        topo_true_wavelengths = y_test_wavelengths[topo_test_mask]
        
        topo_results[topo_name] = {
            'accuracy': topo_accuracy,
            'samples': topo_sample_count,
            'pred_wavelengths': topo_pred_wavelengths,
            'true_wavelengths': topo_true_wavelengths
        }
        
        status = "✅" if topo_accuracy >= 0.8 else "⚠️" if topo_accuracy >= 0.6 else "❌"
        avg_pred_wavelength = np.mean(topo_pred_wavelengths)
        avg_true_wavelength = np.mean(topo_true_wavelengths)
        
        print(f"  {status} {topo_name.upper()}: {topo_accuracy:.4f} (樣本數: {topo_sample_count})")
        print(f"      平均預測波長: {avg_pred_wavelength:.1f}, 平均真實波長: {avg_true_wavelength:.1f}")
    else:
        print(f"  ⚠️ {topo_name.upper()}: 無測試樣本")

# ====== 結果視覺化 ======
def plot_comprehensive_results():
    """繪製完整的訓練與評估結果"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 訓練準確率曲線
    ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax1.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80% Target')
    ax1.set_title('Training & Validation Accuracy', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 訓練損失曲線
    ax2.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_title('Training & Validation Loss', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 各拓撲準確率對比
    if topo_results:
        topos = list(topo_results.keys())
        accuracies = [topo_results[t]['accuracy'] for t in topos]
        colors = ['green' if acc >= 0.8 else 'orange' if acc >= 0.6 else 'red' for acc in accuracies]
        
        bars = ax3.bar([t.upper() for t in topos], accuracies, color=colors, alpha=0.8)
        ax3.set_title('Accuracy by Topology', fontsize=14)
        ax3.set_ylabel('Accuracy')
        ax3.set_ylim(0, 1)
        ax3.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80% Target')
        
        # 在柱狀圖上標註數值
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. 預測信心度分佈
    confidence_scores = np.max(y_pred, axis=1)
    ax4.hist(confidence_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.axvline(x=0.8, color='red', linestyle='--', alpha=0.7, label='80% Confidence')
    ax4.set_title('Prediction Confidence Distribution', fontsize=14)
    ax4.set_xlabel('Confidence Score')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/comprehensive_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_comprehensive_results()

# ====== 儲存訓練結果 ======
results_summary = {
    'overall_test_accuracy': test_acc,
    'topology_results': topo_results,
    'model_type': 'robust_mlp',
    'model_params': {
        'input_dim': X_scaled.shape[1],
        'num_classes': num_classes,
        'architecture': 'MLP: 512->256->128->classes'
    },
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'label_type': 'max_wavelength_index',
    'label_mapping': label_mapping,
    'reverse_label_mapping': reverse_label_mapping,
    'original_unique_labels': unique_labels.tolist()
}

with open(f'{RESULTS_DIR}/results_summary.pkl', 'wb') as f:
    pickle.dump(results_summary, f)

# ====== 最終診斷報告 ======
print("\n" + "="*70)
print("🎯 拓撲感知DNN訓練完成報告")
print("="*70)

print(f"🔧 模型架構: 穩健MLP (512→256→128→{num_classes})")
print(f"📊 整體測試準確率: {test_acc:.4f}")
print(f"📈 標籤類別數: {num_classes} (從數千類簡化)")
print(f"🏷️ 標籤映射: {dict(sorted(reverse_label_mapping.items()))}")

print(f"\n📊 各拓撲詳細性能:")
all_above_80 = True
total_weighted_acc = 0
total_samples = 0

for topo, result in topo_results.items():
    accuracy = result['accuracy']
    samples = result['samples']
    status = "✅" if accuracy >= 0.8 else "⚠️" if accuracy >= 0.6 else "❌"
    
    print(f"  {status} {topo.upper()}:")
    print(f"      準確率: {accuracy:.4f}")
    print(f"      樣本數: {samples}")
    print(f"      狀態: {'達標' if accuracy >= 0.8 else '需改進'}")
    
    total_weighted_acc += accuracy * samples
    total_samples += samples
    
    if accuracy < 0.8:
        all_above_80 = False

weighted_avg_acc = total_weighted_acc / total_samples if total_samples > 0 else 0

print(f"\n📈 綜合統計:")
print(f"  加權平均準確率: {weighted_avg_acc:.4f}")
print(f"  達標拓撲數量: {sum(1 for result in topo_results.values() if result['accuracy'] >= 0.8)}/{len(topo_results)}")

if all_above_80:
    print(f"\n🎉 恭喜！所有拓撲都達到80%以上準確率目標！")
    print(f"   標籤簡化策略成功：從數千類降至{num_classes}類")
    print(f"   波長預測範圍：{min(unique_labels)} ~ {max(unique_labels)}")
else:
    print(f"\n⚠️ 部分拓撲未完全達標，但相比原來的0%已有巨大改善")
    under_performing = [topo for topo, result in topo_results.items() if result['accuracy'] < 0.8]
    print(f"   需要重點關注: {', '.join(under_performing)}")

print(f"\n📁 所有結果已儲存至 {RESULTS_DIR}/")
print(f"📁 模型已儲存至 {RESULTS_DIR}/{MODEL_NAME}")
print(f"📁 特徵標準化器已儲存至 {SCALER_NAME}")

# ====== 優化建議 ======
print(f"\n💡 後續優化建議:")
print(f"  1. 如有拓撲未達標，可考慮增加該拓撲的訓練樣本")
print(f"  2. 可嘗試調整學習率或網絡架構")
print(f"  3. 監控過擬合：驗證準確率 vs 訓練準確率")

high_confidence_count = np.sum(np.max(y_pred, axis=1) > 0.8)
print(f"  4. 高信心預測比例: {high_confidence_count}/{len(y_pred)} ({high_confidence_count/len(y_pred)*100:.1f}%)")

# 顯示預測 vs 真實的波長分佈統計
print(f"\n📊 波長使用統計:")
for orig_wavelength in sorted(unique_labels):
    mapped_class = label_mapping[orig_wavelength]
    pred_count = np.sum(y_pred_classes == mapped_class)
    true_count = np.sum(y_test_classes == mapped_class)
    print(f"  波長 {orig_wavelength}: 預測 {pred_count} 次, 實際 {true_count} 次")