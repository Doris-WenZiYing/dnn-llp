import pandas as pd
from collections import Counter

# === 參數設定 ===
INPUT_CSV = "dataset88.csv"
OUTPUT_CSV = "filtered_dataset88.csv"
MIN_COUNT = 1       # label 出現次數過少門檻
MAX_COUNT = 5     # label 出現次數過多門檻

# === 讀取資料 ===
df = pd.read_csv(INPUT_CSV)
label_counts = Counter(df['label'])

# === 篩選 label 在合理範圍內的資料 ===
valid_labels = [label for label, count in label_counts.items() if MIN_COUNT <= count <= MAX_COUNT]
filtered_df = df[df['label'].isin(valid_labels)].reset_index(drop=True)

# === 儲存結果 ===
filtered_df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ 已儲存篩選後的資料集至 {OUTPUT_CSV}，原始樣本數: {len(df)} → 篩選後: {len(filtered_df)}")
print(f"保留的 label 數量: {len(valid_labels)}（介於 {MIN_COUNT} 到 {MAX_COUNT} 次之間）")
