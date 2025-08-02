import pickle
from collections import defaultdict, Counter

TOP_K = 5  # 想查看出現頻率最高的前 K 個 label

# 讀取 label map
with open("label_mappings.pkl", "rb") as f:
    label_dict = pickle.load(f)

label_to_solution = label_dict["label_to_solution"]

# 若你有訓練用 CSV，可以讀來統計 label 頻率
import pandas as pd
df = pd.read_csv("train_dataset8_weighted.csv")  # 改成你的資料集檔名
label_counts = Counter(df["label"])

print(f"📊 出現頻率最高的前 {TOP_K} 個 labels：")
for label_id, count in label_counts.most_common(TOP_K):
    print(f"\n🧾 Label {label_id}（出現 {count} 次）:")
    sol = label_to_solution[label_id]
    for (s, d), (p_id, w) in sorted(sol.items()):
        print(f"  Demand {s} → {d} → path_{p_id}, λ{w}")
