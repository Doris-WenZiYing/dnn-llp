import numpy as np
import pandas as pd
import os

N_NODES = 8                   # ← 改為 12 個節點
N_SAMPLES = 1000                # ← 改為 1000 個樣本
VARIATION = 0.5              # 維持不變
OUTPUT_DIR = "tms_8nodes"     # ← 建議改個資料夾名稱避免混淆

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 建立 12x12 的 Canonical TM ===
# 這裡我們使用隨機整數模擬流量，對角線為 0（不會送自己）

np.random.seed(42)  # 為了結果可重現，可加上 random seed
canonical_tm = np.random.randint(20, 120, size=(N_NODES, N_NODES))
np.fill_diagonal(canonical_tm, 0)  # 自己到自己不送資料

# === 批次產生 noisy TM ===
for i in range(N_SAMPLES):
    noise = np.random.normal(loc=0.0, scale=VARIATION, size=(N_NODES, N_NODES))
    noisy_tm = canonical_tm * (1 + noise)
    noisy_tm = np.clip(noisy_tm, 0, None).astype(int)

    df = pd.DataFrame(noisy_tm)
    df.to_csv(f"{OUTPUT_DIR}/tm_{i:04d}.csv", index=False, header=False)

    if i % 100 == 0:
        print(f"已產生 {i} / {N_SAMPLES} 筆 TMs")

print("✅ 全部 Traffic Matrices 產生完畢！")
