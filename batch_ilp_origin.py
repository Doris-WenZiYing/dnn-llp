import os
import numpy as np
import pandas as pd
import networkx as nx
from pulp import *

# ====== 參數設定 ======
N = 5
K_PATHS = 3
MAX_WAVELENGTH = 10
INPUT_DIR = "tms_5nodes"
OUTPUT_CSV = "train_dataset_origin.csv"

# ====== 建立拓樸 ======
G = nx.complete_graph(N, create_using=nx.DiGraph())
for u, v in G.edges():
    G[u][v]['length'] = 1  # 可自訂距離權重

# ====== 建立候選路徑表（每組 demand 最多 K 條）======
from networkx.algorithms.simple_paths import all_simple_paths

def k_shortest_paths(graph, source, target, k):
    all_paths = list(all_simple_paths(graph, source, target, cutoff=6))  # 長度限制防止爆炸
    all_paths = sorted(all_paths, key=lambda p: len(p))  # 短的排前面
    return all_paths[:k]

paths = {}
for s in range(N):
    for d in range(N):
        if s != d:
            try:
                paths[(s, d)] = k_shortest_paths(G, s, d, K_PATHS)
            except:
                paths[(s, d)] = []

# ====== 開始處理每筆 TM ======
X_data = []
Y_data = []

for file in sorted(os.listdir(INPUT_DIR)):
    if not file.endswith(".csv"):
        continue

    tm_path = os.path.join(INPUT_DIR, file)
    tm = pd.read_csv(tm_path, header=None).values
    flat_tm = tm.flatten()

    # ====== 建立 ILP ======
    model = LpProblem("RWA_ILP", LpMinimize)
    x = {}

    for (s, d), path_list in paths.items():
        for p_id, path in enumerate(path_list):
            for w in range(MAX_WAVELENGTH):
                var = LpVariable(f"x_{s}_{d}_{p_id}_{w}", 0, 1, LpBinary)
                x[(s, d, p_id, w)] = var

    model += lpSum(x.values()), "Minimize_Total_Resources"

    # 每個 demand 只能選一個 (path, wavelength)
    for (s, d), path_list in paths.items():
        model += lpSum(x[(s, d, p_id, w)] for p_id in range(len(path_list)) for w in range(MAX_WAVELENGTH)) == 1

    # link-wavelength 不得衝突
    for u, v in G.edges():
        for w in range(MAX_WAVELENGTH):
            conflict_terms = []
            for (s, d), path_list in paths.items():
                for p_id, path in enumerate(path_list):
                    if (u, v) in zip(path, path[1:]):
                        conflict_terms.append(x[(s, d, p_id, w)])
            if conflict_terms:
                model += lpSum(conflict_terms) <= 1

    # ====== 求解 ======
    model.solve()

    # ====== 記錄資料 (flattened TM → label) ======
    for (s, d), path_list in paths.items():
        for p_id, path in enumerate(path_list):
            for w in range(MAX_WAVELENGTH):
                if x[(s, d, p_id, w)].varValue == 1:
                    label = p_id * MAX_WAVELENGTH + w
                    X_data.append(flat_tm)
                    Y_data.append(label)

    print(f"✅ {file} 處理完成，共產生 {len(paths)*1} 筆樣本")

# ====== 存成訓練檔 ======
df = pd.DataFrame(X_data)
df['label'] = Y_data
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n🎯 完成訓練資料儲存：{OUTPUT_CSV}（共 {len(Y_data)} 筆）")
