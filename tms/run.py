import numpy as np
import pandas as pd
import networkx as nx
from pulp import *

# ========== 載入 TM ==========
tm_file = "tm_0000.csv"  # 改成你要測的
tm = pd.read_csv(tm_file, header=None).values
N = tm.shape[0]  # 節點數量

# ========== 建立拓樸 ==========
# fully connected mesh
G = nx.complete_graph(N, create_using=nx.DiGraph())
for u, v in G.edges():
    G[u][v]['length'] = 1  # 可加距離限制

# ========== 路徑候選 ==========
# 每個 demand 只取最短路徑（簡化版本）
paths = {}
for s in range(N):
    for d in range(N):
        if s != d:
            try:
                path = nx.shortest_path(G, s, d, weight='length')
                paths[(s, d)] = [path]
            except:
                paths[(s, d)] = []

# ========== ILP 定義 ==========
MAX_WAVELENGTH = 10
model = LpProblem("RWA_ILP", LpMinimize)

# 決策變數：是否為 demand (s,d) 選擇 path 與 wavelength
x = {}
for (s, d), path_list in paths.items():
    for p_id, path in enumerate(path_list):
        for w in range(MAX_WAVELENGTH):
            var = LpVariable(f"x_{s}_{d}_{p_id}_{w}", 0, 1, LpBinary)
            x[(s, d, p_id, w)] = var

# 目標：最小化使用的 wavelength 數（可自訂）
model += lpSum(x.values()), "Minimize_Wavelength_Usage"

# 限制：每筆 demand 選一條路徑與一個 wavelength
for (s, d), path_list in paths.items():
    model += lpSum(x[(s, d, p_id, w)] for p_id in range(len(path_list)) for w in range(MAX_WAVELENGTH)) == 1

# 限制：每個 link, wavelength 不得衝突
for u, v in G.edges():
    for w in range(MAX_WAVELENGTH):
        conflict_terms = []
        for (s, d), path_list in paths.items():
            for p_id, path in enumerate(path_list):
                if (u, v) in zip(path, path[1:]):
                    conflict_terms.append(x[(s, d, p_id, w)])
        if conflict_terms:
            model += lpSum(conflict_terms) <= 1

# 求解
model.solve()

# ========== 輸出 RWA Label ==========
print("\n[Routing and Wavelength Assignment]")
for (s, d), path_list in paths.items():
    for p_id, path in enumerate(path_list):
        for w in range(MAX_WAVELENGTH):
            if x[(s, d, p_id, w)].varValue == 1:
                print(f"Demand {s}->{d} | Path: {path} | Wavelength: {w}")
