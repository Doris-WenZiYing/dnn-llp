import os
import numpy as np
import pandas as pd
import networkx as nx
from pulp import *
import pickle
from collections import Counter
from networkx.algorithms.simple_paths import all_simple_paths

# ====== 可調參數 ======
N = 8
K_PATHS = 3
MAX_WAVELENGTH = 20
INPUT_DIR = "tms_8nodes"
OUTPUT_CSV = "dataset88.csv"
LABEL_MAP_FILE = "label_88mappings.pkl"
INFEASIBLE_LOG = "infeasible_tms.csv"
DEBUG = True
CUTOFF_LEN = 6 # 🔧 調高以確保找到長路徑

# ====== 建立非對稱拓樸 ======
# ====== 建立雙向拓樸 ======
G = nx.DiGraph()
G.add_nodes_from(range(N))

edges_with_lengths = [
    (0, 1, 5),
    (1, 2, 5),
    (2, 3, 5),
    (3, 4, 5),
    (4, 5, 5),
    (5, 6, 10),
    (6, 7, 10),
    (7, 0, 10),
]

# 加上反向邊
for u, v, length in edges_with_lengths:
    G.add_edge(u, v, length=length)
    G.add_edge(v, u, length=length)  # 新增這一行，建立雙向


# ====== 生成 K-shortest paths ======
def k_shortest_paths(graph, source, target, k, cutoff=CUTOFF_LEN):
    all_paths = list(all_simple_paths(graph, source, target, cutoff=cutoff))
    all_paths = sorted(all_paths, key=lambda p: sum(graph[u][v]['length'] for u, v in zip(p, p[1:])))
    return all_paths[:k]

paths = {}
demand_list = []
for s in range(N):
    for d in range(N):
        if s != d:
            path_list = k_shortest_paths(G, s, d, K_PATHS)
            paths[(s, d)] = path_list
            if path_list:
                demand_list.append((s, d))
            elif DEBUG:
                print(f"⚠️ 無法找到 {s} → {d} 的路徑")

print(f"📦 共 {len(demand_list)} 個 (s→d) demands。")

# ====== 建立 label 對應表 ======
solution_to_label = {}
label_to_solution = {}
next_label_id = 0

def get_solution_signature(solution_dict, demand_list):
    parts = []
    for (s, d) in sorted(demand_list):
        if (s, d) in solution_dict:
            p_id, = solution_dict[(s, d)]
            parts.append(f"{s}_{d}_{p_id}")
        else:
            parts.append(f"{s}_{d}_None")
    return "|".join(parts)

def get_or_create_label(solution_dict, demand_list):
    global next_label_id
    signature = get_solution_signature(solution_dict, demand_list)
    if signature in solution_to_label:
        return solution_to_label[signature]
    else:
        label_id = next_label_id
        solution_to_label[signature] = label_id
        label_to_solution[label_id] = dict(solution_dict)
        next_label_id += 1
        return label_id

# ====== 計算路徑成本 ======
def path_cost(path):
    return sum(G[u][v]['length'] for u, v in zip(path, path[1:]))

# ====== 主處理流程 ======
X_data, Y_data = [], []
infeasible_files = []
label_counter = Counter()

all_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")])
print(f"\n🧶 開始處理 {len(all_files)} 筆 Traffic Matrix...\n")

for idx, file in enumerate(all_files):
    tm_path = os.path.join(INPUT_DIR, file)
    tm = pd.read_csv(tm_path, header=None).values
    flat_tm = tm.flatten()

    model = LpProblem("RWA_ILP", LpMinimize)
    x = {}

    for (s, d), path_list in paths.items():
        for p_id, path in enumerate(path_list):
            for w in range(MAX_WAVELENGTH):
                x[(s, d, p_id, w)] = LpVariable(f"x_{s}_{d}_{p_id}_{w}", 0, 1, LpBinary)

    cost_terms = []
    for (s, d), path_list in paths.items():
        if not path_list:
            continue
        for p_id, path in enumerate(path_list):
            for w in range(MAX_WAVELENGTH):
                cost_terms.append(tm[s][d] * path_cost(path) * ((w + 1)**2) * x[(s, d, p_id, w)])
    model += lpSum(cost_terms)

    for (s, d), path_list in paths.items():
        if not path_list:
            continue
        if tm[s][d] > 0:
            model += lpSum(x[(s, d, p_id, w)] for p_id in range(len(path_list)) for w in range(MAX_WAVELENGTH)) == 1
        else:
            for p_id in range(len(path_list)):
                for w in range(MAX_WAVELENGTH):
                    model += x[(s, d, p_id, w)] == 0

    for u, v in G.edges():
        for w in range(MAX_WAVELENGTH):
            model += lpSum(
                x[(s, d, p_id, w)]
                for (s, d), path_list in paths.items()
                for p_id, path in enumerate(path_list)
                if (u, v) in zip(path, path[1:])
            ) <= 1

    model.solve()

    if LpStatus[model.status] == 'Optimal':
        solution_dict = {}
        for (s, d), path_list in paths.items():
            for p_id, path in enumerate(path_list):
                for w in range(MAX_WAVELENGTH):
                    if x[(s, d, p_id, w)].varValue == 1:
                        solution_dict[(s, d)] = (p_id, w)

        tm_label = get_or_create_label(solution_dict, demand_list)
        X_data.append(flat_tm)
        Y_data.append(tm_label)
        label_counter[tm_label] += 1

        if DEBUG:
            print(f"✅ {file:<20} → Label {tm_label:<3} ｜目前總 label 數: {next_label_id}")
    else:
        infeasible_files.append(file)
        print(f"❌ {file:<20} 無法求解")

# ====== 儲存結果 ======
df = pd.DataFrame(X_data)
df['label'] = Y_data
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n🌟 已儲存訓練集至 {OUTPUT_CSV}、樣本數: {len(Y_data)}、類別數: {len(label_counter)}")

with open(LABEL_MAP_FILE, 'wb') as f:
    pickle.dump({
        'solution_to_label': solution_to_label,
        'label_to_solution': label_to_solution,
        'demand_list': demand_list,
        'paths': paths
    }, f)
print(f"🧠 label 映對表已儲存至: {LABEL_MAP_FILE}")

if infeasible_files:
    pd.DataFrame({'infeasible_tm': infeasible_files}).to_csv(INFEASIBLE_LOG, index=False)
    print(f"⚠️ 共 {len(infeasible_files)} 筆 TM 無解，已紀錄至 {INFEASIBLE_LOG}")

# ====== 印出最常見 label 前幾名 ======
print("\n📊 最常見的 5 個解決方案（Label）:")
for label_id, count in label_counter.most_common(5):
    print(f"  Label {label_id:<3} 出現次數: {count:<4}")
    solution = label_to_solution[label_id]
    for (s, d), (p_id, w) in list(solution.items())[:3]:
        print(f"     Demand {s}→{d} → path_{p_id}, λ{w}")
    if len(solution) > 3:
        print(f"     ... (還有 {len(solution)-3} 個 demands)\n")
