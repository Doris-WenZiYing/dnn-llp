import os
import numpy as np
import pandas as pd
import networkx as nx
from pulp import *
import pickle
from collections import Counter
from networkx.algorithms.simple_paths import all_simple_paths

# ====== 保持原有參數 ======
N = 8
K_PATHS = 3
MAX_WAVELENGTH = 10
INPUT_DIR = "./tms_8nodes_enhanced"
OUTPUT_CSV = "./enhanced_dataset_all_topologies.csv"
LABEL_MAP_FILE = "./enhanced_label_mappings.pkl"
TOPOLOGY_FILE = "./topology_info.pkl"
DEBUG = True
CUTOFF_LEN = 6

# ====== 載入拓撲資訊 ======
with open(TOPOLOGY_FILE, 'rb') as f:
    topology_info = pickle.load(f)

# ====== 保持原有的路徑生成邏輯 ======
def k_shortest_paths(graph, source, target, k, cutoff=CUTOFF_LEN):
    """完全相同的路徑生成邏輯"""
    try:
        all_paths = list(all_simple_paths(graph, source, target, cutoff=cutoff))
        all_paths = sorted(all_paths, key=lambda p: sum(graph[u][v]['length'] for u, v in zip(p, p[1:])))
        return all_paths[:k]
    except:
        return []

def generate_paths_for_topology(G):
    """完全相同的路徑生成邏輯"""
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
    return paths, demand_list

# ====== 保持原有的標籤映射邏輯 ======
def get_solution_signature(solution_dict, demand_list):
    """保持原有的簽名生成邏輯"""
    parts = []
    for (s, d) in sorted(demand_list):
        if (s, d) in solution_dict:
            p_id, w = solution_dict[(s, d)]
            parts.append(f"{s}_{d}_{p_id}_{w}")
        else:
            parts.append(f"{s}_{d}_None")
    return "|".join(parts)

def get_or_create_label(solution_dict, demand_list, solution_to_label, label_to_solution, next_label_id):
    """保持原有的標籤生成邏輯"""
    signature = get_solution_signature(solution_dict, demand_list)
    if signature in solution_to_label:
        return solution_to_label[signature], next_label_id
    else:
        label_id = next_label_id
        solution_to_label[signature] = label_id
        label_to_solution[label_id] = dict(solution_dict)
        next_label_id += 1
        return label_id, next_label_id

# ====== 保持原有的路徑成本計算 ======
def path_cost(path, G):
    """保持原有的成本計算"""
    return sum(G[u][v]['length'] for u, v in zip(path, path[1:]))

# ====== 拓撲編碼（唯一新增部分）======
topo_encoding = {'full_mesh': 0, 'ring': 1, 'mesh': 2, 'random': 3}

# ====== 處理每種拓撲 ======
all_X_data, all_Y_data = [], []
global_solution_to_label = {}
global_label_to_solution = {}
global_next_label_id = 0
global_label_counter = Counter()

for topo_name, topo_data in topology_info.items():
    print(f"\n🔄 處理拓撲: {topo_name.upper()}")
    
    G = topo_data['graph']
    topo_code = topo_encoding[topo_name]  # 拓撲編碼
    
    # 使用原有的路徑生成
    paths, demand_list = generate_paths_for_topology(G)
    print(f"📦 {topo_name} 拓撲共 {len(demand_list)} 個有效 demands")
    
    topo_dir = f"{INPUT_DIR}/{topo_name}"
    if not os.path.exists(topo_dir):
        print(f"⚠️ 目錄不存在: {topo_dir}")
        continue
    
    topo_files = sorted([f for f in os.listdir(topo_dir) if f.endswith(".csv")])
    infeasible_count = 0
    
    for idx, file in enumerate(topo_files):
        tm_path = os.path.join(topo_dir, file)
        tm = pd.read_csv(tm_path, header=None).values
        flat_tm = tm.flatten()
        
        # ====== 完全相同的ILP模型 ======
        model = LpProblem(f"RWA_ILP_{topo_name}", LpMinimize)
        x = {}
        
        for (s, d), path_list in paths.items():
            for p_id, path in enumerate(path_list):
                for w in range(MAX_WAVELENGTH):
                    x[(s, d, p_id, w)] = LpVariable(f"x_{s}_{d}_{p_id}_{w}", 0, 1, LpBinary)
        
        # 保持原有的目標函數
        cost_terms = []
        for (s, d), path_list in paths.items():
            if not path_list:
                continue
            for p_id, path in enumerate(path_list):
                for w in range(MAX_WAVELENGTH):
                    cost = tm[s][d] * path_cost(path, G) * ((w + 1)**2)
                    cost_terms.append(cost * x[(s, d, p_id, w)])
        
        if cost_terms:
            model += lpSum(cost_terms)
        
        # 保持原有的約束條件
        for (s, d), path_list in paths.items():
            if not path_list:
                continue
            if tm[s][d] > 0:
                model += lpSum(x[(s, d, p_id, w)] for p_id in range(len(path_list)) for w in range(MAX_WAVELENGTH)) == 1
            else:
                for p_id in range(len(path_list)):
                    for w in range(MAX_WAVELENGTH):
                        model += x[(s, d, p_id, w)] == 0
        
        # 保持原有的波長衝突約束
        for u, v in G.edges():
            for w in range(MAX_WAVELENGTH):
                conflict_vars = []
                for (s, d), path_list in paths.items():
                    for p_id, path in enumerate(path_list):
                        if len(path) > 1 and (u, v) in zip(path, path[1:]):
                            conflict_vars.append(x[(s, d, p_id, w)])
                if conflict_vars:
                    model += lpSum(conflict_vars) <= 1
        
        # ====== 保持原有的求解 ======
        model.solve(PULP_CBC_CMD(msg=0))
        
        if LpStatus[model.status] == 'Optimal':
            solution_dict = {}
            for (s, d), path_list in paths.items():
                for p_id, path in enumerate(path_list):
                    for w in range(MAX_WAVELENGTH):
                        if x[(s, d, p_id, w)].varValue == 1:
                            solution_dict[(s, d)] = (p_id, w)
            
            # 使用原有的標籤生成邏輯
            tm_label, global_next_label_id = get_or_create_label(
                solution_dict, demand_list,
                global_solution_to_label, global_label_to_solution,
                global_next_label_id
            )
            
            # ====== 關鍵修改：只添加拓撲編碼到特徵 ======
            enhanced_features = np.concatenate([
                flat_tm,          # 保持原有的64維TM特徵
                [topo_code]       # 只添加1維拓撲編碼 (0,1,2,3)
            ])
            
            all_X_data.append(enhanced_features)
            all_Y_data.append(tm_label)
            global_label_counter[tm_label] += 1
            
            if DEBUG and idx % 100 == 0:
                print(f"✅ {topo_name}: {file:<20} → Label {tm_label:<3}")
        else:
            infeasible_count += 1
            if DEBUG and infeasible_count <= 5:
                print(f"❌ {topo_name}: {file:<20} 無法求解")
    
    print(f"📊 {topo_name} 完成，無解數量: {infeasible_count}")

# ====== 儲存結果 ======
df = pd.DataFrame(all_X_data)
df['label'] = all_Y_data
df.to_csv(OUTPUT_CSV, index=False)

print(f"\n🌟 已儲存增強訓練集至 {OUTPUT_CSV}")
print(f"   樣本數: {len(all_Y_data)}")
print(f"   特徵維度: {len(all_X_data[0]) if all_X_data else 0} (原64維TM + 1維拓撲)")
print(f"   類別數: {len(global_label_counter)}")

# 儲存標籤映射（保持原有格式）
with open(LABEL_MAP_FILE, 'wb') as f:
    pickle.dump({
        'solution_to_label': global_solution_to_label,
        'label_to_solution': global_label_to_solution,
        'topology_info': topology_info,
        'paths': {},  # 簡化儲存
        'topo_encoding': topo_encoding
    }, f)

print(f"🧠 標籤映射已儲存至: {LABEL_MAP_FILE}")

# ====== 分析數據平衡 ======
print("\n📊 各拓撲標籤分佈：")
topo_label_stats = {}
for label_id, count in global_label_counter.most_common():
    # 統計每個拓撲的標籤數量
    found_topo = "unknown"
    for topo_name in topology_info.keys():
        topo_samples = sum(1 for i, y in enumerate(all_Y_data) if y == label_id and all_X_data[i][-1] == topo_encoding[topo_name])
        if topo_samples > 0:
            found_topo = topo_name
            break
    
    if found_topo not in topo_label_stats:
        topo_label_stats[found_topo] = 0
    topo_label_stats[found_topo] += count

for topo, count in topo_label_stats.items():
    print(f"  {topo}: {count} 個解決方案")

print(f"\n🎯 最常見的 5 個解決方案：")
for label_id, count in global_label_counter.most_common(5):
    print(f"  Label {label_id:<3}: {count} 次")

# 檢查數據平衡
balanced = all(count > len(all_Y_data) * 0.15 for count in topo_label_stats.values())
print(f"\n{'✅' if balanced else '⚠️'} 數據平衡檢查: {'通過' if balanced else '需要調整'}")