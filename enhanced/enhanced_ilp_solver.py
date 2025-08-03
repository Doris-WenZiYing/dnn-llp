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

# ====== 保持原有的路徑成本計算 ======
def path_cost(path, G):
    """保持原有的成本計算"""
    return sum(G[u][v]['length'] for u, v in zip(path, path[1:]))

# ====== 拓撲編碼（唯一新增部分）======
topo_encoding = {'full_mesh': 0, 'ring': 1, 'mesh': 2, 'random': 3}

# ====== 處理每種拓撲 ======
all_X_data, all_Y_data = [], []
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
        
        # ====== 關鍵修改：簡化標籤生成 ======
        if LpStatus[model.status] == 'Optimal':
            # 找出使用的最大波長索引
            max_wavelength_used = -1
            
            # 遍歷所有解變數，找出被選中的最大波長
            for (s, d), path_list in paths.items():
                for p_id, path in enumerate(path_list):
                    for w in range(MAX_WAVELENGTH):
                        if x[(s, d, p_id, w)].varValue == 1:
                            if w > max_wavelength_used:
                                max_wavelength_used = w
            
            # 使用最大波長索引作為簡化標籤
            tm_label = max_wavelength_used
            
            # ====== 保持原有的特徵組合：64維TM + 1維拓撲 ======
            enhanced_features = np.concatenate([
                flat_tm,          # 保持原有的64維TM特徵
                [topo_code]       # 只添加1維拓撲編碼 (0,1,2,3)
            ])
            
            all_X_data.append(enhanced_features)
            all_Y_data.append(tm_label)
            global_label_counter[tm_label] += 1
            
            if DEBUG and idx % 100 == 0:
                print(f"✅ {topo_name}: {file:<20} → Max Wavelength {tm_label}")
        else:
            infeasible_count += 1
            if DEBUG and infeasible_count <= 5:
                print(f"❌ {topo_name}: {file:<20} 無法求解")
    
    print(f"📊 {topo_name} 完成，無解數量: {infeasible_count}")

# ====== 儲存結果 ======
df = pd.DataFrame(all_X_data)
df['label'] = all_Y_data
df.to_csv(OUTPUT_CSV, index=False)

print(f"\n🌟 已儲存簡化訓練集至 {OUTPUT_CSV}")
print(f"   樣本數: {len(all_Y_data)}")
print(f"   特徵維度: {len(all_X_data[0]) if all_X_data else 0} (64維TM + 1維拓撲)")
print(f"   類別數: {len(global_label_counter)} (簡化後)")

# ====== 儲存簡化的標籤映射 ======
simplified_label_info = {
    'label_type': 'max_wavelength_index',
    'label_range': f'0 to {max(all_Y_data) if all_Y_data else -1}',
    'topology_encoding': topo_encoding,
    'feature_dimensions': {'tm_size': 64, 'topo_size': 1},
    'total_samples': len(all_Y_data),
    'total_classes': len(global_label_counter)
}

with open(LABEL_MAP_FILE, 'wb') as f:
    pickle.dump(simplified_label_info, f)

print(f"🧠 簡化標籤映射已儲存至: {LABEL_MAP_FILE}")

# ====== 分析簡化後的數據分佈 ======
print("\n📊 簡化標籤分佈：")
for label_id, count in global_label_counter.most_common():
    percentage = count / len(all_Y_data) * 100
    print(f"  最大波長 {label_id}: {count} 次 ({percentage:.1f}%)")

print(f"\n📊 各拓撲樣本分佈：")
topo_stats = {name: 0 for name in topo_encoding.keys()}
for i, features in enumerate(all_X_data):
    topo_code = int(features[-1])  # 最後一維是拓撲編碼
    topo_name = [name for name, code in topo_encoding.items() if code == topo_code][0]
    topo_stats[topo_name] += 1

for topo_name, count in topo_stats.items():
    percentage = count / len(all_Y_data) * 100 if all_Y_data else 0
    print(f"  {topo_name.upper()}: {count} 個樣本 ({percentage:.1f}%)")

# 檢查數據平衡
min_topo_samples = min(topo_stats.values()) if topo_stats.values() else 0
max_topo_samples = max(topo_stats.values()) if topo_stats.values() else 0
balance_ratio = min_topo_samples / max_topo_samples if max_topo_samples > 0 else 0

print(f"\n{'✅' if balance_ratio > 0.7 else '⚠️'} 拓撲數據平衡度: {balance_ratio:.2f}")
print(f"✅ 標籤簡化成功: 從數千類降至 {len(global_label_counter)} 類")