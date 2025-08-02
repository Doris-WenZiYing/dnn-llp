# import numpy as np
# import pandas as pd
# import os
# import networkx as nx
# import pickle

# # ====== 參數設定 - 保持原有樣本數 ======
# N_NODES = 8                   
# N_SAMPLES = 1000              # 保持不變
# VARIATION = 0.5               # 保持不變
# OUTPUT_DIR = "./tms_8nodes_enhanced"
# TOPOLOGY_FILE = "./topology_info.pkl"

# # ====== 建立輸出目錄 ======
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # ====== 建立多種拓撲結構 ======
# def create_full_mesh(n):
#     """全連接拓撲"""
#     G = nx.complete_graph(n, create_using=nx.DiGraph())
#     for u, v in G.edges():
#         G[u][v]['length'] = 1
#     return G

# def create_ring_topology(n):
#     """環形拓撲"""
#     G = nx.DiGraph()
#     G.add_nodes_from(range(n))
#     for i in range(n):
#         G.add_edge(i, (i+1) % n, length=1)
#         G.add_edge((i+1) % n, i, length=1)  # 雙向
#     return G

# def create_mesh_topology(n):
#     """網格拓撲 (2x4 網格)"""
#     G = nx.DiGraph()
#     G.add_nodes_from(range(n))
    
#     # 2x4 網格連接 - 更規律的連接
#     connections = [
#         (0, 1), (1, 2), (2, 3),          # 第一行
#         (4, 5), (5, 6), (6, 7),          # 第二行  
#         (0, 4), (1, 5), (2, 6), (3, 7),  # 垂直連接
#         (1, 4), (2, 5), (3, 6)           # 對角連接增加連通性
#     ]
    
#     for u, v in connections:
#         G.add_edge(u, v, length=1)
#         G.add_edge(v, u, length=1)  # 雙向
#     return G

# def create_random_topology(n, density=0.5):
#     """隨機拓撲 - 確保強連通"""
#     while True:
#         G = nx.erdos_renyi_graph(n, density, directed=True)
#         G = nx.DiGraph(G)
        
#         # 確保強連通
#         if nx.is_strongly_connected(G):
#             break
#         # 如果不連通，添加必要的邊
#         G = G.to_undirected()
#         G = nx.DiGraph(G)
        
#         # 手動添加一些邊確保連通
#         for i in range(n):
#             next_node = (i + 1) % n
#             G.add_edge(i, next_node)
#             G.add_edge(next_node, i)
    
#     for u, v in G.edges():
#         G[u][v]['length'] = 1  # 保持統一長度
#     return G

# # ====== 生成拓撲 ======
# topologies = {
#     'full_mesh': create_full_mesh(N_NODES),
#     'ring': create_ring_topology(N_NODES), 
#     'mesh': create_mesh_topology(N_NODES),
#     'random': create_random_topology(N_NODES)
# }

# # ====== 儲存拓撲資訊 (簡化版) ======
# topology_info = {}
# for topo_name, G in topologies.items():
#     topology_info[topo_name] = {
#         'graph': G,
#         'edge_count': G.number_of_edges(),
#         'density': nx.density(G)
#     }

# with open(TOPOLOGY_FILE, 'wb') as f:
#     pickle.dump(topology_info, f)

# print(f"✅ 拓撲資訊已儲存至 {TOPOLOGY_FILE}")

# # ====== 使用相同的TM生成邏輯 ======
# np.random.seed(42)  # 保持一致性
# canonical_tm = np.random.randint(20, 120, size=(N_NODES, N_NODES))
# np.fill_diagonal(canonical_tm, 0)  

# # 為每種拓撲生成相同的TM數據
# for topo_name, G in topologies.items():
#     topo_dir = f"{OUTPUT_DIR}/{topo_name}"
#     os.makedirs(topo_dir, exist_ok=True)
    
#     # 重設隨機種子確保每種拓撲使用相同的TM模式
#     np.random.seed(42)
    
#     for i in range(N_SAMPLES):
#         noise = np.random.normal(loc=0.0, scale=VARIATION, size=(N_NODES, N_NODES))
#         noisy_tm = canonical_tm * (1 + noise)
#         noisy_tm = np.clip(noisy_tm, 0, None).astype(int)
        
#         # 對於不連通的節點對，將流量設為0
#         for s in range(N_NODES):
#             for d in range(N_NODES):
#                 if s != d and not nx.has_path(G, s, d):
#                     noisy_tm[s][d] = 0
        
#         df = pd.DataFrame(noisy_tm)
#         df.to_csv(f"{topo_dir}/tm_{i:04d}.csv", index=False, header=False)
        
#         if i % 200 == 0:
#             print(f"已為 {topo_name} 拓撲產生 {i} / {N_SAMPLES} 筆 TMs")

# print("✅ 所有拓撲的 Traffic Matrices 產生完畢！")

# # ====== 印出拓撲統計資訊 ======
# print("\n📊 拓撲結構統計：")
# for topo_name, G in topologies.items():
#     print(f"\n{topo_name.upper()}:")
#     print(f"  節點數: {G.number_of_nodes()}")
#     print(f"  邊數: {G.number_of_edges()}")
#     print(f"  密度: {nx.density(G):.3f}")
#     print(f"  強連通: {nx.is_strongly_connected(G)}")

import numpy as np
import pandas as pd
import os

N_NODES = 8                   # ← 改為 12 個節點
N_SAMPLES = 1000                # ← 改為 1000 個樣本
VARIATION = 0.5              # 維持不變
OUTPUT_DIR = "./tms_8nodes_enhanced"
TOPOLOGY_FILE = "./topology_info.pkl"

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
