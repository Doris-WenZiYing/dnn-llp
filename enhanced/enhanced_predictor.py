import numpy as np
import pandas as pd
import pickle
import joblib
import networkx as nx
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ====== 配置 ======
MODEL_PATH = "./training_results/topology_aware_dnn_model.keras"   # 當前目錄下
SCALER_PATH = "./enhanced_scaler.pkl"                              # 當前目錄下
LABEL_MAP_PATH = "./enhanced_label_mappings.pkl"                   # 當前目錄下
TOPOLOGY_INFO_PATH = "./topology_info.pkl"                         # 當前目錄下
RESULTS_DIR = "./prediction_results"                               # 當前目錄下

os.makedirs(RESULTS_DIR, exist_ok=True)

class TopologyAwarePredictor:
    """拓撲感知RWA預測器"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_info = None
        self.topology_info = None
        self.feature_dims = None
        
    def load_model_components(self):
        """載入所有模型組件"""
        print("📥 載入模型組件...")
        
        # 載入訓練好的模型
        self.model = load_model(MODEL_PATH)
        print(f"✅ 模型載入成功")
        
        # 載入特徵標準化器
        self.scaler = joblib.load(SCALER_PATH)
        print(f"✅ 標準化器載入成功")
        
        # 載入標籤映射
        with open(LABEL_MAP_PATH, 'rb') as f:
            self.label_info = pickle.load(f)
        self.feature_dims = self.label_info['feature_dimensions']
        print(f"✅ 標籤映射載入成功，包含 {len(self.label_info['label_to_solution'])} 個解決方案")
        
        # 載入拓撲資訊
        with open(TOPOLOGY_INFO_PATH, 'rb') as f:
            self.topology_info = pickle.load(f)
        print(f"✅ 拓撲資訊載入成功，包含 {len(self.topology_info)} 種拓撲")
    
    def extract_topology_features(self, G):
        """提取單個拓撲的特徵（與訓練時保持一致）"""
        n = G.number_of_nodes()
        features = {}
        
        # 鄰接矩陣
        adj_matrix = nx.adjacency_matrix(G).todense()
        features['adjacency_matrix'] = np.array(adj_matrix)
        
        # 最短路徑矩陣
        try:
            shortest_paths = dict(nx.all_pairs_shortest_path_length(G))
            path_matrix = np.full((n, n), float('inf'))
            for i in range(n):
                for j in range(n):
                    if j in shortest_paths[i]:
                        path_matrix[i][j] = shortest_paths[i][j]
            max_path = np.max(path_matrix[path_matrix != float('inf')])
            path_matrix[path_matrix == float('inf')] = max_path + 1
        except:
            path_matrix = np.full((n, n), n)
        
        features['shortest_path_matrix'] = path_matrix
        
        # 節點度數
        in_degrees = [G.in_degree(i) for i in range(n)]
        out_degrees = [G.out_degree(i) for i in range(n)]
        features['node_degrees'] = np.array([in_degrees, out_degrees]).T
        
        # 聚類係數
        clustering = nx.clustering(G.to_undirected())
        features['clustering_coeffs'] = np.array([clustering[i] for i in range(n)])
        
        # 連通性指標
        features['edge_density'] = nx.density(G)
        features['average_path_length'] = np.mean(path_matrix[path_matrix < n])
        
        return features
    
    def create_enhanced_features(self, tm, topology_name):
        """建立增強特徵向量"""
        flat_tm = tm.flatten()
        
        if topology_name not in self.topology_info:
            raise ValueError(f"未知拓撲類型: {topology_name}")
        
        topo_features = self.topology_info[topology_name]['features']
        
        # 組合所有特徵
        enhanced_features = np.concatenate([
            flat_tm,
            topo_features['adjacency_matrix'].flatten(),
            topo_features['shortest_path_matrix'].flatten(),
            topo_features['node_degrees'].flatten(),
            topo_features['clustering_coeffs'],
            [topo_features['edge_density']],
            [topo_features['average_path_length']]
        ])
        
        return enhanced_features.reshape(1, -1)
    
    def predict_rwa(self, tm, topology_name):
        """預測給定TM和拓撲的RWA解決方案"""
        # 建立增強特徵
        enhanced_features = self.create_enhanced_features(tm, topology_name)
        
        # 標準化
        features_scaled = self.scaler.transform(enhanced_features)
        
        # 預測
        prediction = self.model.predict(features_scaled, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # 查找對應的解決方案
        solution = None
        if predicted_class in self.label_info['label_to_solution']:
            solution_data = self.label_info['label_to_solution'][predicted_class]
            solution = solution_data['solution']
            predicted_topology = solution_data['topology']
        else:
            predicted_topology = "Unknown"
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'solution': solution,
            'predicted_topology': predicted_topology,
            'input_topology': topology_name
        }
    
    def test_all_topologies(self, tm_index=0):
        """測試同一個TM在所有拓撲下的表現"""
        print(f"\n🧪 測試TM {tm_index} 在所有拓撲下的預測結果...")
        
        results = {}
        tm_file = f"./tms_8nodes_enhanced/full_mesh/tm_{tm_index:04d}.csv"
        
        if not os.path.exists(tm_file):
            print(f"❌ TM文件不存在: {tm_file}")
            return None
        
        # 載入基準TM
        tm = pd.read_csv(tm_file, header=None).values
        
        for topo_name in self.topology_info.keys():
            try:
                result = self.predict_rwa(tm, topo_name)
                results[topo_name] = result
                
                status = "✅" if result['confidence'] > 0.8 else "⚠️"
                print(f"  {status} {topo_name.upper()}: "
                      f"Label={result['predicted_class']}, "
                      f"Confidence={result['confidence']:.3f}")
                
            except Exception as e:
                print(f"  ❌ {topo_name.upper()}: 預測失敗 - {str(e)}")
                results[topo_name] = None
        
        return results
    
    def detailed_solution_analysis(self, solution, topology_name):
        """詳細分析預測的解決方案"""
        if solution is None:
            return "無有效解決方案"
        
        G = self.topology_info[topology_name]['graph']
        analysis = []
        
        analysis.append(f"拓撲: {topology_name.upper()}")
        analysis.append(f"路由方案數量: {len(solution)}")
        analysis.append("\n詳細路由安排:")
        
        for (s, d), (path_id, wavelength) in solution.items():
            try:
                # 這裡需要根據實際路徑資訊來顯示完整路徑
                analysis.append(f"  需求 {s}→{d}: 路徑ID={path_id}, 波長={wavelength}")
            except:
                analysis.append(f"  需求 {s}→{d}: 路徑ID={path_id}, 波長={wavelength}")
        
        return "\n".join(analysis)
    
    def batch_evaluation(self, num_samples=50):
        """批量評估模型在所有拓撲上的性能"""
        print(f"\n📊 批量評估 {num_samples} 個樣本...")
        
        topology_accuracies = {topo: [] for topo in self.topology_info.keys()}
        overall_results = []
        
        for i in range(min(num_samples, 100)):  # 限制最大測試數量
            try:
                tm_results = self.test_all_topologies(i)
                if tm_results:
                    for topo, result in tm_results.items():
                        if result and result['confidence'] > 0.5:  # 只考慮有信心的預測
                            topology_accuracies[topo].append(result['confidence'])
                    overall_results.append(tm_results)
            except:
                continue
        
        # 計算統計結果
        stats = {}
        for topo, confidences in topology_accuracies.items():
            if confidences:
                stats[topo] = {
                    'mean_confidence': np.mean(confidences),
                    'std_confidence': np.std(confidences),
                    'samples': len(confidences),
                    'accuracy_80plus': np.mean(np.array(confidences) >= 0.8)
                }
            else:
                stats[topo] = {
                    'mean_confidence': 0,
                    'std_confidence': 0,
                    'samples': 0,
                    'accuracy_80plus': 0
                }
        
        return stats, overall_results
    
    def visualize_results(self, stats):
        """視覺化批量評估結果"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        topologies = list(stats.keys())
        
        # 平均信心度
        mean_confs = [stats[topo]['mean_confidence'] for topo in topologies]
        bars1 = ax1.bar(topologies, mean_confs)
        ax1.set_title('Average Prediction Confidence by Topology')
        ax1.set_ylabel('Confidence')
        ax1.set_ylim(0, 1)
        ax1.axhline(y=0.8, color='red', linestyle='--', label='80% Threshold')
        
        for bar, conf in zip(bars1, mean_confs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{conf:.3f}', ha='center', va='bottom')
        ax1.legend()
        
        # 80%以上準確率比例
        acc_80plus = [stats[topo]['accuracy_80plus'] for topo in topologies]
        bars2 = ax2.bar(topologies, acc_80plus)
        ax2.set_title('Percentage of Predictions with 80%+ Confidence')
        ax2.set_ylabel('Percentage')
        ax2.set_ylim(0, 1)
        
        for bar, acc in zip(bars2, acc_80plus):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.2%}', ha='center', va='bottom')
        
        # 樣本數量
        samples = [stats[topo]['samples'] for topo in topologies]
        ax3.bar(topologies, samples)
        ax3.set_title('Number of Valid Predictions')
        ax3.set_ylabel('Count')
        
        # 信心度分佈箱型圖（需要原始數據）
        ax4.text(0.5, 0.5, 'Confidence Distribution\n(Requires raw data)', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Confidence Distribution by Topology')
        
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig

def main():
    """主要執行函數"""
    predictor = TopologyAwarePredictor()
    
    try:
        # 載入模型組件
        predictor.load_model_components()
        
        # 單一TM測試
        print("\n" + "="*60)
        print("🎯 單一TM多拓撲測試")
        print("="*60)
        
        single_results = predictor.test_all_topologies(tm_index=0)
        
        if single_results:
            print("\n📋 詳細解決方案分析:")
            for topo, result in single_results.items():
                if result and result['solution']:
                    print(f"\n{topo.upper()}:")
                    analysis = predictor.detailed_solution_analysis(
                        result['solution'], topo
                    )
                    print(analysis)
        
        # 批量評估
        print("\n" + "="*60)
        print("📊 批量性能評估")
        print("="*60)
        
        stats, batch_results = predictor.batch_evaluation(num_samples=20)
        
        print("\n各拓撲性能統計:")
        for topo, stat in stats.items():
            status = "✅" if stat['accuracy_80plus'] >= 0.8 else "❌"
            print(f"{status} {topo.upper()}:")
            print(f"    平均信心度: {stat['mean_confidence']:.3f}")
            print(f"    80%+信心度比例: {stat['accuracy_80plus']:.2%}")
            print(f"    有效樣本數: {stat['samples']}")
        
        # 視覺化結果
        predictor.visualize_results(stats)
        
        # 儲存結果
        with open(f'{RESULTS_DIR}/evaluation_stats.pkl', 'wb') as f:
            pickle.dump({'stats': stats, 'batch_results': batch_results}, f)
        
        # 最終評估
        print("\n" + "="*60)
        print("🏆 最終評估結果")
        print("="*60)
        
        all_above_80 = all(stat['accuracy_80plus'] >= 0.8 for stat in stats.values())
        avg_performance = np.mean([stat['accuracy_80plus'] for stat in stats.values()])
        
        print(f"整體平均性能: {avg_performance:.2%}")
        
        if all_above_80:
            print("🎉 恭喜！所有拓撲都達到80%以上高信心度預測比例！")
        else:
            print("⚠️ 部分拓撲未達到80%目標")
            
        print(f"\n📁 詳細結果已儲存至 {RESULTS_DIR}/")
        
    except FileNotFoundError as e:
        print(f"❌ 檔案未找到: {e}")
        print("請確認已執行完整的訓練流程")
    except Exception as e:
        print(f"❌ 執行錯誤: {e}")

if __name__ == "__main__":
    main()