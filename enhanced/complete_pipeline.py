#!/usr/bin/env python3
"""
DNN+ILP 光網路RWA完整執行流程
作者: 基於原始代碼優化
目標: 讓DNN在所有拓撲下都能達到80%以上準確率
支援: python3 和 python 命令
"""

import os
import sys
import subprocess
import time
from pathlib import Path

class RWAPipeline:
    """RWA完整流程管理器"""
    
    def __init__(self, python_cmd='python3'):
        self.python_cmd = python_cmd  # 支援 python3 或 python
        self.steps = [
            ("1. 生成多拓撲數據", "enhanced_datamake.py"),
            ("2. ILP求解訓練數據", "enhanced_ilp_solver.py"),
            ("3. 訓練拓撲感知DNN", "enhanced_dnn_training.py"),
            ("4. 性能評估與測試", "enhanced_predictor.py")
        ]
        
    def check_python_command(self):
        """檢查Python命令是否可用"""
        try:
            result = subprocess.run([self.python_cmd, '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {self.python_cmd} 可用: {result.stdout.strip()}")
                return True
            else:
                print(f"❌ {self.python_cmd} 不可用")
                return False
        except FileNotFoundError:
            print(f"❌ {self.python_cmd} 命令未找到")
            return False
        
    def check_dependencies(self):
        """檢查依賴套件"""
        required_packages = {
            'tensorflow': 'tensorflow',
            'pandas': 'pandas', 
            'numpy': 'numpy',
            'scikit-learn': 'sklearn',  # 套件名和導入名不同
            'matplotlib': 'matplotlib',
            'networkx': 'networkx',
            'pulp': 'pulp',
            'seaborn': 'seaborn',
            'joblib': 'joblib'
        }
        
        print("🔍 檢查依賴套件...")
        missing_packages = []
        
        for package_name, import_name in required_packages.items():
            try:
                __import__(import_name)
                print(f"  ✅ {package_name}")
            except ImportError:
                print(f"  ❌ {package_name} (未安裝)")
                missing_packages.append(package_name)
        
        if missing_packages:
            print(f"\n請安裝缺失的套件:")
            print(f"{self.python_cmd} -m pip install {' '.join(missing_packages)}")
            return False
        
        print("✅ 所有依賴套件已安裝")
        return True
    
    def run_step(self, step_name, script_name):
        """執行單一步驟"""
        print(f"\n{'='*60}")
        print(f"🚀 執行: {step_name}")
        print(f"{'='*60}")
        
        if not os.path.exists(script_name):
            print(f"❌ 腳本不存在: {script_name}")
            print("請確認所有代碼文件都已正確建立")
            return False
        
        start_time = time.time()
        
        try:
            # 使用指定的Python命令執行腳本
            print(f"執行命令: {self.python_cmd} {script_name}")
            result = subprocess.run([self.python_cmd, script_name], 
                                  capture_output=True, text=True)
            
            elapsed_time = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ {step_name} 完成 (耗時: {elapsed_time:.1f}秒)")
                if result.stdout:
                    print("輸出:")
                    print(result.stdout)
                return True
            else:
                print(f"❌ {step_name} 失敗")
                if result.stderr:
                    print("錯誤訊息:")
                    print(result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ 執行錯誤: {e}")
            print(f"提示: 請確認 {self.python_cmd} 命令可用")
            return False
    
    def create_script_files(self):
        """建立所有必要的腳本文件"""
        print("📝 請確認以下腳本文件已建立:")
        
        scripts = [
            "enhanced_datamake.py",
            "enhanced_ilp_solver.py", 
            "enhanced_dnn_training.py",
            "enhanced_predictor.py"
        ]
        
        all_exist = True
        for script in scripts:
            if os.path.exists(script):
                print(f"  ✅ {script}")
            else:
                print(f"  ❌ {script} (缺失)")
                all_exist = False
        
        return all_exist
    
    def check_results(self):
        """檢查執行結果"""
        print("\n🔍 檢查執行結果...")
        
        expected_files = [
            "tms_8nodes_enhanced/",
            "topology_info.pkl",
            "enhanced_dataset_all_topologies.csv",
            "enhanced_label_mappings.pkl",
            "training_results/",
            "prediction_results/"
        ]
        
        results = {}
        for item in expected_files:
            if item.endswith("/"):
                # 目錄
                exists = os.path.isdir(item)
                if exists:
                    try:
                        file_count = len([f for f in os.listdir(item) 
                                        if os.path.isfile(os.path.join(item, f))])
                        print(f"  ✅ {item} (包含 {file_count} 個文件)")
                    except:
                        print(f"  ✅ {item} (目錄存在)")
                else:
                    print(f"  ❌ {item} (目錄不存在)")
            else:
                # 文件
                exists = os.path.exists(item)
                if exists:
                    try:
                        size_mb = os.path.getsize(item) / 1024 / 1024
                        print(f"  ✅ {item} ({size_mb:.1f} MB)")
                    except:
                        print(f"  ✅ {item} (文件存在)")
                else:
                    print(f"  ❌ {item} (文件不存在)")
            
            results[item] = exists
        
        return results
    
    def show_final_report(self):
        """顯示最終報告"""
        print("\n" + "="*80)
        print("🎯 DNN+ILP 光網路RWA 專案執行完成報告")
        print("="*80)
        
        # 讀取訓練結果
        results_file = "training_results/results_summary.pkl"
        if os.path.exists(results_file):
            try:
                import pickle
                with open(results_file, 'rb') as f:
                    results = pickle.load(f)
                
                print(f"整體測試準確率: {results['overall_test_accuracy']:.4f}")
                print("\n各拓撲準確率:")
                
                all_above_80 = True
                for topo, result in results['topology_results'].items():
                    accuracy = result['accuracy']
                    status = "✅" if accuracy >= 0.8 else "❌"
                    print(f"  {status} {topo.upper()}: {accuracy:.4f}")
                    if accuracy < 0.8:
                        all_above_80 = False
                
                if all_above_80:
                    print("\n🎉 恭喜！所有拓撲都達到80%以上準確率目標！")
                else:
                    print("\n⚠️ 部分拓撲未達到80%目標，建議進一步優化")
                
            except Exception as e:
                print(f"無法讀取訓練結果: {e}")
        
        # 顯示使用指南
        print("\n📖 使用指南:")
        print("1. 查看訓練曲線: training_results/training_results.png")
        print("2. 查看預測結果: prediction_results/evaluation_results.png")
        print(f"3. 使用模型預測: {self.python_cmd} enhanced_predictor.py")
        print("4. 查看詳細日誌: 檢查各步驟的輸出")
        
        print("\n📁 重要文件:")
        print("- 訓練好的模型: training_results/topology_aware_dnn_model.keras")
        print("- 特徵標準化器: enhanced_scaler.pkl")
        print("- 標籤映射: enhanced_label_mappings.pkl")
        print("- 拓撲資訊: topology_info.pkl")
    
    def run_complete_pipeline(self):
        """執行完整流程"""
        print("🌟 DNN+ILP 光網路RWA 拓撲感知解決方案")
        print("目標: 讓DNN在所有拓撲下都能達到80%以上準確率")
        print(f"使用Python命令: {self.python_cmd}\n")
        
        # 檢查Python命令
        if not self.check_python_command():
            print("❌ Python命令檢查失敗")
            return False
        
        # 檢查依賴
        if not self.check_dependencies():
            return False
        
        # 檢查腳本文件
        if not self.create_script_files():
            print("\n❌ 請先建立所有必要的腳本文件")
            return False
        
        # 執行各步驟
        total_start_time = time.time()
        
        for step_name, script_name in self.steps:
            success = self.run_step(step_name, script_name)
            if not success:
                print(f"\n❌ 流程在 '{step_name}' 步驟失敗")
                return False
        
        total_time = time.time() - total_start_time
        
        # 檢查結果
        self.check_results()
        
        # 顯示報告
        self.show_final_report()
        
        print(f"\n⏱️ 總執行時間: {total_time/60:.1f} 分鐘")
        print("🎯 流程執行完成！")
        
        return True

def main():
    """主函數"""
    print("🌟 DNN+ILP 光網路RWA 拓撲感知解決方案")
    print("="*50)
    
    # 選擇Python命令
    print("選擇Python命令:")
    print("1. python3 (推薦)")
    print("2. python")
    print("3. 自動檢測")
    
    while True:
        choice = input("請選擇Python命令 (1-3): ").strip()
        
        if choice == "1":
            python_cmd = "python3"
            break
        elif choice == "2":
            python_cmd = "python"
            break
        elif choice == "3":
            # 自動檢測可用的Python命令
            for cmd in ["python3", "python"]:
                try:
                    result = subprocess.run([cmd, '--version'], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        python_cmd = cmd
                        print(f"✅ 自動檢測到: {cmd}")
                        break
                except:
                    continue
            else:
                print("❌ 無法檢測到可用的Python命令")
                return
            break
        else:
            print("請輸入有效選項 (1-3)")
    
    pipeline = RWAPipeline(python_cmd=python_cmd)
    
    print(f"\n使用命令: {python_cmd}")
    print("選擇執行模式:")
    print("1. 完整流程 (推薦)")
    print("2. 檢查環境")
    print("3. 僅檢查結果")
    print("4. 退出")
    
    while True:
        choice = input("\n請選擇 (1-4): ").strip()
        
        if choice == "1":
            pipeline.run_complete_pipeline()
            break
        elif choice == "2":
            pipeline.check_python_command()
            pipeline.check_dependencies()
            pipeline.create_script_files()
            break
        elif choice == "3":
            pipeline.check_results()
            pipeline.show_final_report()
            break
        elif choice == "4":
            print("再見！")
            break
        else:
            print("請輸入有效選項 (1-4)")

if __name__ == "__main__":
    main()