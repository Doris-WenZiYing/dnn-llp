import numpy as np
import pandas as pd
import pickle
import joblib
from tensorflow.keras.models import load_model

def load_tm(tm_path):
    """讀取 TM 並轉成 flat vector"""
    tm = pd.read_csv(tm_path, header=None).values
    tm_flat = tm.flatten().reshape(1, -1)
    return tm_flat, tm

def predict_rwa(tm_flat, model, scaler, label_to_solution):
    """預測 label 並對應回 (path_id, wavelength) 解"""
    tm_scaled = scaler.transform(tm_flat)
    prediction = model.predict(tm_scaled)
    label = np.argmax(prediction)
    if label not in label_to_solution:
        return label, None
    return label, label_to_solution[label]

def main():
    # === 檔案路徑 ===
    model_path = "final_model_8nodes.keras"
    label_map_path = "label_8mappings.pkl"
    scaler_path = "scaler.pkl"
    tm_input_path = "tm_input.csv"
    output_csv = "rwa_prediction.csv"

    # === 載入模型、label 對應表、scaler ===
    print("🔄 載入模型與 label 對應表...")
    model = load_model(model_path)
    with open(label_map_path, "rb") as f:
        label_dict = pickle.load(f)
    label_to_solution = label_dict["label_to_solution"]
    paths = label_dict["paths"]  # ⬅️ 必須是完整節點路徑
    scaler = joblib.load(scaler_path)

    # === 載入 TM 並預測 ===
    print("📥 載入 traffic matrix...")
    tm_flat, _ = load_tm(tm_input_path)

    print("🤖 預測中...")
    label, solution = predict_rwa(tm_flat, model, scaler, label_to_solution)

    print(f"\n✅ 預測 Label: {label}")
    if solution is None:
        print("❌ 沒有找到對應的解。")
        return

    # === 建立 RWA 表格 ===
    print("📦 預測的 (Path, λ) 安排如下：")
    records = []
    for (s, d), (p_id, lamb) in solution.items():
        try:
            path_nodes = paths[(s, d)][p_id]
        except (KeyError, IndexError):
            print(f"⚠️ 路徑資訊錯誤：{s} → {d} (p_id={p_id})")
            continue
        path_str = "→".join(map(str, path_nodes))
        print(f"  Demand {s} → {d} ➜ {path_str}, λ{lamb}")
        records.append({
            "source": s,
            "destination": d,
            "full_path": path_str,
            "wavelength": lamb
        })

    df = pd.DataFrame(records)[["source", "destination", "full_path", "wavelength"]]
    df.to_csv(output_csv, index=False)
    print(f"\n📄 RWA 安排已儲存至：{output_csv}")

if __name__ == "__main__":
    main()
