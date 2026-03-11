import re
import matplotlib.pyplot as plt
import os

def plot_loss(log_file="train.log", output_path="report/loss_curve.png"):
    if not os.path.exists(log_file):
        print(f"Error: {log_file} not found. Did the training finish?")
        return

    with open(log_file, "r") as f:
        content = f.read()
    
    # 提取命令行 tqdm 打印的 loss 值
    losses = [float(x) for x in re.findall(r'loss=([0-9\.]+)', content)]
    
    if not losses:
        print("No loss values found. Please ensure training completed successfully.")
        return
        
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="Raw Loss", color="#1f77b4", alpha=0.3)
    
    # 平滑曲线
    window = max(1, len(losses) // 20)
    smoothed = [sum(losses[max(0, i-window):i+1])/len(losses[max(0, i-window):i+1]) for i in range(len(losses))]
    plt.plot(smoothed, label="Smoothed Loss", color="red", linewidth=2)
    
    plt.title("BOFT Training Loss over Steps (MSE)")
    plt.xlabel("Logging Steps")
    plt.ylabel("Loss")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Loss curve saved as {output_path}")

if __name__ == "__main__":
    plot_loss()
