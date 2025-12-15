# run_experiment.py
import os
import sys
import subprocess
import datetime
import utils

# ================= 实验配置 =================
EXP_NAME = "High_Indel_Test"
SEQ_LENGTH = 150
NUM_CLUSTERS = 50
READS_PER_CLUSTER = 20
CLOVER_PROCESSES = 0
# ===========================================

def run():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. 准备文件夹
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(current_dir, "Experiments", f"{timestamp}_{EXP_NAME}")
    
    dir_raw = os.path.join(base_dir, "01_RawData")
    dir_clover = os.path.join(base_dir, "02_CloverOut")
    dir_feddna = os.path.join(base_dir, "03_FedDNA_In")
    
    for d in [dir_raw, dir_clover, dir_feddna]:
        os.makedirs(d, exist_ok=True)
        
    print(f"🚀 开始实验: {EXP_NAME}")
    print(f"📂 实验目录: {base_dir}")
    
    # 2. 生成数据
    print("\n[Step 1] 生成模拟数据...")
    raw_reads_path, gt_path = utils.generate_data(
        dir_raw, NUM_CLUSTERS, READS_PER_CLUSTER, SEQ_LENGTH
    )
    print(f"✅ 数据已生成: {raw_reads_path}")
    
    # 3. 运行 Clover
    print("\n[Step 2] 运行 Clover 聚类...")
    
    # 【修改点 1】: 传给 Clover 的路径（去掉 .txt，因为 Clover 会自动加）
    clover_out_arg = os.path.join(dir_clover, "clover_result") 
    # 【修改点 2】: 我们预期的实际文件路径（Clover 加完后缀后的样子）
    clover_out_real = clover_out_arg + ".txt"
    
    env = os.environ.copy()
    clover_repo_path = os.path.join(current_dir, "Clover")
    env["PYTHONPATH"] = clover_repo_path + os.pathsep + env.get("PYTHONPATH", "")
    
    cmd = [
        sys.executable, "-m", "clover.main",
        "-I", raw_reads_path,
        "-O", clover_out_arg,   # <--- 传不带后缀的
        "-L", str(SEQ_LENGTH),
        "-P", str(CLOVER_PROCESSES),
        "--no-tag"
    ]
    
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"❌ Clover 运行失败: {e}")
        return

    # 检查文件
    if not os.path.exists(clover_out_real):
        print(f"❌ 严重错误: 文件未找到！")
        print(f"   预期路径: {clover_out_real}")
        print(f"   目录内容: {os.listdir(dir_clover)}")
        return
    else:
        print(f"✅ Clover 运行完毕: {clover_out_real}")

    # 4. 转换格式
    print("\n[Step 3] 转换为 FedDNA 格式...")
    try:
        # 【修改点 3】: 这里读取的是实际存在的 real 路径
        count, final_path = utils.clover_to_feddna(
            clover_out_real, raw_reads_path, dir_feddna
        )
        print("-" * 40)
        print(f"🎉 实验圆满结束！")
        print(f"📊 有效簇数量: {count}")
        print(f"👉 FedDNA 输入文件已就绪: {dir_feddna}")
        print("-" * 40)
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run()