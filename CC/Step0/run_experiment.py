import os
import sys
import subprocess
import datetime
import utils  

# ================= 实验配置 =================
EXP_NAME = "Cluster_GT_Test"
SEQ_LENGTH = 150
NUM_CLUSTERS = 10000      
READS_PER_CLUSTER = 100 
CLOVER_PROCESSES = 0
REF_TYPE = "diverse" 
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
        
    print(f"🚀 开始实验: {EXP_NAME} (Mode: {REF_TYPE})")
    
    # 2. 生成数据 (接收 3 个返回值)
    print("\n[Step 1] 生成数据 (含 Cluster-Level GT)...")
    # 注意这里解包了 3 个变量
    raw_reads_path, read_gt_path, cluster_gt_path = utils.generate_data(
        output_dir=dir_raw, 
        num_clusters=NUM_CLUSTERS, 
        reads_per_cluster=READS_PER_CLUSTER, 
        seq_len=SEQ_LENGTH,
        reference_type=REF_TYPE
    )
    
    print(f"✅ 数据就绪。")
    print(f"   - Raw Reads: {os.path.basename(raw_reads_path)}")
    print(f"   - Cluster GT (Key!): {os.path.basename(cluster_gt_path)}")
    
    # 3. 运行 Clover
    print("\n[Step 2] 运行 Clover 聚类...")
    clover_out_file = os.path.join(dir_clover, "clover_result")
    clover_out_real = clover_out_file + ".txt"
    
    env = os.environ.copy()
    clover_repo_path = os.path.join(current_dir, "Clover")
    env["PYTHONPATH"] = clover_repo_path + os.pathsep + env.get("PYTHONPATH", "")
    
    cmd = [
        sys.executable, "-m", "clover.main",
        "-I", raw_reads_path,
        "-O", clover_out_file,
        "-L", str(SEQ_LENGTH),
        "-P", str(CLOVER_PROCESSES),
        "--no-tag"
    ]
    
    try:
        subprocess.run(cmd, check=True, env=env)
        print(f"✅ Clover 运行完毕")
    except subprocess.CalledProcessError as e:
        print(f"❌ Clover 运行失败: {e}")
        return

    # 4. 转换格式
    print("\n[Step 3] 转换为 FedDNA 格式...")
    try:
        count, final_path = utils.clover_to_feddna(
            clover_out_real, raw_reads_path, dir_feddna
        )
        print("-" * 40)
        print(f"🎉 实验完成！")
        print(f"📊 有效簇数量: {count} (Clover 聚类结果)")
        print(f"🎯 真实簇数量: {NUM_CLUSTERS} (Ground Truth)")
        print(f"👉 数据目录: {base_dir}")
        print("-" * 40)
    except Exception as e:
        print(f"❌ 转换失败: {e}")

if __name__ == "__main__":
    run()