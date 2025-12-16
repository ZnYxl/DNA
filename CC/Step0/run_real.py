import os
import sys
import subprocess
import datetime
import utils  # 复用之前的工具包

# ================= 实验配置 =================
EXP_NAME = "Real_Data_ERR1816980"
# 【注意】这里的长度请根据上一步的统计结果修改！
SEQ_LENGTH = 150              
CLOVER_PROCESSES = 0          # 0=单进程 (最稳)
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
        
    print(f"🚀 开始实战实验: {EXP_NAME}")
    
    # 2. 准备数据 (不生成，直接搬运你转换好的 raw_reads.txt)
    # 假设你把转换好的 raw_reads.txt 放在了和脚本同级目录下
    source_file = "raw_reads.txt" 
    target_file = os.path.join(dir_raw, "raw_reads.txt")
    
    if os.path.exists(source_file):
        print(f"\n[Step 1] 加载真实数据: {source_file}")
        # 复制文件进去
        with open(source_file, 'r') as f_src, open(target_file, 'w') as f_dst:
            f_dst.write(f_src.read())
        print(f"✅ 数据已就位: {target_file}")
    else:
        print(f"❌ 错误: 找不到 {source_file}，请确保你运行了转换脚本！")
        return
    
    # 3. 运行 Clover
    print("\n[Step 2] 运行 Clover 聚类 (这可能需要几分钟到几小时)...")
    
    clover_out_file = os.path.join(dir_clover, "clover_result") # 不带后缀
    clover_out_real = clover_out_file + ".txt"
    
    env = os.environ.copy()
    clover_repo_path = os.path.join(current_dir, "Clover")
    env["PYTHONPATH"] = clover_repo_path + os.pathsep + env.get("PYTHONPATH", "")
    
    # 注意：真实数据量大，建议先用 head -n 1000 raw_reads.txt > test.txt 测试一下
    cmd = [
        sys.executable, "-m", "clover.main",
        "-I", target_file,
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
            clover_out_real, target_file, dir_feddna
        )
        print("-" * 40)
        print(f"🎉 实战处理完成！")
        print(f"📊 有效簇数量: {count}")
        print(f"👉 下一步: 可以在 Step1 中训练这个新数据集了！")
        print("-" * 40)
    except Exception as e:
        print(f"❌ 转换失败: {e}")

if __name__ == "__main__":
    run()