# run_experiment.py - 改进版
import os
import sys
import subprocess
import datetime
import utils

# ================= 改进的实验配置 =================
EXP_NAME = "Improved_Clover_Pipeline"
SEQ_LENGTH = 150
NUM_CLUSTERS = 8        # 🔥 减少簇数，提高成功率
READS_PER_CLUSTER = 25  # 🔥 增加每簇reads数
CLOVER_PROCESSES = 0
REFERENCE_TYPE = "diverse"  # 🔥 新增：参考序列类型 ("diverse" 或 "motif")
MIN_DISTANCE = 0.25     # 🔥 新增：簇间最小距离
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
        
    print(f"🚀 开始改进实验: {EXP_NAME}")
    print(f"📂 实验目录: {base_dir}")
    print(f"⚙️  实验参数:")
    print(f"   簇数: {NUM_CLUSTERS}")
    print(f"   每簇reads: {READS_PER_CLUSTER}")
    print(f"   序列长度: {SEQ_LENGTH}")
    print(f"   参考序列类型: {REFERENCE_TYPE}")
    print(f"   最小簇间距离: {MIN_DISTANCE}")
    
    # 2. 生成改进数据
    print("\n[Step 1] 生成改进的模拟数据...")
    try:
        raw_reads_path, gt_path = utils.generate_data(
            output_dir=dir_raw,
            num_clusters=NUM_CLUSTERS,
            reads_per_cluster=READS_PER_CLUSTER,
            seq_len=SEQ_LENGTH,
            reference_type=REFERENCE_TYPE,
            min_distance=MIN_DISTANCE
        )
        print(f"✅ 改进数据已生成: {raw_reads_path}")
        
        # 验证数据质量
        if utils.validate_generated_data(gt_path, raw_reads_path):
            print("✅ 数据质量验证通过")
        else:
            print("⚠️  数据质量验证有问题，但继续执行")
            
    except Exception as e:
        print(f"❌ 数据生成失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. 运行 Clover
    print("\n[Step 2] 运行 Clover 聚类...")
    
    clover_out_arg = os.path.join(dir_clover, "clover_result") 
    clover_out_real = clover_out_arg + ".txt"
    
    env = os.environ.copy()
    clover_repo_path = os.path.join(current_dir, "Clover")
    env["PYTHONPATH"] = clover_repo_path + os.pathsep + env.get("PYTHONPATH", "")
    
    cmd = [
        sys.executable, "-m", "clover.main",
        "-I", raw_reads_path,
        "-O", clover_out_arg,
        "-L", str(SEQ_LENGTH),
        "-P", str(CLOVER_PROCESSES),
        "--no-tag"
    ]
    
    print(f"🔧 Clover命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, env=env, 
                              capture_output=True, text=True)
        print("✅ Clover运行成功")
        if result.stdout:
            print(f"   输出: {result.stdout[:200]}...")
        if result.stderr:
            print(f"   警告: {result.stderr[:200]}...")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Clover 运行失败: {e}")
        print(f"   返回码: {e.returncode}")
        if e.stdout:
            print(f"   标准输出: {e.stdout}")
        if e.stderr:
            print(f"   错误输出: {e.stderr}")
        return

    # 检查输出文件
    if not os.path.exists(clover_out_real):
        print(f"❌ 严重错误: Clover输出文件未找到！")
        print(f"   预期路径: {clover_out_real}")
        print(f"   目录内容: {os.listdir(dir_clover)}")
        return
    else:
        # 检查文件大小
        file_size = os.path.getsize(clover_out_real)
        print(f"✅ Clover 输出文件存在: {clover_out_real} ({file_size} bytes)")
        
        # 预览文件内容
        try:
            with open(clover_out_real, 'r') as f:
                preview = f.read(200)
                print(f"   文件预览: {preview}...")
        except Exception as e:
            print(f"   无法预览文件: {e}")

    # 4. 转换格式
    print("\n[Step 3] 转换为 FedDNA 格式...")
    try:
        count, final_path = utils.clover_to_feddna(
            clover_out_real, raw_reads_path, dir_feddna
        )
        
        # 5. 生成实验总结
        summary_path = os.path.join(base_dir, "experiment_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"=== 改进实验总结 ===\n")
            f.write(f"实验名称: {EXP_NAME}\n")
            f.write(f"时间戳: {timestamp}\n")
            f.write(f"实验目录: {base_dir}\n\n")
            
            f.write(f"=== 实验参数 ===\n")
            f.write(f"簇数: {NUM_CLUSTERS}\n")
            f.write(f"每簇reads: {READS_PER_CLUSTER}\n")
            f.write(f"序列长度: {SEQ_LENGTH}\n")
            f.write(f"参考序列类型: {REFERENCE_TYPE}\n")
            f.write(f"最小簇间距离: {MIN_DISTANCE}\n\n")
            
            f.write(f"=== 结果统计 ===\n")
            f.write(f"Clover有效簇数: {count}\n")
            f.write(f"原始簇数: {NUM_CLUSTERS}\n")
            f.write(f"聚类成功率: {count/NUM_CLUSTERS*100:.1f}%\n")
            f.write(f"FedDNA输入文件: {final_path}\n")
        
        print("-" * 50)
        print(f"🎉 改进实验圆满结束！")
        print(f"📊 Clover有效簇数: {count}/{NUM_CLUSTERS} ({count/NUM_CLUSTERS*100:.1f}%)")
        print(f"👉 FedDNA 输入文件已就绪: {dir_feddna}")
        print(f"📋 实验总结: {summary_path}")
        print(f"📈 数据统计: {os.path.join(dir_raw, 'data_stats.txt')}")
        print("-" * 50)
        
        # 6. 为后续神经网络训练准备数据
        prepare_for_neural_network(base_dir, dir_feddna)
        
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()

def prepare_for_neural_network(base_dir, feddna_dir):
    """🔥 为神经网络训练准备数据"""
    try:
        # 创建神经网络数据目录
        nn_dir = os.path.join(base_dir, "04_NeuralNet_Ready")
        os.makedirs(nn_dir, exist_ok=True)
        
        # 复制FedDNA格式的文件
        import shutil
        read_file = os.path.join(feddna_dir, "read.txt")
        ref_file = os.path.join(feddna_dir, "reference.txt")
        
        if os.path.exists(read_file) and os.path.exists(ref_file):
            shutil.copy2(read_file, nn_dir)
            shutil.copy2(ref_file, nn_dir)
            
            # 创建配置文件
            config_path = os.path.join(nn_dir, "training_config.txt")
            with open(config_path, 'w') as f:
                f.write("=== 神经网络训练配置建议 ===\n")
                f.write(f"数据目录: {nn_dir}\n")
                f.write(f"序列长度: {SEQ_LENGTH}\n")
                f.write(f"预期簇数: 根据Clover结果调整\n\n")
                
                f.write("=== 训练参数建议 ===\n")
                f.write("batch_size: 1\n")
                f.write("learning_rate: 1e-4\n")
                f.write("max_epochs: 10\n")
                f.write("convergence_threshold: 0.08\n")
                f.write("min_epochs: 4\n\n")
                
                f.write("=== 模型参数建议 ===\n")
                f.write("hidden_dim: 128\n")
                f.write("num_layers: 3\n")
                f.write("num_heads: 8\n")
                f.write("dropout: 0.1\n")
                f.write("contrastive_dim: 64\n")
            
            print(f"🧠 神经网络数据已准备: {nn_dir}")
            print(f"📋 训练配置建议: {config_path}")
            
    except Exception as e:
        print(f"⚠️  神经网络数据准备失败: {e}")

if __name__ == "__main__":
    run()
