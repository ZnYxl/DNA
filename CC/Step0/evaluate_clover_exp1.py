import os
import sys
import array
from collections import defaultdict, Counter

# ================= 配置区域 =================
# 1. 原始带标签的数据文件 (Ground Truth)
# 注意：这里是 exp_1 的路径
GT_FILE = "/mnt/st_data/liangxinyi/code/CC/Step0/给师妹的clover数据集/exp_1/exp1_tags_reads.txt"

# 2. Clover 聚类输出文件 (Prediction)
# 对应 run_real_data_exp1_Fixed.py 生成的结果
CLOVER_OUT_FILE = "./Experiments/exp_1_Real/02_CloverOut/clover_result_merged.txt"

# ===========================================

class CompactGroundTruth:
    def __init__(self):
        self.tag_to_id = {}    # 唯一 Tag (str) -> int ID
        self.id_to_tag = []    # int ID -> 唯一 Tag (str)
        # 使用 'I' (unsigned int) 数组存储，极省内存
        self.read_labels = array.array('I') 
        
    def load(self, file_path):
        print(f"📖 [1/3] 正在加载 Ground Truth (exp_1)...")
        if not os.path.exists(file_path):
            print(f"❌ 错误: 找不到文件 {file_path}")
            sys.exit(1)

        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                parts = line.strip().split()
                if not parts:
                    self.read_labels.append(0)
                    continue
                
                # exp_1 格式: Tag Sequence (例如 "6512 TCCT...")
                tag = parts[0]
                
                if tag not in self.tag_to_id:
                    new_id = len(self.id_to_tag) + 1 
                    self.tag_to_id[tag] = new_id
                    self.id_to_tag.append(tag)
                
                self.read_labels.append(self.tag_to_id[tag])
                
                if (i + 1) % 1000000 == 0:
                    print(f"   已索引 {i + 1} 行...", end='\r')
                        
        print(f"\n   ✅ GT 加载完成。总 Reads: {len(self.read_labels)}")
        print(f"   - 唯一 Tags 数: {len(self.id_to_tag)}")

    def get_tag_id(self, read_idx):
        if read_idx < 0 or read_idx >= len(self.read_labels):
            return 0
        return self.read_labels[read_idx]


def stream_clover_tokens(file_path):
    """流式解析 Clover 结果"""
    with open(file_path, 'r') as f:
        buffer = ""
        while True:
            chunk = f.read(1024*1024)
            if not chunk: break
            
            cleaned = chunk.replace('[', ' ').replace(']', ' ')\
                           .replace('(', ' ').replace(')', ' ')\
                           .replace(',', ' ').replace("'", " ").replace('"', " ")
            buffer += cleaned
            tokens = buffer.split()
            
            if chunk[-1].isspace() or chunk[-1] in "[](),'\"":
                for t in tokens: yield t
                buffer = ""
            else:
                if len(tokens) > 0:
                    for t in tokens[:-1]: yield t
                    buffer = tokens[-1]
                else: pass
        if buffer.strip(): yield buffer.strip()

def evaluate(gt_data, clover_path):
    print(f"\n🚀 [2/3] 流式处理聚类结果 (Token Stream)...")
    
    cluster_stats = defaultdict(Counter)
    total_reads_clustered = 0
    
    token_stream = stream_clover_tokens(clover_path)
    
    try:
        while True:
            try:
                idx_token = next(token_stream)
                cid_token = next(token_stream)
            except StopIteration:
                break
            
            # Clover 输出的索引通常是 1-based，但也可能是切片时的全局行号
            # 我们脚本里写的是 global_line_idx，所以是 1-based
            line_idx = int(idx_token)
            cluster_id = cid_token 
            
            if str(cluster_id) == '-1':
                continue
            
            # 转换为 0-based 索引去查 GT
            read_idx = line_idx - 1
            true_tag_id = gt_data.get_tag_id(read_idx)
            
            if true_tag_id != 0:
                cluster_stats[str(cluster_id)][true_tag_id] += 1
                total_reads_clustered += 1
                
            if total_reads_clustered % 1000000 == 0 and total_reads_clustered > 0:
                print(f"   已统计 {total_reads_clustered} 个聚类成员...", end='\r')

    except ValueError as e:
        print(f"\n⚠️ 解析警告: 数据格式异常: {e}")

    # === [3/3] 生成报告 ===
    print(f"\n\n📊 正在汇总统计信息...")
    
    correct_reads_count = 0
    recovered_tag_ids = set()
    cluster_purities = []
    
    for cid, counts in cluster_stats.items():
        if not counts: continue
        
        # 找出该簇的主导 Tag
        dominant_tag_id, dominant_count = counts.most_common(1)[0]
        total_in_cluster = sum(counts.values())
        
        purity = dominant_count / total_in_cluster
        cluster_purities.append(purity)
        
        correct_reads_count += dominant_count
        recovered_tag_ids.add(dominant_tag_id)

    total_unique_tags = len(gt_data.id_to_tag)
    avg_purity = sum(cluster_purities) / len(cluster_purities) if cluster_purities else 0
    micro_accuracy = correct_reads_count / total_reads_clustered if total_reads_clustered else 0
    recovery_rate = len(recovered_tag_ids) / total_unique_tags if total_unique_tags else 0
    
    print("\n" + "="*40)
    print("       📊 CLOVER 评估报告 - exp_1 数据集")
    print("="*40)
    print(f"1. 原始 Tag 恢复率 (Recovery Rate):")
    print(f"   {len(recovered_tag_ids)} / {total_unique_tags}  ({recovery_rate*100:.2f}%)")
    print(f"   (注：如果此数值接近 100%，说明去引物策略大获全胜！)")
    
    print(f"\n2. 平均簇纯度 (Average Purity):")
    print(f"   {avg_purity*100:.2f}%")
    
    print(f"\n3. 整体准确率 (Micro Accuracy):")
    print(f"   {micro_accuracy*100:.2f}%")
    
    print(f"\n4. 统计摘要:")
    print(f"   有效聚类 Reads 数: {total_reads_clustered}")
    print(f"   生成的簇数量: {len(cluster_stats)}")
    print("="*40)

if __name__ == "__main__":
    if not os.path.exists(CLOVER_OUT_FILE):
        print(f"❌ 找不到 Clover 输出文件: {CLOVER_OUT_FILE}")
        print(f"   请先运行 run_real_data_exp1_Fixed.py 生成结果。")
        sys.exit(1)

    gt = CompactGroundTruth()
    gt.load(GT_FILE)
    evaluate(gt, CLOVER_OUT_FILE)