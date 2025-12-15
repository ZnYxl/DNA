import os
import collections
import ast

# =================配置区域=================
# 1. 请替换为你刚刚跑出来的 Clover 输出文件路径
# (例如: "output/output_20251215_152757.txt")
CLOVER_OUTPUT_PATH = "/Users/miemie/Library/Mobile Documents/com~apple~CloudDocs/DNA/miemie_DNA/code/Step0/Clover/output/output_20251215_154926.txt"  # <--- 请修改这里！！！

# 2. 请替换为你的原始 Reads 文件路径
RAW_READS_PATH = "/Users/miemie/Library/Mobile Documents/com~apple~CloudDocs/DNA/miemie_DNA/code/Step0/raw_reads.txt"

# 3. 输出目录 (FedDNA 的输入文件夹)
OUTPUT_DIR = "feddna_input"


# =========================================

def load_raw_reads(file_path):
    """读取原始序列文件 ID -> Sequence"""
    reads_dict = {}
    print(f"正在读取原始数据: {file_path}")
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()  # 自动处理空格或Tab
                if len(parts) >= 2:
                    rid = parts[0].strip()
                    seq = parts[1].strip()
                    reads_dict[rid] = seq
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {file_path}")
        exit(1)
    return reads_dict


def parse_clover_output(file_path):
    """
    读取 Clover 输出文件 (兼容列表格式 [('id', 'center'), ...])
    """
    clusters = collections.defaultdict(list)
    print(f"正在解析 Clover 结果: {file_path}")

    try:
        with open(file_path, 'r') as f:
            content = f.read().strip()

            # 情况1: 如果文件内容是 Python 列表格式 [ ... ]
            if content.startswith("[") and content.endswith("]"):
                try:
                    # 使用 ast.literal_eval 安全地把字符串变回列表
                    pairs = ast.literal_eval(content)
                    print(f"  - 识别为列表格式，共包含 {len(pairs)} 条关系")

                    for item in pairs:
                        # 确保取出的 ID 是字符串格式
                        read_id = str(item[0])
                        center_id = str(item[1])

                        # 过滤掉 -1
                        if center_id == '-1' or center_id == -1:
                            continue

                        clusters[center_id].append(read_id)

                except Exception as e:
                    print(f"❌ 列表解析失败，可能是文件截断或格式错误: {e}")
                    exit(1)

            # 情况2: 传统的逐行 CSV 格式
            else:
                print("  - 识别为逐行文本格式")
                # 指针回到文件开头重新读
                f.seek(0)
                for line in f:
                    line = line.strip()
                    if not line: continue

                    # 简单的分割处理
                    parts = line.replace(',', ' ').split()

                    if len(parts) >= 2:
                        read_id = parts[0].strip()
                        center_id = parts[1].strip()

                        if center_id == '-1' or center_id == -1:
                            continue
                        clusters[center_id].append(read_id)

    except FileNotFoundError:
        print(f"❌ 错误: 找不到 Clover 输出文件 {file_path}")
        exit(1)

    print(f"  - 共形成 {len(clusters)} 个有效簇")
    return clusters


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. 加载数据
    raw_reads = load_raw_reads(RAW_READS_PATH)
    clusters = parse_clover_output(CLOVER_OUTPUT_PATH)

    # 2. 准备写入
    out_read_path = os.path.join(OUTPUT_DIR, "read.txt")
    out_ref_path = os.path.join(OUTPUT_DIR, "reference.txt")

    print("正在写入 FedDNA 格式...")

    valid_cluster_count = 0

    with open(out_read_path, 'w') as f_read, open(out_ref_path, 'w') as f_ref:
        for center_id, member_ids in clusters.items():

            # 找到 Center 的序列作为 Reference (伪真值)
            if center_id not in raw_reads:
                print(f"⚠️ 警告: 簇中心 ID {center_id} 在原始数据中找不到序列，跳过该簇。")
                continue

            center_seq = raw_reads[center_id]

            # 写入 reference.txt
            f_ref.write(center_seq + '\n')

            # 写入 read.txt (该簇的所有成员)
            for member_id in member_ids:
                if member_id in raw_reads:
                    seq = raw_reads[member_id]
                    f_read.write(seq + '\n')

            # 写入 FedDNA 专用的簇分隔符
            f_read.write("===============================\n")

            valid_cluster_count += 1

    print("-" * 30)
    print(f"🎉 转换成功！")
    print(f"✅ 生成簇数量: {valid_cluster_count}")
    print(f"📂 结果保存在: {OUTPUT_DIR}/")
    print(f"   - {out_read_path}")
    print(f"   - {out_ref_path}")
    print("-" * 30)
    print("下一步：将这两个文件放入 FedDNA 项目的 dataset 目录中即可开始训练！")


if __name__ == "__main__":
    main()