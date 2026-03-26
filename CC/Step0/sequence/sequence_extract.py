import sys
sys.path.append('/mnt/st_data/linhaiyan/python_pkgs')
import pysam

def extract_columns(bam_file, output_file):
    # 打开 BAM 文件
    bam = pysam.AlignmentFile(bam_file, "rb",ignore_truncation=True)

    with open(output_file, "w") as f:
        # 遍历 BAM 文件中的每个读取序列
        for read in bam:
            # 获取读取序列名称 (QNAME)
            query_sequence = read.query_sequence

            # 获取参考序列名称 (RNAME)
            ref_name = bam.get_reference_name(read.reference_id)

            # 写入到输出文件
            f.write(f"{ref_name}\t{query_sequence}\n")

    bam.close()

# 示例：使用你的 BAM 文件路径和输出文件路径

bam_file_path = "/mnt/st_data/linhaiyan/Sequencing_data_first_dimension/sample_mapped.bam"
output_file_path = "/mnt/st_data/linhaiyan/Sequencing_data_first_dimension/output.txt"
extract_columns(bam_file_path, output_file_path)
