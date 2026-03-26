# def load_references(ref_file):
#     with open(ref_file, 'r') as f:
#         references = [line.strip() for line in f if line.strip()]
#     return references
#
# def load_reads(reads_file):
#     with open(reads_file, 'r') as f:
#         content = f.read()
#     clusters = content.strip().split('===============================')
#     clusters = [[line.strip() for line in block.strip().split('\n') if line.strip()] for block in clusters if block.strip()]
#     return clusters
#
# def write_grouped_output(references, clusters, output_file):
#     with open(output_file, 'w') as f:
#         for ref, reads in zip(references, clusters):
#             f.write(ref + '\n')
#             f.write('****\n')
#             for read in reads:
#                 f.write(read + '\n')
#             f.write('\n')  # 空行分隔不同簇
#
# # === 主程序入口 ===
# if __name__ == '__main__':
#     reference_file = '/mnt/st_data/linhaiyan/sequence/test/reference.txt'
#     reads_file = '/mnt/st_data/linhaiyan/sequence/test/reads.txt'
#     output_file = '/mnt/st_data/linhaiyan/sequence/test/I.txt'
#
#     references = load_references(reference_file)
#     clusters = load_reads(reads_file)
#
#     assert len(references) == len(clusters), f"参考序列数量({len(references)})与reads簇数({len(clusters)})不一致"
#
#     write_grouped_output(references, clusters, output_file)
#     print(f"写入完成，输出文件为 {output_file}")


def load_references(ref_file):
    with open(ref_file, 'r') as f:
        references = [line.strip() for line in f if line.strip()]
    return references

def load_reads(reads_file):
    with open(reads_file, 'r') as f:
        content = f.read()
    clusters = content.strip().split('===============================')
    clusters = [[line.strip() for line in block.strip().split('\n') if line.strip()]
                for block in clusters if block.strip()]
    return clusters

def write_grouped_output(references, clusters, output_file, max_units=10000):
    with open(output_file, 'w') as f:
        for i, (ref, reads) in enumerate(zip(references, clusters)):
            if i >= max_units:
                break
            f.write(ref + '\n')
            f.write('*****\n')
            for read in reads:
                f.write(read + '\n')
            f.write('\n\n')  # 两个空行分隔不同单元

# === 主程序入口 ===
if __name__ == '__main__':
    reference_file = '/mnt/st_data/linhaiyan/feddna/Dataset/S/test/reference.txt'
    reads_file = '/mnt/st_data/linhaiyan/feddna/Dataset/S/test/reads.txt'
    output_file = '/mnt/st_data/linhaiyan/recon/S1.txt'

    references = load_references(reference_file)
    clusters = load_reads(reads_file)

    assert len(references) == len(clusters), f"参考序列数量({len(references)})与reads簇数({len(clusters)})不一致"

    write_grouped_output(references, clusters, output_file, max_units=10000)
    print(f"写入完成，共写入前 1000 个单元，输出文件为 {output_file}")



