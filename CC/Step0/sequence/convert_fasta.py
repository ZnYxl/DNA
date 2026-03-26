# # 读取reads.txt文件中的序列
# with open(r'/mnt/st_data/linhaiyan/PE_AYB/ref.txt', 'r') as file:
#     sequences = file.readlines()
#
# # 创建并写入reads.fasta文件
# with open(r'/mnt/st_data/linhaiyan/PE_AYB/ref.fasta', 'w') as fasta_file:
#     for i, sequence in enumerate(sequences, start=1):
#         fasta_file.write(f">{i}\n{sequence.strip()}\n")

# 读取序列文件
# with open(r'/amax/linhaiyan_data/data-nbt17-master/id20.refs.txt', 'r') as file:
#     sequences = file.readlines()
#
# # 处理后的序列列表
# processed_sequences = []
#
# # 遍历每一个序列，删除前缀和后缀，并加上编号
# for i, seq in enumerate(sequences, start=1):
#     # 删除前后的空白字符并删除5'-和-3'
#     clean_seq = seq.strip().replace("5'-", "").replace("-3'", "")
#     # 添加编号
#     processed_sequences.append(f">{i}")
#     processed_sequences.append(clean_seq)
#
# # 将处理后的序列列表转换为字符串
# output = "\n".join(processed_sequences)
#
# # 写入到新的FASTA文件中
# with open(r'/amax/linhaiyan_data/data-nbt17-master/id20.refs.fasta', 'w') as file:
#     file.write(output)

input_path = '/mnt/st_data/linhaiyan/data-nbt17/id20.refs.fasta'
output_path = '/mnt/st_data/linhaiyan/data-nbt17/id20.fasta'

with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
    for line in fin:
        line = line.strip()
        if line.startswith('>'):
            # 去掉 >seq 前缀，只保留数字部分
            # 比如 >seq1 -> >1
            # 先去掉开头的 '>'
            header = line[1:]
            # 去掉seq，只保留后面的数字
            number = header.replace('seq', '')
            fout.write(f'>{number}\n')
        else:
            fout.write(line + '\n')


