# import pandas as pd
#
# def extract_sequences_to_txt(excel_file, output_file):
#     # 读取Excel文件
#     df = pd.read_excel(excel_file)
#
#     # 提取 A 列和 B 列
#     index_column = df['Index']
#     strand_sequences_column = df['Strand Sequences']
#
#     # 将索引和序列写入输出文件
#     with open(output_file, 'w') as f:
#         for index, sequence in zip(index_column, strand_sequences_column):
#             f.write(f"{index}\t{sequence}\n")
#
#     print(f"提取完成，结果保存在 {output_file} 文件中。")
#
# # 调用函数
# extract_sequences_to_txt('/amax/linhaiyan_data/P10_5_BDDP210000009/41467_2022_33046_MOESM5_ESM.xlsx', '/amax/linhaiyan_data/P10_5_BDDP210000009/excel.txt')


def filter_sequences(output_file, excel_file, reference_file):
    # 读取output.txt中的数字
    with open(output_file, 'r') as f:
        output_lines = f.read().strip().splitlines()
        output_numbers = {line.split('\t')[0] for line in output_lines}  # 使用集合存储数字

    # 读取excel.txt并过滤未出现的数字
    with open(excel_file, 'r') as f:
        excel_lines = f.read().strip().splitlines()

    filtered_sequences = []
    for line in excel_lines:
        parts = line.split('\t')
        if len(parts) == 2:
            index, sequence = parts
            if index in output_numbers:
                filtered_sequences.append(sequence)  # 只添加序列

    # 将结果写入reference.txt
    with open(reference_file, 'w') as f:
        for sequence in filtered_sequences:
            f.write(sequence + '\n')  # 只写入序列

    # 输出保留的序列数量
    print(f"过滤完成，结果保存在 {reference_file} 文件中。")
    print(f"保留的序列数量: {len(filtered_sequences)}")




# 调用函数
filter_sequences('output.txt', '/amax/linhaiyan_data/P10_5_BDDP210000009/excel.txt', '/amax/linhaiyan_data/P10_5_BDDP210000009/reference.txt')