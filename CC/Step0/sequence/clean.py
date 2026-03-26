
input_file = '/mnt/st_data/linhaiyan/Sequencing_data_first_dimension/output.txt'
output_file = '/mnt/st_data/linhaiyan/Sequencing_data_first_dimension/clean.txt'

with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
    for line in f_in:
        line = line.strip()
        if not line:
            continue  # 跳过空行
        parts = line.split('\t', 1)  # 分割序号和序列
        if len(parts) != 2:
            continue  # 如果格式不对，跳过该行
        seq = parts[1]
        if 'N' in seq:
            continue  # 跳过含 N 的序列
        f_out.write(seq + '\n')





