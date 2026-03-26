def read_fasta(file_path):
    with open(file_path, 'r') as file:
        sequences = {}
        current_seq = ''
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                current_seq = line[1:]
                sequences[current_seq] = ''
            else:
                sequences[current_seq] += line
    return sequences


def main():
    fasta_file = '/amax/linhaiyan_data/data-nbt17-master/id20.refs.fasta'
    sequences = read_fasta(fasta_file)

    sequence_lengths = {}

    for seq_id, seq in sequences.items():
        length = len(seq)
        if length in sequence_lengths:
            sequence_lengths[length] += 1
        else:
            sequence_lengths[length] = 1

    print(f"总共有 {len(sequences)} 条序列")
    for length, count in sequence_lengths.items():
        print(f"长度为 {length} 的序列有 {count} 条")


if __name__ == "__main__":
    main()
