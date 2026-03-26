import sys
sys.path.append('/mnt/st_data/linhaiyan/python_pkgs')
import pysam
import re


# def count_mismatches_and_bases(bam_file):
#     bam = pysam.AlignmentFile(bam_file, "rb")
#     total_mismatch = 0
#     total_bases = 0
#     read_count = 0
#
#     for read in bam.fetch():
#         if read.is_unmapped:
#             continue
#         read_count += 1
#
#         # 计算 M 总长度（match + mismatch）
#         cigar = read.cigartuples  # List of (operation, length)
#         if cigar:
#             for op, length in cigar:
#                 if op == 0:  # M operation
#                     total_bases += length
#
#         # 统计 MD 字段的替换错误数
#         tags = dict(read.tags)
#         md = tags.get("MD", "")
#         mismatches = re.findall(r'[A-Z]', md)
#         total_mismatch += len(mismatches)
#
#     bam.close()
#
#     print(f"Total mapped reads: {read_count}")
#     print(f"Total mismatch (substitution) errors: {total_mismatch}")
#     print(f"Total matched bases (from CIGAR M): {total_bases}")
#     print(f"Average mismatch per read: {total_mismatch / read_count:.4f}")
#     print(f"Substitution rate (mismatch / total matched bases): {total_mismatch / total_bases:.6f}")
#
# # 运行脚本
# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         print("Usage: python verify.py <your.bam>")
#     else:
#         count_mismatches_and_bases(sys.argv[1])

def count_alignment_errors(bam_file):
    bam = pysam.AlignmentFile(bam_file, "rb")
    mismatch_total = 0
    insertion_total = 0
    deletion_total = 0
    matched_bases = 0
    read_count = 0

    for read in bam.fetch():
        if read.is_unmapped:
            continue
        read_count += 1

        cigar = read.cigartuples
        if cigar:
            for op, length in cigar:
                if op == 0:  # M
                    matched_bases += length
                elif op == 1:  # I
                    insertion_total += length
                elif op == 2:  # D
                    deletion_total += length

        tags = dict(read.tags)
        md = tags.get("MD", "")
        mismatches = re.findall(r'[A-Z]', md)
        mismatch_total += len(mismatches)

    bam.close()

    print(f"Total mapped reads: {read_count}")
    print(f"Matched bases (M): {matched_bases}")
    print(f"Substitutions (mismatches): {mismatch_total}")
    print(f"Insertions (I): {insertion_total}")
    print(f"Deletions (D): {deletion_total}")
    print()
    print(f"Substitution rate: {mismatch_total / matched_bases:.6f}")
    print(f"Insertion rate: {insertion_total / matched_bases:.6f}")
    print(f"Deletion rate: {deletion_total / matched_bases:.6f}")

# Run
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python verify.py <your.bam>")
    else:
        count_alignment_errors(sys.argv[1])