def read_save_txt(file_path, file_save, extract=10):
    with open(file_path, 'r') as f:
        item = []
        cnt = 0
        k = f.readlines()  # 一次读取所有行
        for i in k:  # 一次取一行给i
            cnt += 1
            if cnt <= extract:  # 提取前多少行
                for j in i.split():  # .split是将字符串分隔开，一次取一行中的一个字符串
                    item.append(j)
                    item.append('\t')  # 加上空格
                item.append(str('\n'))

    with open(file_save, 'w') as f:
        for i in item:
            f.write(str(i))


if __name__ == "__main__":
    file_path = r"/mnt/st_data/linhaiyan/data-nbt17/clean.txt"
    file_save = r"verify.txt"
    print("开始读取文件...")
    read_save_txt(file_path, file_save, 500)  # 文件路径 ， 保存路径  ， 提取前多少行
    print("文件已成功保存为:", file_save)
