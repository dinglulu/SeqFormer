import os
import re
import subprocess


def convert_dict_to_input_txt(cluster_data, output_file='FOR_ITR_INPUT.txt'):
    """
    将字典格式的数据转换为 input.txt 格式
    每个类之间有两个空行
    """
    with open(output_file, 'w') as file:
        for cluster_id, cluster_info in cluster_data.items():
            # 写入参考序列
            file.write(f"{cluster_info['refs']}\n")
            # 添加一个空行
            file.write("****\n")
            # 添加一个空行

            # 写入每个seq
            for seq in cluster_info['seqs']:
                file.write(f"{seq}\n")
            file.write("\n")
            file.write("\n")


def convert_to_unix_format_python(file_path):
    """
    使用 Python 将文件从 Windows 格式 (CRLF) 转换为 Unix 格式 (LF)
    """
    with open(file_path, 'r', newline='') as file:
        content = file.read()

    # 替换所有的 \r\n 为 \n
    content = content.replace('\r\n', '\n')

    # 将转换后的内容写回文件
    with open(file_path, 'w', newline='') as file:
        file.write(content)

    print(f"{file_path} has been converted to Unix format using Python.")


def run_itr_commands(itr_path, itr_exe_path):
    """
    运行 itr 命令
    """

    # 运行 itr 命令，指定 itr 可执行文件的相对路径
    out_dir = os.path.join(itr_path, 'out')  # 拼接 itr 路径和 'out' 文件夹

    subprocess.run([itr_exe_path, "FOR_ITR_INPUT.txt", out_dir], check=True)


def parse_itr_output(out_dir):
    """
    解析 itr 输出文件夹中的所有文件，提取每个 Cluster 的信息
    """
    cluster_info = []

    # 遍历 out 目录中的所有文件
    for file_name in os.listdir(out_dir):
        file_path = os.path.join(out_dir, file_name)

        if os.path.isfile(file_path) and file_name.endswith('.txt'):  # 假设输出是 txt 文件
            with open(file_path, 'r') as file:
                content = file.read()

                # 查找每个 Cluster 的信息
                clusters = re.findall(r"Cluster Num: (\d+)(.*?)Distance: (\d+)", content, re.DOTALL)

                for cluster in clusters:
                    cluster_num = int(cluster[0])
                    seqs = cluster[1].strip().split('\n')
                    distance = int(cluster[2])
                    cluster_info.append({
                        'cluster_num': cluster_num,
                        'seqs': seqs,
                        'distance': distance
                    })

    return cluster_info


def calculate_statistics(cluster_info):
    """
    计算成功类和失败类的比例以及编辑距离的总和
    """
    total_clusters = len(cluster_info)

    # 假设编辑距离小于等于 1 的为成功，其他为失败
    success_clusters = [cluster for cluster in cluster_info if cluster['distance'] < 1]
    fail_clusters = [cluster for cluster in cluster_info if cluster['distance'] >=1]

    # 计算比例
    success_proportion = len(success_clusters) / total_clusters if total_clusters else 0
    fail_proportion = len(fail_clusters) / total_clusters if total_clusters else 0

    # 计算所有编辑距离的总和
    total_distance = sum(cluster['distance'] for cluster in cluster_info)

    return success_proportion, fail_proportion, total_distance


def ITR_TEST(cluster_data,itr_path,itr_exe_path):
    # 示例字典数据（可视作来自json）
    # cluster_data = {
    #     'cluster_id1': {
    #         'seqs': [
    #             'TTCGAAGAAAAATACCTATTTACGGCGTTTTACTCACACCATGACTTCTAATAGTCATTAAAATAGATGTGCGCATGATCGACGATCCGTTTAGACACCATCAAAGCAAAGTCCCGGATGTCAGCAGAGCCTCAACACGAGAATAGGCAACGCATAAACCCAACGCACTTTTAGAAATAGGATATTATGCGCCCGCTAGAACTATATAACATGCCTGTAGGGAACGGCGTTAGGTGAGCTGCTAATGTCCCATATCGTCC',
    #             'TTCGAAGAAAAATACCTATTTACGGCGTTTTACTCACACCATGACTTCTAATAGTCATTAAAATAGATGTGCGCATGATCGACGATCCGTTTAGACACCATCAAAGCAAAGTCCCGGATGTCAGCAGAGCCTCAACACGAGAATAGGCAACGCATAAACCCAACGCACTTTTAGAAATAGGATATTACGCGCCCGCTAAAACTATATAACATGCCTGTAGGGAACGGCGTTAGGTGAGCTGCTAATGTCCCATATCGTCC'
    #         ],
    #         'quals': [
    #             'IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII',
    #             'IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII'
    #         ],
    #         'refs': 'TTCGAAGAAAAATACCTATTTACGGCGTTTTACTCACACCATGACTTCTAATAGTCATTAAAATAGATGTGCGCATGATCGACGATCCGTTTAGACACCATCAAAGCAAAGTCCCGGATGTCAGCAGAGCCTCAACACGAGAATAGGCAACGCATAAACCCAACGCACTTTTAGAAATAGGATATTATGCGCCCGCTAGAACTATATAACATGCCTGTAGGGAACGGCGTTAGGTGAGCTGCTAATGTCCCATATCGTCC'
    #     },
    #     'cluster_id2': {
    #         'seqs': [
    #             'GGTGAGCTGCTAATGTCCCATATCGTCC',
    #             'GGTGAGCTGCTAATGTCCCATATCGTCC'
    #         ],
    #         'quals': [
    #             'IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII',
    #             'IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII'
    #         ],
    #         'refs': 'GGTGAGCTGCTAATGTCCCATATCGTCC'
    #     }
    # }

    # 步骤 1：转换字典数据为 input.txt
    cluster_data=cluster_data
    # 步骤 1：转换字典数据为 input.txt
    convert_dict_to_input_txt(cluster_data)

    # 将 input.txt 转换为 Unix 格式
    convert_to_unix_format_python('FOR_ITR_INPUT.txt')

    # 步骤 2：运行 ITR 命令,需要修改地址

    run_itr_commands(itr_path, itr_exe_path)

    # 步骤 3：解析 itr 输出文件
    out_dir = os.path.join(itr_path, 'out')  # itr 输出文件夹路径
    cluster_info = parse_itr_output(out_dir)

    # 计算统计数据
    success_proportion, fail_proportion, total_distance = calculate_statistics(cluster_info)

    # 输出统计数据
    print(f"成功类占比: {success_proportion:.2%}")
    print(f"失败类占比: {fail_proportion:.2%}")
    print(f"总编辑距离: {total_distance}")
    return success_proportion, fail_proportion, total_distance

# 调用主函数
if __name__ == "__main__":
    # 示例字典数据（可视作来自json）
    cluster_data = {
        'cluster_id1': {
            'seqs': [
                'TTCGAAGAAAAATACCTATTTACGGCGTTTTACTCACACCATGACTTCTAATAGTCATTAAAATAGATGTGCGCATGATCGACGATCCGTTTAGACACCATCAAAGCAAAGTCCCGGATGTCAGCAGAGCCTCAACACGAGAATAGGCAACGCATAAACCCAACGCACTTTTAGAAATAGGATATTATGCGCCCGCTAGAACTATATAACATGCCTGTAGGGAACGGCGTTAGGTGAGCTGCTAATGTCCCATATCGTCC',
                'TTCGAAGAAAAATACCTATTTACGGCGTTTTACTCACACCATGACTTCTAATAGTCATTAAAATAGATGTGCGCATGATCGACGATCCGTTTAGACACCATCAAAGCAAAGTCCCGGATGTCAGCAGAGCCTCAACACGAGAATAGGCAACGCATAAACCCAACGCACTTTTAGAAATAGGATATTACGCGCCCGCTAAAACTATATAACATGCCTGTAGGGAACGGCGTTAGGTGAGCTGCTAATGTCCCATATCGTCC'
            ],
            'quals': [
                'IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII',
                'IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII'
            ],
            'refs': 'TTCGAAGAAAAATACCTATTTACGGCGTTTTACTCACACCATGACTTCTAATAGTCATTAAAATAGATGTGCGCATGATCGACGATCCGTTTAGACACCATCAAAGCAAAGTCCCGGATGTCAGCAGAGCCTCAACACGAGAATAGGCAACGCATAAACCCAACGCACTTTTAGAAATAGGATATTATGCGCCCGCTAGAACTATATAACATGCCTGTAGGGAACGGCGTTAGGTGAGCTGCTAATGTCCCATATCGTCC'
        },
        'cluster_id2': {
            'seqs': [
                'GGTGAGCTGCTAATGTCCCATATCGTCC',
                'GGTGAGCTGCTAATGTCCCATATCGTCC'
            ],
            'quals': [
                'IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII',
                'IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII'
            ],
            'refs': 'GGTGAGCTGCTAATGTCCCATATCGTCC'
        }
    }
    itr_path = os.path.join("./Reconstruction", "Iterative")  # itr 的路径
    itr_exe_path = os.path.join("./Reconstruction", "Iterative", "DNA")  # itr 可执行文件的路径

    success_proportion, fail_proportion, total_distance=ITR_TEST(cluster_data,itr_path,itr_exe_path)
