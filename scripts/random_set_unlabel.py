import json
import random
import sys

def update_split(input_file, output_file, split_ratio=0.2):
    # 读取原始 JSON 文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 筛选出 "split" 为 "train" 的元素
    train_images = [img for img in data.get('images', []) if img.get('split') == 'train']

    # 计算需要设置为 "unlabel" 的数量
    num_to_update = int(len(train_images) * split_ratio)

    # 随机选择五分之一的图像，将其 "split" 设置为 "unlabel"
    selected_images = random.sample(train_images, num_to_update)
    for img in selected_images:
        img['split'] = 'unlabel'

    # 更新原数据中的 "train" 图像列表
    for img in train_images:
        # 如果该图片已经被更新，跳过，否则保留原本的 "split" 值
        if img not in selected_images:
            img['split'] = 'train'

    # 保存更新后的数据为新文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"转换完成，结果已保存到 {output_file}")

if __name__ == "__main__":
    # 确保传递了正确数量的参数
    if len(sys.argv) != 4:
        print("Usage: python script.py <input_file> <output_file> <ratio>")
        sys.exit(1)
    
    # 获取命令行参数
    input_file = sys.argv[1]  # 第一个参数是输入文件路径
    output_file = sys.argv[2]  # 第二个参数是输出文件路径
    ratio = float(sys.argv[3])  # 第三个参数是比例，转换为浮动类型

    # 调用函数处理数据
    update_split(input_file, output_file, ratio)
