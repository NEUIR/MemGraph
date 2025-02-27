import json
import re
import sys
from tqdm import tqdm


def parse_patent_jsonl(input_file, output_file):
    """
    解析包含专利数据的JSONL文件，拆分为指令、原始专利摘要、选项和正确答案

    参数:
        input_file (str): 输入的JSONL文件路径
        output_file (str): 输出的JSON文件路径
    """
    parsed_data = []

    # 读取并解析JSONL文件
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line_num, line in enumerate(tqdm(lines, desc="解析专利数据")):
            try:
                # 解析JSON
                data = json.loads(line.strip())
                instruction = data.get('instruction', '')
                output = data.get('output', '')

                # 保存原始指令
                original_instruction = instruction

                # 判断是中文还是英文
                is_chinese = "请从" in instruction

                # 提取指令部分
                if is_chinese:
                    instruct_pattern = r'^(请从.*?该序号是\?)'
                else:
                    instruct_pattern = r'^(Please select.*?Which number is\?)'

                instruct_match = re.search(instruct_pattern, instruction, re.DOTALL)
                if instruct_match:
                    instruct = instruct_match.group(1)
                    remaining_text = instruction[len(instruct):]
                else:
                    print(f"警告: 行 {line_num + 1} 无法识别指令部分")
                    # 简单地将第一行作为指令
                    parts = instruction.split('\n', 1)
                    instruct = parts[0] + '\n'
                    remaining_text = parts[1] if len(parts) > 1 else ''

                # 寻找选项A的位置
                option_a_pos = remaining_text.find("\nA: ")
                if option_a_pos == -1:
                    print(f"警告: 行 {line_num + 1} 无法找到选项A")
                    continue

                # 提取原始专利摘要
                original_patent = remaining_text[:option_a_pos].strip()

                # 提取所有选项
                options_text = remaining_text[option_a_pos:]
                option_pattern = r'\n([A-D]): (.*?)(?=\n[A-D]: |\Z)'
                options_matches = re.findall(option_pattern, options_text, re.DOTALL)

                if not options_matches:
                    print(f"警告: 行 {line_num + 1} 无法提取选项")
                    continue

                options = {match[0]: match[1].strip() for match in options_matches}

                # 将结果添加到列表
                parsed_data.append({
                    'original_instruction': original_instruction,  # 原始完整指令
                    'instruct': instruct,  # 提取的指令部分
                    'original_patent': original_patent,  # 原始专利摘要
                    'option_A': options.get('A', "未找到选项A"),
                    'option_B': options.get('B', "未找到选项B"),
                    'option_C': options.get('C', "未找到选项C"),
                    'option_D': options.get('D', "未找到选项D"),
                    'correct_answer': output  # 正确答案
                })

            except Exception as e:
                print(f"错误: 行 {line_num + 1} 处理异常: {str(e)}")
                continue

    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(parsed_data, f, ensure_ascii=False, indent=2)

    print(f"处理完成: 已成功解析 {len(parsed_data)}/{len(lines)} 条专利数据，保存到 {output_file}")


def test_sample(sample_json):
    """测试单个样本解析"""
    with open('temp_sample.jsonl', 'w', encoding='utf-8') as f:
        f.write(sample_json)

    parse_patent_jsonl('temp_sample.jsonl', 'temp_result.json')

    with open('temp_result.json', 'r', encoding='utf-8') as f:
        result = json.load(f)

    return result[0] if result else None


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python parse_patents.py <input_jsonl_file> <output_json_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    parse_patent_jsonl(input_file, output_file)