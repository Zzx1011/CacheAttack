import json
import re

def clean_html(text):
    # 移除 HTML 标签（如 <P>, <Table> 等）并清理多余空格
    clean = re.sub(r'<[^>]+>', '', text)
    return " ".join(clean.split())

def filter_and_extract_qa(input_path, output_path):
    """
    读取 Simplified NQ JSONL，仅保留有 short_answers 的数据。
    """
    extracted_count = 0
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            data = json.loads(line)
            
            # 1. 检查是否有短答案 (Simplified NQ 的标注结构)
            valid_annotations = [a for a in data.get('annotations', []) if a.get('short_answers')]
            
            if not valid_annotations:
                continue  # 跳过没有短答案的行
            
            # 2. 提取文本
            question = data['question_text']
            all_tokens = data['document_tokens']
            
            # 提取该条目下所有的短答案文本
            current_answers = []
            for annot in valid_annotations:
                for sa in annot['short_answers']:
                    # 从字典列表中提取 'token' 字段
                    tokens = [all_tokens[i]['token'] for i in range(sa['start_token'], sa['end_token'])]
                    ans_text = clean_html(" ".join(tokens))
                    if ans_text and ans_text not in current_answers:
                        current_answers.append(ans_text)
            
            # 3. 如果提取到了有效文本，则保存
            if current_answers:
                result = {
                    "question": question,
                    "answers": current_answers
                }
                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                extracted_count += 1
                
    print(f"处理完成！成功提取出 {extracted_count} 条含有短答案的问答对。")

# 使用方法
filter_and_extract_qa('v1.0-simplified_nq-dev-all.jsonl', 'cleaned_qa.jsonl')