import easyocr
import cv2
import re

#统一使用半角符号
def replace_fullwidth_symbols(text):
    translation_table = str.maketrans(
        '：；“”‘’（）【】，。！？',  # 全角符号
        ':;""\'\'()[],.!?'  # 对应的半角符号
    )
    return text.translate(translation_table)

def preprocessData(data):
    #去除空格
    data = [string.strip() for string in data]
    data = [string.replace(' ', '') for string in data]

    #修复时间格式
    new_data = []
    for string in data:
        if re.search(r'\d{4}-\d{2}-\d{2}\d{2}:\d{2}', string):
            fixed_string = re.sub(r'(\d{4}-\d{2}-\d{2})(\d{2}:\d{2})', r'\1 \2', string)
            new_data.append(fixed_string)
        else:
            new_data.append(string)
    data = new_data

    data = [replace_fullwidth_symbols(string) for string in data]
    #处理冒号前后内容分离
    new_data = []
    i = 0
    while i < len(data):
        if data[i].endswith(':') and i+1 < len(data):
            new_data.append(data[i] + data[i+1])
            i += 2
        else:
            new_data.append(data[i])
            i += 1
    data = new_data
    
    return data

def extract_information(data, category, keys):
    result = None
    
    for item in data:
        if category in item:
            result = {'类别': category}
            break
        
    if result == None:
        return None

    for item in data:
        for key in keys:
            for item in data:
                if key + ':' in item:
                    value = item.replace(key + ':', '').strip()
                    result[key] = value
                    break

    return result

def process_cases(data, cases):
    for case in cases:
        category = case['category']
        keys = case['keys']

        output = extract_information(data, category, keys)
        if output:
            return output

    return None

#imgpath = r"I:\\Lab\\CareFileManager\\叶帆病历\\202306\\图像 (227).jpg".encode("utf-8")
imgpath = "I:\\Lab\\CareFileManager\\test.jpg"

cases = [
    {
        'category': '病历',
        'keys': ['姓名', '就诊时间', '就诊科室'],
    },
    {
        'category': '出院记录',
        'keys': ['姓名', '科室', '入院时间', '出院时间'],
    },
    {
        'category': '疾病证明',
        'keys': ['姓名', '科室', '日期'],
    },
    {
        'category': '监测报告',
        'keys': ['姓名', '科室', '时间'],
    },
    {
        'category': '报告',
        'keys': ['姓名', '项目', '日期'],
    },
    {
        'category': '注射单',
        'keys': ['姓名', '科室', '日期'],
    },
    {
        'category': '处方',
        'keys': ['姓名', '科别', '日期'],
    },             
]

reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
result = reader.readtext(imgpath, detail=0)
data = preprocessData(result)
output = process_cases(data, cases)
print(output)    
