import easyocr
import cv2
import re
import os
import numpy as np

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
    {
        'category': '病案',
        'keys': ['姓名', '时间'],
    },              
]

reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)

def rotate_image(image):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 边缘检测
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # 检测直线
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)

    if lines is not None and len(lines) >= 2:
        # 计算旋转角度
        angle = lines[0][0][1] * 180 / np.pi - 90
        angle = np.clip(angle, -45, 45)  # 限制旋转角度在-45至45度之间

        # 旋转图像
        center = (image.shape[1] // 2, image.shape[0] // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
        return rotated
    
    return image


def Recognize(image):
    img = rotate_image(image)
    result = reader.readtext(img, detail=0)
    data = preprocessData(result)
    output = process_cases(data, cases)
    print(output)
    
def search_and_recognize(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.jpg'):
                file_path = os.path.join(root, file)
                print("Processing file:", file_path)
                cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
                Recognize(cv_img)

search_and_recognize("I:\\Lab\\CareFileMgr")