import easyocr
import cv2
import re
import os
import numpy as np

DebugFlag = False

#统一使用半角符号
def replace_fullwidth_symbols(text):
    translation_table = str.maketrans(
        '：；“”‘’（）【】，。！？',  # 全角符号
        ':;""\'\'()[],.!?'  # 对应的半角符号
    )
    return text.translate(translation_table)

def filter_space(text):
    corrected_text = re.sub(r'(?<![0-9])\s+(?![0-9])', ' ', text) #特判日期格式中间的空格
    corrected_text = re.sub(r'\s{2,}', ' ', corrected_text)
    corrected_text = re.sub(r'(?<=\D)\s+(?=\D)', '', corrected_text)
    return corrected_text

def preprocessData(data):
    #去除空格
    data = [string.strip() for string in data]
    #data = [string.replace(' ', '') for string in data]
    data = [filter_space(text) for text in data]

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
            result[key] = "?"
            index = 0
            for item in data:
                if key in item:
                    if ":" in item:
                        suffix = item.split(key + ':', 1)[-1].strip()
                        value = suffix if suffix else "?"  # 处理空字符串的情况
                    else:
                        value = data[index + 1]
                        
                    result[key] = value
                    break
                index += 1

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

import matplotlib.pyplot as plt
import copy

def rotate_image(image):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 边缘检测
    edges = cv2.Canny(gray, 50, 80, apertureSize=3)

    # 检测直线
    lines = cv2.HoughLines(edges, 1, np.pi / 180 / 8, threshold=600)
    
    angle_sum = 0
    avg_angle = 0
    count = 0
    
    if lines is not None and len(lines) > 0:
        for line in lines:
            rho, theta = line[0]
            angle = np.rad2deg(theta)
            angle -= 90
            while angle < 0:
                angle += 360
                            
            if angle > 45 and angle < (90 + 45): 
                angle -= 90
                
        if DebugFlag:    
            print("angle=", angle)
            
        angle_sum += angle
        count += 1
        avg_angle = angle_sum / count
    
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    if DebugFlag:
        DebugImg = copy.copy(image)
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(DebugImg, (x1, y1), (x2, y2), (0, 0, 255), 2)      

    matrix = cv2.getRotationMatrix2D(center, -avg_angle, 1.0) #顺时针
    rotated = cv2.warpAffine(image, matrix, (w, h))
    if DebugFlag: 
        # 显示图像
        DebugImg = cv2.warpAffine(DebugImg, matrix, (w, h))    
        DebugImg = cv2.cvtColor(DebugImg, cv2.COLOR_BGR2RGB)
        
        plt.subplot(121)
        plt.imshow(edges)
        plt.subplot(122)
        plt.imshow(DebugImg)
        plt.axis('off')  # 可选：关闭坐标轴
        plt.show()

    return rotated


def Recognize(image):
    image = rotate_image(image)
    result = reader.readtext(image, detail=0)
    data = preprocessData(result)
    
    if DebugFlag:
      print(data)
      
    output = process_cases(data, cases)
    print(output)
    
def singe_recognize(file_path):
    print("Processing file:", file_path)
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    Recognize(cv_img)    
    
def search_and_recognize(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.jpg'):
                file_path = os.path.join(root, file)
                singe_recognize(file_path)

DebugFlag=True   

if DebugFlag:
    singe_recognize(r"I:\Lab\CareFileMgr\叶帆病历\202306\图像 (234).jpg")
else: 
    search_and_recognize("I:\\Lab\\CareFileMgr\\")