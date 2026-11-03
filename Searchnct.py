import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import time
import pdfplumber
import io
import os

# --- 核心配置 ---
# 使用 r"" 前缀防止路径中的特殊字符（如空格、斜杠）导致报错
INPUT_PATH = r"E:/一匹午马/DTx/小张的方案呢？/研究/TracetotheSource/260115-NCT试验/NCT_Trial.xlsx"

# 自动定位：获取输入文件所在的文件夹路径，并将结果也放在那里
BASE_DIR = os.path.dirname(INPUT_PATH)
OUTPUT_PATH = os.path.join(BASE_DIR, "FDA_NCT_Results.xlsx")

COLUMN_NAME = "设备编号"  # 确保 Excel 中这一列的名字准确无误

def extract_nct_from_text(text):
    """从文本中提取所有 NCT 编号"""
    if not text: return []
    # 匹配 NCT 后跟 8 位数字，允许中间有空格
    return list(set(re.findall(r'NCT\s?\d{8}', text, re.IGNORECASE)))

def get_nct_data(k_number):
    """核心逻辑：从 FDA HTML 页面获取，若无则解析 PDF Summary"""
    print(f"-> 正在检索编号: {k_number}...", end=" ", flush=True)
    base_url = "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm"
    search_url = f"{base_url}?ID={k_number}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        resp = requests.get(search_url, headers=headers, timeout=15)
        if resp.status_code != 200: 
            print("HTTP 错误")
            return f"HTTP {resp.status_code}"
        
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # 1. 尝试从网页 HTML 中提取
        ncts = extract_nct_from_text(resp.text)
        
        # 2. 如果网页没找到，寻找 PDF Summary 链接
        if not ncts:
            summary_link = soup.find('a', href=re.compile(r'\.pdf'))
            if summary_link:
                pdf_url = summary_link['href']
                if not pdf_url.startswith('http'):
                    pdf_url = "https://www.accessdata.fda.gov" + pdf_url
                
                # 下载并解析 PDF 内容
                pdf_resp = requests.get(pdf_url, headers=headers, timeout=20)
                with pdfplumber.open(io.BytesIO(pdf_resp.content)) as pdf:
                    pdf_text = "".join([page.extract_text() or "" for page in pdf.pages])
                    ncts = extract_nct_from_text(pdf_text)
        
        result = ", ".join(ncts) if ncts else "Not Found"
        print(f"完成 ({result})")
        return result

    except Exception as e:
        print(f"失败: {e}")
        return f"Error: {str(e)}"

def main():
    # 检查输入文件是否存在
    if not os.path.exists(INPUT_PATH):
        print(f"错误：找不到输入文件，请检查路径是否正确：\n{INPUT_PATH}")
        return

    print("="*50)
    print(f"读取文件: {INPUT_PATH}")
    print(f"结果将保存至: {OUTPUT_PATH}")
    print("="*50)

    # 1. 读取 Excel
    try:
        df = pd.read_excel(INPUT_PATH)
    except Exception as e:
        print(f"无法读取 Excel 文件: {e}")
        return

    if COLUMN_NAME not in df.columns:
        print(f"错误：Excel 中找不到名为 '{COLUMN_NAME}' 的列！")
        print(f"现有列名: {list(df.columns)}")
        return

    # 2. 遍历执行
    nct_results = []
    for index, row in df.iterrows():
        k_num = str(row[COLUMN_NAME]).strip()
        # 跳过空值
        if k_num == "nan" or not k_num:
            nct_results.append("")
            continue
            
        nct_info = get_nct_data(k_num)
        nct_results.append(nct_info)
        
        # 频率控制，防止被封 IP
        time.sleep(1.2)
    
    # 3. 保存结果
    df['NCT试验编号'] = nct_results
    try:
        df.to_excel(OUTPUT_PATH, index=False)
        print("\n" + "="*50)
        print(f"任务全部完成！")
        print(f"文件位置: {OUTPUT_PATH}")
        print("="*50)
    except PermissionError:
        print("\n错误：无法保存结果文件！请先关闭已经打开的 FDA_NCT_Results.xlsx 再次运行。")

if __name__ == "__main__":
    main()