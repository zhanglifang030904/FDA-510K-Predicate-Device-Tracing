import requests
import pandas as pd
import time

INPUT_FILE = r"E:/一匹午马/DTx/小张的方案呢？/研究/TracetotheSource/数据收集_Decide_Date/完整数据DecideDate.xlsx"
OUTPUT_FILE = "1230_FDA_Dates_Output.xlsx"
COLUMN_NAME = "设备编号"

FDA_API = "https://api.fda.gov/device/510k.json?search=k_number:"

def query_fda_api(k_number):
    url = FDA_API + k_number
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None, None
        data = r.json()
        if "results" not in data or len(data["results"]) == 0:
            return None, None

        record = data["results"][0]
        date_received = record.get("date_received", None)
        decision_date = record.get("decision_date", None)

        return date_received, decision_date

    except Exception:
        return None, None


def main():
    print(f"正在读取：{INPUT_FILE}")
    df = pd.read_excel(INPUT_FILE)

    if COLUMN_NAME not in df.columns:
        raise ValueError(f"Excel 中找不到列：{COLUMN_NAME}")

    df["Date Received"] = ""
    df["Decision Date"] = ""

    for i, row in df.iterrows():
        k = str(row[COLUMN_NAME]).strip()

        if not k or k.lower() == "nan":
            continue

        print(f"查询：{k} ... ", end="")
        dr, dd = query_fda_api(k)

        print(f"结果 → Received={dr}, Decision={dd}")

        df.at[i, "Date Received"] = dr
        df.at[i, "Decision Date"] = dd

        time.sleep(0.3)  # 限速

    print(f"✅ 完成，正在保存到 {OUTPUT_FILE}")
    df.to_excel(OUTPUT_FILE, index=False)
    print("✅ 全部完成！")

if __name__ == "__main__":
    main()
