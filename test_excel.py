from pathlib import Path
import openpyxl

data_dir = Path("data")

print("当前工作目录：", Path.cwd())
print("data目录下的xlsx文件：", list(data_dir.glob("*.xlsx")))

for file in data_dir.glob("*.xlsx"):
    print("\n正在读取：", file)
    wb = openpyxl.load_workbook(file, data_only=True)
    print("工作表：", wb.sheetnames)