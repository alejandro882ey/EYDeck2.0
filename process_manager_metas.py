import pandas as pd
import re

def debug_excel_read(excel_path):
    try:
        xls = pd.ExcelFile(excel_path)
        df = pd.read_excel(xls, sheet_name='METAS MANAGERS', header=None)
        print("--- ANSR Table (head) ---")
        print(df.head(10))
        print("--- Horas Table (head) ---")
        print(df.iloc[60:70])
        print("--- RPH Table (head) ---")
        print(df.iloc[120:130])
    except FileNotFoundError:
        print(f"Error: The file was not found at {excel_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    excel_file_path = r"C:\Users\CK624GF\OneDrive - EY\Documents\2025\Metas Mensualizadas internas EY Venezuela FY26 v2.xlsx"
    # Temporarily changing the script to debug
    debug_excel_read(excel_file_path)