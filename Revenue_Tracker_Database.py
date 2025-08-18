import pandas as pd
import re
from datetime import datetime, timedelta
from pandasgui import show

# Define file paths
file_path_engagement_list = r"C:\Users\CK624GF\OneDrive - EY\Documents\2025\Automation01_ANSR\Engagement List - Jul 18.xlsb"
file_path_dif_updated = r"C:\Users\CK624GF\OneDrive - EY\Documents\2025\Automation01_ANSR\Acumulado Detalle de perdida cambiaria FY26 a Julio 18-07-2025.xlsx"
file_path_revenue_days = r"C:\Users\CK624GF\OneDrive - EY\Documents\2025\Automation01_ANSR\Revenue Days Report (10) Jul 18.xlsx"

# Extract date from filename
match = re.search(r'([A-Za-z]+)\s(\d{1,2})', file_path_engagement_list)
if match:
    month_str = match.group(1)
    day = int(match.group(2))
    month = datetime.strptime(month_str, "%b").month
    year = 2025
    week_ending_date = datetime(year, month, day)
else:
    raise ValueError("Could not extract date from filename.")

week_start_date = week_ending_date - timedelta(days=4)

# Load data
engagement_df = pd.read_excel(file_path_engagement_list, sheet_name='DATA ENG LIST', engine='pyxlsb')
dif_df = pd.read_excel(file_path_dif_updated, sheet_name='DATA DIFERENCIAL', engine='openpyxl')
revenue_df = pd.read_excel(file_path_revenue_days, sheet_name='RevenueDays', engine='openpyxl')

# Filter columns
engagement_cols = [
    "EngagementID", "Engagement", "EngagementPartner", "EngagementManager",
    "Client", "EngagementServiceLine", "EngagementSubServiceLine",
    "FYTD_ChargedHours", "FYTD_DirectCostAmt", "FYTD_ANSRAmt",
    "MTD_ChargedHours", "MTD_DirectCostAmt", "MTD_ANSRAmt", "CP_ANSRAmt"
]
engagement_df = engagement_df[engagement_cols]
engagement_df["Duplicate EngagementID"] = engagement_df["EngagementID"].duplicated(keep=False).astype(int)
engagement_df["Week"] = week_ending_date.date()

# Ensure numeric types for key financial and hours columns
numeric_cols = [
    "FYTD_ChargedHours", "FYTD_DirectCostAmt", "FYTD_ANSRAmt",
    "MTD_ChargedHours", "MTD_DirectCostAmt", "MTD_ANSRAmt", "CP_ANSRAmt"
]

for col in numeric_cols:
    engagement_df[col] = pd.to_numeric(engagement_df[col], errors='coerce')


# Select relevant columns
dif_cols = [
    "Socio", "Gerente", "Perdida al tipo de cambio Monitor",
    "Fecha de Cobro", "Engagement", "Dif_Div"
]
dif_df = dif_df[dif_cols]

# Convert date column
dif_df["Fecha de Cobro"] = pd.to_datetime(dif_df["Fecha de Cobro"], errors='coerce')

# Rename for merge
dif_df.rename(columns={
    "Engagement": "EngagementID",
    "Socio": "EngagementPartner",
    "Gerente": "EngagementManager"
}, inplace=True)

# Merge keys
merge_keys = ["EngagementID", "EngagementPartner", "EngagementManager"]

# Group and sum 'Perdida al tipo de cambio Monitor'
grouped_sum = dif_df.groupby(merge_keys, as_index=False)["Perdida al tipo de cambio Monitor"].sum()

# Preserve other columns and remove duplicates
preserved_data = dif_df.drop(columns=["Perdida al tipo de cambio Monitor"]).drop_duplicates(subset=merge_keys)

# Merge grouped sum back into the preserved data
dif_df= pd.merge(preserved_data, grouped_sum, on=merge_keys, how="left")

# Convert 'Perdida al tipo de cambio Monitor' to numeric
dif_df["Perdida al tipo de cambio Monitor"] = pd.to_numeric(
    dif_df["Perdida al tipo de cambio Monitor"], errors='coerce'
)

# Merge with engagement_df
merged_df = pd.merge(engagement_df, dif_df, on=merge_keys, how="left")

# Use the correct column name after merge
merged_df["diferencial_final"] = merged_df["Perdida al tipo de cambio Monitor"]

# Apply Dif_Div logic only to duplicated EngagementIDs
condition = merged_df["Dif_Div"].notna() & (merged_df["Duplicate EngagementID"] == 1)

for eid in merged_df.loc[condition, "EngagementID"].unique():
    group = merged_df[merged_df["EngagementID"] == eid]
    with_pct = group[group["Dif_Div"].notna()]
    without_pct = group[group["Dif_Div"].isna()]
    if not with_pct.empty and not without_pct.empty:
        pct = with_pct["Dif_Div"].values[0]
        original = with_pct["Perdida al tipo de cambio Monitor"].values[0]
        transformed = original * pct
        remaining = original - transformed
        merged_df.loc[with_pct.index, "diferencial_final"] = transformed
        merged_df.loc[without_pct.index, "diferencial_final"] = remaining

# Ensure numeric types
merged_df["FYTD_ANSRAmt"] = pd.to_numeric(merged_df["FYTD_ANSRAmt"], errors='coerce')
merged_df["diferencial_final"] = pd.to_numeric(merged_df["diferencial_final"], errors='coerce')
merged_df["diferencial_final"] = -merged_df["diferencial_final"]

# Recalculate FYTD_ANSR_Sintetico
# Replace NaN with 0 before subtraction
merged_df["FYTD_ANSR_Sintetico"] = merged_df["FYTD_ANSRAmt"] - merged_df["diferencial_final"].fillna(0)

# Skip the first 8 rows and keep columns 0 to 25
revenue_df = revenue_df.iloc[8:, :26]

# Set row 9 as header
revenue_df.columns = revenue_df.iloc[0]
revenue_df = revenue_df[1:]

# Reset index to avoid issues
revenue_df.reset_index(drop=True, inplace=True)

# Rename columns with index prefix to ensure uniqueness
revenue_df.columns = [f"{i}{str(col).strip()}" for i, col in enumerate(revenue_df.columns)]

# Find the column that contains 'Employee Country/Region'
country_col = next((col for col in revenue_df.columns if "Employee Country/Region" in col), None)
if country_col is None:
    raise KeyError("Could not find a column containing 'Employee Country/Region' in revenue_df.")

# Filter for Venezuela
venezuela_df = revenue_df[revenue_df[country_col].str.contains("Venezuela", case=False, na=False)]

# Find the column that contains 'Employee' for merging
employee_col = next((col for col in venezuela_df.columns if "Employee" in col), None)
if employee_col is None:
    raise KeyError("Could not find a column containing 'Employee' in venezuela_df.")

# Merge with merged_df using EngagementPartner and employee_col
merged_df = pd.merge(
    merged_df,
    venezuela_df,
    left_on="EngagementPartner",
    right_on=employee_col,
    how="left"
)


# Calculated columns
merged_df["Margin"] = merged_df["FYTD_ANSR_Sintetico"] - merged_df["FYTD_DirectCostAmt"]
merged_df["Margin_%"] = merged_df["Margin"] / merged_df["FYTD_ANSR_Sintetico"]
merged_df["RPH"] = merged_df["FYTD_ANSR_Sintetico"] / merged_df["FYTD_ChargedHours"]

# Rename Venezuela columns with "P"
for col in venezuela_df.columns:
    if col in merged_df.columns:
        merged_df.rename(columns={col: f"{col} P"}, inplace=True)

# Save output
merged_df.to_csv(r"C:\Users\CK624GF\OneDrive - EY\Documents\2025\Automation01_ANSR\Final_Database.csv", index=False)

show(engagement_df,venezuela_df, dif_df, merged_df)
