import pandas as pd
import re
from datetime import datetime, timedelta
import sys
import os

# Django setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'dashboard_django.settings')
import django
django.setup()

from core_dashboard.models import RevenueEntry, Client, Area, SubArea, Contract
from django.db import transaction
from decimal import Decimal

def _load_file(file_path, expected_columns, max_header_rows=10, sheet_name=None):
    """
    Loads a file and dynamically finds the header row based on expected columns.
    Can dynamically find the sheet if sheet_name is None or not found.
    """
    print(f"Attempting to load file: {file_path} (Sheet: {sheet_name}) and find header with expected columns: {expected_columns}", file=sys.stderr)

    # For Excel files, try to find the sheet dynamically if sheet_name is not provided or not found
    if file_path.endswith(('.xls', '.xlsx', '.xlsb')):
        xl = pd.ExcelFile(file_path, engine='openpyxl' if file_path.endswith(('.xls', '.xlsx')) else 'pyxlsb')
        
        sheets_to_try = []
        if sheet_name and sheet_name in xl.sheet_names:
            sheets_to_try.append(sheet_name)
        else:
            sheets_to_try.extend(xl.sheet_names) # Try all sheets if specific one not found or not provided

        for current_sheet_name in sheets_to_try:
            try:
                # Read the file without a header initially, reading enough rows to find the header
                temp_df = xl.parse(current_sheet_name, header=None, nrows=max_header_rows + 1)
                
                # Iterate through potential header rows
                found_header_row = -1
                for i in range(min(max_header_rows, len(temp_df))):
                    current_header = temp_df.iloc[i].astype(str).tolist()
                    # Check if all expected columns are present in the current header candidate
                    if all(col in current_header for col in expected_columns):
                        found_header_row = i
                        print(f"Found header for expected columns in sheet '{current_sheet_name}' at row {found_header_row}", file=sys.stderr)
                        # Reload the file with the identified header row and sheet
                        df = xl.parse(current_sheet_name, header=found_header_row)
                        # Clean column names (remove leading/trailing spaces)
                        df.columns = df.columns.str.strip()
                        print(f"Successfully loaded {file_path} (Sheet: {current_sheet_name}) with header at row {found_header_row}. Columns: {df.columns.tolist()}", file=sys.stderr)
                        return df
            except Exception as e:
                print(f"Error parsing sheet '{current_sheet_name}': {e}", file=sys.stderr)
        
        raise ValueError(f"Could not find a sheet containing all expected columns {expected_columns} in file {file_path} within the first {max_header_rows} rows of any sheet.")
    
    # Original CSV loading logic (if not Excel)
    elif file_path.endswith('.csv'):
        try:
            temp_df = pd.read_csv(file_path, header=None, encoding='utf-8', nrows=max_header_rows + 1)
        except UnicodeDecodeError:
            temp_df = pd.read_csv(file_path, header=None, encoding='latin1', nrows=max_header_rows + 1)
        
        # Iterate through potential header rows
        found_header_row = -1
        for i in range(min(max_header_rows, len(temp_df))):
            current_header = temp_df.iloc[i].astype(str).tolist()
            if all(col in current_header for col in expected_columns):
                found_header_row = i
                break

        if found_header_row == -1:
            raise ValueError(f"Could not find a header row containing all expected columns {expected_columns} in file {file_path} within the first {max_header_rows} rows. Found columns in first row: {temp_df.iloc[0].tolist()}")

        # Reload the file with the identified header row
        try:
            df = pd.read_csv(file_path, header=found_header_row, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, header=found_header_row, encoding='latin1')
        
        # Clean column names (remove leading/trailing spaces)
        df.columns = df.columns.str.strip()

        print(f"Successfully loaded {file_path} with header at row {found_header_row}. Columns: {df.columns.tolist()}", file=sys.stderr)
        return df
    else:
        raise ValueError(f"Unsupported file type: {file_path}. Only .csv, .xls, .xlsx, .xlsb are supported.")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python process_uploaded_data.py <engagement_path> <dif_path> <revenue_path> <upload_date_str>", file=sys.stderr)
        sys.exit(1)

    engagement_path = sys.argv[1]
    dif_path = sys.argv[2]
    revenue_path = sys.argv[3]
    upload_date_str = sys.argv[4]

    try:
        week_ending_date = pd.to_datetime(upload_date_str).date()

        # Define expected columns for each file type
        engagement_expected_cols = [
            "EngagementID", "Engagement", "EngagementPartner", "EngagementManager",
            "Client", "EngagementServiceLine", "EngagementSubServiceLine",
            "FYTD_ChargedHours", "FYTD_DirectCostAmt", "FYTD_ANSRAmt",
            "MTD_ChargedHours", "MTD_DirectCostAmt", "MTD_ANSRAmt", "CP_ANSRAmt"
        ]
        dif_expected_cols = [
            "Socio", "Gerente", "Perdida al tipo de cambio Monitor",
            "Fecha de Cobro", "Engagement"
        ]
        # For revenue, the original script skips 8 rows and then takes row 9 as header.
        # The key columns used later are 'Employee Country/Region' and 'Employee'.
        # We will look for these in the header.
        revenue_expected_cols = ["Employee Country/Region", "Employee"]

        # Load data using the dynamic header finding function, specifying sheet names
        engagement_df = _load_file(engagement_path, expected_columns=engagement_expected_cols, sheet_name='DATA ENG LIST')
        dif_df = _load_file(dif_path, expected_columns=dif_expected_cols, sheet_name='DATA DIFERENCIAL')
        revenue_df = _load_file(revenue_path, expected_columns=revenue_expected_cols, sheet_name='RevenueDays')

        # Filter columns
        # The _load_file function already ensures these columns are present.
        # We still filter to ensure order and only keep necessary columns.
        engagement_df = engagement_df[engagement_expected_cols]
        engagement_df["Duplicate EngagementID"] = engagement_df["EngagementID"].duplicated(keep=False).astype(int)
        engagement_df["Week"] = week_ending_date

        # Ensure numeric types for key financial and hours columns
        numeric_cols = [
            "FYTD_ChargedHours", "FYTD_DirectCostAmt", "FYTD_ANSRAmt",
            "MTD_ChargedHours", "MTD_DirectCostAmt", "MTD_ANSRAmt", "CP_ANSRAmt"
        ]

        for col in numeric_cols:
            engagement_df[col] = pd.to_numeric(engagement_df[col], errors='coerce')


        # Select relevant columns
        # The _load_file function already ensures these columns are present.
        dif_df = dif_df[dif_expected_cols]

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

        # Ensure Dif_Div column exists
        if "Dif_Div" not in dif_df.columns:
            dif_df["Dif_Div"] = None

        # Define EngagementID and EngagementPartner-based percentage rules
        engagement_partner_pct = {
            "E-68504587": 0.5,
            "E-68827483": 0.5,
            "E-68826677": {
                "Hector Azocar": 0.7,
                "Eduardo Sabater": 0.3
            },
            "E-68820284": {
                "Hector Azocar": 0.7,
                "Eduardo Sabater": 0.3
            }
        }

        # Apply conditional percentages based on EngagementID and EngagementPartner
        for idx, row in dif_df.iterrows():
            eid = row.get("EngagementID")
            partner = row.get("EngagementPartner")

            if eid in engagement_partner_pct:
                pct_info = engagement_partner_pct[eid]
                if isinstance(pct_info, dict):
                    if partner in pct_info:
                        dif_df.at[idx, "Dif_Div"] = pct_info[partner]
                else:
                    dif_df.at[idx, "Dif_Div"] = pct_info
        
        # Merge with engagement_df
        merged_df = pd.merge(engagement_df, dif_df, on=merge_keys, how="left")

        # Use the correct column name after merge
        merged_df["diferencial_final"] = merged_df["Perdida al tipo de cambio Monitor"]

        # Apply Dif_Div logic only if the column exists
        if 'Dif_Div' in merged_df.columns:
            print("Found 'Dif_Div' column, applying related logic.", file=sys.stderr)
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
        else:
            print("Warning: 'Dif_Div' column not found. Skipping related logic.", file=sys.stderr)

        # Ensure numeric types
        merged_df["FYTD_ANSRAmt"] = pd.to_numeric(merged_df["FYTD_ANSRAmt"], errors='coerce')
        merged_df["diferencial_final"] = pd.to_numeric(merged_df["diferencial_final"], errors='coerce')
        merged_df["diferencial_final"] = -merged_df["diferencial_final"]

        # Recalculate FYTD_ANSR_Sintetico
        # Replace NaN with 0 before subtraction
        merged_df["FYTD_ANSR_Sintetico"] = merged_df["FYTD_ANSRAmt"] - merged_df["diferencial_final"].fillna(0)

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

        print(f"DEBUG: merged_df columns after 'P' renaming: {merged_df.columns.tolist()}", file=sys.stderr)

        

        # Delete all columns after "25Billings FYTD P"
        # Find the index of "25Billings FYTD P"
        try:
            col_to_keep_until_index = merged_df.columns.get_loc("25Billings FYTD P")
            # Keep columns from the beginning up to and including "25Billings FYTD P"
            merged_df = merged_df.iloc[:, :col_to_keep_until_index + 1]
        except KeyError:
            print("Warning: '25Billings FYTD P' column not found. No columns will be deleted.", file=sys.stderr)

        print(f"DEBUG: merged_df columns after deletion: {merged_df.columns.tolist()}", file=sys.stderr)

        # Renaming based on specific column names as requested by user
        specific_rename_map = {
            "Total Revenue Days P": "Total Revenue Days P CP",
            "Billed Revenue Days P": "Billed Revenue Days P CP",
            "Unbilled Revenue Days P": "Unbilled Revenue Days P CP",
            "Total Revenue Days.1 P": "Total Revenue Days.1 P MTD",
            "Billed Revenue Days.1 P": "Billed Revenue Days.1 P MTD",
            "Unbilled Revenue Days.1 P": "Unbilled Revenue Days.1 P MTD",
            "Total Revenue Days.2 P": "Total Revenue Days.2 P FYTD",
            "Billed Revenue Days.2 P": "Billed Revenue Days.2 P FYTD",
            "Unbilled Revenue Days.2 P": "Unbilled Revenue Days.2 P FYTD",
            "Total Revenue Days.3 P": "Total Revenue Days.3 P 52WKS",
            "Billed Revenue Days.3 P": "Billed Revenue Days.3 P 52WKS",
            "Unbilled Revenue Days.3 P": "Unbilled Revenue Days.3 P 52WKS",
        }
        merged_df.rename(columns=specific_rename_map, inplace=True)
        print(f"DEBUG: merged_df columns after index-based renaming: {merged_df.columns.tolist()}", file=sys.stderr)

        # New deletion based on column index as requested
        # Columns 48 to 67 (inclusive) are 0-indexed 47 to 66
        columns_to_drop_start_idx = 47
        columns_to_drop_end_idx = 66

        if len(merged_df.columns) > columns_to_drop_start_idx:
            cols_to_drop = merged_df.columns[columns_to_drop_start_idx : columns_to_drop_end_idx + 1]
            merged_df.drop(columns=cols_to_drop, inplace=True)
        print(f"DEBUG: merged_df columns after final index-based deletion: {merged_df.columns.tolist()}", file=sys.stderr)

        # Determine output directory from engagement_path
        output_dir = os.path.dirname(engagement_path)
        output_filename = f"Final_Database_{week_ending_date.strftime('%Y-%m-%d')}.csv"
        output_path = os.path.join(output_dir, output_filename)

        try:
            merged_df.to_csv(output_path, index=False)
            print(f"Successfully processed and saved to {output_path}", file=sys.stderr)
        except Exception as e:
            print(f"Error saving Final_Database.csv: {e}", file=sys.stderr)
            sys.exit(1) # Exit with error code if saving fails

        # --- Create and save partner revenue days mapping ---
        print("Creating partner revenue days mapping...", file=sys.stderr)
        revenue_days_col_index = 30 # Assuming column 31 is 0-indexed 30
        if len(merged_df.columns) > revenue_days_col_index:
            revenue_days_col_name = merged_df.columns[revenue_days_col_index]
            print(f"Using column '{revenue_days_col_name}' for revenue days.", file=sys.stderr)

            partner_revenue_days = {}
            for index, row in merged_df.iterrows():
                partner = row.get('EngagementPartner')
                revenue_days = row.get(revenue_days_col_name)
                if partner and pd.notna(revenue_days):
                    partner_revenue_days[partner] = revenue_days
            
            media_root = os.path.dirname(os.path.dirname(output_dir))
            json_output_path = os.path.join(media_root, 'revenue_days.json')

            try:
                import json
                with open(json_output_path, 'w') as f:
                    json.dump(partner_revenue_days, f, indent=4)
                print(f"Successfully saved partner revenue days to {json_output_path}", file=sys.stderr)
            except Exception as e:
                print(f"Error saving partner revenue days JSON: {e}", file=sys.stderr)
        else:
            print(f"Warning: Column at index {revenue_days_col_index} not found. Cannot create revenue days mapping.", file=sys.stderr)


        # --- Import data into Django models ---
        print("Starting data import into Django models...", file=sys.stderr)
        try:
            with transaction.atomic():
                # Clear existing data for the given week to prevent duplicates
                # Assuming 'Week' column in merged_df corresponds to 'date' in RevenueEntry
                # and we want to replace data for that specific week.
                # If 'Week' is a single date for the entire upload, then clear all data for that date.
                # If 'Week' can contain multiple dates, then clear based on unique dates in merged_df.
                # For simplicity, let's assume 'Week' is the upload_date_str for all entries.
                RevenueEntry.objects.filter(date=week_ending_date).delete()
                print(f"Cleared existing RevenueEntry data for week: {week_ending_date}", file=sys.stderr)

                for index, row in merged_df.iterrows():
                    # Get or create related objects
                    client_obj, _ = Client.objects.get_or_create(name=row['Client'])
                    area_obj, _ = Area.objects.get_or_create(name=row['EngagementServiceLine'])
                    sub_area_obj, _ = SubArea.objects.get_or_create(name=row['EngagementSubServiceLine'], area=area_obj)
                    # Contract might need more specific logic if it's not just by name
                    # For now, let's assume contract name is unique per client or can be created simply
                    contract_name = row.get('Engagement', None) # Use Engagement as contract name if available
                    contract_obj = None
                    if contract_name:
                        contract_obj, _ = Contract.objects.get_or_create(
                            client=client_obj, name=contract_name,
                            defaults={'value': 0, 'start_date': week_ending_date, 'end_date': week_ending_date}
                        )

                    # Create RevenueEntry instance
                    RevenueEntry.objects.create(
                        date=row['Week'],
                        client=client_obj,
                        contract=contract_obj,
                        area=area_obj,
                        sub_area=sub_area_obj,
                        revenue=None if pd.isna(row.get('FYTD_ANSRAmt')) else Decimal(row.get('FYTD_ANSRAmt')),
                        engagement_partner=row.get('EngagementPartner', ''),
                        engagement_manager=row.get('EngagementManager', ''),
                        collections=None if pd.isna(row.get('Billings FYTD P')) else Decimal(row.get('Billings FYTD P')),
                        billing=None if pd.isna(row.get('Billings CP P')) else Decimal(row.get('Billings CP P')),
                        bcv_rate=None if pd.isna(row.get('BCV Rate')) else Decimal(row.get('BCV Rate')),
                        monitor_rate=None if pd.isna(row.get('Monitor Rate')) else Decimal(row.get('Monitor Rate')),
                        engagement_id=row.get('EngagementID', ''),
                        engagement=row.get('Engagement', ''),
                        engagement_service_line=row.get('EngagementServiceLine', ''),
                        engagement_sub_service_line=row.get('EngagementSubServiceLine', ''),
                        fytd_charged_hours=None if pd.isna(row.get('FYTD_ChargedHours')) else row.get('FYTD_ChargedHours', 0.0),
                        fytd_direct_cost_amt=None if pd.isna(row.get('FYTD_DirectCostAmt')) else row.get('FYTD_DirectCostAmt', 0.0),
                        fytd_ansr_amt=None if pd.isna(row.get('FYTD_ANSRAmt')) else row.get('FYTD_ANSRAmt', 0.0),
                        mtd_charged_hours=None if pd.isna(row.get('MTD_ChargedHours')) else row.get('MTD_ChargedHours', 0.0),
                        mtd_direct_cost_amt=None if pd.isna(row.get('MTD_DirectCostAmt')) else row.get('MTD_DirectCostAmt', 0.0),
                        mtd_ansr_amt=None if pd.isna(row.get('MTD_ANSRAmt')) else row.get('MTD_ANSRAmt', 0.0),
                        cp_ansr_amt=None if pd.isna(row.get('CP_ANSRAmt')) else row.get('CP_ANSRAmt', 0.0),
                        duplicate_engagement_id=None if pd.isna(row.get('Duplicate EngagementID')) else row.get('Duplicate EngagementID', 0),
                        original_week_string=None if pd.isna(row.get('Week')) else row.get('Week').strftime('%Y-%m-%d'),
                        periodo_fiscal=row.get('Periodo Fiscal', ''), # Placeholder
                        fecha_cobro=None if pd.isna(row.get('Fecha de Cobro')) else row.get('Fecha de Cobro').strftime('%Y-%m-%d'),
                        dif_div=None if pd.isna(row.get('Dif_Div')) else row.get('Dif_Div', 0.0),
                        perdida_tipo_cambio_monitor=None if pd.isna(row.get('Perdida al tipo de cambio Monitor')) else row.get('Perdida al tipo de cambio Monitor', 0.0),
                        fytd_diferencial_final=None if pd.isna(row.get('diferencial_final')) else row.get('diferencial_final', 0.0),
                        fytd_ansr_sintetico=None if pd.isna(row.get('FYTD_ANSR_Sintetico')) else row.get('FYTD_ANSR_Sintetico', 0.0),
                        total_revenue_days_p_cp=None if pd.isna(row.get('Total Revenue Days P CP')) else row.get('Total Revenue Days P CP', 0.0),
                    )
            print("Data import into Django models completed successfully.", file=sys.stderr)
        except Exception as e:
            print(f"Error importing data into Django models: {e}", file=sys.stderr)
            sys.exit(1) # Exit with error code if import fails

    except Exception as e:
        print(f"Error during processing: {e}", file=sys.stderr)
        sys.exit(1)