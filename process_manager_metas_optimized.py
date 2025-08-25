import pandas as pd
import re

def process_manager_metas_optimized(excel_path, output_csv_path):
    """
    Processes the METAS MANAGERS sheet from the Excel file to create an optimized CSV
    with proper SL and SSL information for each manager.
    """
    try:
        # Read the Excel file
        xls = pd.ExcelFile(excel_path)
        df = pd.read_excel(xls, sheet_name='METAS MANAGERS', header=None)
        
        def extract_table_data(start_row, end_row, value_col_name):
            """
            Extract data from a specific table range and convert to long format, including totals.
            """
            # Extract the table data
            table_data = df.iloc[start_row:end_row+1, 0:16]  # A to P columns (0-15)
            
            # Set up column names based on the header row (first row of each table)
            header_row = table_data.iloc[0]
            column_names = ['Manager', 'SL', 'SSL', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 
                           'Noviembre', 'Diciembre', 'Enero', 'Febrero', 'Marzo', 'Abril', 
                           'Mayo', 'Junio', 'Total']
            
            # Skip the header row and create dataframe with proper columns
            data_rows = table_data.iloc[1:].copy()
            data_rows.columns = column_names
            
            # Remove rows where Manager is NaN or empty
            data_rows = data_rows.dropna(subset=['Manager'])
            data_rows = data_rows[data_rows['Manager'].astype(str).str.strip() != '']
            
            # Convert monthly and total columns to numeric, handling any formatting issues
            month_columns = ['Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre',
                           'Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio']
            
            for col in month_columns + ['Total']:
                data_rows[col] = pd.to_numeric(data_rows[col], errors='coerce')

            # Extract totals
            totals_df = data_rows[['Manager', 'SL', 'SSL', 'Total']].copy()
            totals_df['Mes'] = 'Total'
            totals_df.rename(columns={'Total': value_col_name}, inplace=True)

            # Melt the dataframe to convert from wide to long format
            melted_df = data_rows.melt(
                id_vars=['Manager', 'SL', 'SSL'], 
                value_vars=month_columns,
                var_name='Mes', 
                value_name=value_col_name
            )
            
            # Apply month-year formatting
            def format_month_year(month_name):
                month_map_25 = ['Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
                month_map_26 = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio']
                
                if month_name in month_map_25:
                    return f"{month_name} 25"
                elif month_name in month_map_26:
                    return f"{month_name} 26"
                else:
                    return month_name
            
            melted_df['Mes'] = melted_df['Mes'].apply(format_month_year)

            # Combine monthly data and totals
            final_table_df = pd.concat([melted_df, totals_df], ignore_index=True)
            
            return final_table_df
        
        # Extract data from each table
        print("Extracting ANSR Goal data...")
        ansr_df = extract_table_data(0, 57, 'ANSR Goal')  # A1:P58 (0-indexed: 0-57)
        
        print("Extracting Horas Goal data...")
        horas_df = extract_table_data(61, 118, 'Horas Goal')  # A62:P119 (0-indexed: 61-118)
        
        print("Extracting RPH Goal data...")
        rph_df = extract_table_data(122, 179, 'RPH Goal')  # A123:P180 (0-indexed: 122-179)
        
        # Merge the dataframes
        print("Merging dataframes...")
        merged_df = pd.merge(ansr_df, horas_df, on=['Manager', 'SL', 'SSL', 'Mes'], how='outer')
        final_df = pd.merge(merged_df, rph_df, on=['Manager', 'SL', 'SSL', 'Mes'], how='outer')
        
        # Define custom month order for sorting
        month_order = ['Julio 25', 'Agosto 25', 'Septiembre 25', 'Octubre 25', 'Noviembre 25', 'Diciembre 25',
                       'Enero 26', 'Febrero 26', 'Marzo 26', 'Abril 26', 'Mayo 26', 'Junio 26', 'Total']
        
        # Convert 'Mes' column to categorical type for custom sorting
        final_df['Mes'] = pd.Categorical(final_df['Mes'], categories=month_order, ordered=True)
        
        # Sort the DataFrame by 'Manager' and then by 'Mes'
        final_df = final_df.sort_values(by=['Manager', 'Mes']).reset_index(drop=True)
        
        # Clean up any remaining NaN values in numeric columns
        for col in ['ANSR Goal', 'Horas Goal', 'RPH Goal']:
            if col in final_df.columns:
                final_df[col] = final_df[col].fillna(0)
        
        # Save to CSV
        final_df.to_csv(output_csv_path, index=False)
        print(f"Optimized MANAGERS data saved to {output_csv_path}")
        print(f"Total records: {len(final_df)}")
        print(f"Unique managers: {final_df['Manager'].nunique()}")
        print(f"SL categories: {final_df['SL'].unique()}")
        print(f"SSL categories: {final_df['SSL'].unique()}")
        
        return final_df
        
    except FileNotFoundError:
        print(f"Error: The file was not found at {excel_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        return None

# Main execution
if __name__ == "__main__":
    excel_file_path = r"C:\Users\CK624GF\OneDrive - EY\Documents\2025\Metas Mensualizadas internas EY Venezuela FY26 v2.xlsx"
    output_csv_file = 'metas_MANAGERS.csv'
    
    result_df = process_manager_metas_optimized(excel_file_path, output_csv_file)
    
    if result_df is not None:
        print("\nFirst few rows of the processed data:")
        print(result_df.head(13))
        print("\nLast few rows of the processed data:")
        print(result_df.tail(13))
        print("\nData types:")
        print(result_df.dtypes)