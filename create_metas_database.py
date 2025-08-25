
import pandas as pd

def create_metas_database(excel_path, output_csv_path):
    """
    Reads the 'METAS SL' sheet from an Excel file, processes three tables,
    merges them, and saves the result to a CSV file.
    """
    try:
        xls = pd.ExcelFile(excel_path)
        df = pd.read_excel(xls, sheet_name='METAS SL', header=None)

        # Extract SL names (assuming they are the same for all tables, from rows 4-9)
        sl_names = df.iloc[3:9, 0].reset_index(drop=True)

        # Helper function to process a table from the sheet
        def process_table(start_row, value_name):
            # Extract table data (6 rows, 13 columns for months)
            table_data = df.iloc[start_row:start_row+6, 1:14]
            
            # Set column headers from row 3 (B3:N3)
            table_data.columns = df.iloc[2, 1:14]
            
            # Add the 'SL' column with the extracted names
            table_data['SL'] = sl_names
            
            # Unpivot the table from wide to long format
            melted_table = table_data.melt(id_vars='SL', var_name='Month', value_name=value_name)
            return melted_table

        # Process the three tables for ANSR, Horas, and RPH
        ansr_df = process_table(start_row=3, value_name='ANSR Meta')
        horas_df = process_table(start_row=11, value_name='Horas Meta')
        rph_df = process_table(start_row=19, value_name='RPH')

        # Merge the three dataframes into one
        merged_df = pd.merge(ansr_df, horas_df, on=['SL', 'Month'])
        final_df = pd.merge(merged_df, rph_df, on=['SL', 'Month'])

        # Save the final dataframe to a CSV file
        final_df.to_csv(output_csv_path, index=False)
        print(f"Data processed successfully and saved to {output_csv_path}")

    except FileNotFoundError:
        print(f"Error: The file was not found at {excel_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    excel_file_path = r"C:\Users\CK624GF\OneDrive - EY\Documents\2025\Metas Mensualizadas internas EY Venezuela FY26 v2.xlsx"
    output_csv = 'metas_database.csv'
    create_metas_database(excel_file_path, output_csv)
