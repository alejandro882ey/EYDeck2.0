
import pandas as pd

def create_metas_v2(excel_path, output_csv_path):
    """
    Reads the 'METAS SL' sheet, aggregates the data for each of the three tables
    by month, and saves the result to a new CSV file.
    """
    try:
        xls = pd.ExcelFile(excel_path)
        df = pd.read_excel(xls, sheet_name='METAS SL', header=None)

        # Helper function to process and aggregate a table
        def aggregate_table(start_row, header_row, goal_name):
            # Extract the numeric data part of the table
            table_data = df.iloc[start_row:start_row+6, 1:14]
            
            # Set the column headers from the specified header row
            table_data.columns = df.iloc[header_row, 1:14]
            
            # Convert all data to numeric, coercing errors will turn non-numerics into NaN
            numeric_data = table_data.apply(pd.to_numeric, errors='coerce')
            
            # Calculate the sum for each month (column)
            monthly_sum = numeric_data.sum()
            
            # Convert the series to a dataframe and name the column
            agg_df = monthly_sum.to_frame(name=goal_name)
            return agg_df

        # Process and aggregate each of the three tables
        ansr_agg_df = aggregate_table(start_row=3, header_row=2, goal_name='ANSR Goal SL')
        horas_agg_df = aggregate_table(start_row=11, header_row=10, goal_name='Horas Goal SL')
        rph_agg_df = aggregate_table(start_row=19, header_row=18, goal_name='RPH Goal SL')

        # Combine the aggregated dataframes
        final_df = pd.concat([ansr_agg_df, horas_agg_df, rph_agg_df], axis=1)
        
        # Set the index name to 'Mes' and then reset it to become a column
        final_df.index.name = 'Mes'
        final_df.reset_index(inplace=True)

        # Save the final dataframe to a CSV file
        final_df.to_csv(output_csv_path, index=False)
        print(f"Data aggregated successfully and saved to {output_csv_path}")

    except FileNotFoundError:
        print(f"Error: The file was not found at {excel_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    excel_file_path = r"C:\Users\CK624GF\OneDrive - EY\Documents\2025\Metas Mensualizadas internas EY Venezuela FY26 v2.xlsx"
    output_csv = 'metas_database_v2.csv'
    create_metas_v2(excel_file_path, output_csv)
