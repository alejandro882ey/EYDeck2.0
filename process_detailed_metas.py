
import pandas as pd
import io
import re

def process_detailed_metas(output_csv_path):
    # ANSR Data
    ansr_data_lines = [
        "ANSR.Julio.Agosto.Septiembre.Octubre.Noviembre.Diciembre.Enero.Febrero.Marzo.Abril.Mayo.Junio.Total.",
        "AUDIT 375,589  274,026  393,853  638,820  553,325  354,018  577,773  653,300  615,207  642,811  538,266  591,910  6,208,900 ",
        "CNS 38,931  36,571  36,431  51,152  52,115  39,706  55,592  38,646  31,171  39,976  38,565  56,989  515,846 ",
        "TAX 322,317  313,066  476,410  625,342  526,751  352,748  626,949  663,754  685,639  582,326  520,564  528,069  6,223,935 ",
        "TECH ASSURANCE 11,600  9,818  15,796  40,260  45,100  25,898  36,784  26,836  20,240  11,901  22,810  21,344  288,386 ",
        "SAT 9,000  7,500  6,390  7,877  8,432  5,105  9,600  10,800  14,998  10,184  10,418  10,271  110,575 ",
        "Total general 757,436  640,981  928,880  1,363,451  1,185,723  777,476  1,306,697  1,393,336  1,367,256  1,287,198  1,130,623  1,208,584  13,347,641 "
    ]
    ansr_data_str = "\n".join(ansr_data_lines)

    # Horas Data
    horas_data_lines = [
        "Horas: HorasSuma de JulioSuma de AgostoSuma de SeptiembreSuma de OctubreSuma de NoviembreSuma de DiciembreSuma de EneroSuma de FebreroSuma de MarzoSuma de AbrilSuma de Mayo Suma de JunioSuma de Total",
        "AUDIT 12,390  8,998  11,935  19,275  16,600  11,249  17,584  19,933  18,723  18,416  15,322  16,551  186,976 ",
        "CNS 1,162  1,076  1,034  1,478  1,400  1,110  1,393  1,143  1,072  1,334  1,248  1,671  15,123 ",
        "TAX 8,255  8,310  11,267  13,914  12,373  8,581  14,241  15,371  15,603  13,734  12,400  12,473  146,520 ",
        "TECH ASSURANCE 352  298  395  1,007  1,128  719  968  706  440  300  634  593  7,538 ",
        "SAT 300  250  213  263  291  170  320  360  500  335  347  342  3,691 ",
        "Total general 22,459  18,932  24,844  35,936  31,792  21,830  34,505  37,513  36,338  34,119  29,951  31,631  359,848 "
    ]
    horas_data_str = "\n".join(horas_data_lines)

    # RPH Data
    rph_data_lines = [
        "RPH: RPHJulio.Agosto.Septiembre.Octubre.Noviembre.Diciembre.Enero.Febrero.Marzo.Abril.Mayo.Junio.Total.",
        "AUDIT 30  30  33  33  33  31  33  33  33  35  35  36  33 ",
        "CNS 33  34  35  35  37  36  40  34  29  30  31  34  34 ",
        "TAX 39  38  42  45  43  41  44  43  44  42  42  42  42 ",
        "TECH ASSURANCE 33  33  40  40  40  36  38  38  46  40  36  36  38 ",
        "SAT 30  30  30  30  29  30  30  30  30  30  30  30  30 ",
        "Total general 34  34  37  38  37  36  38  37  38  38  38  38  37"
    ]
    rph_data_str = "\n".join(rph_data_lines)

    def parse_table(data_str, value_col_name):
        lines = data_str.strip().split('\n')
        
        # Clean header
        header_line = lines[0]
        
        # Special handling for Horas header
        if "HorasSuma de Julio" in header_line: # Unique identifier for Horas header
            column_names = ['Horas', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre',
                            'Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Total']
        # Special handling for RPH header
        elif "RPH: RPHJulio" in header_line: # Unique identifier for RPH header
            column_names = ['RPH', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre',
                            'Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Total']
        else: # General handling for ANSR
            header_line = header_line.replace('.', ' ').replace('Suma de ', '').replace(':', '')
            if "Mayo  Junio" in header_line:
                header_line = header_line.replace("Mayo  Junio", "Mayo Junio")
            column_names = header_line.split()

        # Read the data lines
        data_rows = []
        for line in lines[1:]:
            # Find the index of the first digit to separate SL name from numerical data
            first_digit_index = -1
            for i, char in enumerate(line):
                if char.isdigit():
                    first_digit_index = i
                    break
            
            if first_digit_index != -1:
                sl_name = line[0:first_digit_index].strip()
                numerical_data_str = line[first_digit_index:].strip()
                
                # Split the numerical data string by one or more spaces
                numerical_parts = re.split(r'\s+', numerical_data_str)
                
                parts = [sl_name] + numerical_parts
            else:
                # Fallback for lines without numbers (should not happen with this data)
                parts = re.split(r'\s+', line.strip())
            
            data_rows.append(parts)

        # Create DataFrame from the parsed data
        df = pd.DataFrame(data_rows, columns=column_names)
        
        # Melt the dataframe
        id_vars = df.columns[0] # First column is the SL
        melted_df = df.melt(id_vars=[id_vars], var_name='Mes', value_name=value_col_name)
        
        # Rename the first column to 'SL' for consistency
        melted_df = melted_df.rename(columns={id_vars: 'SL'})
        
        # --- Apply month-year formatting ---
        def format_month_year(month_name):
            if month_name == 'Total':
                return 'Total'
            
            month_map_25 = ['Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
            month_map_26 = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio']
            
            if month_name in month_map_25:
                return f"{month_name} 25"
            elif month_name in month_map_26:
                return f"{month_name} 26"
            else:
                return month_name # Return as is if not a recognized month
        
        melted_df['Mes'] = melted_df['Mes'].apply(format_month_year)
        # --- End month-year formatting ---

        # Convert numeric columns, handling commas
        if value_col_name != 'RPH Goal': # RPH is integer, others have commas
            melted_df[value_col_name] = melted_df[value_col_name].astype(str).str.replace(',', '').astype(float)
        else:
            melted_df[value_col_name] = pd.to_numeric(melted_df[value_col_name], errors='coerce')

        return melted_df

    ansr_df = parse_table(ansr_data_str, 'ANSR Goal')
    horas_df = parse_table(horas_data_str, 'Horas Goal')
    rph_df = parse_table(rph_data_str, 'RPH Goal')

    # Merge the dataframes
    # Assuming 'SL' and 'Mes' are the common keys
    merged_df = pd.merge(ansr_df, horas_df, on=['SL', 'Mes'], how='outer')
    final_df = pd.merge(merged_df, rph_df, on=['SL', 'Mes'], how='outer')

    # Define custom month order for sorting
    month_order = ['Julio 25', 'Agosto 25', 'Septiembre 25', 'Octubre 25', 'Noviembre 25', 'Diciembre 25',
                   'Enero 26', 'Febrero 26', 'Marzo 26', 'Abril 26', 'Mayo 26', 'Junio 26', 'Total']
    
    # Convert 'Mes' column to categorical type for custom sorting
    final_df['Mes'] = pd.Categorical(final_df['Mes'], categories=month_order, ordered=True)

    # Sort the DataFrame by 'SL' and then by 'Mes'
    final_df = final_df.sort_values(by=['SL', 'Mes']).reset_index(drop=True)

    # Save to CSV
    final_df.to_csv(output_csv_path, index=False)
    print(f"Processed detailed data saved to {output_csv_path}")

# --- Main execution ---
if __name__ == "__main__":
    output_csv_file = 'metas_detailed_database.csv'
    process_detailed_metas(output_csv_file)
