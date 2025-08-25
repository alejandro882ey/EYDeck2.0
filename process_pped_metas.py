
import pandas as pd
import io
import re

def process_pped_metas(output_csv_path):
    # PPED ANSR Data
    pped_ansr_data_lines = [
        "PPED ansrJulioAgostoSeptiembreOctubreNoviembreDiciembreEneroFebreroMarzoAbrilMayo JunioTotal",
        "Aguilera, Valentina 11,600  9,818  15,796  40,260  45,100  25,898  36,784  26,836  20,240  11,901  22,810  21,344  288,386",
        "Azocar, Hector 40,235  29,412  44,229  75,853  65,650  41,759  68,599  79,018  73,034  72,491  60,090  63,028  713,396",
        "Benitez, Luis 8,056  21,133  55,775  56,394  57,138  24,789  36,563  61,972  65,071  75,606  81,184  76,040  619,722",
        "Casanova, Edmundo 130,689  125,687  124,557  137,949  129,075  127,946  149,369  131,446  139,987  146,614  131,574  138,545  1,613,439",
        "Cedeño, Anmar 35,754  26,136  39,303  67,405  58,339  37,108  60,959  70,218  64,901  64,417  53,398  56,009  633,947",
        "Cuni, Javier (AUD) 27,836  20,348  30,599  52,477  45,419  28,890  47,459  54,667  50,528  50,151  41,572  43,605  493,552",
        "Cuni, Javier (SAT) 9,000  7,500  6,390  7,877  8,432  5,105  9,600  10,800  14,998  10,184  10,418  10,271  110,576",
        "Total Javier Cuni 36,836  27,848  36,989  60,355  53,852  33,995  57,059  65,467  65,526  60,335  51,991  53,876  604,128",
        "Fernandez, Juan 38,933  36,571  36,431  51,152  52,115  39,706  55,592  38,646  31,171  39,976  38,564  56,988  515,846",
        "Gomez, Damian 28,329  25,281  62,753  103,990  81,579  44,913  121,382  122,996  115,017  73,780  60,422  56,029  896,468",
        "Jimenez, Ivette 20,538  18,328  45,496  75,393  59,144  32,562  88,002  89,172  83,387  53,490  43,806  40,621  649,940",
        "Lopez, Miguel (AUD) 27,734  20,273  30,487  52,285  45,253  28,785  47,285  54,467  50,343  49,968  41,420  43,445  491,746",
        "Lopez, Miguel (FAAS) 62,208  44,946  49,368  48,020  41,990  28,766  43,472  37,848  46,360  78,200  70,240  101,000  652,418",
        "Total Lopez, Miguel 89,942  65,219  79,855  100,305  87,243  57,551  90,757  92,315  96,703  128,168  111,660  144,445  1,144,164",
        "Marin, Erika 32,858  24,019  36,120  61,946  53,614  34,103  56,022  64,531  59,644  59,200  49,073  51,472  582,600",
        "Medina, Saul 38,599  26,409  50,480  73,934  44,570  37,121  48,386  39,769  65,623  64,453  56,697  69,564  615,605",
        "Niño, Daniela 29,617  21,650  32,557  55,836  48,326  30,739  50,496  58,166  53,761  53,361  44,233  46,395  525,135",
        "Ossenkopp, Gigliola 22,339  16,330  24,557  42,115  36,451  23,186  38,088  43,873  40,550  40,248  33,363  34,995  396,094",
        "Ruiz, Jhon E 68,522  50,090  75,324  129,182  111,806  71,118  116,828  134,572  124,382  123,456  102,337  107,340  1,214,957",
        "Sabater, Eduardo AUD 28,485  20,822  31,312  53,700  46,478  29,564  48,565  55,941  51,705  51,320  42,541  44,621  505,054",
        "Sabater, Eduardo TAX 14,521  13,965  13,840  15,328  14,342  14,216  16,597  14,605  15,554  16,290  14,619  15,394  179,271",
        "Total Sabater, Eduardo 43,005  34,787  45,152  69,028  60,819  43,780  65,162  70,546  67,259  67,611  57,160  60,015  684,325",
        "Sanchez, Yordalmir 8,853  7,900  19,610  32,497  25,493  14,035  37,932  38,436  35,943  23,056  18,882  17,509  280,146",
        "Velazquez, Jose GCR 7,082  6,320  15,688  25,998  20,395  11,228  30,345  30,749  28,754  18,445  15,106  14,007  224,117",
        "Velazquez, Jose ITTS Advisory 6,756  4,622  8,835  12,941  7,801  6,497  8,470  6,960  11,487  11,281  9,924  12,178  107,752",
        "Total Velazquez, Jose 13,838  10,942  24,523  38,938  28,196  17,725  38,815  37,709  40,241  29,726  25,029  26,186  331,870",
        "Zerpa, Ruben GCR 6,020  5,372  13,335  22,098  17,335  9,544  25,794  26,137  24,441  15,678  12,840  11,906  190,500",
        "Zerpa, Ruben PAS 52,874  58,049  66,038  68,821  69,878  29,898  64,109  101,512  100,375  83,631  75,510  76,276  846,972",
        "Total Zerpa, Jose 58,894  63,421  79,373  90,919  87,214  39,442  89,903  127,648  124,816  99,309  88,350  88,182  1,037,472",
        "Total 757,436  640,981  928,880  1,363,451  1,185,723  777,476  1,306,697  1,393,336  1,367,256  1,287,198  1,130,623  1,208,584  13,347,641"
    ]
    pped_ansr_data_str = "\n".join(pped_ansr_data_lines)

    # PPED Horas Data
    pped_horas_data_lines = [
        "PPED HorasJulioAgostoSeptiembreOctubreNoviembreDiciembreEneroFebreroMarzoAbrilMayo JunioTotal",
        "Aguilera, Valentina 352  298  395  1,007  1,128  719  968  706  440  300  634  593  7,538",
        "Azocar, Hector 1,341  980  1,340  2,299  1,989  1,347  2,111  2,431  2,247  2,113  1,742  1,801  21,742",
        "Benitez, Luis 200  523  1,381  1,397  1,415  614  906  1,535  1,612  1,872  2,011  1,883  15,348",
        "Casanova, Edmundo 4,316  4,151  4,114  4,556  4,263  4,225  4,933  4,341  4,623  4,842  4,345  4,576  53,285",
        "Cedeño, Anmar 1,192  871  1,191  2,043  1,768  1,197  1,876  2,161  1,997  1,878  1,548  1,600  19,321",
        "Cuni, Javier (AUD) 928  678  927  1,590  1,376  932  1,460  1,682  1,555  1,462  1,205  1,246  15,042",
        "Cuni, Javier (SAT) 300  250  213  263  291  170  320  360  500  335  347  342  3,691",
        "Total Javier Cuni 1,228  928  1,140  1,853  1,667  1,102  1,780  2,042  2,055  1,797  1,552  1,588  18,733",
        "Fernandez, Juan 1,162  1,076  1,034  1,478  1,400  1,110  1,393  1,143  1,072  1,334  1,248  1,671  15,123",
        "Gomez, Damian 506  452  1,121  1,858  1,457  802  2,168  2,197  2,055  1,318  1,079  1,001  16,014",
        "Jimenez, Ivette 367  327  813  1,347  1,057  582  1,572  1,593  1,490  956  783  726  11,610",
        "Lopez, Miguel (AUD) 924  676  924  1,584  1,371  929  1,455  1,676  1,549  1,457  1,201  1,241  14,987",
        "Lopez, Miguel (FAAS) 1,944  1,362  1,496  1,372  1,105  757  1,144  996  1,220  1,955  1,756  2,525  17,632",
        "Total Lopez, Miguel 2,868  2,038  2,420  2,956  2,476  1,686  2,599  2,672  2,769  3,412  2,957  3,766  32,619",
        "Marin, Erika 1,095  801  1,095  1,877  1,625  1,100  1,724  1,986  1,835  1,726  1,422  1,471  17,756",
        "Medina, Saul 318  217  416  609  367  306  398  327  540  531  467  573  5,068",
        "Niño, Daniela 987  722  987  1,692  1,464  992  1,554  1,790  1,654  1,556  1,282  1,326  16,004",
        "Ossenkopp, Gigliola 745  544  744  1,276  1,105  748  1,172  1,350  1,248  1,173  967  1,000  12,072",
        "Ruiz, Jhon E 2,284  1,670  2,283  3,915  3,388  2,294  3,595  4,141  3,827  3,599  2,966  3,067  37,028",
        "Sabater, Eduardo AUD 949  694  949  1,627  1,408  954  1,494  1,721  1,591  1,496  1,233  1,241  15,392",
        "Sabater, Eduardo TAX 480  461  457  506  474  469  548  482  514  538  483  508  5,921",
        "Total Sabater, Eduardo 1,429  1,155  1,406  2,133  1,882  1,423  2,042  2,204  2,105  2,034  1,716  1,783  21,313",
        "Sanchez, Yordalmir 139  124  308  511  401  221  596  604  565  362  297  275  4,404",
        "Velazquez, Jose GCR 127  113  280  464  364  201  542  549  514  329  270  250  4,003",
        "Velazquez, Jose ITTS Advisory 31  21  41  60  36  30  39  32  53  52  46  57  500",
        "Total Velazquez, Jose 158  134  321  524  401  231  581  582  567  382  316  307  4,503",
        "Zerpa, Ruben GCR 127  113  280  464  364  201  542  549  514  329  270  250  4,003",
        "Zerpa, Ruben PAS 1,646  1,807  2,056  2,142  2,175  931  1,996  3,160  3,125  2,603  2,351  2,374  26,366",
        "Total Zerpa, Jose 1,772  1,920  2,336  2,607  2,540  1,131  2,538  3,709  3,638  2,933  2,620  2,625  30,369",
        "Total 22,459  18,932  24,844  35,936  31,792  21,830  34,505  37,513  36,338  34,119  29,951  31,630  359,849"
    ]
    pped_horas_data_str = "\n".join(pped_horas_data_lines)

    # PPED RPH Data
    pped_rph_data_lines = [
        "PPED rphJulioAgostoSeptiembreOctubreNoviembreDiciembreEneroFebreroMarzoAbrilMayo JunioTotal",
        "Aguilera, Valentina 33  33  40  40  40  36  38  38  46  40  36  36  38",
        "Azocar, Hector 30  30  33  33  33  31  33  33  33  34  35  35  33",
        "Benitez, Luis 40  40  40  40  40  40  40  40  40  40  40  40  40",
        "Casanova, Edmundo 30  30  30  30  30  30  30  30  30  30  30  30  30",
        "Cedeño, Anmar 30  30  33  33  33  31  33  33  33  34  35  35  33",
        "Cuni, Javier (AUD) 30  30  33  33  33  31  33  33  33  34  35  35  33",
        "Cuni, Javier (SAT) 30  30  30  30  29  30  30  30  30  30  30  30  30",
        "Total Javier Cuni 30  30  32  33  32  31  32  32  32  34  33  34  32",
        "Fernandez, Juan 33  34  35  35  37  36  40  34  29  30  31  34  34",
        "Gomez, Damian 56  56  56  56  56  56  56  56  56  56  56  56  56",
        "Jimenez, Ivette 56  56  56  56  56  56  56  56  56  56  56  56  56",
        "Lopez, Miguel (AUD) 30  30  33  33  33  31  33  33  33  34  35  35  33",
        "Lopez, Miguel (FAAS) 32  33  33  35  38  38  38  38  38  40  40  40  37",
        "Total Lopez, Miguel 31  32  33  34  35  34  35  35  35  38  38  38  35",
        "Marin, Erika 30  30  33  33  33  31  33  33  33  34  35  35  33",
        "Medina, Saul 121  121  121  121  121  121  121  121  121  121  121  121  121",
        "Niño, Daniela 30  30  33  33  33  31  33  33  33  34  35  35  33",
        "Ossenkopp, Gigliola 30  30  33  33  33  31  33  33  33  34  35  35  33",
        "Ruiz, Jhon E 30  30  33  33  33  31  33  33  33  34  35  35  33",
        "Sabater, Eduardo AUD 30  30  33  33  33  31  33  33  33  34  35  35  33",
        "Sabater, Eduardo TAX 30  30  30  30  30  30  30  30  30  30  30  30  30",
        "Total Sabater, Eduardo 30  30  32  32  32  31  32  32  32  33  33  34  32",
        "Sanchez, Yordalmir 64  64  64  64  64  64  64  64  64  64  64  64  64",
        "Velazquez, Jose GCR 56  56  56  56  56  56  56  56  56  56  56  56  56",
        "Velazquez, Jose ITTS Advisory 216  216  216  216  216  216  216  216  216  216  216  216  216",
        "Total Velazquez, Jose 88  81  76  74  70  77  67  65  71  78  79  85  74",
        "Zerpa, Ruben GCR 48  48  48  48  48  48  48  48  48  48  48  48  48",
        "Zerpa, Ruben PAS 32  32  32  32  32  32  32  32  32  32  32  32  32",
        "Total Zerpa, Jose 33  33  34  35  34  35  35  34  34  34  34  34  34",
        "Total 34  34  37  38  37  36  38  37  38  38  38  38  37"
    ]
    pped_rph_data_str = "\n".join(pped_rph_data_lines)

    def parse_table(data_str, value_col_name):
        lines = data_str.strip().split('\n')
        
        # Define column names explicitly for PPED tables
        column_names = ['Partner', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre',
                        'Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Total']

        # Read the data lines
        data_rows = []
        for line in lines[1:]:
            # Find the index of the first digit to separate Partner name from numerical data
            first_digit_index = -1
            for i, char in enumerate(line):
                if char.isdigit():
                    first_digit_index = i
                    break
            
            if first_digit_index != -1:
                partner_name = line[0:first_digit_index].strip()
                numerical_data_str = line[first_digit_index:].strip()
                
                # Split the numerical data string by one or more spaces
                numerical_parts = re.split(r'\s+', numerical_data_str)
                
                parts = [partner_name] + numerical_parts
            else:
                # Fallback for lines without numbers (should not happen with this data)
                parts = re.split(r'\s+', line.strip())
            
            data_rows.append(parts)

        # Create DataFrame from the parsed data
        df = pd.DataFrame(data_rows, columns=column_names)
        
        # Melt the dataframe
        id_vars = df.columns[0] # First column is the Partner
        melted_df = df.melt(id_vars=[id_vars], var_name='Mes', value_name=value_col_name)
        
        # Rename the first column to 'Partner' for consistency
        melted_df = melted_df.rename(columns={id_vars: 'Partner'})
        
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
        melted_df[value_col_name] = melted_df[value_col_name].astype(str).str.replace(',', '').astype(float)

        return melted_df

    pped_ansr_df = parse_table(pped_ansr_data_str, 'ANSR Goal PPED')
    pped_horas_df = parse_table(pped_horas_data_str, 'Horas Goal PPED')
    pped_rph_df = parse_table(pped_rph_data_str, 'RPH Goal PPED')

    # Merge the dataframes
    merged_df = pd.merge(pped_ansr_df, pped_horas_df, on=['Partner', 'Mes'], how='outer')
    final_df = pd.merge(merged_df, pped_rph_df, on=['Partner', 'Mes'], how='outer')

    # Define custom month order for sorting
    month_order = ['Julio 25', 'Agosto 25', 'Septiembre 25', 'Octubre 25', 'Noviembre 25', 'Diciembre 25',
                   'Enero 26', 'Febrero 26', 'Marzo 26', 'Abril 26', 'Mayo 26', 'Junio 26', 'Total']
    
    # Convert 'Mes' column to categorical type for custom sorting
    final_df['Mes'] = pd.Categorical(final_df['Mes'], categories=month_order, ordered=True)

    # Sort the DataFrame by 'Partner' and then by 'Mes'
    final_df = final_df.sort_values(by=['Partner', 'Mes']).reset_index(drop=True)

    # Save to CSV
    final_df.to_csv(output_csv_path, index=False)
    print(f"Processed PPED data saved to {output_csv_path}")

# --- Main execution ---
if __name__ == "__main__":
    output_csv_file = 'metas_PPED.csv'
    process_pped_metas(output_csv_file)
