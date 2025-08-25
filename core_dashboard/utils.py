import pandas as pd
import numpy as np
import datetime

def generate_mock_data(num_days=500):
    """Genera un DataFrame con datos económicos simulados para Venezuela."""
    np.random.seed(42) # Para reproducibilidad
    dates = pd.bdate_range(end=datetime.date.today(), periods=num_days)
    df = pd.DataFrame(index=dates)

    # Official_Rate: Depreciación controlada
    df['Official_Rate'] = 100 * np.exp(np.cumsum(np.random.normal(0.0002, 0.0005, num_days)))
    df['Official_Rate'] = df['Official_Rate'].apply(lambda x: max(x, 100)) 

    # Parallel_Rate: Depreciación más volátil y rápida
    df['Parallel_Rate'] = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.0015, num_days)))
    df['Parallel_Rate'] = df['Parallel_Rate'] * (1 + np.random.normal(0.001, 0.005, num_days)).cumsum() 
    df['Parallel_Rate'] = df.apply(lambda row: max(row['Parallel_Rate'], row['Official_Rate'] * 1.05), axis=1) # Siempre por encima del oficial

    # IBC_Index: Índice bursátil con crecimiento y volatilidad
    df['IBC_Index'] = 1000 * np.exp(np.cumsum(np.random.normal(0.001, 0.01, num_days)))
    df['IBC_Index'] = df['IBC_Index'] + np.sin(np.linspace(0, 20, num_days)) * 100 

    # EMBI_Risk: Índice de riesgo país (valores altos, volátil)
    df['EMBI_Risk'] = 2000 + np.cumsum(np.random.normal(0.5, 5, num_days))
    df['EMBI_Risk'] = df['EMBI_Risk'].apply(lambda x: max(x, 1500)) 

    # LATAM_Index_Benchmark: Benchmark para mercados latinoamericanos
    df['LATAM_Index_Benchmark'] = 500 * np.exp(np.cumsum(np.random.normal(0.0008, 0.008, num_days)))

    # Asegurar que no haya valores negativos
    for col in ['Official_Rate', 'Parallel_Rate', 'IBC_Index', 'EMBI_Risk', 'LATAM_Index_Benchmark']:
        df[col] = df[col].apply(lambda x: max(x, 1.0)) 

    return df

def get_fiscal_month_year(report_date):
    """
    Determines the fiscal month and year for a given report date based on a simplified rule.
    If the report date is in the first week (days 1-7) of a calendar month, it belongs to the previous fiscal month.
    Otherwise, it belongs to the current calendar month.
    """
    month_map = {
        1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio',
        7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
    }

    current_month = report_date.month
    current_year = report_date.year

    if report_date.day <= 7:
        # Belongs to the previous fiscal month
        fiscal_month_num = current_month - 1
        fiscal_year = current_year
        if fiscal_month_num == 0: # If current month is January, previous is December of previous year
            fiscal_month_num = 12
            fiscal_year -= 1
    else:
        # Belongs to the current fiscal month
        fiscal_month_num = current_month
        fiscal_year = current_year

    fiscal_month_name = month_map.get(fiscal_month_num)
    # For fiscal year, use last two digits as per metas_SL.csv format (e.g., 'Julio 25')
    fiscal_year_short = fiscal_year % 100

    return f"{fiscal_month_name} {fiscal_year_short}"