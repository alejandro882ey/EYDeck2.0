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
