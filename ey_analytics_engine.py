import pandas as pd
import requests
from bs4 import BeautifulSoup
from pytrends.request import TrendReq
import plotly.graph_objects as go
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import VAR
from arch import arch_model
import numpy as np
import random # Added this import
from datetime import date, timedelta
import warnings

warnings.filterwarnings("ignore")

# --- PART 1: DATA ACQUISITION MODULE ---

from pyDolarVenezuela import Monitor

def get_exchange_rates():
    """Fetches official and parallel exchange rates using pyDolarVenezuela.Monitor."""
    print("NOTE: get_exchange_rates is fetching official and parallel rates using pyDolarVenezuela.Monitor.")
    official_rate = None
    parallel_rate = None
    try:
        monitor_data = Monitor().get_monitor()
        if 'bcv' in monitor_data and 'price' in monitor_data['bcv']:
            official_rate = monitor_data['bcv']['price']
            print(f"Official Rate from pyDolarVenezuela: {official_rate}")
        if 'enparalelovzla' in monitor_data and 'price' in monitor_data['enparalelovzla']:
            parallel_rate = monitor_data['enparalelovzla']['price']
            print(f"Parallel Rate from pyDolarVenezuela: {parallel_rate}")
    except Exception as e:
        print(f"Error fetching exchange rates from pyDolarVenezuela.Monitor: {e}. Using dummy rates.")

    # Fallback to dummy rates if fetching fails
    if official_rate is None:
        official_rate = 36.0 + random.uniform(-0.5, 0.5)
    if parallel_rate is None:
        parallel_rate = official_rate * (1 + random.uniform(0.03, 0.07)) # Use official for parallel fallback

    return {
        'Parallel_Rate': round(parallel_rate, 2),
        'Official_Rate': round(official_rate, 2)
    }

def get_commodities_and_indices(api_key_alpha, api_key_te):
    """
    Fetches commodity prices and financial indices.
    - Oil Price: Alpha Vantage API
    - IBC Index: Scraped from Bolsa de Valores de Caracas (BVC)
    - EMBI & LATAM: Simulated with more dynamic behavior (fallback for IBC if scraping fails).
    """
    results = {}
    # Oil Price (Brent) from Alpha Vantage
    try:
        url_alpha = f'https://www.alphavantage.co/query?function=BRENT&interval=daily&apikey={api_key_alpha}'
        response = requests.get(url_alpha, timeout=5)
        response.raise_for_status()
        data = response.json()
        # Taking the most recent data point
        latest_brent = data['data'][0]
        results['Oil_Price_Brent'] = float(latest_brent['value']) if latest_brent['value'] != "." else None
    except requests.exceptions.Timeout:
        print("Error fetching Brent oil price from Alpha Vantage: Request timed out.")
        results['Oil_Price_Brent'] = None # Handle case where API fails
    except Exception as e:
        print(f"Error fetching Brent oil price from Alpha Vantage: {e}")
        results['Oil_Price_Brent'] = None # Handle case where API fails

    # IBC Index (Scraping from BVC)
    ibc_index = None
    bvc_url = "http://www.bolsadecaracas.com/"
    try:
        print(f"Attempting to scrape IBC Index from {bvc_url}")
        response = requests.get(bvc_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Look for the element containing the IBC Index. This is a common pattern, might need adjustment.
        # Example: <span id="ctl00_ContentPlaceHolder1_lblIBC">1.234.567,89</span>
        ibc_element = soup.find('span', id='ctl00_ContentPlaceHolder1_lblIBC')
        if ibc_element:
            ibc_price_str = ibc_element.text.strip().replace('.', '').replace(',', '.')
            ibc_index = float(ibc_price_str)
            print(f"IBC Index scraped: {ibc_index}")
        else:
            print("Could not find IBC Index element on BVC page.")

    except requests.exceptions.Timeout:
        print(f"Error scraping IBC Index from {bvc_url}: Request timed out.")
    except requests.exceptions.RequestException as e:
        print(f"Error scraping IBC Index from {bvc_url}: {e}")
    except Exception as e:
        print(f"Unexpected error during IBC Index scraping: {e}")

    # Fallback for IBC Index if scraping fails or is not found
    if ibc_index is None:
        print("WARNING: IBC Index scraping failed. Using dynamic simulation for IBC Index.")
        last_ibc = getattr(get_commodities_and_indices, 'last_ibc', 65000.0)
        ibc_change = random.uniform(-500, 500)
        ibc_index = last_ibc + ibc_change
        setattr(get_commodities_and_indices, 'last_ibc', ibc_index)
    results['IBC_Index'] = ibc_index

    # EMBI & LATAM (Simulated - More Dynamic) - Existing logic
    print("NOTE: IBC Index, EMBI Risk, and LATAM Index data are simulated. Real-time, free, and reliable APIs for these are not readily available, and web scraping is highly fragile.")
    last_embi = getattr(get_commodities_and_indices, 'last_embi', 18000.0)
    embi_change = random.uniform(-50, 50)
    results['EMBI_Risk'] = last_embi + embi_change
    setattr(get_commodities_and_indices, 'last_embi', results['EMBI_Risk'])

    last_latam = getattr(get_commodities_and_indices, 'last_latam', 1200.0)
    latam_change = random.uniform(-10, 10)
    results['LATAM_Index'] = last_latam + latam_change
    setattr(get_commodities_and_indices, 'last_latam', results['LATAM_Index'])

    return results

from fredapi import Fred # This import needs to be added at the top

def get_inflation_data(fred_api_key):
    """
    Fetches Venezuela inflation data from FRED API.
    """
    print("NOTE: Attempting to fetch inflation data from FRED API.")
    try:
        fred = Fred(api_key=fred_api_key)
        # CPI for Venezuela (example series ID, might need to be verified)
        # Search for "Venezuela CPI" on FRED website to find the correct series ID
        # Example: CPALTT01VEA661N (Consumer Price Index: All Items for Venezuela)
        # Or: VECPIALLMINMEI (Consumer Price Index, All items, Monthly, Venezuela)
        # Let's use a common one, but user might need to adjust
        inflation_data = fred.get_series('VECPIALLMINMEI') # Example series ID
        if inflation_data is not None and not inflation_data.empty:
            latest_inflation = inflation_data.iloc[-1]
            print(f"Latest FRED Inflation Data: {latest_inflation}")
            return latest_inflation
        else:
            print("FRED API returned no inflation data. Using varied dummy data.")
            return round(2.0 + random.uniform(-1.0, 1.0), 2) # Varied dummy data
    except Exception as e:
        print(f"Error fetching inflation data from FRED API: {e}. Using varied dummy data.")
        return round(2.0 + random.uniform(-1.0, 1.0), 2) # Varied dummy data

def get_alternative_data(fred_api_key):
    """
    Fetches alternative data:
    - Inflation: Scraped from Observatorio Venezolano de Finanzas (OVF)
    - Google Trends: Search interest for "dolar paralelo"
    """
    results = {}
    # Monthly Inflation (using varied dummy data due to OVF instability)
    # print("NOTE: Monthly Inflation is using varied dummy data due to OVF scraping instability.")
    # results['Monthly_Inflation_OVF'] = round(2.0 + random.uniform(-1.0, 1.0), 2) # Varied dummy data
    
    # New: Fetch inflation from FRED API
    results['Monthly_Inflation_OVF'] = get_inflation_data(fred_api_key)

    results['Monthly_Inflation_OVF'] = get_inflation_data(fred_api_key)

    # Google Trends for "dolar paralelo"
    try:
        pytrends = TrendReq(hl='en-US', tz=360)
        pytrends.build_payload(kw_list=['dolar paralelo'], geo='VE', timeframe='today 1-m')
        trends_df = pytrends.interest_over_time()
        if not trends_df.empty:
            results['Google_Trends_Dolar'] = trends_df['dolar paralelo'].iloc[-1]
        else:
            results['Google_Trends_Dolar'] = round(75 + random.uniform(-15, 15), 2) # Varied dummy data
    except Exception as e:
        print(f"Could not fetch Google Trends data: {e}. Using varied dummy data.")
        results['Google_Trends_Dolar'] = round(75 + random.uniform(-15, 15), 2) # Varied dummy data

    return results

    print(f"DEBUG: get_alternative_data returning: {results}")
    return results

def get_economic_indicators(api_key_alpha, api_key_te):
    """
    Placeholder for fetching economic indicators (using more varied dummy data).
    """
    print("NOTE: get_economic_indicators is using more varied dummy data.")
    return {
        'Economic_Indicator_1': round(100.0 + random.uniform(-20, 20), 2), # Varied dummy value
        'Economic_Indicator_2': round(50.0 + random.uniform(-10, 10), 2)  # Varied dummy value
    }

def get_competitive_news_mentions(news_api_key):
    """
    Fetches recent news headlines mentioning the "Big Four" in Venezuela using NewsAPI.org.
    """
    results = {}
    query = "EY Venezuela OR PwC Venezuela OR Deloitte Venezuela OR KPMG Venezuela"
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=relevancy&apiKey={news_api_key}"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        articles = data.get('articles', [])
        headlines = []
        for article in articles:
            headlines.append({
                'title': article.get('title'),
                'source': article.get('source', {}).get('name')
            })
        results['Competitive_News_Mentions'] = headlines
    except requests.exceptions.Timeout:
        print("Error fetching competitive news mentions from NewsAPI.org: Request timed out. Using varied dummy data.")
        results['Competitive_News_Mentions'] = [
            {'title': f'Dummy News {random.randint(1, 100)} about EY', 'source': 'Simulated News Source'},
            {'title': f'PwC in the headlines {random.randint(1, 100)}', 'source': 'Simulated Press'},
            {'title': f'Deloitte\'s latest insights {random.randint(1, 100)}', 'source': 'Simulated Journal'}
        ] # Varied dummy data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching competitive news mentions from NewsAPI.org: {e}. Using varied dummy data.")
        results['Competitive_News_Mentions'] = [
            {'title': f'Dummy News {random.randint(1, 100)} about EY', 'source': 'Simulated News Source'},
            {'title': f'PwC in the headlines {random.randint(1, 100)}', 'source': 'Simulated Press'},
            {'title': f'Deloitte\'s latest insights {random.randint(1, 100)}', 'source': 'Simulated Journal'}
        ] # Varied dummy data
    print("NOTE: NewsAPI.org data requires a valid API key and may be subject to rate limits.")
    return results
    print("NOTE: NewsAPI.org data requires a valid API key and may be subject to rate limits.")
    return results

def get_competitive_brand_interest():
    """
    Compares public search interest for the "Big Four" brands in Venezuela using pytrends.
    """
    results = {}
    keywords = ["EY Venezuela", "PwC Venezuela", "Deloitte Venezuela", "KPMG Venezuela"]
    try:
        pytrends = TrendReq(hl='en-US', tz=360)
        pytrends.build_payload(kw_list=keywords, geo='VE', timeframe='today 3-m') # Last 3 months for better trend
        trends_df = pytrends.interest_over_time()
        if not trends_df.empty:
            # Drop the 'isPartial' column if it exists
            if 'isPartial' in trends_df.columns:
                trends_df = trends_df.drop(columns=['isPartial'])
            results['Competitive_Brand_Interest'] = trends_df.to_dict('records') # Convert to list of dicts for easier JSON serialization
        else:
            results['Competitive_Brand_Interest'] = {}
    except Exception as e:
        print(f"Could not fetch Google Trends data: {e}. Using dummy data.")
        # Fallback to dummy data if API fails
        dummy_data = []
        today = date.today()
        for i in range(90): # 90 days for 3 months
            current_date = today - timedelta(days=i)
            dummy_data.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'EY Venezuela': 50 + i % 10,
                'PwC Venezuela': 45 + (i + 2) % 10,
                'Deloitte Venezuela': 55 + (i + 5) % 10,
                'KPMG Venezuela': 40 + (i + 7) % 10
            })
        results['Competitive_Brand_Interest'] = dummy_data
    return results
    return results

def simulate_talent_market_scrape():
    """
    Simulates a scrape of LinkedIn Jobs to gauge hiring activity for the "Big Four".
    Returns a randomized but realistic dictionary of active job postings.
    """
    print("NOTE: Talent market data is simulated.")
    import random
    return {
        'EY_Job_Postings': random.randint(10, 30),
        'PwC_Job_Postings': random.randint(8, 25),
        'Deloitte_Job_Postings': random.randint(12, 35),
        'KPMG_Job_Postings': random.randint(7, 20)
    }

def fetch_all_data(historical_csv_path='historical_data.csv'):
    """
    Master function to acquire all data, combine it, and append to historical data.
    """
    print("--- Starting Data Acquisition Module ---")
    # --- API Key Placeholders ---
    # IMPORTANT: Replace with your actual API keys
    API_KEY_ALPHA_VANTAGE = "DPIQFHYTTJBXG5GY" # Provided by user
    API_KEY_TRADING_ECONOMICS = "YOUR_TRADING_ECONOMICS_API_KEY" # Placeholder
    API_KEY_NEWSAPI = "92599cece5f84986b96a118450b0edd4" # Provided by user
    FRED_API_KEY = "612df2427b77591bb911c3f7a843377e" # Provided by user

    # Fetch all data points
    economic_data = get_economic_indicators(API_KEY_ALPHA_VANTAGE, API_KEY_TRADING_ECONOMICS)
    alt_data = get_alternative_data(FRED_API_KEY)
    competitive_news = get_competitive_news_mentions(API_KEY_NEWSAPI)
    competitive_brand = get_competitive_brand_interest()
    talent_market = simulate_talent_market_scrape()

    # Combine into a single dictionary for today's data
    today_data = {
        **economic_data,
        **alt_data,
        'Competitive_News_Mentions': [competitive_news.get('Competitive_News_Mentions', [])],
        'Competitive_Brand_Interest': [competitive_brand.get('Competitive_Brand_Interest', {})],
        **talent_market
    }
    today_df = pd.DataFrame(today_data, index=[pd.to_datetime(date.today())])

    # Load and append to historical data
    try:
        historical_df = pd.read_csv(historical_csv_path, parse_dates=True)
        # Assuming the first column is the date and it's unnamed, rename it to 'Date'
        if historical_df.columns[0] == 'Unnamed: 0':
            historical_df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
        historical_df.set_index('Date', inplace=True)

        # Ensure the new row is only added if the index (date) doesn't already exist
        if today_df.index[0] not in historical_df.index:
            updated_df = pd.concat([historical_df, today_df])
            updated_df.to_csv(historical_csv_path) # Save back to CSV
            print(f"Successfully fetched and appended data for {date.today()}.")
        else:
            updated_df = historical_df
            print(f"Data for {date.today()} already exists. Using historical data.")
    except FileNotFoundError:
        print(f"Warning: '{historical_csv_path}' not found. Creating a new one with dummy historical data.")
        # Create a dummy dataframe with enough data for analysis
        dates = pd.to_datetime([date.today() - timedelta(days=i) for i in range(30)])
        dummy_data = {
            'Date': dates,
            'Parallel_Rate': [38 + random.uniform(-2, 2) for _ in range(30)],
            'Official_Rate': [36 + random.uniform(-1, 1) for _ in range(30)],
            'IBC_Index': [65000 + random.uniform(-500, 500) for _ in range(30)],
            'EMBI_Risk': [18000 + random.uniform(-100, 100) for _ in range(30)],
            'LATAM_Index': [1200 + random.uniform(-50, 50) for _ in range(30)],
            'Competitive_News_Mentions': [[] for _ in range(30)],
            'Competitive_Brand_Interest': [{} for _ in range(30)],
            'EY_Job_Postings': [random.randint(10, 30) for _ in range(30)],
            'PwC_Job_Postings': [random.randint(8, 25) for _ in range(30)],
            'Deloitte_Job_Postings': [random.randint(12, 35) for _ in range(30)],
            'KPMG_Job_Postings': [random.randint(7, 20) for _ in range(30)],
        }
        updated_df = pd.DataFrame(dummy_data)
        updated_df.set_index('Date', inplace=True)
        updated_df.to_csv(historical_csv_path)

    # Ensure LATAM_Index exists and fill any missing values
    if 'LATAM_Index' not in updated_df.columns:
        updated_df['LATAM_Index'] = 1200 # Default placeholder if not present
    
    # Forward-fill any missing values for analysis continuity
    updated_df.ffill(inplace=True)

    print("--- Data Acquisition Complete ---")
    return updated_df



# --- PART 2: ANALYTICS & VISUALIZATION MODULE ---

def generate_dashboard_analytics(df):
    """
    Master function to perform all econometric analyses and generate visualizations.
    """
    print("--- Starting Analytics & Visualization Module ---")
    df.dropna(inplace=True) # Ensure no NaNs are passed to models
    dashboard_output = {
        "Trends": {},
        "Projections": {},
        "Estimates": {},
        "Benchmarking": {},
        "Competitive_Landscape": {} # Added new section
    }

    # --- 2.1 Analysis for "Trends" Section ---
    try:
        # Moving Averages
        df['EMA_5'] = df['Parallel_Rate'].ewm(span=5, adjust=False).mean()
        df['EMA_20'] = df['Parallel_Rate'].ewm(span=20, adjust=False).mean()
        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(x=df.index, y=df['Parallel_Rate'], mode='lines', name='Parallel Rate'))
        fig_ma.add_trace(go.Scatter(x=df.index, y=df['EMA_5'], mode='lines', name='5-Day EMA'))
        fig_ma.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], mode='lines', name='20-Day EMA'))
        fig_ma.update_layout(title='Parallel Exchange Rate & Moving Averages', template='plotly_white')
        dashboard_output['Trends']['moving_averages_chart'] = fig_ma
    except Exception as e:
        print(f"Error generating Moving Averages chart: {e}")
        fig_placeholder = go.Figure()
        fig_placeholder.update_layout(title='Moving Averages Chart Not Available', template='plotly_white', annotations=[dict(text=f"Could not generate chart.<br>Error: {e}", showarrow=False)])
        dashboard_output['Trends']['moving_averages_chart'] = fig_placeholder

    try:
        # HP Filter
        cycle, trend = hpfilter(df['Parallel_Rate'], lamb=129600) # Lambda for daily data
        fig_hp = go.Figure()
        fig_hp.add_trace(go.Scatter(x=df.index, y=df['Parallel_Rate'], mode='lines', name='Original Series'))
        fig_hp.add_trace(go.Scatter(x=df.index, y=trend, mode='lines', name='HP Trend'))
        fig_hp.add_trace(go.Scatter(x=df.index, y=cycle, mode='lines', name='HP Cycle'))
        fig_hp.update_layout(title='Hodrick-Prescott Filter Decomposition', template='plotly_white')
        dashboard_output['Trends']['hp_filter_chart'] = fig_hp
    except Exception as e:
        print(f"Error generating HP Filter chart: {e}")
        fig_placeholder = go.Figure()
        fig_placeholder.update_layout(title='HP Filter Chart Not Available', template='plotly_white', annotations=[dict(text=f"Could not generate chart.<br>Error: {e}", showarrow=False)])
        dashboard_output['Trends']['hp_filter_chart'] = fig_placeholder

    try:
        # GARCH Volatility
        returns = df['IBC_Index'].pct_change().dropna() * 100
        garch_model_fit = arch_model(returns, vol='Garch', p=1, q=1).fit(disp='off')
        volatility = garch_model_fit.conditional_volatility
        fig_garch = go.Figure()
        fig_garch.add_trace(go.Scatter(x=volatility.index, y=volatility, mode='lines', name='GARCH Volatility'))
        fig_garch.update_layout(title='IBC Index GARCH(1,1) Volatility', template='plotly_white')
        dashboard_output['Trends']['garch_volatility_chart'] = fig_garch
        dashboard_output['Trends']['latest_volatility'] = volatility.iloc[-1]
    except Exception as e:
        print(f"Error generating GARCH Volatility chart: {e}")
        fig_placeholder = go.Figure()
        fig_placeholder.update_layout(title='GARCH Volatility Chart Not Available', template='plotly_white', annotations=[dict(text=f"Could not generate chart.<br>Error: {e}", showarrow=False)])
        dashboard_output['Trends']['garch_volatility_chart'] = fig_placeholder
        dashboard_output['Trends']['latest_volatility'] = 'N/A'

    # --- 2.2 Analysis for "Projections" Section ---
    try:
        # ARIMA Forecast
        arima_model = ARIMA(df['Parallel_Rate'], order=(5,1,0)).fit()
        forecast_arima = arima_model.get_forecast(steps=5)
        forecast_index = pd.date_range(start=df.index[-1], periods=6, freq='D')[1:]
        fig_arima = go.Figure()
        fig_arima.add_trace(go.Scatter(x=df.index, y=df['Parallel_Rate'], mode='lines', name='Historical'))
        fig_arima.add_trace(go.Scatter(x=forecast_index, y=forecast_arima.predicted_mean, mode='lines', name='Forecast'))
        fig_arima.add_trace(go.Scatter(x=forecast_index, y=forecast_arima.conf_int().iloc[:, 0], fill=None, mode='lines', line_color='rgba(0,100,80,0.2)', name='Lower CI'))
        fig_arima.add_trace(go.Scatter(x=forecast_index, y=forecast_arima.conf_int().iloc[:, 1], fill='tonexty', mode='lines', line_color='rgba(0,100,80,0.2)', name='Upper CI'))
        fig_arima.update_layout(title='ARIMA(5,1,0) Forecast for Parallel Rate', template='plotly_white')
        dashboard_output['Projections']['arima_forecast_chart'] = fig_arima
        dashboard_output['Projections']['arima_forecast_values'] = forecast_arima.predicted_mean
    except Exception as e:
        print(f"Error generating ARIMA Forecast chart: {e}")
        fig_placeholder = go.Figure()
        fig_placeholder.update_layout(title='ARIMA Forecast Chart Not Available', template='plotly_white', annotations=[dict(text=f"Could not generate chart.<br>Error: {e}", showarrow=False)])
        dashboard_output['Projections']['arima_forecast_chart'] = fig_placeholder
        dashboard_output['Projections']['arima_forecast_values'] = []

    try:
        # Holt-Winters Forecast
        hw_model = ExponentialSmoothing(df['IBC_Index'], trend='add', seasonal=None).fit()
        forecast_hw = hw_model.forecast(steps=5)
        fig_hw = go.Figure()
        fig_hw.add_trace(go.Scatter(x=df.index, y=df['IBC_Index'], mode='lines', name='Historical'))
        fig_hw.add_trace(go.Scatter(x=forecast_hw.index, y=forecast_hw, mode='lines', name='Forecast', line={'dash': 'dash'}))
        fig_hw.update_layout(title='Holt-Winters Forecast for IBC Index', template='plotly_white')
        dashboard_output['Projections']['holt_winters_chart'] = fig_hw
        dashboard_output['Projections']['holt_winters_values'] = forecast_hw
    except Exception as e:
        print(f"Error generating Holt-Winters Forecast chart: {e}")
        fig_placeholder = go.Figure()
        fig_placeholder.update_layout(title='Holt-Winters Forecast Chart Not Available', template='plotly_white', annotations=[dict(text=f"Could not generate chart.<br>Error: {e}", showarrow=False)])
        dashboard_output['Projections']['holt_winters_chart'] = fig_placeholder
        dashboard_output['Projections']['holt_winters_values'] = []

    # --- 2.3 Analysis for "Estimates" Section ---
    try:
        # Exchange Rate Spread
        df['Spread'] = ((df['Parallel_Rate'] - df['Official_Rate']) / df['Official_Rate']) * 100
        fig_spread = go.Figure()
        fig_spread.add_trace(go.Scatter(x=df.index, y=df['Spread'], mode='lines', name='Spread (%)'))
        fig_spread.update_layout(title='Exchange Rate Spread (Parallel vs. Official)', template='plotly_white')
        dashboard_output['Estimates']['spread_chart'] = fig_spread
        dashboard_output['Estimates']['latest_spread'] = df['Spread'].iloc[-1]
    except Exception as e:
        print(f"Error generating Exchange Rate Spread chart: {e}")
        fig_placeholder = go.Figure()
        fig_placeholder.update_layout(title='Exchange Rate Spread Chart Not Available', template='plotly_white', annotations=[dict(text=f"Could not generate chart.<br>Error: {e}", showarrow=False)])
        dashboard_output['Estimates']['spread_chart'] = fig_placeholder
        dashboard_output['Estimates']['latest_spread'] = 'N/A'

    # VAR Model & IRF
    var_data = df[['Parallel_Rate', 'IBC_Index', 'EMBI_Risk']].pct_change().dropna()
    print(f"Shape of var_data before VAR model: {var_data.shape}")
    print(f"var_data head:\n{var_data.head()}\n")
    # VAR model requires at least n_lags + 1 observations (here, 2 + 1 = 3)
    if var_data.shape[0] < 3 or var_data.empty:
        print("WARNING: Insufficient data for VAR model. Skipping VAR model and IRF.")
        dashboard_output['Estimates']['var_irf_info'] = "VAR model skipped due to insufficient data."
        fig_placeholder = go.Figure()
        fig_placeholder.update_layout(title='VAR IRF Chart Not Available', template='plotly_white', annotations=[dict(text="Could not generate chart.<br>Reason: Insufficient data for VAR model.", showarrow=False)])
        dashboard_output['Estimates']['var_irf_chart'] = fig_placeholder
    else:
        try:
            var_model = VAR(var_data).fit(2)
            irf = var_model.irf(10) # 10 periods ahead
            fig_irf = irf.plot(orth=False) # orth=False for standard errors
            # This plot is a matplotlib figure, would need conversion for a pure web dash.
            # For this script, we'll just note that it's generated.
            print("Generated VAR Impulse-Response Function plot (matplotlib).")
            dashboard_output['Estimates']['var_irf_info'] = "IRF plot generated. See console output or saved figure."
            # Let's create a simple plotly version for one IRF
            irf_pr_on_ibc = irf.irfs[1, 0, :]
            fig_irf_plotly = go.Figure()
            fig_irf_plotly.add_trace(go.Scatter(y=irf_pr_on_ibc, mode='lines', name='Response of IBC to Parallel Rate Shock'))
            fig_irf_plotly.update_layout(title='IRF: Response of IBC Index to Parallel Rate Shock', template='plotly_white')
            dashboard_output['Estimates']['var_irf_chart'] = fig_irf_plotly
        except np.linalg.LinAlgError as e:
            print(f"WARNING: VAR model estimation failed due to: {e}. Skipping IRF calculation.")
            dashboard_output['Estimates']['var_irf_info'] = "VAR model skipped due to data issues (matrix not positive definite)."
            fig_placeholder = go.Figure()
            fig_placeholder.update_layout(title='VAR IRF Chart Not Available', template='plotly_white', annotations=[dict(text=f"Could not generate chart.<br>Reason: {e}", showarrow=False)])
            dashboard_output['Estimates']['var_irf_chart'] = fig_placeholder
        except Exception as e: # Catch any other unexpected errors
            print(f"WARNING: Unexpected error during VAR model estimation: {e}. Skipping IRF calculation.")
            dashboard_output['Estimates']['var_irf_info'] = "VAR model skipped due to unexpected error."
            fig_placeholder = go.Figure()
            fig_placeholder.update_layout(title='VAR IRF Chart Not Available', template='plotly_white', annotations=[dict(text=f"Could not generate chart.<br>Reason: {e}", showarrow=False)])
            dashboard_output['Estimates']['var_irf_chart'] = fig_placeholder


    # --- 2.4 Analysis for "Benchmarking" Section ---
    try:
        # Performance vs. LATAM
        df_norm = df[['IBC_Index', 'LATAM_Index']].dropna()
        df_norm = (df_norm / df_norm.iloc[0]) * 100 # Normalize to 100
        fig_bench = go.Figure()
        fig_bench.add_trace(go.Scatter(x=df_norm.index, y=df_norm['IBC_Index'], mode='lines', name='IBC Index (Venezuela)'))
        fig_bench.add_trace(go.Scatter(x=df_norm.index, y=df_norm['LATAM_Index'], mode='lines', name='LATAM Index (Benchmark)'))
        fig_bench.update_layout(title='Performance: IBC vs. LATAM Index (Normalized)', template='plotly_white')
        dashboard_output['Benchmarking']['benchmark_chart'] = fig_bench
    except Exception as e:
        print(f"Error generating Performance vs. LATAM chart: {e}")
        fig_placeholder = go.Figure()
        fig_placeholder.update_layout(title='Performance vs. LATAM Chart Not Available', template='plotly_white', annotations=[dict(text=f"Could not generate chart.<br>Error: {e}", showarrow=False)])
        dashboard_output['Benchmarking']['benchmark_chart'] = fig_placeholder

    try:
        # Forecast Surprise
        one_day_forecast = arima_model.forecast(steps=1).iloc[0]
        actual_value = df['Parallel_Rate'].iloc[-1]
        surprise = actual_value - one_day_forecast
        dashboard_output['Benchmarking']['forecast_surprise'] = surprise
    except Exception as e:
        print(f"Error calculating Forecast Surprise: {e}")
        dashboard_output['Benchmarking']['forecast_surprise'] = 'N/A'

    # --- 2.5 NEW - Analysis for "Competitive Landscape" ---
    try:
        # Share of Voice (SoV) in Media
        news_mentions = df['Competitive_News_Mentions'].iloc[-1] if 'Competitive_News_Mentions' in df.columns and not df['Competitive_News_Mentions'].empty else []
        
        ey_mentions = sum(1 for item in news_mentions if item and ('EY' in item.get('title', '') or 'EY' in item.get('source', '')))
        pwc_mentions = sum(1 for item in news_mentions if item and ('PwC' in item.get('title', '') or 'PwC' in item.get('source', '')))
        deloitte_mentions = sum(1 for item in news_mentions if item and ('Deloitte' in item.get('title', '') or 'Deloitte' in item.get('source', '')))
        kpmg_mentions = sum(1 for item in news_mentions if item and ('KPMG' in item.get('title', '') or 'KPMG' in item.get('source', '')))

        total_mentions = ey_mentions + pwc_mentions + deloitte_mentions + kpmg_mentions
        
        if total_mentions > 0:
            sov_data = [
                ey_mentions / total_mentions * 100,
                pwc_mentions / total_mentions * 100,
                deloitte_mentions / total_mentions * 100,
                kpmg_mentions / total_mentions * 100,
            ]
            sov_labels = ['EY', 'PwC', 'Deloitte', 'KPMG']
            fig_sov = go.Figure(data=[go.Pie(labels=sov_labels, values=sov_data, hole=.3)])
            fig_sov.update_layout(title_text="Share of Voice en Medios", template='plotly_white')
            dashboard_output['Competitive_Landscape']['share_of_voice_chart'] = fig_sov
            dashboard_output['Competitive_Landscape']['share_of_voice_percentages'] = dict(zip(sov_labels, sov_data))
        else:
            dashboard_output['Competitive_Landscape']['share_of_voice_chart'] = "No data for Share of Voice."
            dashboard_output['Competitive_Landscape']['share_of_voice_percentages'] = {}
    except Exception as e:
        print(f"Error generating Share of Voice chart: {e}")
        fig_placeholder = go.Figure()
        fig_placeholder.update_layout(title='Share of Voice Chart Not Available', template='plotly_white', annotations=[dict(text=f"Could not generate chart.<br>Error: {e}", showarrow=False)])
        dashboard_output['Competitive_Landscape']['share_of_voice_chart'] = fig_placeholder
        dashboard_output['Competitive_Landscape']['share_of_voice_percentages'] = {}

    try:
        # Brand Interest Index
        brand_interest_data = df['Competitive_Brand_Interest'].iloc[-1] if 'Competitive_Brand_Interest' in df.columns and not df['Competitive_Brand_Interest'].empty else []
        
        if brand_interest_data:
            # Convert list of dicts to DataFrame
            brand_interest_df = pd.DataFrame(brand_interest_data)
            # Assuming the 'date' column is the index in the original pytrends df
            # Need to ensure the index is datetime for plotting
            if 'date' in brand_interest_df.columns:
                brand_interest_df['date'] = pd.to_datetime(brand_interest_df['date'])
                brand_interest_df = brand_interest_df.set_index('date')
            
            fig_brand_interest = go.Figure()
            for col in brand_interest_df.columns:
                fig_brand_interest.add_trace(go.Scatter(x=brand_interest_df.index, y=brand_interest_df[col], mode='lines', name=col))
            fig_brand_interest.update_layout(title_text="Índice de Interés de Marca (Google Trends)", template='plotly_white')
            dashboard_output['Competitive_Landscape']['brand_interest_chart'] = fig_brand_interest
        else:
            dashboard_output['Competitive_Landscape']['brand_interest_chart'] = "No data for Brand Interest Index."
    except Exception as e:
        print(f"Error generating Brand Interest Index chart: {e}")
        fig_placeholder = go.Figure()
        fig_placeholder.update_layout(title='Brand Interest Index Chart Not Available', template='plotly_white', annotations=[dict(text=f"Could not generate chart.<br>Error: {e}", showarrow=False)])
        dashboard_output['Competitive_Landscape']['brand_interest_chart'] = fig_placeholder

    try:
        # Talent Acquisition Index
        talent_data = df['EY_Job_Postings'].iloc[-1] if 'EY_Job_Postings' in df.columns else None # Assuming these are directly in df
        
        if talent_data is not None: # Check if any talent data is available
            job_postings = {
                'EY': df['EY_Job_Postings'].iloc[-1] if 'EY_Job_Postings' in df.columns else 0,
                'PwC': df['PwC_Job_Postings'].iloc[-1] if 'PwC_Job_Postings' in df.columns else 0,
                'Deloitte': df['Deloitte_Job_Postings'].iloc[-1] if 'Deloitte_Job_Postings' in df.columns else 0,
                'KPMG': df['KPMG_Job_Postings'].iloc[-1] if 'KPMG_Job_Postings' in df.columns else 0,
            }
            firms = list(job_postings.keys())
            counts = list(job_postings.values())

            fig_talent = go.Figure(data=[go.Bar(x=firms, y=counts)])
            fig_talent.update_layout(title_text="Índice de Contratación Activa (Proxy)", template='plotly_white')
            dashboard_output['Competitive_Landscape']['talent_acquisition_chart'] = fig_talent
        else:
            dashboard_output['Competitive_Landscape']['talent_acquisition_chart'] = "No data for Talent Acquisition Index."
    except Exception as e:
        print(f"Error generating Talent Acquisition Index chart: {e}")
        fig_placeholder = go.Figure()
        fig_placeholder.update_layout(title='Talent Acquisition Index Chart Not Available', template='plotly_white', annotations=[dict(text=f"Could not generate chart.<br>Error: {e}", showarrow=False)])
        dashboard_output['Competitive_Landscape']['talent_acquisition_chart'] = fig_placeholder

    print("--- Analytics & Visualization Complete ---")
    return dashboard_output

# --- PART 3: MAIN EXECUTION BLOCK ---

if __name__ == "__main__":
    print("=====================================================")
    print("EY VENEZUELA ANALYTICS DASHBOARD - ENGINE START")
    print("=====================================================")

    # 1. Get the complete, up-to-date DataFrame
    master_df = fetch_all_data(historical_csv_path='historical_data.csv')

    # 2. Pass the DataFrame to the analytics module
    # Ensure there's enough data to run the models
    if len(master_df) > 20: # A reasonable threshold for time series models
        final_dashboard_data = generate_dashboard_analytics(master_df)

        # 3. Print the final structured output
        print("\n--- FINAL DASHBOARD OUTPUT (STRUCTURE) ---")
        for section, content in final_dashboard_data.items():
            print(f"\n[{section}]")
            for key, value in content.items():
                if isinstance(value, go.Figure):
                    print(f"  - {key}: Plotly Figure")
                else:
                    print(f"  - {key}: {value}")
        print("\n=====================================================")
        print("ENGINE RUN COMPLETE")
        print("=====================================================")

    else:
        print("Not enough historical data for analysis.")