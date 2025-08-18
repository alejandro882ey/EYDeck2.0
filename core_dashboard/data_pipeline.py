
import pandas as pd
import numpy as np
import datetime
import requests
from bs4 import BeautifulSoup
import json
import random # For simulating data

# --- Part 1: Fetching Fundamental Daily Data ---

# 1.1. Exchange Rates (Parallel and Official)
def fetch_exchange_rates():
    """
    Simulates fetching daily values for parallel and official USD/VES rates
    from pydolarve.org API.
    """
    print("Fetching exchange rates...")
    try:
        # Simulate API response
        # In a real scenario, you would make a requests.get() call here
        # response = requests.get("https://pydolarve.org/api/v1/dollar/")
        # response.raise_for_status()
        # data = response.json()

        # Mock data based on typical pydolarve.org structure
        mock_data = {
            "status": "success",
            "message": "Data fetched successfully",
            "last_update": datetime.datetime.now().isoformat(),
            "monitor": {
                "bcv": {"promedio": round(random.uniform(35.0, 40.0), 4)},
                "enparalelovzla": {"promedio": round(random.uniform(38.0, 45.0), 4)},
                "dolartoday": {"promedio": round(random.uniform(38.5, 46.0), 4)},
            }
        }
        
        # Extract values
        official_rate = mock_data["monitor"]["bcv"]["promedio"]
        # For parallel, take an average or specific one, here using enparalelovzla
        parallel_rate = mock_data["monitor"]["enparalelovzla"]["promedio"] 
        
        print(f"  Official Rate (mock): {official_rate}")
        print(f"  Parallel Rate (mock): {parallel_rate}")
        return {'Parallel_Rate': parallel_rate, 'Official_Rate': official_rate}
    except Exception as e:
        print(f"Error fetching exchange rates (mock): {e}")
        return None

# 1.2. Oil Price (Brent Crude)
def fetch_oil_price(api_key):
    """
    Simulates fetching the latest daily price for Brent crude oil from Alpha Vantage API.
    """
    print("Fetching oil price...")
    # API_KEY = api_key # Use the provided API key
    try:
        # Simulate API response
        # response = requests.get(f"https://www.alphavantage.co/query?function=BRENT&interval=daily&apikey={api_key}")
        # response.raise_for_status()
        # data = response.json()

        # Mock data based on typical Alpha Vantage BRENT structure
        mock_data = {
            "name": "Brent Crude Oil",
            "interval": "daily",
            "unit": "USD",
            "data": [
                {"date": (datetime.date.today() - datetime.timedelta(days=0)).isoformat(), "value": round(random.uniform(80.0, 90.0), 2)},
                {"date": (datetime.date.today() - datetime.timedelta(days=1)).isoformat(), "value": round(random.uniform(80.0, 90.0), 2)},
            ]
        }
        oil_price = float(mock_data["data"][0]["value"])
        print(f"  Oil Price Brent (mock): {oil_price}")
        return {'Oil_Price_Brent': oil_price}
    except Exception as e:
        print(f"Error fetching oil price (mock): {e}")
        return None

# 1.3. Caracas Stock Index (IBC) & EMBI Risk
def fetch_professional_data(api_key):
    """
    Simulates fetching data from a professional service like Trading Economics API
    for IBC Index and EMBI Risk.
    """
    print("Fetching professional data (IBC, EMBI)...")
    # API_KEY = api_key # Use the provided API key
    try:
        # Simulate API calls
        # ibc_response = requests.get(f"https://api.tradingeconomics.com/markets/symbol/IBVC:IND?c={api_key}")
        # embi_response = requests.get(f"https://api.tradingeconomics.com/markets/indicator/EMBI?country=Venezuela&c={api_key}")
        # ibc_response.raise_for_status()
        # embi_response.raise_for_status()

        # Hardcoded mock data as requested
        ibc_index = 65000.0 + random.uniform(-1000, 1000)
        embi_risk = 18000.0 + random.uniform(-500, 500)

        print(f"  IBC Index (mock): {ibc_index}")
        print(f"  EMBI Risk (mock): {embi_risk}")
        return {'IBC_Index': ibc_index, 'EMBI_Risk': embi_risk}
    except Exception as e:
        print(f"Error fetching professional data (mock): {e}")
        return None

# --- Part 2: Fetching Macroeconomic & Alternative Data ---

# 2.1. Inflation Data (Web Scraping)
def scrape_inflation_ovf():
    """
    Simulates scraping inflation data from Observatorio Venezolano de Finanzas (OVF).
    """
    print("Scraping OVF inflation data...")
    # URL = "https://observatoriovenezolanodefinanzas.com/"
    try:
        # Simulate web scraping
        # response = requests.get(URL)
        # soup = BeautifulSoup(response.text, 'html.parser')
        # Find elements containing inflation data (this part is highly dependent on website structure)
        # For example:
        # monthly_tag = soup.find('div', class_='monthly-inflation')
        # ytd_tag = soup.find('div', class_='ytd-inflation')
        # monthly_value = float(monthly_tag.text.replace('%', '').replace(',', '.'))
        # ytd_value = float(ytd_tag.text.replace('%', '').replace(',', '.'))

        # Hardcoded mock data
        monthly_inflation = round(random.uniform(1.5, 3.0), 2)
        ytd_inflation = round(random.uniform(15.0, 30.0), 2)

        print(f"  Monthly Inflation OVF (mock): {monthly_inflation}%")
        print(f"  YTD Inflation OVF (mock): {ytd_inflation}%")
        return {'Monthly_Inflation_OVF': monthly_inflation, 'YTD_Inflation_OVF': ytd_inflation}
    except Exception as e:
        print(f"Error scraping OVF inflation data (mock): {e}")
        return None

# 2.2. Google Trends Data
def fetch_google_trends():
    """
    Simulates fetching Google Trends data for "dolar paralelo" using pytrends.
    """
    print("Fetching Google Trends data...")
    try:
        # from pytrends.request import TrendReq
        # pytrends = TrendReq(hl='es-VE', tz=240)
        # kw_list = ["dolar paralelo"]
        # pytrends.build_payload(kw_list, cat=0, timeframe='now 7-d', geo='VE', gprop='')
        # df_trends = pytrends.interest_over_time()
        # interest_value = df_trends[kw_list[0]].iloc[-1] if not df_trends.empty else 0

        # Hardcoded mock data
        interest_value = random.randint(40, 80)

        print(f"  Google Trends 'dolar paralelo' (mock): {interest_value}")
        return {'Google_Trends_Dolar': interest_value}
    except Exception as e:
        print(f"Error fetching Google Trends data (mock): {e}")
        return None

# --- Part 3: Assembling the Final DataFrame ---
def get_daily_economic_data(alpha_vantage_api_key='YOUR_ALPHA_VANTAGE_KEY', trading_economics_api_key='YOUR_TRADING_ECONOMICS_KEY'):
    """
    Calls all data-fetching functions and consolidates data into a single pandas DataFrame.
    """
    all_data = {}

    # Fetch data from each source
    exchange_rates = fetch_exchange_rates()
    if exchange_rates:
        all_data.update(exchange_rates)

    oil_price = fetch_oil_price(alpha_vantage_api_key)
    if oil_price:
        all_data.update(oil_price)

    professional_data = fetch_professional_data(trading_economics_api_key)
    if professional_data:
        all_data.update(professional_data)

    inflation_data = scrape_inflation_ovf()
    if inflation_data:
        all_data.update(inflation_data)

    google_trends_data = fetch_google_trends()
    if google_trends_data:
        all_data.update(google_trends_data)

    # Add today's date as the index
    today = datetime.date.today()
    final_df = pd.DataFrame([all_data], index=[today])
    
    return final_df

if __name__ == "__main__":
    # Define API key placeholders
    ALPHA_VANTAGE_API_KEY = 'YOUR_ALPHA_VANTAGE_KEY_HERE'
    TRADING_ECONOMICS_API_KEY = 'YOUR_TRADING_ECONOMICS_KEY_HERE'

    print("--- Starting Data Pipeline ---")
    daily_data_df = get_daily_economic_data(ALPHA_VANTAGE_API_KEY, TRADING_ECONOMICS_API_KEY)
    print("
--- Final Daily Economic Data DataFrame ---")
    print(daily_data_df)
    print("
--- Data Pipeline Finished ---")
