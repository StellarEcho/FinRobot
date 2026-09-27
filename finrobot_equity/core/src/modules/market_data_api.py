#!/usr/bin/env python
# coding: utf-8

import yfinance as yf
import pandas as pd
import requests
import datetime
import os

# Assuming common_utils.py is in the same parent directory (src/modules)
from .common_utils import get_api_key, load_config 

# FMP retired its legacy /api/v3 and /api/v4 endpoints on 2025-08-31. Accounts
# created after that date only get the replacement "stable" API, which is
# query-string based (symbol=...) instead of path based (/NVDA).
FMP_BASE_URL = "https://financialmodelingprep.com/stable"

# The stable API also renamed a number of fields. Several downstream modules
# (financial_data_processor, valuation_engine, enhanced_chart_generator,
# create_equity_report, generate_pdf_report) still read the legacy names, so we
# keep the legacy names as aliases at the source instead of rewriting every
# consumer. Maps legacy name -> stable name.
RATIOS_ALIASES = {
    "priceEarningsRatio": "priceToEarningsRatio",
    "debtEquityRatio": "debtToEquityRatio",
    "priceEarningsToGrowthRatio": "priceToEarningsGrowthRatio",
    "payoutRatio": "dividendPayoutRatio",
    "debtRatio": "debtToAssetsRatio",
    "interestCoverage": "interestCoverageRatio",
}
KEY_METRICS_ALIASES = {
    "enterpriseValueOverEBITDA": "evToEBITDA",
}
# Fields the stable API moved from one endpoint to the other.
RATIOS_FROM_KEY_METRICS = ["returnOnEquity"]
KEY_METRICS_FROM_RATIOS = {
    "peRatio": "priceToEarningsRatio",
    "pbRatio": "priceToBookRatio",
}
# Company profile renames between the legacy and stable APIs.
PROFILE_ALIASES = {
    "mktCap": "marketCap",
    "volAvg": "averageVolume",
    "exchangeShortName": "exchange",
    "lastDiv": "lastDividend",
}


def _add_alias_columns(df: pd.DataFrame | None, aliases: dict) -> pd.DataFrame | None:
    """Copies stable-API fields onto their legacy column names."""
    if df is None or df.empty:
        return df
    for legacy, stable in aliases.items():
        if legacy not in df.columns and stable in df.columns:
            df[legacy] = df[stable]
    return df


def _cross_fill_moved_fields(ratios_df: pd.DataFrame | None, key_metrics_df: pd.DataFrame | None) -> None:
    """Copies fields that the stable API moved between ratios and key-metrics."""
    for frame in (ratios_df, key_metrics_df):
        if frame is None or frame.empty or "date" not in frame.columns:
            return

    key_metrics_by_date = key_metrics_df.drop_duplicates(subset="date").set_index("date")
    for column in RATIOS_FROM_KEY_METRICS:
        if column not in ratios_df.columns and column in key_metrics_by_date.columns:
            ratios_df[column] = ratios_df["date"].map(key_metrics_by_date[column])

    ratios_by_date = ratios_df.drop_duplicates(subset="date").set_index("date")
    for legacy, stable in KEY_METRICS_FROM_RATIOS.items():
        if legacy not in key_metrics_df.columns and stable in ratios_by_date.columns:
            key_metrics_df[legacy] = key_metrics_df["date"].map(ratios_by_date[stable])


def fetch_yfinance_volume(ticker: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    """Fetches historical trading volume data using yfinance."""
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if stock_data.empty:
            print(f"No data returned from yfinance for {ticker} between {start_date} and {end_date}")
            return None
        stock_data = stock_data[["Volume"]]
        stock_data.reset_index(inplace=True)
        stock_data["Date"] = pd.to_datetime(stock_data["Date"])
        return stock_data
    except Exception as e:
        print(f"Error fetching yfinance volume for {ticker}: {e}")
        return None

def fetch_fmp_enterprise_value(ticker: str, api_key: str, limit: int = 2000) -> pd.DataFrame | None:
    """Fetches historical enterprise value from Financial Modeling Prep API."""
    url = f"{FMP_BASE_URL}/enterprise-values?symbol={ticker}&limit={limit}&apikey={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if not data:
            print(f"No EV data returned from FMP for {ticker}.")
            return None
        
        df = pd.DataFrame(data)
        if "date" not in df.columns or "enterpriseValue" not in df.columns:
            print(f"FMP EV data for {ticker} does not contain expected columns ('date', 'enterpriseValue'). Response: {data}")
            return None
            
        df["date"] = pd.to_datetime(df["date"])
        df = df[["date", "enterpriseValue"]].sort_values(by="date").reset_index(drop=True)
        return df
    except requests.exceptions.RequestException as e:
        print(f"Error fetching FMP EV for {ticker}: {e}")
        return None
    except (KeyError, ValueError) as e:
        print(f"Error processing FMP EV data for {ticker}: {e}. Response: {data if 'data' in locals() else 'N/A'}")
        return None

def get_fmp_ratios_and_key_metrics(ticker: str, api_key: str, period: str = "annual", limit: int = 5) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Fetches financial ratios and key metrics from FMP API."""
    ratios_df, key_metrics_df = None, None
    try:
        # Ratios
        ratios_url = f"{FMP_BASE_URL}/ratios?symbol={ticker}&period={period}&limit={limit}&apikey={api_key}"
        response_ratios = requests.get(ratios_url)
        response_ratios.raise_for_status()
        ratios_data = response_ratios.json()
        if ratios_data:
            ratios_df = pd.DataFrame(ratios_data)
            ratios_df["date"] = pd.to_datetime(ratios_df["date"])
            ratios_df["year"] = ratios_df["date"].dt.year

        # Key Metrics
        key_metrics_url = f"{FMP_BASE_URL}/key-metrics?symbol={ticker}&period={period}&limit={limit}&apikey={api_key}"
        response_key_metrics = requests.get(key_metrics_url)
        response_key_metrics.raise_for_status()
        key_metrics_data = response_key_metrics.json()
        if key_metrics_data:
            key_metrics_df = pd.DataFrame(key_metrics_data)
            key_metrics_df["date"] = pd.to_datetime(key_metrics_df["date"])
            key_metrics_df["year"] = key_metrics_df["date"].dt.year

        # Keep legacy field names working for downstream consumers.
        ratios_df = _add_alias_columns(ratios_df, RATIOS_ALIASES)
        key_metrics_df = _add_alias_columns(key_metrics_df, KEY_METRICS_ALIASES)
        _cross_fill_moved_fields(ratios_df, key_metrics_df)
        # The stable ratios endpoint dropped the v3 calendarYear column.
        if ratios_df is not None and not ratios_df.empty and "calendarYear" not in ratios_df.columns:
            ratios_df["calendarYear"] = ratios_df["date"].dt.year

    except requests.exceptions.RequestException as e:
        print(f"Error fetching FMP ratios/key metrics for {ticker}: {e}")
    except (KeyError, ValueError) as e:
        print(f"Error processing FMP ratios/key metrics data for {ticker}: {e}")
        
    return ratios_df, key_metrics_df

def get_fmp_income_statement(ticker: str, api_key: str, period: str = "annual", limit: int = 5) -> pd.DataFrame | None:
    """Fetches income statement data from FMP API."""
    try:
        url = f"{FMP_BASE_URL}/income-statement?symbol={ticker}&period={period}&limit={limit}&apikey={api_key}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if not data:
            print(f"No income statement data from FMP for {ticker}.")
            return None
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        return df
    except requests.exceptions.RequestException as e:
        print(f"Error fetching FMP income statement for {ticker}: {e}")
        return None
    except (KeyError, ValueError) as e:
        print(f"Error processing FMP income statement data for {ticker}: {e}")
        return None

def get_fmp_balance_sheet(ticker: str, api_key: str, period: str = "annual", limit: int = 5) -> pd.DataFrame | None:
    """Fetches balance sheet data from FMP API."""
    try:
        url = f"{FMP_BASE_URL}/balance-sheet-statement?symbol={ticker}&period={period}&limit={limit}&apikey={api_key}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if not data:
            print(f"No balance sheet data from FMP for {ticker}.")
            return None
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        return df
    except requests.exceptions.RequestException as e:
        print(f"Error fetching FMP balance sheet for {ticker}: {e}")
        return None
    except (KeyError, ValueError) as e:
        print(f"Error processing FMP balance sheet data for {ticker}: {e}")
        return None

def get_fmp_cash_flow_statement(ticker: str, api_key: str, period: str = "annual", limit: int = 5) -> pd.DataFrame | None:
    """Fetches cash flow statement data from FMP API."""
    try:
        url = f"{FMP_BASE_URL}/cash-flow-statement?symbol={ticker}&period={period}&limit={limit}&apikey={api_key}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if not data:
            print(f"No cash flow data from FMP for {ticker}.")
            return None
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        return df
    except requests.exceptions.RequestException as e:
        print(f"Error fetching FMP cash flow for {ticker}: {e}")
        return None
    except (KeyError, ValueError) as e:
        print(f"Error processing FMP cash flow data for {ticker}: {e}")
        return None

def get_comprehensive_financial_data(ticker: str, api_key: str, period: str = "annual", limit: int = 5) -> dict:
    """Fetches all three financial statements for a company."""
    print(f"Fetching comprehensive financial data for {ticker}...")
    
    financial_data = {
        'income_statement': get_fmp_income_statement(ticker, api_key, period, limit),
        'balance_sheet': get_fmp_balance_sheet(ticker, api_key, period, limit),
        'cash_flow': get_fmp_cash_flow_statement(ticker, api_key, period, limit),
        'ratios': None,
        'key_metrics': None
    }
    
    # Also get ratios and key metrics
    ratios_df, key_metrics_df = get_fmp_ratios_and_key_metrics(ticker, api_key, period, limit)
    financial_data['ratios'] = ratios_df
    financial_data['key_metrics'] = key_metrics_df

    # The stable ratios endpoint dropped cashFlowToDebtRatio
    # (operating cash flow / total debt); derive it from the statements above.
    if ratios_df is not None and not ratios_df.empty and "cashFlowToDebtRatio" not in ratios_df.columns:
        cash_flow_df = financial_data['cash_flow']
        balance_df = financial_data['balance_sheet']
        try:
            if cash_flow_df is not None and balance_df is not None \
                    and not cash_flow_df.empty and not balance_df.empty:
                operating_cash_flow = cash_flow_df.drop_duplicates(subset="year").set_index("year")["operatingCashFlow"]
                total_debt = balance_df.drop_duplicates(subset="year").set_index("year")["totalDebt"]
                ratios_df["cashFlowToDebtRatio"] = (
                    ratios_df["year"].map(operating_cash_flow) / ratios_df["year"].map(total_debt)
                )
        except Exception as e:
            print(f"Could not derive cashFlowToDebtRatio for {ticker}: {e}")

    return financial_data

def combine_peer_financial_data(tickers: list[str], api_key: str, years_limit: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combines EBITDA and EV/EBITDA for a list of peer tickers."""
    all_peers_data = {}
    for ticker in tickers:
        income_df = get_fmp_income_statement(ticker, api_key, limit=years_limit)
        _, key_metrics_df = get_fmp_ratios_and_key_metrics(ticker, api_key, limit=years_limit)
        
        ticker_data = {}
        if income_df is not None and not income_df.empty:
            for _, row in income_df.iterrows():
                year = row["year"]
                if year not in ticker_data: ticker_data[year] = {}
                ticker_data[year]["EBITDA"] = row.get("ebitda")

        if key_metrics_df is not None and not key_metrics_df.empty:
            for _, row in key_metrics_df.iterrows():
                year = row["year"]
                if year not in ticker_data: ticker_data[year] = {}
                ticker_data[year]["EV/EBITDA"] = row.get("enterpriseValueOverEBITDA")
        
        if ticker_data:
            all_peers_data[ticker] = ticker_data

    ebitda_records = []
    for ticker, yearly_data in all_peers_data.items():
        for year, metrics in yearly_data.items():
            if "EBITDA" in metrics and metrics["EBITDA"] is not None:
                ebitda_records.append({"ticker": ticker, "year": year, "EBITDA": metrics["EBITDA"]})
    df_ebitda_all = pd.DataFrame(ebitda_records)
    df_ebitda_pivot = pd.DataFrame()
    if not df_ebitda_all.empty:
        df_ebitda_pivot = df_ebitda_all.pivot(index="year", columns="ticker", values="EBITDA").sort_index()

    ev_ebitda_records = []
    for ticker, yearly_data in all_peers_data.items():
        for year, metrics in yearly_data.items():
            if "EV/EBITDA" in metrics and metrics["EV/EBITDA"] is not None:
                ev_ebitda_records.append({"ticker": ticker, "year": year, "EV/EBITDA": metrics["EV/EBITDA"]})
    df_ev_ebitda_all = pd.DataFrame(ev_ebitda_records)
    df_ev_ebitda_pivot = pd.DataFrame()
    if not df_ev_ebitda_all.empty:
        df_ev_ebitda_pivot = df_ev_ebitda_all.pivot(index="year", columns="ticker", values="EV/EBITDA").sort_index()
        
    return df_ebitda_pivot, df_ev_ebitda_pivot

def project_ebitda_for_peers(df_ebitda_historical: pd.DataFrame, num_projection_years: int = 1) -> pd.DataFrame:
    """Projects EBITDA for future years based on average historical YoY growth."""
    df_projected = df_ebitda_historical.copy()
    if df_projected.empty:
        return df_projected

    last_historical_year = df_projected.index.max()
    
    for company in df_projected.columns:
        historical_values = df_projected[company].dropna()
        if len(historical_values) < 2:
            print(f"Not enough historical EBITDA data for {company} to project.")
            continue
        
        growth_rates = historical_values.pct_change().dropna()
        if growth_rates.empty or all(g == 0 for g in growth_rates):
            avg_growth_rate = 0 
        else:
            avg_growth_rate = growth_rates.mean()

        current_ebitda = historical_values.iloc[-1]
        for i in range(1, num_projection_years + 1):
            projection_year = last_historical_year + i
            current_ebitda = current_ebitda * (1 + avg_growth_rate)
            df_projected.loc[projection_year, company] = current_ebitda
            
    return df_projected.sort_index()

def get_fmp_current_price(ticker: str, api_key: str) -> float | None:
    """Fetches the latest stock price from Financial Modeling Prep API."""
    url = f"{FMP_BASE_URL}/quote?symbol={ticker}&apikey={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data and isinstance(data, list) and len(data) > 0:
            quote_data = data[0]
            if "price" in quote_data and quote_data["price"] is not None:
                return float(quote_data["price"])
            elif quote_data.get("previousClose") is not None:
                print(f"Current price not available for {ticker}, using previous close: {quote_data['previousClose']}")
                return float(quote_data["previousClose"])
            print(f"Price data not found in FMP quote for {ticker}. Response: {quote_data}")
            return None
        else:
            print(f"No data or unexpected format returned from FMP /quote for {ticker}. Response: {data}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching FMP current price for {ticker}: {e}")
        return None
    except (KeyError, ValueError, TypeError) as e:
        print(f"Error processing FMP current price data for {ticker}: {e}. Response: {data if 'data' in locals() else 'N/A'}")
        return None

def get_analyst_insights(ticker: str, api_key: str = None) -> tuple[str | None, float | None]:
    """
    Fetches analyst rating and target price using FMP API.
    
    Args:
        ticker: Stock ticker symbol
        api_key: FMP API key (optional, will try to use existing functions if not provided)
    
    Returns:
        Tuple of (rating, target_price)
    """
    rating = None
    target_price = None
    
    try:
        # Try to get rating from FMP
        if api_key:
            rating = get_fmp_analyst_rating(ticker, api_key)
            target_price = get_fmp_target_price(ticker, api_key)
            
            if rating:
                print(f"[INFO] For {ticker} - Rating: {rating}")
            if target_price:
                print(f"[INFO] For {ticker} - Target Price: {target_price}")
        else:
            print(f"[WARN] No API key provided for get_analyst_insights. Skipping.")
            
    except Exception as e:
        print(f"[ERROR] Error in get_analyst_insights for {ticker}: {e}")
        
    return rating, target_price

def get_fmp_target_price(ticker: str, api_key: str) -> float | None:
    """Fetches the latest analyst target price from Financial Modeling Prep API."""
    url = f"{FMP_BASE_URL}/price-target-consensus?symbol={ticker}&apikey={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data and isinstance(data, list) and len(data) > 0:
            consensus = data[0]
            # The stable endpoint returns a single consensus record rather than
            # a time series of individual analyst reports.
            for field in ("targetConsensus", "targetMedian"):
                if consensus.get(field) is not None:
                    return float(consensus[field])
            print(f"Price target not found in the latest FMP data for {ticker}. Data: {consensus}")
            return None
        else:
            print(f"No target price data or unexpected format returned from FMP for {ticker}. Response: {data}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching FMP target price for {ticker}: {e}")
        return None
    except (KeyError, ValueError, TypeError) as e:
        print(f"Error processing FMP target price data for {ticker}: {e}. Response: {data if 'data' in locals() else 'N/A'}")
        return None

def get_fmp_analyst_rating(ticker: str, api_key: str) -> str | None:
    """Fetches the latest analyst rating (e.g., Buy, Hold, Sell) from Financial Modeling Prep API."""
    # The stable API exposes analyst grades under /grades; /upgrades-downgrades
    # exists but returns an empty result set.
    url = f"{FMP_BASE_URL}/grades?symbol={ticker}&limit=10&apikey={api_key}"
    try:
        response = requests.get(url)
        if response.status_code in (401, 402, 403, 404):
            return _get_fmp_rating_consensus(ticker, api_key)
        response.raise_for_status()
        data = response.json()
        if data and isinstance(data, list) and len(data) > 0:
            for item in data:
                try:
                    # The stable endpoint renamed publishedDate to date.
                    item['parsedDate'] = pd.to_datetime(item.get('publishedDate') or item.get('date'))
                except Exception as e:
                    print(f"Warning: Could not parse the publication date for {ticker}: {e}")
                    item['parsedDate'] = None
            
            valid_data = [item for item in data if item['parsedDate'] is not None]

            if not valid_data:
                print(f"No valid published dates found in FMP analyst rating data for {ticker}.")
                return _get_fmp_rating_consensus(ticker, api_key)

            latest_rating_info = sorted(valid_data, key=lambda x: x['parsedDate'], reverse=True)[0]
            
            if "newGrade" in latest_rating_info and latest_rating_info["newGrade"]:
                return str(latest_rating_info["newGrade"])
            elif "action" in latest_rating_info and latest_rating_info["action"]:
                 print(f"newGrade not found, using action: {latest_rating_info['action']} for {ticker}")
                 return str(latest_rating_info["action"])
            else:
                print(f"Analyst rating (newGrade or action) not found in the latest FMP data for {ticker}. Data: {latest_rating_info}")
                return _get_fmp_rating_consensus(ticker, api_key)
        else:
            print(f"No analyst rating data or unexpected format returned from FMP for {ticker}. Response: {data}")
            return _get_fmp_rating_consensus(ticker, api_key)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching FMP analyst rating for {ticker}: {e}")
        return None
    except (KeyError, ValueError, TypeError) as e:
        print(f"Error processing FMP analyst rating data for {ticker}: {e}. Response: {data if 'data' in locals() else 'N/A'}")
        return None


def _get_fmp_rating_consensus(ticker: str, api_key: str) -> str | None:
    """Falls back to the aggregate analyst consensus when no grade history exists."""
    url = f"{FMP_BASE_URL}/grades-consensus?symbol={ticker}&apikey={api_key}"
    try:
        response = requests.get(url)
        if response.status_code in (401, 402, 403, 404):
            return None
        response.raise_for_status()
        data = response.json()
        if data and isinstance(data, list) and data[0].get("consensus"):
            return str(data[0]["consensus"])
    except Exception as e:  # noqa: BLE001
        print(f"Could not fetch FMP rating consensus for {ticker}: {e}")
    return None

def get_fmp_company_profile(ticker: str, api_key: str) -> dict | None:
    """Fetches comprehensive company profile data from FMP API."""
    url = f"{FMP_BASE_URL}/profile?symbol={ticker}&apikey={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data and isinstance(data, list) and len(data) > 0:
            profile = data[0]  # Profile returns a list with one item
            for legacy, stable in PROFILE_ALIASES.items():
                if legacy not in profile and stable in profile:
                    profile[legacy] = profile[stable]
            return profile
        else:
            print(f"No profile data returned for {ticker}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching company profile for {ticker}: {e}")
        return None
    except (KeyError, ValueError, TypeError) as e:
        print(f"Error processing company profile data for {ticker}: {e}")
        return None

def get_fmp_market_cap(ticker: str, api_key: str) -> float | None:
    """Fetches current market capitalization from FMP API."""
    url = f"{FMP_BASE_URL}/market-capitalization?symbol={ticker}&apikey={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data and isinstance(data, list) and len(data) > 0:
            market_cap = data[0].get('marketCap')
            return float(market_cap) if market_cap else None
        else:
            print(f"No market cap data returned for {ticker}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching market cap for {ticker}: {e}")
        return None
    except (KeyError, ValueError, TypeError) as e:
        print(f"Error processing market cap data for {ticker}: {e}")
        return None

def get_comprehensive_company_metrics(ticker: str, api_key: str) -> dict:
    """Fetches all key company metrics needed for equity report from FMP API."""
    print(f"Fetching comprehensive company metrics for {ticker}...")
    
    metrics = {
        'share_price': None,
        'target_price': None,
        'market_cap': None,
        'volume': None,
        'fwd_pe': None,
        'pb_ratio': None,
        'dividend_yield': None,
        'free_float': None,
        'roe': None,
        'net_debt_to_equity': None,
        'rating': None,
        'beta': None,
        'sector': None,
        'industry': None,
        'exchange': None,
        '52w_range': None,
        'shares_outstanding': None,
    }
    
    # 1. Get current price and basic quote data
    current_price = get_fmp_current_price(ticker, api_key)
    if current_price:
        metrics['share_price'] = current_price
    
    # 2. Get target price and rating
    target_price = get_fmp_target_price(ticker, api_key)
    if target_price:
        metrics['target_price'] = target_price
    
    rating = get_fmp_analyst_rating(ticker, api_key)
    if rating:
        metrics['rating'] = rating
    
    # 3. Get company profile data
    profile = get_fmp_company_profile(ticker, api_key)
    if profile:
        metrics['market_cap'] = profile.get('mktCap', 0) / 1e9 if profile.get('mktCap') else None  # Convert to billions
        metrics['volume'] = profile.get('volAvg', 0) / 1e6 if profile.get('volAvg') else None  # Convert to millions
        metrics['beta'] = profile.get('beta')
        metrics['sector'] = profile.get('sector', 'N/A')
        metrics['industry'] = profile.get('industry', 'N/A')
        metrics['exchange'] = profile.get('exchangeShortName', 'N/A')
    
    # 4. Get detailed quote data (volume, 52w range, shares outstanding)
    try:
        quote_url = f"{FMP_BASE_URL}/quote?symbol={ticker}&apikey={api_key}"
        response = requests.get(quote_url)
        response.raise_for_status()
        quote_data = response.json()
        if quote_data and isinstance(quote_data, list) and len(quote_data) > 0:
            quote = quote_data[0]
            # Volume
            if not metrics['volume']:
                avg_volume = quote.get('avgVolume')
                if avg_volume:
                    metrics['volume'] = avg_volume / 1e6  # Convert to millions
            # 52-week range
            year_high = quote.get('yearHigh')
            year_low = quote.get('yearLow')
            if year_high is not None and year_low is not None:
                metrics['52w_range'] = f"${year_low:.2f} - ${year_high:.2f}"
            # Shares outstanding. The stable quote endpoint dropped this field,
            # so derive it from market cap / price when it is absent.
            shares_out = quote.get('sharesOutstanding')
            if not shares_out:
                quote_market_cap = quote.get('marketCap')
                quote_price = quote.get('price')
                if quote_market_cap and quote_price:
                    shares_out = quote_market_cap / quote_price
            if shares_out:
                metrics['shares_outstanding'] = float(shares_out)
    except Exception as e:
        print(f"Warning: Could not fetch quote data: {e}")
    
    # 5. Get financial ratios
    try:
        ratios_df, key_metrics_df = get_fmp_ratios_and_key_metrics(ticker, api_key, limit=1)
        
        if ratios_df is not None and not ratios_df.empty:
            latest_ratios = ratios_df.iloc[0]
            metrics['pb_ratio'] = latest_ratios.get('priceToBookRatio')
            metrics['roe'] = latest_ratios.get('returnOnEquity')
            if latest_ratios.get('returnOnEquity'):
                metrics['roe'] = latest_ratios['returnOnEquity'] * 100  # Convert to percentage
            metrics['net_debt_to_equity'] = latest_ratios.get('debtEquityRatio')
            # Try to get forward P/E from ratios
            metrics['fwd_pe'] = latest_ratios.get('priceEarningsRatio')
        
        if key_metrics_df is not None and not key_metrics_df.empty:
            latest_key_metrics = key_metrics_df.iloc[0]
            # Override with key metrics if available
            if latest_key_metrics.get('peRatio'):
                metrics['fwd_pe'] = latest_key_metrics['peRatio']
            if latest_key_metrics.get('pbRatio'):
                metrics['pb_ratio'] = latest_key_metrics['pbRatio']
    except Exception as e:
        print(f"Warning: Could not fetch financial ratios: {e}")
    
    # 6. Get dividend yield from profile or financial data
    if profile and profile.get('lastDiv'):
        if current_price and profile['lastDiv'] > 0:
            # Calculate dividend yield: (annual dividend / current price) * 100
            annual_dividend = profile['lastDiv']  # Assuming this is annual
            metrics['dividend_yield'] = (annual_dividend / current_price) * 100
    
    # 7. Get shares outstanding for free float calculation (approximate)
    try:
        if profile and profile.get('sharesOutstanding'):
            # Most companies have high free float, use a reasonable estimate
            metrics['free_float'] = 95.0  # Default assumption for large companies
    except Exception as e:
        print(f"Warning: Could not calculate free float: {e}")
    
    # Fill in any remaining None values with sensible defaults
    if metrics['free_float'] is None:
        metrics['free_float'] = 95.0
    if metrics['sector'] is None:
        metrics['sector'] = 'Technology'  # Default for many stocks
    if metrics['rating'] is None:
        metrics['rating'] = 'N/A'
    
    print(f"Successfully fetched metrics for {ticker}")
    return metrics


def get_technical_indicators(ticker: str, api_key: str) -> dict:
    """从 FMP 获取历史价格并计算 SMA50/200、RSI14、MACD、成交量信号。"""
    result = {
        'sma50': None, 'sma200': None, 'rsi14': None,
        'macd': None, 'macd_signal': None, 'macd_histogram': None,
        'avg_volume_20d': None, 'latest_volume': None,
        'price': None,
        'ma_signal': 'N/A', 'rsi_signal': 'N/A',
        'macd_signal_label': 'N/A', 'volume_signal': 'N/A',
        'overall_signal': 'N/A',
    }
    try:
        import numpy as np
        url = f"{FMP_BASE_URL}/historical-price-eod/full?symbol={ticker}&apikey={api_key}"
        resp = requests.get(url, timeout=30)
        data = resp.json()
        # The stable endpoint returns a flat list of EOD rows; the legacy one
        # wrapped them in {"historical": [...]}.
        if isinstance(data, dict):
            prices = data.get('historical', [])
        else:
            prices = data or []
        if len(prices) < 50:
            return result

        df = pd.DataFrame(prices).sort_values('date').reset_index(drop=True)
        close = df['close'].astype(float)
        volume = df['volume'].astype(float)

        result['price'] = close.iloc[-1]

        # SMA
        sma50 = close.rolling(50).mean().iloc[-1]
        sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None
        result['sma50'] = round(sma50, 2)
        if sma200 is not None:
            result['sma200'] = round(sma200, 2)

        # MA signal
        price = close.iloc[-1]
        if sma200 is not None:
            if price > sma50 > sma200:
                result['ma_signal'] = 'Bullish'
            elif price < sma50 < sma200:
                result['ma_signal'] = 'Bearish'
            elif price > sma200:
                result['ma_signal'] = 'Neutral-Bullish'
            else:
                result['ma_signal'] = 'Neutral-Bearish'
        elif price > sma50:
            result['ma_signal'] = 'Bullish'
        else:
            result['ma_signal'] = 'Bearish'

        # RSI 14
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi_val = rsi.iloc[-1]
        result['rsi14'] = round(rsi_val, 1)
        if rsi_val > 70:
            result['rsi_signal'] = 'Overbought'
        elif rsi_val < 30:
            result['rsi_signal'] = 'Oversold'
        elif rsi_val > 55:
            result['rsi_signal'] = 'Bullish'
        elif rsi_val < 45:
            result['rsi_signal'] = 'Bearish'
        else:
            result['rsi_signal'] = 'Neutral'

        # MACD (12, 26, 9)
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9).mean()
        histogram = macd_line - signal_line
        result['macd'] = round(macd_line.iloc[-1], 2)
        result['macd_signal'] = round(signal_line.iloc[-1], 2)
        result['macd_histogram'] = round(histogram.iloc[-1], 2)
        if macd_line.iloc[-1] > signal_line.iloc[-1] and histogram.iloc[-1] > 0:
            result['macd_signal_label'] = 'Bullish'
        elif macd_line.iloc[-1] < signal_line.iloc[-1] and histogram.iloc[-1] < 0:
            result['macd_signal_label'] = 'Bearish'
        else:
            result['macd_signal_label'] = 'Neutral'

        # Volume
        avg_vol = volume.tail(20).mean()
        latest_vol = volume.iloc[-1]
        result['avg_volume_20d'] = round(avg_vol)
        result['latest_volume'] = round(latest_vol)
        vol_ratio = latest_vol / avg_vol if avg_vol > 0 else 1
        if vol_ratio > 1.5:
            result['volume_signal'] = 'High Activity'
        elif vol_ratio < 0.5:
            result['volume_signal'] = 'Low Activity'
        else:
            result['volume_signal'] = 'Normal'

        # Overall signal
        signals = [result['ma_signal'], result['rsi_signal'],
                   result['macd_signal_label']]
        bullish = sum(1 for s in signals if 'Bullish' in s or s == 'Oversold')
        bearish = sum(1 for s in signals if 'Bearish' in s or s == 'Overbought')
        if bullish >= 2:
            result['overall_signal'] = 'Bullish'
        elif bearish >= 2:
            result['overall_signal'] = 'Bearish'
        else:
            result['overall_signal'] = 'Neutral'

        print(f"✅ Computed technical indicators for {ticker}: {result['overall_signal']}")
    except Exception as e:
        print(f"⚠️ Could not compute technical indicators: {e}")
    return result


def get_company_news(ticker: str, api_key: str, days_back: int = 5, limit: int = 50) -> list[dict] | None:
    """
    Fetches recent company news from FMP API.
    
    Args:
        ticker: Stock ticker symbol
        api_key: FMP API key
        days_back: Number of days to look back for news (default: 5)
        limit: Maximum number of news articles to fetch (default: 50)
    
    Returns:
        List of dictionaries containing filtered news data, or None if error occurs
    """
    try:
        from datetime import datetime, timedelta
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Format dates for API
        from_date = start_date.strftime('%Y-%m-%d')
        to_date = end_date.strftime('%Y-%m-%d')
        
        # FMP Stock News API endpoint
        url = f"{FMP_BASE_URL}/news/stock"
        params = {
            'symbols': ticker,
            'from': from_date,
            'to': to_date,
            'limit': limit,
            'apikey': api_key
        }
        
        print(f"Fetching news for {ticker} from {from_date} to {to_date}...")
        response = requests.get(url, params=params)
        if response.status_code in (401, 402, 403):
            print(f"News endpoint is not included in the current FMP subscription "
                  f"(HTTP {response.status_code}); continuing without news data.")
            return None
        response.raise_for_status()
        data = response.json()
        
        if not data:
            print(f"No news data returned from FMP for {ticker}.")
            return None
        
        # Filter to keep only required fields
        filtered_news = []
        for article in data:
            filtered_article = {
                'symbol': article.get('symbol'),
                'title': article.get('title'),
                'publishedDate': article.get('publishedDate'),
                'text': article.get('text'),
                'site': article.get('site'),
                'url': article.get('url')
            }
            filtered_news.append(filtered_article)
        
        print(f"Successfully fetched {len(filtered_news)} news articles for {ticker}")
        return filtered_news
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news for {ticker}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error fetching news for {ticker}: {e}")
        return None


if __name__ == "__main__":
    print("Testing market_data_api.py...")
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path_test = os.path.join(current_script_dir, "..", "..", "config", "config.ini")
    
    fmp_api_key_for_test = "YOUR_FMP_KEY_HERE" 
    if not os.path.exists(config_path_test):
        print(f"Test config file not found at {config_path_test}. Creating a dummy one for structure test.")
        os.makedirs(os.path.dirname(config_path_test), exist_ok=True)
        with open(config_path_test, "w") as f:
            f.write("[API_KEYS]\n")
            f.write("fmp_api_key = YOUR_FMP_KEY_HERE\n")
        print("Please put a valid FMP API key in the dummy config.ini to run live tests.")
    else:
        try:
            test_config = load_config(config_path_test)
            fmp_api_key_for_test = get_api_key(test_config, section="API_KEYS", key="fmp_api_key")
        except Exception as e:
            print(f"Error loading test config: {e}")

    if fmp_api_key_for_test != "YOUR_FMP_KEY_HERE" and fmp_api_key_for_test:
        print(f"\nUsing FMP API Key: {fmp_api_key_for_test[:5]}... for live FMP tests")
        
        print("\nTesting get_comprehensive_financial_data for AAPL...")
        financial_data = get_comprehensive_financial_data("AAPL", fmp_api_key_for_test)
        for statement_type, df in financial_data.items():
            if df is not None and not df.empty:
                print(f"{statement_type}: {len(df)} years of data")
            else:
                print(f"{statement_type}: No data")
    else:
        print("\nSkipping live FMP API tests. Please provide a valid API key in config.ini.")

    print("\nmarket_data_api.py tests complete.")
