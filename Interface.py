import pandas as pd
import yfinance as yf
from functools import lru_cache
import time
import os
import matplotlib.pyplot as plt

#Nice Numbers
def nice_num(num, currency: str = "", percent: bool = False) -> str:
    if num is None or (isinstance(num, float) and pd.isna(num)):
        return "N/A"
    try:
        num = float(num)
    except Exception:
        return str(num)
    if percent:
        return f"{num * 100:.2f}%"
    billion, million = 1_000_000_000, 1_000_000
    if 0 < abs(num) < 1 and not currency:
        return f"{num * 100:.2f}%"
    if abs(num) >= billion:
        return f"{currency}{num / billion:.2f}B"
    if abs(num) >= million:
        return f"{currency}{num / million:.2f}M"
    return f"{currency}{num:,.2f}"

#Table Formatting
def print_section(title: str, df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("-" * 60)
    print(df.to_string(index=False))
    print("=" * 60)

def dict_to_df(d: dict, title_col="Metric", value_col="Value") -> pd.DataFrame:
    df = pd.DataFrame([(k, v) for k, v in d.items()], columns=[title_col, value_col])
    df[title_col] = df[title_col].astype(str)
    df[value_col] = df[value_col].astype(str)
    df.sort_values(by=title_col, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

#Industry Comparison
@lru_cache(maxsize=1)
def sp500_tickers() -> list[str]:
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", flavor="lxml")
        return tables[0]["Symbol"].tolist()
    except Exception:
        return []

def find_peers(ticker: str, level: str = "industry", max_peers: int = 5) -> list[str]:
    try:
        base = yf.Ticker(ticker).info.get(level)
    except Exception:
        return []
    peers = []
    for sym in sp500_tickers():
        if sym == ticker:
            continue
        try:
            info = yf.Ticker(sym).info
            if info.get(level) == base:
                peers.append(sym)
        except Exception:
            continue
        if len(peers) >= max_peers:
            break
    return peers

#Peer Metrics
METRIC_MAP = {
    "P/E Ratio": "trailingPE",
    "P/S Ratio": "priceToSalesTrailing12Months",
    "P/B Ratio": "priceToBook",
    "EV/EBITDA": "enterpriseToEbitda",
    "EBITDA": "ebitda",
    "Revenue": "totalRevenue",
    "Gross Margin": "grossMargins",
    "Net Margin": "profitMargins",
    "ROA": "returnOnAssets",
    "ROE": "returnOnEquity",
    "Debt to Equity": "debtToEquity",
    "Current Ratio": "currentRatio",
    "Quick Ratio": "quickRatio",
    "Dividend Yield": "dividendYield",
    "Beta": "beta"}

#Peer Info 
def collect_row(ticker: str, info_cache: dict) -> dict | None:
    try:
        info = info_cache.get(ticker)
        if not info:
            info = yf.Ticker(ticker).info
            info_cache[ticker] = info
        return {k: info.get(v) for k, v in METRIC_MAP.items()}
    except Exception:
        return None

#Averaging Peer Info
def peer_averages(tickers: list[str]) -> dict:
    info_cache = {}
    rows = [r for t in tickers if (r := collect_row(t, info_cache))]
    if not rows:
        return {k: None for k in METRIC_MAP}
    return pd.DataFrame(rows, dtype=float).mean(numeric_only=True).to_dict()

def get_price_change(ticker: str, periods=["1y"]) -> dict:
    changes = {}
    try:
        data = yf.Ticker(ticker).history(period="5y")
        if data.empty:
            return {p: None for p in periods}
        for p in periods:
            delta = {"1m": 21, "3m": 63, "6m": 126, "1y": 252, "5y": len(data)}.get(p, 252)
            if len(data) >= delta:
                start = data["Close"].iloc[-delta]
                end = data["Close"].iloc[-1]
                changes[p] = ((end - start) / start) * 100
            else:
                changes[p] = None
    except:
        changes = {p: None for p in periods}
    return changes

#Graphing on Excel 
def generate_stock_chart(ticker_symbol: str, hist_1y, currency_symbol: str = "$"):
    if hist_1y.empty:
        return None

    plt.figure(figsize=(8, 4))
    plt.plot(hist_1y.index, hist_1y["Close"], label=f"{ticker_symbol} Price", linewidth=2)
    plt.title(f"{ticker_symbol} - 1Y Price Chart")
    plt.xlabel("Date")
    plt.ylabel(f"Price ({currency_symbol})")
    plt.grid(True)
    plt.tight_layout()

    chart_path = f"{ticker_symbol}_chart.png"
    plt.savefig(chart_path)
    plt.close()
    return chart_path

def generate_peer_bar_chart(df_perf: pd.DataFrame, ticker_symbol: str):
    peer_df = df_perf[~df_perf["Ticker"].isin([ticker_symbol, "S&P 500", "NASDAQ", "Dow Jones"])].copy()

    try:
        peer_df["Change"] = peer_df["1Y Change"].str.replace('%', '', regex=False).astype(float)
    except Exception:
        return None

    if peer_df.empty:
        return None

    plt.figure(figsize=(8, 4))
    plt.bar(peer_df["Ticker"], peer_df["Change"], color='skyblue')
    plt.title("Peer 1Y Performance")
    plt.ylabel("Return (%)")
    plt.tight_layout()

    peer_chart_path = f"{ticker_symbol}_peer_chart.png"
    plt.savefig(peer_chart_path)
    plt.close()
    return peer_chart_path

#Main Code 
def get_company_financials(ticker_symbol: str):
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        currency_symbols = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "CHF": "Fr.",
                            "CAD": "C$", "AUD": "A$", "HKD": "HK$", "CNY": "¥"}
        currency_code = info.get("financialCurrency", "USD")
        currency_symbol = currency_symbols.get(currency_code, currency_code + " ")

        industry_peers = find_peers(ticker_symbol, "industry")
        sector_peers = find_peers(ticker_symbol, "sector")
        industry_avg = peer_averages(industry_peers)
        sector_avg = peer_averages(sector_peers)

        reverse_map = {v: k for k, v in METRIC_MAP.items()}

        def compare(metric_key, percent=False, is_currency=False):
            val = info.get(metric_key)
            label = reverse_map.get(metric_key, metric_key)
            return [
                nice_num(val, currency_symbol if is_currency else "", percent),
                nice_num(industry_avg.get(label), currency_symbol if is_currency else "", percent),
                nice_num(sector_avg.get(label), currency_symbol if is_currency else "", percent)]

        column_labels = ["Metric", "Company", "Industry Avg", "Sector Avg"]

        #Valuation
        valuation = pd.DataFrame([
            ["P/E Ratio"] + compare("trailingPE"),
            ["P/S Ratio"] + compare("priceToSalesTrailing12Months"),
            ["P/B Ratio"] + compare("priceToBook"),
            ["EV/EBITDA"] + compare("enterpriseToEbitda"),
            ["Enterprise Value", nice_num(info.get("enterpriseValue"), currency_symbol), "N/A", "N/A"],
            ["EPS", nice_num(info.get("earningsPerShare"), currency_symbol), "N/A", "N/A"]], columns=column_labels)
        
        #Profitability
        profit = pd.DataFrame([
            ["EBITDA"] + compare("ebitda", is_currency=True),
            ["Revenue"] + compare("totalRevenue", is_currency=True),
            ["Gross Margin"] + compare("grossMargins", percent=True),
            ["Net Margin"] + compare("profitMargins", percent=True),
            ["Return on Assets"] + compare("returnOnAssets", percent=True),
            ["Return on Equity"] + compare("returnOnEquity", percent=True)], columns=column_labels)

        #Financial Health
        health = pd.DataFrame([
            ["Debt to Equity"] + compare("debtToEquity"),
            ["Current Ratio"] + compare("currentRatio"),
            ["Quick Ratio"] + compare("quickRatio"),
            ["Interest Coverage", nice_num(info.get("interestCoverage")), "N/A", "N/A"]], columns=column_labels)

        #Additional Metrics 
        additional = pd.DataFrame([
            ["Dividend Yield"] + compare("dividendYield", percent=True),
            ["Beta (Volatility)"] + compare("beta"),
            ["Market Cap", nice_num(info.get("marketCap"), currency_symbol), "N/A", "N/A"],
            ["Shares Outstanding", nice_num(info.get("sharesOutstanding")), "N/A", "N/A"]], columns=column_labels)
        
        #Analyst Sentiment
        analyst_data = {
            "Analyst Rating": info.get("recommendationKey", "N/A").title(),
            "Rating Score (1-5)": info.get("recommendationMean", "N/A"),
            "Number of Analysts": info.get("numberOfAnalystOpinions", "N/A"),
            "Target Price (Mean)": nice_num(info.get("targetMeanPrice"), currency_symbol),
            "Target Price (Low)": nice_num(info.get("targetLowPrice"), currency_symbol),
            "Target Price (High)": nice_num(info.get("targetHighPrice"), currency_symbol),}
 
        #Price Momentum
        momentum = {
            "52 Week High": nice_num(info.get("fiftyTwoWeekHigh"), currency_symbol),
            "52 Week Low": nice_num(info.get("fiftyTwoWeekLow"), currency_symbol),
            "50-Day MA": nice_num(info.get("fiftyDayAverage"), currency_symbol),
            "200-Day MA": nice_num(info.get("twoHundredDayAverage"), currency_symbol)}
        
        hist_1y = ticker.history(period="1y")
        if not hist_1y.empty:
            start_price = hist_1y["Close"].iloc[0]
            end_price = hist_1y["Close"].iloc[-1]
            change_pct = ((end_price - start_price) / start_price) * 100
            momentum["52-Week Change %"] = f"{change_pct:.2f}%"
        else:
            momentum["52-Week Change %"] = "N/A"

        #Peer Price Comparison 
        peers_to_compare = industry_peers[:5] + sector_peers[:5] + ["^GSPC", "^IXIC", "^DJI"]
        perf_data = {}
        for peer in peers_to_compare:
            changes = get_price_change(peer, periods=["1y"])
            change = changes["1y"]
            label = {"^GSPC": "S&P 500", "^IXIC": "NASDAQ", "^DJI": "Dow Jones"}.get(peer, peer)
            perf_data[label] = f"{change:.2f}%" if change is not None else "N/A"

        company_change = get_price_change(ticker_symbol, periods=["1y"])["1y"]
        perf_data[ticker_symbol] = f"{company_change:.2f}%" if company_change is not None else "N/A"

        df_perf = dict_to_df(perf_data, "Ticker", "1Y Change")

        #Chart Generation
        chart_path = generate_stock_chart(ticker_symbol, hist_1y, currency_symbol)
        peer_chart_path = generate_peer_bar_chart(df_perf, ticker_symbol)

        #Export to Excel 
        filename = f"{ticker_symbol}_valuation_dashboard.xlsx"
        with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
            workbook = writer.book
            bold = workbook.add_format({'bold': True})
            green_fmt = workbook.add_format({'font_color': 'green'})
            red_fmt = workbook.add_format({'font_color': 'red'})
            
        #Stock Overview 
            overview = dict_to_df({
                "Ticker": ticker_symbol,
                "Company Name": info.get("longName", "N/A"),
                "Sector": info.get("sector", "N/A"),
                "Industry": info.get("industry", "N/A"),
                "Currency": currency_code,
                "Market Cap": nice_num(info.get("marketCap"), currency_symbol),
                "52W Change": momentum.get("52-Week Change %")})
            
            overview.to_excel(writer, sheet_name="Dashboard Overview", index=False)
            overview_ws = writer.sheets["Dashboard Overview"]
            overview_ws.set_column("A:A", 25, bold)
            overview_ws.set_column("B:B", 30)
            
            #Potential Insights 
            insights = []
            if info.get("trailingPE") and info.get("trailingPE") > sector_avg.get("P/E Ratio", 0):
                insights.append("P/E Ratio is above sector average — may be overvalued")
            if info.get("profitMargins") and info.get("profitMargins") > sector_avg.get("Net Margin", 0):
                insights.append("Strong Net Margin relative to sector")
            if info.get("debtToEquity") and info.get("debtToEquity") > 150:
                insights.append("High debt-to-equity ratio — watch leverage risk")
            if info.get("recommendationKey", "").lower() in ["strong_buy", "buy"]:
                insights.append("Positive analyst sentiment: Buy-rated")
            if not insights:
                insights.append("No major insights triggered")

            for i, insight in enumerate(insights, start=len(overview) + 3):
                overview_ws.write(f"A{i}", f"Insight {i - len(overview) + 1}", bold)
                overview_ws.write(f"B{i}", insight)

            summary = "Summary: "
            if any("overvalued" in i for i in insights): summary += "Valuation appears stretched. "
            if any("Net Margin" in i for i in insights): summary += "Profitability is strong. "
            if any("Buy-rated" in i for i in insights): summary += "Analysts are optimistic. "
            if any("debt-to-equity" in i for i in insights): summary += "Financial leverage is high. "
            if summary == "Summary: ": summary += "No significant highlights."
            final_row = len(overview) + len(insights) + 5
            overview_ws.write(f"A{final_row}", "Final Summary", bold)
            overview_ws.write(f"B{final_row}", summary)

            tables = {
           "Valuation": valuation,
           "Profitability": profit,
           "Financial Health": health,
           "Additional Metrics": additional,
           "Analyst Sentiment": dict_to_df(analyst_data),
           "Price Momentum": dict_to_df(momentum),
           "Relative Performance": df_perf}

            for name, df in tables.items():
                df.to_excel(writer, sheet_name=name[:31], index=False)
                sheet = writer.sheets[name[:31]]
                sheet.set_column("A:A", 25, bold)
                sheet.set_column("B:D", 20)
                for row in range(1, len(df) + 1):
                    sheet.conditional_format(f"B{row+1}", {'type': 'cell', 'criteria': '>=', 'value': 0, 'format': green_fmt})
                    sheet.conditional_format(f"B{row+1}", {'type': 'cell', 'criteria': '<', 'value': 0, 'format': red_fmt})

            #Insert stock price chart into Dashboard Overview
            if chart_path and os.path.exists(chart_path):
                overview_ws = writer.sheets.get("Dashboard Overview")
                if overview_ws:
                    overview_ws.insert_image("E2", chart_path, {"x_scale": 0.7, "y_scale": 0.7})

            #Insert peer bar chart into Relative Performance
            if peer_chart_path and os.path.exists(peer_chart_path):
                perf_ws = writer.sheets.get("Relative Performance")
                if perf_ws:
                    perf_ws.insert_image("E2", peer_chart_path, {"x_scale": 0.7, "y_scale": 0.7})

        #Clean up chart image files
        if chart_path and os.path.exists(chart_path):
            os.remove(chart_path)
        if peer_chart_path and os.path.exists(peer_chart_path):
            os.remove(peer_chart_path)
        os.system(f'start EXCEL.EXE "{filename}"' if os.name == 'nt' else f'open "{filename}"')

    except Exception as e:
        print(f"❌ Error processing {ticker_symbol}: {e}")

#User Friendly 
import tkinter as tk
from tkinter import messagebox

def on_submit():
    ticker = entry.get().strip().upper()
    if ticker:
        root.withdraw()  # hide the GUI window while processing
        try:
            get_company_financials(ticker)
        except Exception as e:
            messagebox.showerror("Error", f"Something went wrong:\n{e}")
        root.deiconify()
    else:
        messagebox.showwarning("Missing Input", "Please enter a stock ticker.")

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Industrials Dashboard Generator")

    tk.Label(root, text="Enter a stock ticker (e.g. NVDA):").pack(padx=20, pady=(20, 5))
    entry = tk.Entry(root, width=30)
    entry.pack(padx=20, pady=5)
    tk.Button(root, text="Generate Dashboard", command=on_submit).pack(pady=(10, 20))

    root.mainloop()