import streamlit as st
st.set_page_config(page_title="Live DeFi Risk Checker + USDT News Sentiment", layout="wide")

import pandas as pd
import requests
import pickle
import numpy as np
from datetime import datetime
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon
nltk.download('vader_lexicon')

# --- Utility ---
def safe_float(val):
    if isinstance(val, pd.Series):
        return float(val.iloc[0])
    return float(val)

# --- Load Model and Scaler ---
@st.cache_resource
def load_model_scaler():
    model_path = r"C:\Users\shant\OneDrive\Desktop\models\modelxg.pkl"
    scaler_path = r"C:\Users\shant\OneDrive\Desktop\models\scaler.pkl"
    model = pickle.load(open(model_path, "rb"))
    scaler = pickle.load(open(scaler_path, "rb"))
    return model, scaler

model, scaler = load_model_scaler()

# --- Fetch Etherscan Transactions ---
def fetch_etherscan_transactions(address, api_key):
    try:
        url = f"https://api.etherscan.io/api?module=account&action=txlist&address={address}&startblock=0&endblock=99999999&sort=desc&apikey={api_key}"
        res = requests.get(url).json()
        return pd.DataFrame(res["result"]) if res.get("status") == "1" else pd.DataFrame()
    except Exception as e:
        st.error(e)
        return pd.DataFrame()

# --- Predict Risk ---
def predict_risks_on_live_data(df):
    try:
        df = df.copy().astype({
            'value': 'float', 'gasPrice': 'float',
            'gasUsed': 'float', 'nonce': 'int'
        })

        df['timeStamp'] = pd.to_datetime(df['timeStamp'].astype(int), unit='s')
        df = df.sort_values(by='timeStamp')

        df['time_diff_seconds'] = df['timeStamp'].diff().dt.total_seconds()
        df['time_diff_seconds'].fillna(0, inplace=True)
        df['from_address_transaction_count'] = df.groupby('from').cumcount()

        base_features = [
            'gasPrice',
            'gasUsed',
            'from_address_transaction_count',
            'time_diff_seconds'
        ]

        df_input = df[base_features]
        df_scaled = scaler.transform(df_input)

        df['prediction_label'] = pd.Series(model.predict(df_scaled)).map({1: '⚠️ Risky', 0: '✅ Safe'})

        return df[['hash', 'from', 'to', 'value', 'gasPrice', 'timeStamp', 'prediction_label'] + base_features].head(10)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return pd.DataFrame()

# --- Fetch News & Sentiment ---
def fetch_usdt_news_and_sentiment():
    query = "Tether USDT"
    rss_url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}"
    feed = requests.get(rss_url).content
    df = pd.read_xml(feed, xpath="//item")
    df = df[['title', 'pubDate', 'link']].head(5)
    sia = SentimentIntensityAnalyzer()
    df['sentiment'] = df['title'].apply(lambda t: sia.polarity_scores(t)['compound'])
    avg = df['sentiment'].mean()
    return df, avg

# --- UI Title ---
st.title("🛡️ Live DeFi Risk Checker + USDT News Sentiment")

# --- Layout: Two Columns ---
col1, col2 = st.columns(2)

# --- Column 1: Risk Checker ---
with col1:
    st.header("⚙️ Risk Checker")
    address = st.text_input("Wallet Address", key="wallet")
    api_key = st.text_input("Etherscan API Key", type="password", key="api")

    if st.button("Check Risk"):
        if not address or not api_key:
            st.warning("Please enter both wallet address and API key.")
        else:
            df = fetch_etherscan_transactions(address, api_key)
            if df.empty:
                st.error("No transactions found or invalid API key/address.")
            else:
                res = predict_risks_on_live_data(df)
                if not res.empty:
                    st.success("🔍 Risk analysis completed.")
                    for _, row in res.iterrows():
                        with st.expander(f"{row.hash[:10]}… — {row.prediction_label}"):
                            st.write(f"**From:** {row['from']}")
                            st.write(f"**To:** {row['to']}")
                            st.write(f"**Timestamp:** {row['timeStamp']}")
                            st.write(f"**Value (Wei):** {safe_float(row['value']):.2f}")
                            st.write(f"**Gas Price (Wei):** {safe_float(row['gasPrice']):.2f}")
                            st.write("**Features Used:**")
                            st.json({
                                "gasPrice": safe_float(row['gasPrice']),
                                "gasUsed": safe_float(row['gasUsed']),
                                "from_address_transaction_count": safe_float(row['from_address_transaction_count']),
                                "time_diff_seconds": safe_float(row['time_diff_seconds'])
                            })

# --- Column 2: Sentiment Dashboard ---
with col2:
    st.header("📰 USDT Sentiment")

    if st.button("🔄 Refresh Sentiment News"):
        st.session_state['refresh_news'] = True

    if 'refresh_news' not in st.session_state:
        st.session_state['refresh_news'] = False

    if st.session_state['refresh_news']:
        news_df, avg_sent = fetch_usdt_news_and_sentiment()
        st.session_state['news_df'] = news_df
        st.session_state['avg_sent'] = avg_sent
        st.session_state['refresh_news'] = False
    else:
        if 'news_df' not in st.session_state:
            news_df, avg_sent = fetch_usdt_news_and_sentiment()
            st.session_state['news_df'] = news_df
            st.session_state['avg_sent'] = avg_sent
        else:
            news_df = st.session_state['news_df']
            avg_sent = st.session_state['avg_sent']

    sent_label = "Negative 🔻" if avg_sent < 0 else "Positive 🔺" if avg_sent > 0 else "Neutral ⚪"
    st.metric("Average Sentiment Score", round(avg_sent, 2), sent_label)

    for _, r in news_df.iterrows():
        score = round(r['sentiment'], 3)
        tag = "🔻 Negative" if score < -0.05 else "🔺 Positive" if score > 0.05 else "⚪ Neutral"
        st.markdown(f"- [{r['title']}]({r['link']}) — *{tag}* (`Score: {score}`)")
