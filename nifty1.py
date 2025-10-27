import requests
import pandas as pd
import time
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import pytz
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================
IST = pytz.timezone("Asia/Kolkata")

CLIENT_ID = '1100244268'
ACCESS_TOKEN = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzYxNjMyNzgzLCJpYXQiOjE3NjE1NDYzODMsInRva2VuQ29uc3VtZXJUeXBlIjoiU0VMRiIsIndlYmhvb2tVcmwiOiIiLCJkaGFuQ2xpZW50SWQiOiIxMTAwMjQ0MjY4In0.7UBEy-qO0x_Ux3G_OKH036RRz1_JU7t67RcKWL4_L-y4bhyVQ_z6HGhdbaQ1HHGx5XyFYclhLdLjWDDky9Cjog'

HEADERS = {
    'client-id': CLIENT_ID,
    'access-token': ACCESS_TOKEN,
    'Content-Type': 'application/json'
}

# BIG MONEY THRESHOLDS
MIN_NOTIONAL = 5_000_000  # ₹5M minimum
BLOCK_TRADE_MULTIPLIER = 8
DARK_POOL_IMPACT = 0.025
MEGA_FLOW = 10_000_000  # ₹10M

st.set_page_config(page_title="HFT Algo Scanner", layout="wide", initial_sidebar_state="collapsed")

# ============================================
# SESSION STATE
# ============================================
if "flow_log" not in st.session_state:
    st.session_state.flow_log = []

if "previous_data" not in st.session_state:
    st.session_state.previous_data = {}

if "timeframe" not in st.session_state:
    st.session_state.timeframe = "30S"  # Default HFT mode

# ============================================
# API FUNCTIONS
# ============================================
def get_expiry_dates():
    url = "https://api.dhan.co/v2/optionchain/expirylist"
    payload = {"UnderlyingScrip": 13, "UnderlyingSeg": "IDX_I"}
    try:
        response = requests.post(url, json=payload, headers=HEADERS, timeout=10)
        return response.json()['data']
    except:
        return []

def fetch_option_chain(expiry):
    url = "https://api.dhan.co/v2/optionchain"
    payload = {"UnderlyingScrip": 13, "UnderlyingSeg": "IDX_I", "Expiry": expiry}
    try:
        response = requests.post(url, json=payload, headers=HEADERS, timeout=10)
        time.sleep(2)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

# ============================================
# SWEEP DETECTION
# ============================================
def detect_sweeps(df):
    """Detect multi-strike sweeps"""
    sweeps = {}
    
    if len(df) < 5:
        return sweeps
    
    for opt_type in ['CE', 'PE']:
        type_data = df[df['Type'] == opt_type]
        
        # High volume across multiple strikes
        high_vol = type_data[type_data['Volume'] > 100]
        if len(high_vol) >= 3:
            total_vol = high_vol['Volume'].sum()
            avg_change = high_vol['LTP_Change'].mean()
            
            if total_vol > 500 and abs(avg_change) > 2:
                sweeps[opt_type] = {
                    'strikes': len(high_vol),
                    'volume': total_vol,
                    'direction': 'BUY' if avg_change > 0 else 'SELL'
                }
    
    return sweeps

# ============================================
# BIG MONEY CLASSIFICATION
# ============================================
def classify_big_money(row, avg_volume):
    """Strict institutional flow classification"""
    
    notional = row['Notional']
    volume = row['Volume']
    oi_change = row['OI_Change']
    ltp_change = row['LTP_Change']
    iv_change = row['IV_Change']
    opt_type = row['Type']
    
    # Filter: Minimum notional
    if notional < MIN_NOTIONAL:
        return None, 0, None, 0
    
    volume_ratio = volume / avg_volume if avg_volume > 0 else 1
    price_impact = abs(ltp_change) / (volume + 1) * 1000
    
    # MEGA FLOW (₹10M+)
    if notional >= MEGA_FLOW:
        if ltp_change > 2:
            # MEGA BUY: Green for CE, Red for PE
            if opt_type == "CE":
                return f"🐋 MEGA CE BUY", 98, "#008000", 11  # Green
            else:
                return f"🐋 MEGA PE BUY", 98, "#FF0000", 11  # Red
        else:
            # MEGA WRITE: Light Red for CE, Light Green for PE
            if opt_type == "CE":
                return f"🦈 MEGA CE WRITE", 95, "#FF6B6B", 10  # Light Red
            else:
                return f"🦈 MEGA PE WRITE", 95, "#90EE90", 10  # Light Green
    
    # DARK POOL
    if price_impact < DARK_POOL_IMPACT and volume > 150 and abs(oi_change) > 5:
        return f"🟣 Dark Pool {opt_type}", 92, "#4B0082", 10
    
    # BLOCK TRADE - CE BUY (Dark Green)
    if volume_ratio > BLOCK_TRADE_MULTIPLIER and ltp_change > 3:
        if opt_type == "CE":
            return "🚀 Inst. Call Buy", 88, "#006400", 9  # Dark Green
        else:
            return "🐻 Inst. Put Buy", 88, "#8B0000", 9  # Dark Red
    
    # HEAVY WRITE - CE WRITE (Light Red), PE WRITE (Light Green)
    if abs(oi_change) > 10 and ltp_change < -2 and volume > 120:
        if opt_type == "CE":
            return "📉 Heavy Call Write", 85, "#FF6B6B", 8  # Light Red
        else:
            return "📈 Heavy Put Write", 85, "#90EE90", 8  # Light Green
    
    # UNUSUAL ACTIVITY
    if volume_ratio > 6 and abs(iv_change) > 8:
        return f"🔥 UOA {opt_type}", 80, "#FF6347", 7
    
    # SIGNIFICANT FLOW
    if notional > 7_000_000:
        return f"💰 Inst. {opt_type}", 75, "#DAA520", 6
    
    return None, 0, None, 0

# ============================================
# DATA PROCESSING
# ============================================
def process_option_data(option_chain):
    """Process option chain and detect big money flows"""
    
    if not option_chain or "data" not in option_chain:
        return None, None
    
    option_data = option_chain["data"]["oc"]
    underlying = option_chain["data"]["last_price"]
    
    # Find ATM
    atm = min(option_data.keys(), key=lambda x: abs(float(x) - underlying))
    atm = float(atm)
    
    # ATM ± 5 strikes
    strike_range = (atm - 250, atm + 250)
    
    data_list = []
    previous = st.session_state.previous_data
    
    for strike, contracts in option_data.items():
        strike_price = float(strike)
        
        if not (strike_range[0] <= strike_price <= strike_range[1]):
            continue
        
        for opt_type, opt_data in [("CE", contracts.get("ce", {})), ("PE", contracts.get("pe", {}))]:
            if not opt_data:
                continue
            
            oi = opt_data.get("oi", 0)
            ltp = opt_data.get("last_price", 0)
            volume = opt_data.get("volume", 0)
            iv = opt_data.get("implied_volatility", 0)
            
            key = f"{strike_price}_{opt_type}"
            prev = previous.get(key, {})
            
            # Calculate changes
            oi_change = ((oi - prev.get('oi', oi)) / oi * 100) if oi else 0
            ltp_change = ((ltp - prev.get('ltp', ltp)) / ltp * 100) if ltp else 0
            iv_change = ((iv - prev.get('iv', iv)) / iv * 100) if iv else 0
            
            notional = oi * ltp * 75  # Lot size
            
            data_list.append({
                'Timestamp': datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
                'StrikePrice': strike_price,
                'Type': opt_type,
                'OI': oi,
                'LTP': ltp,
                'Volume': volume,
                'IV': iv,
                'OI_Change': oi_change,
                'LTP_Change': ltp_change,
                'IV_Change': iv_change,
                'Notional': notional,
                'Underlying': underlying
            })
            
            # Update previous
            previous[key] = {'oi': oi, 'ltp': ltp, 'iv': iv}
    
    df = pd.DataFrame(data_list)
    
    if df.empty:
        return df, underlying
    
    # Detect sweeps
    sweeps = detect_sweeps(df)
    
    # Classify flows
    avg_volume = df.groupby(['StrikePrice', 'Type'])['Volume'].mean().mean()
    
    flow_results = df.apply(lambda row: classify_big_money(row, avg_volume), axis=1)
    
    df['Flow_Type'] = flow_results.apply(lambda x: x[0])
    df['Confidence'] = flow_results.apply(lambda x: x[1])
    df['Color'] = flow_results.apply(lambda x: x[2])
    df['Priority'] = flow_results.apply(lambda x: x[3])
    
    # Filter only institutional flows
    df = df[df['Flow_Type'].notna()].copy()
    
    # Add sweep info
    for opt_type, sweep_data in sweeps.items():
        df.loc[df['Type'] == opt_type, 'Is_Sweep'] = True
    
    # Add to log
    if not df.empty:
        for _, row in df.iterrows():
            st.session_state.flow_log.append(row.to_dict())
        
        # Keep last 1000 records
        st.session_state.flow_log = st.session_state.flow_log[-1000:]
    
    return df, underlying

# ============================================
# VISUALIZATION
# ============================================
def render_hft_scanner():
    """Render the HFT Algo Scanner"""
    
    st.markdown("""
        <h1 style='text-align: center; color: #1a1a1a; margin-bottom: 0;'>
            🐋 HFT ALGO SCANNER
        </h1>
        <p style='text-align: center; color: #666; margin-top: 5px;'>
            Real-Time Institutional Flow Detection
        </p>
        <div style='background: #f0f8ff; padding: 15px; border-radius: 10px; margin: 20px 0;'>
            <h4 style='margin: 0 0 10px 0; color: #1a1a1a;'>📊 What is Activity Score?</h4>
            <p style='margin: 0; color: #555; line-height: 1.6;'>
                <strong>Activity Score</strong> = (Notional Value × Confidence %) / 100<br>
                It measures the <strong>intensity and reliability</strong> of institutional flows.<br>
                Higher score = Larger money flow with higher confidence of being institutional.<br><br>
                <strong>Color coding:</strong><br>
                🟢 Green = Low activity | 🟡 Yellow = Moderate | 🔴 Red = High institutional activity
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    flow_df = pd.DataFrame(st.session_state.flow_log)
    
    if flow_df.empty:
        st.info("⏳ Scanning for institutional flows... (₹5M+ notional filter active)")
        return
    
    # Time aggregation - Use selected timeframe
    timeframe = st.session_state.timeframe
    
    if timeframe == "30S":
        flow_df['TimeSlot'] = pd.to_datetime(flow_df['Timestamp']).dt.floor('30S').dt.strftime('%H:%M:%S')
    elif timeframe == "1min":
        flow_df['TimeSlot'] = pd.to_datetime(flow_df['Timestamp']).dt.floor('1min').dt.strftime('%H:%M:%S')
    elif timeframe == "2min":
        flow_df['TimeSlot'] = pd.to_datetime(flow_df['Timestamp']).dt.floor('2min').dt.strftime('%H:%M')
    else:  # 5min
        flow_df['TimeSlot'] = pd.to_datetime(flow_df['Timestamp']).dt.floor('5min').dt.strftime('%H:%M')
    
    # Calculate activity metric
    flow_df['Activity'] = flow_df['Notional'] * (flow_df['Confidence'] / 100)
    
    # Aggregate
    agg_df = flow_df.groupby(['TimeSlot', 'Flow_Type', 'Color', 'Priority']).agg({
        'Activity': 'sum',
        'Volume': 'sum',
        'Notional': 'sum',
        'Confidence': 'mean',
        'Underlying': 'last'
    }).reset_index()
    
    agg_df = agg_df.sort_values(['Priority', 'Confidence'], ascending=[False, False])
    
    # Price data
    price_df = flow_df.groupby('TimeSlot')['Underlying'].last().reset_index()
    
    # ============================================
    # METRICS
    # ============================================
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_notional = agg_df['Notional'].sum() / 1_000_000
    total_volume = agg_df['Volume'].sum()
    avg_confidence = agg_df['Confidence'].mean()
    mega_count = len(agg_df[agg_df['Notional'] >= MEGA_FLOW])
    flow_types = len(agg_df['Flow_Type'].unique())
    
    col1.metric("💰 Notional", f"₹{total_notional:.1f}M")
    col2.metric("📊 Volume", f"{total_volume:,.0f}")
    col3.metric("🎯 Confidence", f"{avg_confidence:.0f}%")
    col4.metric("🐋 Mega Flows", mega_count)
    col5.metric("🔥 Flow Types", flow_types)
    
    st.markdown("---")
    
    # ============================================
    # CHART
    # ============================================
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.015,
        row_heights=[0.45, 0.12, 0.43],
        subplot_titles=(
            "💹 Spot Price", 
            "📊 Activity Score",
            "🐋 Institutional Flows"
        )
    )
    
    # ROW 1: Price
    fig.add_trace(
        go.Scatter(
            x=price_df['TimeSlot'],
            y=price_df['Underlying'],
            mode='lines+markers',
            name='Nifty',
            line=dict(color='#1E90FF', width=3),
            marker=dict(size=7, color='#1E90FF'),
            fill='tozeroy',
            fillcolor='rgba(30, 144, 255, 0.08)',
            hovertemplate="<b>%{x}</b><br>₹%{y:,.2f}<extra></extra>"
        ),
        row=1, col=1
    )
    
    # ROW 2: Activity Score
    activity_score = agg_df.groupby('TimeSlot').agg({
        'Activity': 'sum',
        'Notional': 'sum'
    }).reset_index()
    
    colors = ['#FF4500' if x > activity_score['Activity'].quantile(0.75)
              else '#FFD700' if x > activity_score['Activity'].quantile(0.5)
              else '#90EE90' for x in activity_score['Activity']]
    
    fig.add_trace(
        go.Bar(
            x=activity_score['TimeSlot'],
            y=activity_score['Activity'],
            name='Activity',
            marker=dict(color=colors, line=dict(color='#1a1a1a', width=0.5)),
            hovertemplate="<b>%{x}</b><br>Score: %{y:,.0f}<extra></extra>"
        ),
        row=2, col=1
    )
    
    # ROW 3: Flow Bars
    for flow_type in agg_df['Flow_Type'].unique():
        flow_data = agg_df[agg_df['Flow_Type'] == flow_type]
        color = flow_data['Color'].iloc[0]
        conf = flow_data['Confidence'].mean()
        
        fig.add_trace(
            go.Bar(
                x=flow_data['TimeSlot'],
                y=flow_data['Activity'],
                name=f"{flow_type} ({conf:.0f}%)",
                marker=dict(color=color, line=dict(color='#000', width=0.3)),
                opacity=0.92,
                hovertemplate=(
                    f"<b>{flow_type}</b><br>"
                    "%{x}<br>"
                    "Activity: %{y:,.0f}<br>"
                    "Vol: %{customdata[0]:,.0f}<br>"
                    "₹%{customdata[1]:.1f}M<extra></extra>"
                ),
                customdata=np.column_stack((
                    flow_data['Volume'],
                    flow_data['Notional'] / 1_000_000
                ))
            ),
            row=3, col=1
        )
    
    # Layout
    fig.update_xaxes(
        rangeslider_visible=False,
        showgrid=True,
        gridcolor='rgba(128,128,128,0.15)',
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridcolor='rgba(128,128,128,0.15)',
    )
    
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Score", row=2, col=1)
    fig.update_yaxes(title_text="Flow Intensity", row=3, col=1)
    
    # Dynamic x-axis label based on timeframe
    tf_label = {
        "30S": "Time (30-sec)",
        "1min": "Time (1-min)",
        "2min": "Time (2-min)",
        "5min": "Time (5-min)"
    }
    fig.update_xaxes(
        title_text=tf_label.get(st.session_state.timeframe, "Time"), 
        row=3, col=1
    )
    
    fig.update_layout(
        barmode='stack',
        height=950,
        plot_bgcolor='#fafafa',
        paper_bgcolor='white',
        font=dict(color='#1a1a1a', size=11, family='Arial'),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01,
            bgcolor="rgba(255,255,255,0.97)",
            bordercolor="#1a1a1a",
            borderwidth=1.5,
            font=dict(size=10)
        ),
        hovermode='x unified',
        margin=dict(l=50, r=150, t=80, b=50)
    )
    
    st.plotly_chart(fig, use_container_width=True, config={
        "scrollZoom": True,
        "displayModeBar": True,
        "displaylogo": False
    })
    
    # ============================================
    # FLOW TABLE
    # ============================================
    with st.expander("📋 Detailed Flow Breakdown", expanded=False):
        top_flows = agg_df.nlargest(15, 'Notional')[
            ['TimeSlot', 'Flow_Type', 'Volume', 'Notional', 'Confidence']
        ].copy()
        
        top_flows['Notional'] = top_flows['Notional'].apply(lambda x: f"₹{x/1_000_000:.2f}M")
        top_flows['Confidence'] = top_flows['Confidence'].apply(lambda x: f"{x:.0f}%")
        top_flows.columns = ['Time', 'Flow Type', 'Volume', 'Notional', 'Conf.']
        
        st.dataframe(
            top_flows, 
            use_container_width=True, 
            hide_index=True,
            height=400
        )

# ============================================
# MAIN
# ============================================
def main():
    # Minimal sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        expiry_dates = get_expiry_dates()
        if not expiry_dates:
            st.error("Cannot fetch expiry dates")
            return
        
        selected_expiry = st.selectbox("Expiry", expiry_dates, index=0)
        
        st.markdown("---")
        
        # HFT Timeframe selector
        timeframe_options = {
            "30 Seconds (HFT)": "30S",
            "1 Minute": "1min",
            "2 Minutes": "2min",
            "5 Minutes": "5min"
        }
        
        selected_tf = st.selectbox(
            "📊 Timeframe", 
            list(timeframe_options.keys()),
            index=0
        )
        
        st.session_state.timeframe = timeframe_options[selected_tf]
        
        st.markdown("---")
        
        refresh_interval = st.slider("Refresh (sec)", 10, 120, 30)
        auto_refresh = st.checkbox("Auto Refresh", value=True)
        
        st.markdown("---")
        
        if st.button("🔄 Refresh Now", use_container_width=True):
            st.rerun()
        
        if st.button("🗑️ Clear Log", use_container_width=True):
            st.session_state.flow_log = []
            st.success("Log cleared!")
            time.sleep(1)
            st.rerun()
    
    # Fetch and process
    option_chain = fetch_option_chain(selected_expiry)
    
    if option_chain:
        df, underlying = process_option_data(option_chain)
        
        if df is not None:
            # Current price in title
            st.markdown(f"<h3 style='text-align: center; color: #666;'>Nifty: ₹{underlying:,.2f}</h3>", unsafe_allow_html=True)
            
            # Render scanner
            render_hft_scanner()
        else:
            st.error("Error processing data")
    else:
        st.error("Failed to fetch data")
    
    # Auto refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":

    main()


