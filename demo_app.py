import streamlit as st
import pandas as pd
import numpy as np
import time
import random

# Page Config
st.set_page_config(
    page_title="ForteBank Anti-Fraud System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Mock Data Generation ---
@st.cache_data
def load_data():
    # Load predictions
    try:
        df = pd.read_csv("predictions_synthetic_iso.csv")
    except FileNotFoundError:
        # Fallback if file not found
        dates = pd.date_range(start="2024-11-30", periods=100, freq="min")
        df = pd.DataFrame({
            'transdatetime': dates,
            'cst_dim_id': np.random.randint(100000, 999999, 100),
            'amount': np.random.uniform(100, 500000, 100),
            'fraud_probability': np.random.uniform(0, 1, 100),
            'is_fraud_prediction': np.random.choice([0, 1], 100)
        })

    if 'transdatetime' in df.columns:
        df['transdatetime'] = pd.to_datetime(df['transdatetime'])
    
    # Mock Features for Demo
    np.random.seed(42) # For reproducibility
    df['direction_txn_count'] = np.random.randint(1, 50, size=len(df))
    df['is_fake_os'] = np.random.choice([0, 1], size=len(df), p=[0.95, 0.05])
    df['amount_to_avg_30d'] = np.random.uniform(0.1, 15.0, size=len(df))
    df['device_hash'] = [f"dev_{random.randint(1000, 9999)}" for _ in range(len(df))]
    
    # Adjust mock features for fraud cases to make them look realistic
    fraud_mask = df['fraud_probability'] > 0.5
    df.loc[fraud_mask, 'direction_txn_count'] += 20
    df.loc[fraud_mask, 'amount_to_avg_30d'] += 5.0
    df.loc[fraud_mask, 'is_fake_os'] = 1
    
    return df

# --- Sidebar ---
st.sidebar.title("🛡️ Control Panel")
simulation_speed = st.sidebar.slider("Simulation Speed (x)", 1, 100, 10)
st.sidebar.markdown("---")
st.sidebar.info("System Status: **Active** 🟢")
st.sidebar.markdown("### Model Stats")
st.sidebar.text("Model: Stacking Ensemble")
st.sidebar.text("Version: v1.2.0")
st.sidebar.text("Last Retrain: 2 hours ago")

# --- Main Layout ---
st.title("🛡️ ForteBank Fraud Detection System")

# Metrics Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Transactions Processed", "1,245,892", "+124/sec")
with col2:
    st.metric("Fraud Detected", "1,892", "+3")
with col3:
    st.metric("Saved Money (KZT)", "29,990,514", "+8.9M")
with col4:
    st.metric("Avg Response Time", "87ms", "-2ms")

st.markdown("---")

# Split View
left_col, right_col = st.columns([2, 1])

df = load_data()

# Session State for selected transaction
if 'selected_tx_id' not in st.session_state:
    st.session_state.selected_tx_id = None

with left_col:
    st.subheader("📡 Live Transaction Feed")
    
    # Filter
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        filter_status = st.radio("Filter:", ["All", "Fraud Only", "Legit Only"], horizontal=True)
    
    display_df = df.copy()
    if filter_status == "Fraud Only":
        display_df = display_df[display_df['fraud_probability'] > 0.5]
    elif filter_status == "Legit Only":
        display_df = display_df[display_df['fraud_probability'] <= 0.5]
        
    # Styling
    def highlight_fraud(val):
        color = '#ffcccc' if val > 0.5 else '#ccffcc'
        return f'background-color: {color}'

    # Display Table
    # Using simple dataframe display for compatibility, but adding a selection box below
    st.dataframe(
        display_df[['transdatetime', 'cst_dim_id', 'amount', 'fraud_probability']].style.applymap(highlight_fraud, subset=['fraud_probability']),
        use_container_width=True,
        height=500
    )
    
    # Selection
    st.markdown("### Select Transaction to Investigate")
    # Get list of IDs for selection
    tx_options = display_df['cst_dim_id'].astype(str).tolist()
    selected_id_str = st.selectbox("Transaction ID:", tx_options)
    
    if selected_id_str:
        st.session_state.selected_tx_id = int(selected_id_str)

with right_col:
    st.subheader("🔍 Investigation")
    
    if st.session_state.selected_tx_id is not None:
        # Get selected row
        tx_row = df[df['cst_dim_id'] == st.session_state.selected_tx_id]
        
        if not tx_row.empty:
            tx = tx_row.iloc[0]
            
            # Risk Score Card
            risk_score = tx['fraud_probability']
            color = "red" if risk_score > 0.5 else "green"
            st.markdown(f"### Risk Score: <span style='color:{color}'>{risk_score:.2%}</span>", unsafe_allow_html=True)
            
            # Details
            st.write(f"**Amount:** {tx['amount']:,.2f} KZT")
            st.write(f"**User ID:** {tx['cst_dim_id']}")
            st.write(f"**Time:** {tx['transdatetime']}")
            
            st.markdown("---")
            st.markdown("#### 🧠 Explainability (SHAP)")
            
            # Mock SHAP explanation logic
            reasons = []
            if tx['direction_txn_count'] > 20:
                reasons.append(f"⚠️ High velocity to receiver ({tx['direction_txn_count']} txns)")
            if tx['amount_to_avg_30d'] > 3:
                reasons.append(f"⚠️ Amount is {tx['amount_to_avg_30d']:.1f}x of monthly average")
            if tx['is_fake_os'] == 1:
                reasons.append("⚠️ Suspicious OS version (Emulator detected)")
                
            if not reasons and risk_score > 0.5:
                reasons.append("⚠️ Anomalous behavior pattern detected by Graph Network")
            if not reasons and risk_score <= 0.5:
                reasons.append("✅ Transaction looks normal.")
                
            for r in reasons:
                if "⚠️" in r:
                    st.error(r)
                else:
                    st.success(r)
                
            st.markdown("---")
            st.markdown("#### 🤖 AI Assistant")
            
            # Chat Interface
            if "messages" not in st.session_state:
                st.session_state.messages = []
                
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Chat Input
            if prompt := st.chat_input("Ask about this transaction..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    # Simple Rule-Based Response Generation
                    response = ""
                    prompt_lower = prompt.lower()
                    
                    if "why" in prompt_lower:
                        response = f"This transaction was flagged because of the following risk factors:\n" + "\n".join([f"- {r}" for r in reasons])
                    elif "report" in prompt_lower or "sar" in prompt_lower:
                        response = f"**Drafting Suspicious Activity Report (SAR)**\n\n*Subject:* Suspicious Activity - User {tx['cst_dim_id']}\n*Date:* {tx['transdatetime']}\n*Amount:* {tx['amount']} KZT\n\n*Analysis:* The system detected high-risk indicators consistent with money laundering (velocity: {tx['direction_txn_count']}). Recommended action: Block and investigate."
                    elif "graph" in prompt_lower or "connection" in prompt_lower:
                        response = "Graph analysis shows this user is connected to 3 known fraud nodes within 2 hops. This indicates a potential 'Money Mule' ring."
                    else:
                        response = "I am analyzing the transaction context. You can ask me 'Why is this fraud?', 'Show graph connections', or 'Draft SAR report'."
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.error("Transaction not found.")
            
    else:
        st.info("Select a transaction from the list to view details.")
