import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- Page Configuration ---
st.set_page_config(
    page_title="ESG Portfolio Stress Tester",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---
@st.cache_data
def get_stock_data(tickers, start_date, end_date):
    try:
        raw_data = yf.download(tickers, start=start_date, end=end_date, group_by="ticker", auto_adjust=True)
        if raw_data.empty:
            return None, "No data fetched. Check ticker symbols, date range, or your internet connection."
        if len(tickers) == 1:
            data = raw_data[['Close']].rename(columns={"Close": tickers[0]})
        else:
            data = pd.concat([raw_data[t]['Close'].rename(t) for t in tickers if t in raw_data], axis=1)
        if data.empty:
            return None, "Could not retrieve data for the given tickers."
        data = data.ffill().bfill().dropna(axis=1, how='all')
        if data.empty:
            return None, "All selected tickers failed to return valid data after cleaning."
        return data, None
    except Exception as e:
        return None, f"An error occurred: {e}"

def run_monte_carlo_simulation(daily_returns, weights, num_simulations, num_days):
    num_assets = len(weights)
    weights = np.array(weights)
    mean_returns = daily_returns.mean()
    cov_matrix = daily_returns.cov()
    results = np.zeros((num_simulations, num_days))
    for i in range(num_simulations):
        cholesky_decomp = np.linalg.cholesky(cov_matrix)
        random_values = np.random.normal(size=(num_days, num_assets))
        simulated_daily_returns = mean_returns.values + np.inner(cholesky_decomp, random_values).T
        portfolio_returns = np.dot(simulated_daily_returns, weights)
        results[i, :] = portfolio_returns
    return results

def apply_stress_scenario(daily_returns, scenario, shock_value):
    stressed_returns = daily_returns.copy()
    esg_sensitivity = {
        'Carbon Tax': ['XOM', 'CVX', 'SHEL'],
        'Green Energy Boom': ['NEE', 'ENPH', 'TSLA'],
        'Social Responsibility Focus': ['MSFT', 'GOOGL', 'AAPL']
    }
    sensitive_tickers = esg_sensitivity.get(scenario, [])
    for ticker in sensitive_tickers:
        if ticker in stressed_returns.columns:
            stressed_returns[ticker] *= (1 + shock_value)
    return stressed_returns

# --- Streamlit UI ---
# --- Sidebar ---
st.sidebar.title("Portfolio & Simulation Controls")
st.sidebar.header("Step 1: Define Your Portfolio")

tickers_input = st.sidebar.text_area(
    "Enter stock tickers (comma-separated)",
    "AAPL, MSFT, GOOGL, TSLA, XOM"
)
tickers = [ticker.strip().upper() for ticker in tickers_input.split(',') if ticker.strip()]

weights = []
if tickers:
    st.sidebar.subheader("Asset Allocation")
    for ticker in tickers:
        weight = st.sidebar.number_input(f"Weight for {ticker} (%)", min_value=0.0, max_value=100.0, value=100.0/len(tickers), step=1.0)
        weights.append(weight / 100.0)
else:
    st.sidebar.warning("Please enter at least one ticker symbol.")

if tickers and not np.isclose(sum(weights), 1.0):
    st.sidebar.error(f"Weights must sum to 100%. Current sum: {sum(weights)*100:.2f}%")

st.sidebar.header("Step 2: Simulation Settings")
num_simulations = st.sidebar.slider("Number of Simulations", 100, 5000, 1000)
projection_years = st.sidebar.slider("Projection Horizon (Years)", 1, 30, 10)
num_days = projection_years * 252

st.sidebar.header("Step 3: ESG Stress Testing (Optional)")
apply_stress = st.sidebar.checkbox("Apply ESG Stress Test")
scenario = None
shock_value = 0.0
if apply_stress:
    scenario = st.sidebar.selectbox(
        "Select ESG Scenario",
        ["Carbon Tax", "Green Energy Boom", "Social Responsibility Focus"]
    )
    shock_value = st.sidebar.slider("Shock Intensity (%)", -50.0, 50.0, -10.0, 1.0) / 100.0

# --- Main App Body ---
st.title("🌿 Interactive ESG Portfolio Stress Tester")

if st.button("Run Simulation", type="primary"):
    if not tickers:
        st.error("Please enter at least one ticker symbol.")
    elif not np.isclose(sum(weights), 1.0):
        st.error("Portfolio weights must sum to 100%. Please adjust the weights in the sidebar.")
    else:
        with st.spinner("Fetching data and running simulations..."):
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5*365)
            stock_data, error = get_stock_data(tickers, start_date, end_date)
            if error:
                st.error(error)
            else:
                daily_returns = stock_data.pct_change().dropna()
                valid_tickers = daily_returns.columns.tolist()
                valid_weights = [weights[tickers.index(t)] for t in valid_tickers]
                total_valid_weight = sum(valid_weights)
                renormalized_weights = [w / total_valid_weight for w in valid_weights]
                if apply_stress and scenario:
                    daily_returns_stressed = apply_stress_scenario(daily_returns, scenario, shock_value)
                else:
                    daily_returns_stressed = daily_returns
                simulations = run_monte_carlo_simulation(daily_returns_stressed, renormalized_weights, num_simulations, num_days)
                initial_investment = 100000
                cumulative_returns = (1 + simulations).cumprod(axis=1)
                simulated_portfolio_values = initial_investment * cumulative_returns
                final_values = simulated_portfolio_values[:, -1]
                median_final_value = np.median(final_values)
                ci_5 = np.percentile(final_values, 5)
                ci_95 = np.percentile(final_values, 95)
                st.header("Simulation Results")
                col1, col2, col3 = st.columns(3)
                col1.metric("Median Final Value", f"${median_final_value:,.2f}")
                col2.metric("5th Percentile (Worst Case)", f"${ci_5:,.2f}")
                col3.metric("95th Percentile (Best Case)", f"${ci_95:,.2f}")
                fig = go.Figure()
                num_to_plot = min(num_simulations, 100)
                for i in range(num_to_plot):
                    fig.add_trace(go.Scatter(
                        y=simulated_portfolio_values[i, :],
                        mode='lines',
                        line=dict(width=1, color='rgba(173, 216, 230, 0.5)'),
                        hoverinfo='none',
                        showlegend=False
                    ))
                median_path = np.median(simulated_portfolio_values, axis=0)
                ci_5_path = np.percentile(simulated_portfolio_values, 5, axis=0)
                ci_95_path = np.percentile(simulated_portfolio_values, 95, axis=0)
                fig.add_trace(go.Scatter(y=ci_5_path, mode='lines', line=dict(color='red', dash='dash'), name='5th Percentile'))
                fig.add_trace(go.Scatter(y=median_path, mode='lines', line=dict(color='black', width=3), name='Median Outcome'))
                fig.add_trace(go.Scatter(y=ci_95_path, mode='lines', line=dict(color='green', dash='dash'), name='95th Percentile'))
                fig.update_layout(
                    title=f'Portfolio Value Projections over {projection_years} Years',
                    xaxis_title='Trading Days',
                    yaxis_title='Portfolio Value ($)',
                    showlegend=True,
                    legend=dict(x=0, y=1, traceorder="normal"),
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
                st.header("Distribution of Final Portfolio Values")
                hist_fig = go.Figure(data=[go.Histogram(x=final_values, nbinsx=100, name='Distribution')])
                hist_fig.add_vline(x=median_final_value, line_dash="dash", line_color="black", annotation_text="Median")
                hist_fig.add_vline(x=ci_5, line_dash="dash", line_color="red", annotation_text="5th Percentile")
                hist_fig.add_vline(x=ci_95, line_dash="dash", line_color="green", annotation_text="95th Percentile")
                hist_fig.update_layout(
                    title='Frequency Distribution of Final Outcomes',
                    xaxis_title='Final Portfolio Value ($)',
                    yaxis_title='Frequency',
                    template='plotly_white'
                )
                st.plotly_chart(hist_fig, use_container_width=True)
