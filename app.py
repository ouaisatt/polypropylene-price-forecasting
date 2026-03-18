import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Polypropylene Price Forecast",
    page_icon="🏭",
    layout="wide"
)

# ============================================================
# Load data and model
# ============================================================
@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model("pp_model.json")
    return model

@st.cache_data
def load_data():
    merged   = pd.read_csv("merged_final.csv", parse_dates=['month'])
    results  = pd.read_csv("predictions_with_intervals.csv", parse_dates=['month'])
    with open("feature_list.json") as f:
        features = json.load(f)
    with open("best_params.json") as f:
        best_params = json.load(f)
    return merged, results, features, best_params

model                          = load_model()
merged, results, features, best_params = load_data()

# ============================================================
# Sidebar
# ============================================================
st.sidebar.image("https://via.placeholder.com/300x80?text=PP+Forecast", width=300)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Dashboard",
    "Historical Predictions",
    "New Prediction",
    "Feature Importance",
    "Model Info"
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Performance**")
st.sidebar.metric("R²",       "0.8009")
st.sidebar.metric("MAE",      "178.6 CNY/ton")
st.sidebar.metric("MAPE",     "2.30%")
st.sidebar.metric("Coverage", "87.5%")

# ============================================================
# PAGE 1 — Dashboard
# ============================================================
if page == "Dashboard":
    st.title("🏭 Polypropylene Price Forecast Dashboard")
    st.markdown("Predicting monthly polypropylene futures prices (CNY/ton) — DCE Exchange")

    # --- metric cards ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("R² Score",     "0.8009",    delta="vs baseline -10.30")
    with col2:
        st.metric("MAE",          "178.6 CNY", delta="-1444 vs baseline")
    with col3:
        st.metric("MAPE",         "2.30%",     delta="error rate")
    with col4:
        st.metric("Coverage",     "87.5%",     delta="months inside band")

    st.markdown("---")

    # --- main chart ---
    st.subheader("Actual vs Predicted — Test Period (2022–2023)")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=results['month'], y=results['upper_95'],
        mode='lines', line=dict(width=0),
        showlegend=False, name='upper'
    ))
    fig.add_trace(go.Scatter(
        x=results['month'], y=results['lower_95'],
        mode='lines', line=dict(width=0),
        fill='tonexty', fillcolor='rgba(70,130,180,0.15)',
        name='95% confidence interval'
    ))
    fig.add_trace(go.Scatter(
        x=results['month'], y=results['actual'],
        mode='lines+markers',
        line=dict(color='black', width=2.5),
        marker=dict(size=6),
        name='Actual'
    ))
    fig.add_trace(go.Scatter(
        x=results['month'], y=results['predicted'],
        mode='lines+markers',
        line=dict(color='steelblue', width=2, dash='dash'),
        marker=dict(size=5),
        name='Predicted'
    ))

    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Price (CNY/ton)",
        hovermode='x unified',
        height=450,
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- error distribution ---
    st.subheader("Monthly Error Distribution")
    col1, col2 = st.columns(2)

    with col1:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=results['month'].dt.strftime('%Y-%m'),
            y=results['error'],
            marker_color=['red' if e > 0 else 'steelblue'
                          for e in results['error']],
            name='Error (CNY/ton)'
        ))
        fig2.update_layout(
            title="Prediction Error per Month",
            yaxis_title="Error (CNY/ton)",
            height=350
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=results['month'].dt.strftime('%Y-%m'),
            y=results['error_%'].abs(),
            marker_color=['red' if abs(e) > 5 else 'steelblue'
                          for e in results['error_%']],
            name='Error %'
        ))
        fig3.add_hline(y=5, line_dash='dash', line_color='red',
                       annotation_text='5% threshold')
        fig3.update_layout(
            title="Absolute Error % per Month",
            yaxis_title="Error (%)",
            height=350
        )
        st.plotly_chart(fig3, use_container_width=True)

# ============================================================
# PAGE 2 — Historical Predictions
# ============================================================
elif page == "Historical Predictions":
    st.title("📊 Historical Predictions")
    st.markdown("Full table of actual vs predicted prices with confidence intervals")

    # color inside_band column
    def highlight_band(val):
        return 'background-color: #d4edda' if val else 'background-color: #f8d7da'

    display_df = results.copy()
    display_df['month'] = display_df['month'].dt.strftime('%Y-%m')

    st.dataframe(
        display_df.style.applymap(highlight_band, subset=['inside_band']),
        use_container_width=True,
        height=600
    )

    # summary stats
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Months inside band",
                  f"{results['inside_band'].sum()} / {len(results)}")
    with col2:
        st.metric("Max error",
                  f"{results['error_%'].abs().max():.1f}%")
    with col3:
        st.metric("Months with error > 5%",
                  f"{(results['error_%'].abs() > 5).sum()}")

    st.download_button(
        label="Download predictions CSV",
        data=results.to_csv(index=False),
        file_name="pp_predictions.csv",
        mime="text/csv"
    )

# ============================================================
# PAGE 3 — New Prediction
# ============================================================
elif page == "New Prediction":
    st.title("🔮 New Monthly Prediction")
    st.markdown("Enter current market values to predict next month's price")

    st.info("All values should be for the **current month**. "
            "The model will predict the **next month's** price.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Energy & Feedstock")
        oil       = st.number_input("Crude Oil Price (USD/bbl)",
                                    min_value=20.0,  max_value=200.0, value=80.0,  step=0.5)
        gas       = st.number_input("Natural Gas Price (USD/MMBtu)",
                                    min_value=1.0,   max_value=20.0,  value=4.0,   step=0.1)
        naphtha   = st.number_input("Naphtha Price (USD/bbl)",
                                    min_value=20.0,  max_value=200.0, value=75.0,  step=0.5)
        lldpe     = st.number_input("LLDPE Price (CNY/ton)",
                                    min_value=3000.0,max_value=20000.0,value=8000.0,step=50.0)

    with col2:
        st.subheader("Market Indicators")
        usdcny    = st.number_input("USD/CNY Exchange Rate",
                                    min_value=6.0,   max_value=8.0,   value=7.1,   step=0.01)
        copper    = st.number_input("Copper Price (USD/lb)",
                                    min_value=1.0,   max_value=10.0,  value=4.0,   step=0.05)
        bdi       = st.number_input("Baltic Dry Index",
                                    min_value=200.0, max_value=5000.0,value=1500.0,step=10.0)
        inflation = st.number_input("Inflation Rate (%)",
                                    min_value=-2.0,  max_value=15.0,  value=2.0,   step=0.1)

    st.markdown("---")

    if st.button("🔮 Predict Next Month Price", type="primary"):

        # get last row of merged as template
        last_row = merged.iloc[-1].copy()

        # update with user inputs
        last_row['oil']       = oil
        last_row['gas']       = gas
        last_row['naphtha']   = naphtha
        last_row['lldpe']     = lldpe
        last_row['usdcny']    = usdcny
        last_row['copper']    = copper
        last_row['bdi']       = bdi
        last_row['inflation'] = inflation
        last_row['propylene_proxy'] = naphtha - oil

        # build feature vector
        try:
            X_new = pd.DataFrame([last_row])[features]
            pred  = model.predict(X_new)[0]

            # confidence interval using ±MAE*1.96 as simple approximation
            margin = 178.6 * 1.96
            low    = pred - margin
            high   = pred + margin

            st.success("Prediction complete!")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Price",
                          f"{pred:,.0f} CNY/ton")
            with col2:
                st.metric("Lower bound (95%)",
                          f"{low:,.0f} CNY/ton")
            with col3:
                st.metric("Upper bound (95%)",
                          f"{high:,.0f} CNY/ton")

            # gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=pred,
                delta={'reference': merged['product'].iloc[-1],
                       'valueformat': '.0f'},
                gauge={
                    'axis': {'range': [5000, 13000]},
                    'bar':  {'color': 'steelblue'},
                    'steps': [
                        {'range': [5000,  7000], 'color': '#d4edda'},
                        {'range': [7000,  9000], 'color': '#fff3cd'},
                        {'range': [9000, 13000], 'color': '#f8d7da'}
                    ],
                    'threshold': {
                        'line':  {'color': 'red', 'width': 4},
                        'thickness': 0.75,
                        'value': merged['product'].iloc[-1]
                    }
                },
                title={'text': "Predicted PP Price (CNY/ton)"}
            ))
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info("Make sure all input values are filled correctly.")

# ============================================================
# PAGE 4 — Feature Importance
# ============================================================
elif page == "Feature Importance":
    st.title("🔍 Feature Importance")
    st.markdown("Which variables influence the model most")

    importance_data = {
        'feature': ['usdcny_lag2', 'usdcny_lag1', 'usdcny_lag3',
                    'inflation_roll3', 'naphtha_lag3', 'usdcny',
                    'naphtha_diff_lag3', 'inflation_roll6',
                    'bdi_lag3', 'lldpe'],
        'importance': [0.1986, 0.0991, 0.0889,
                       0.0711, 0.0382, 0.0363,
                       0.0353, 0.0318,
                       0.0255, 0.0254]
    }
    imp_df = pd.DataFrame(importance_data).sort_values('importance', ascending=True)

    fig = go.Figure(go.Bar(
        x=imp_df['importance'],
        y=imp_df['feature'],
        orientation='h',
        marker_color='steelblue'
    ))
    fig.update_layout(
        title="Top 10 Feature Importances",
        xaxis_title="Importance Score",
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Key Insights")
    st.markdown("""
    - **USD/CNY exchange rate** dominates with 42% of total importance —
      polypropylene is priced in CNY but produced using USD-priced oil
    - **Lagged values (lag1, lag2, lag3)** confirm that currency effects
      take 1–3 months to flow through to prices
    - **Inflation rolling average** captures medium-term cost pressure
    - **Naphtha lag3** confirms a 3-month feedstock cost delay
    - **BDI lag3** shows shipping costs affect prices with a 3-month delay
    - **LLDPE** as a sister plastic provides demand-side signal
    """)

# ============================================================
# PAGE 5 — Model Info
# ============================================================
elif page == "Model Info":
    st.title("ℹ️ Model Information")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Architecture")
        st.markdown("""
        | Component | Detail |
        |---|---|
        | Algorithm | XGBoost Regressor |
        | Strategy | Walk-Forward (Window-36) |
        | Window size | 36 months |
        | Tuning | Optuna (60 trials) |
        | Features | 92 total |
        | Train period | Feb 2014 – Dec 2021 |
        | Test period | Jan 2022 – Dec 2023 |
        """)

        st.subheader("Best Parameters")
        params_df = pd.DataFrame(
            best_params.items(),
            columns=['Parameter', 'Value']
        )
        st.dataframe(params_df, use_container_width=True)

    with col2:
        st.subheader("Performance Journey")
        journey = pd.DataFrame({
            'Model': [
                'XGBoost baseline',
                'Window-36 XGBoost',
                'Window-36 Optimized',
                'Window-36 + 4 Features',
                'Window-36 + LLDPE + proxy',
                'Final + Confidence Intervals'
            ],
            'MAE': [1622.6, 331.7, 259.8, 231.5, 178.6, 178.6],
            'R²':  [-10.30, 0.3954, 0.5454, 0.6185, 0.8009, 0.8009]
        })
        st.dataframe(journey, use_container_width=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=journey['Model'], y=journey['R²'],
            mode='lines+markers',
            line=dict(color='steelblue', width=2),
            marker=dict(size=8),
            name='R²'
        ))
        fig.add_hline(y=0, line_dash='dash', line_color='red')
        fig.update_layout(
            title="R² improvement across versions",
            yaxis_title="R²",
            height=350,
            xaxis_tickangle=-30
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Data Sources")
        st.markdown("""
        | Source | Data |
        |---|---|
        | Investing.com | Polypropylene, LLDPE, BDI futures |
        | FRED | USD/CNY exchange rate |
        | yfinance | Naphtha (BZ=F), Copper (HG=F) |
        | CPIAI | US Inflation index |
        """)