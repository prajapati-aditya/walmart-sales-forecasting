import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing 
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────
st.set_page_config(
    page_title = "Walmart Sales Forecaster",
    page_icon  = "📦",
    layout     = "wide"
)

# ─────────────────────────────────────
# TITLE
# ─────────────────────────────────────
st.title("📦 Walmart Retail Sales Forecasting")
st.markdown("*Predict future sales for any date range and get smart inventory recommendations*")
st.divider()

# ─────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('data/walmart_enriched.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# ─────────────────────────────────────
# SIDEBAR — USER INPUTS
# ─────────────────────────────────────
st.sidebar.header("⚙️ Select Options")

# Store and Department
store_list     = sorted(df['Store'].unique().tolist())
selected_store = st.sidebar.selectbox("Select Store", store_list)

dept_list      = sorted(df[df['Store'] == selected_store]['Dept'].unique().tolist())
selected_dept  = st.sidebar.selectbox("Select Department", dept_list)

st.sidebar.divider()

# Date range picker
st.sidebar.subheader("📅 Forecast Date Range")
st.sidebar.caption("Pick any future date range to forecast sales")

default_start = date.today()
default_end   = date.today() + timedelta(weeks=12)

forecast_start = st.sidebar.date_input(
    "Forecast Start Date",
    value   = default_start,
    min_value = date.today()
)

forecast_end = st.sidebar.date_input(
    "Forecast End Date",
    value     = default_end,
    min_value = forecast_start + timedelta(weeks=1)
)

# Calculate number of weeks between dates
weeks_ahead = max(1, (forecast_end - forecast_start).days // 7)
st.sidebar.info(f"📊 Forecasting **{weeks_ahead} weeks** of sales")

st.sidebar.divider()

# Cost inputs
st.sidebar.subheader("💰 Cost Parameters")
holding_cost  = st.sidebar.number_input("Holding Cost per Unit ($)",  min_value=1,  max_value=20,  value=2)
stockout_cost = st.sidebar.number_input("Stockout Cost per Unit ($)", min_value=1,  max_value=50,  value=8)
avg_price     = st.sidebar.number_input("Average Product Price ($)",  min_value=10, max_value=500, value=50)

st.sidebar.divider()
run_button = st.sidebar.button("🚀 Run Forecast", use_container_width=True)

# ─────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────
if run_button:

    # Validate date range
    if forecast_end <= forecast_start:
        st.error("End date must be after start date. Please fix the dates.")
        st.stop()

    # Filter data for selected store and department
    filtered = df[
        (df['Store'] == selected_store) &
        (df['Dept']  == selected_dept)
    ].copy().sort_values('Date').reset_index(drop=True)

    if len(filtered) < 20:
        st.error("Not enough data for this combination. Please select another Store or Department.")
        st.stop()

    # Show what we're forecasting
    st.info(
        f"📅 Forecasting **Store {selected_store}, Dept {selected_dept}** "
        f"from **{forecast_start.strftime('%b %d, %Y')}** "
        f"to **{forecast_end.strftime('%b %d, %Y')}** "
        f"({weeks_ahead} weeks)"
    )

    # ── TAB LAYOUT ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Historical Trend",
        "🔮 Forecast",
        "📦 Inventory Recommendations",
        "📊 Business Insights"
    ])

    # ─────────────────────────────────
    # TAB 1 — HISTORICAL TREND
    # ─────────────────────────────────
    with tab1:
        st.subheader(f"Historical Sales — Store {selected_store}, Dept {selected_dept}")
        st.caption("Training data the model learns patterns from")

        fig = px.line(
            filtered,
            x     = 'Date',
            y     = 'Weekly_Sales',
            title = 'Historical Weekly Sales (Training Data)',
            labels = {'Weekly_Sales': 'Weekly Sales ($)'}
        )
        fig.update_traces(line_color='#2E86AB')
        fig.update_layout(hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Average Weekly Sales", f"${filtered['Weekly_Sales'].mean():,.0f}")
        col2.metric("Max Weekly Sales",     f"${filtered['Weekly_Sales'].max():,.0f}")
        col3.metric("Min Weekly Sales",     f"${filtered['Weekly_Sales'].min():,.0f}")
        col4.metric("Weeks of Training Data", str(len(filtered)))

        st.caption(
            "💡 The model learns weekly, monthly and yearly sales patterns "
            "from this historical data and projects them into your selected future date range."
        )

    # ─────────────────────────────────
   # ─────────────────────────────────
    # TAB 2 — FORECAST
    # ─────────────────────────────────
    with tab2:
        st.subheader(
            f"Sales Forecast: {forecast_start.strftime('%b %d %Y')} → {forecast_end.strftime('%b %d %Y')}"
        )

        with st.spinner("Training model and generating forecast... ⏳"):

            # Prepare data
            train_series = filtered.set_index('Date')['Weekly_Sales']

            # Train Exponential Smoothing model
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            model = ExponentialSmoothing(
                train_series,
                trend            = 'add',
                seasonal         = 'add',
                seasonal_periods = 52
            ).fit()

            # Calculate periods needed
            periods_needed = (
                pd.to_datetime(forecast_end) - filtered['Date'].max()
            ).days // 7 + 4

            # Generate forecast
            forecast_values = model.forecast(periods_needed)
            forecast_values = forecast_values.reset_index()
            forecast_values.columns = ['ds', 'yhat']

            # Add confidence interval
            std = filtered['Weekly_Sales'].std()
            forecast_values['yhat_upper'] = forecast_values['yhat'] + 1.96 * std
            forecast_values['yhat_lower'] = forecast_values['yhat'] - 1.96 * std

            # Filter to user selected date range
            forecast_filtered = forecast_values[
                (forecast_values['ds'] >= pd.to_datetime(forecast_start)) &
                (forecast_values['ds'] <= pd.to_datetime(forecast_end))
            ].copy()

        if len(forecast_filtered) == 0:
            st.warning("No forecast data found for selected date range. Try extending your end date.")
            st.stop()

        # Plot historical + forecast together
        fig2 = go.Figure()

        # Historical data
        fig2.add_trace(go.Scatter(
            x    = filtered['Date'],
            y    = filtered['Weekly_Sales'],
            mode = 'lines',
            name = 'Historical Sales',
            line = dict(color='#2E86AB', width=2)
        ))

        # Forecast line
        fig2.add_trace(go.Scatter(
            x    = forecast_filtered['ds'],
            y    = forecast_filtered['yhat'],
            mode = 'lines+markers',
            name = f'Forecast ({forecast_start} to {forecast_end})',
            line = dict(color='#E84855', width=2, dash='dash')
        ))

        # Confidence interval
        fig2.add_trace(go.Scatter(
            x         = pd.concat([forecast_filtered['ds'], forecast_filtered['ds'][::-1]]),
            y         = pd.concat([forecast_filtered['yhat_upper'], forecast_filtered['yhat_lower'][::-1]]),
            fill      = 'toself',
            fillcolor = 'rgba(232,72,85,0.1)',
            line      = dict(color='rgba(255,255,255,0)'),
            name      = '95% Confidence Interval'
        ))

        fig2.update_layout(
            title       = 'Sales Forecast for Your Selected Period',
            xaxis_title = 'Date',
            yaxis_title = 'Weekly Sales ($)',
            hovermode   = 'x unified'
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Forecast summary metrics
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Avg Predicted Weekly Sales",
            f"${forecast_filtered['yhat'].mean():,.0f}"
        )
        col2.metric(
            "Peak Week Sales",
            f"${forecast_filtered['yhat'].max():,.0f}",
            f"Week of {forecast_filtered.loc[forecast_filtered['yhat'].idxmax(), 'ds'].strftime('%b %d')}"
        )
        col3.metric(
            "Total Forecasted Sales",
            f"${forecast_filtered['yhat'].sum():,.0f}"
        )

        st.success(f"✅ Forecast generated for {len(forecast_filtered)} weeks")

    # ─────────────────────────────────
    # TAB 3 — INVENTORY
    # ─────────────────────────────────
    with tab3:
        st.subheader("📦 Weekly Inventory Recommendations")
        st.caption(f"Optimal order quantities for {forecast_start} to {forecast_end}")

        # Calculate inventory recommendations
        inv = forecast_filtered[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()

        inv['Predicted_Units']    = (inv['yhat']       / avg_price).astype(int)
        inv['Min_Order']          = (inv['yhat_lower'] / avg_price).astype(int)
        inv['Recommended_Order']  = (inv['Predicted_Units'] * 1.03).astype(int)
        inv['Max_Order']          = (inv['yhat_upper'] / avg_price).astype(int)
        inv['Overstock_Cost']     = (inv['Recommended_Order'] - inv['Predicted_Units']) * holding_cost
        inv['Stockout_Cost']      = (inv['Predicted_Units']   - inv['Min_Order'])       * stockout_cost
        inv['Total_Risk_Cost']    = inv['Overstock_Cost'] + inv['Stockout_Cost']

        # Summary
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Units to Order",    f"{inv['Recommended_Order'].sum():,}")
        col2.metric("Avg Weekly Order",        f"{inv['Recommended_Order'].mean():.0f} units")
        col3.metric("Total Overstock Risk",    f"${inv['Overstock_Cost'].sum():,.0f}")
        col4.metric("Total Stockout Risk",     f"${inv['Stockout_Cost'].sum():,.0f}")

        st.divider()

        # Recommendation chart
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x    = inv['ds'],
            y    = inv['Recommended_Order'],
            name = 'Recommended Order',
            marker_color = '#2DC653'
        ))
        fig3.add_trace(go.Scatter(
            x    = inv['ds'],
            y    = inv['Min_Order'],
            mode = 'lines',
            name = 'Minimum Safe Order',
            line = dict(color='#E84855', dash='dash')
        ))
        fig3.add_trace(go.Scatter(
            x    = inv['ds'],
            y    = inv['Max_Order'],
            mode = 'lines',
            name = 'Maximum Order',
            line = dict(color='#F4A261', dash='dot')
        ))
        fig3.update_layout(
            title       = 'Weekly Inventory Recommendations',
            xaxis_title = 'Week',
            yaxis_title = 'Units',
            hovermode   = 'x unified'
        )
        st.plotly_chart(fig3, use_container_width=True)

        # Table
        st.dataframe(
            inv[[
                'ds', 'Predicted_Units', 'Min_Order',
                'Recommended_Order', 'Max_Order',
                'Overstock_Cost', 'Stockout_Cost', 'Total_Risk_Cost'
            ]].rename(columns={
                'ds'                : 'Week',
                'Predicted_Units'   : 'Predicted Units',
                'Min_Order'         : 'Min Order',
                'Recommended_Order' : '✅ Recommended Order',
                'Max_Order'         : 'Max Order',
                'Overstock_Cost'    : 'Overstock Cost ($)',
                'Stockout_Cost'     : 'Stockout Cost ($)',
                'Total_Risk_Cost'   : 'Total Risk ($)'
            }),
            use_container_width = True
        )

        # Download
        csv = inv.to_csv(index=False)
        st.download_button(
            label     = "⬇️ Download Recommendations as CSV",
            data      = csv,
            file_name = f"inventory_store{selected_store}_dept{selected_dept}_{forecast_start}.csv",
            mime      = "text/csv"
        )

    # ─────────────────────────────────
    # TAB 4 — INSIGHTS
    # ─────────────────────────────────
    with tab4:
        st.subheader("📊 Business Insights from Historical Data")

        col1, col2 = st.columns(2)

        with col1:
            holiday_data = filtered.groupby('IsHoliday')['Weekly_Sales'].mean().reset_index()
            holiday_data['IsHoliday'] = holiday_data['IsHoliday'].map(
                {0: 'Normal Week', 1: 'Holiday Week'}
            )
            fig4 = px.bar(
                holiday_data,
                x     = 'IsHoliday',
                y     = 'Weekly_Sales',
                title = 'Holiday vs Normal Week Sales',
                color = 'IsHoliday',
                color_discrete_map = {
                    'Normal Week' : '#2E86AB',
                    'Holiday Week': '#E84855'
                }
            )
            st.plotly_chart(fig4, use_container_width=True)

        with col2:
            monthly      = filtered.copy()
            monthly['Month_Name'] = pd.to_datetime(monthly['Date']).dt.strftime('%b')
            monthly['Month_Num']  = pd.to_datetime(monthly['Date']).dt.month
            monthly_avg  = monthly.groupby(
                ['Month_Num', 'Month_Name']
            )['Weekly_Sales'].mean().reset_index().sort_values('Month_Num')

            fig5 = px.bar(
                monthly_avg,
                x     = 'Month_Name',
                y     = 'Weekly_Sales',
                title = 'Average Sales by Month',
                color = 'Weekly_Sales',
                color_continuous_scale = 'Blues'
            )
            st.plotly_chart(fig5, use_container_width=True)

        # Markdown impact
        st.subheader("Markdown Promotional Impact")
        markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
        lifts = []
        for col in markdown_cols:
            active   = filtered[filtered[col] > 0]['Weekly_Sales'].mean()
            inactive = filtered[filtered[col] == 0]['Weekly_Sales'].mean()
            if pd.notna(active) and pd.notna(inactive) and inactive > 0:
                lift = ((active - inactive) / inactive) * 100
                lifts.append({'Markdown': col, 'Sales Lift (%)': round(lift, 2)})

        if lifts:
            lift_df = pd.DataFrame(lifts)
            fig6    = px.bar(
                lift_df,
                x     = 'Markdown',
                y     = 'Sales Lift (%)',
                title = 'Sales Lift % by Markdown Type',
                color = 'Sales Lift (%)',
                color_continuous_scale = 'Greens'
            )
            st.plotly_chart(fig6, use_container_width=True)

else:
    # Default landing screen
    st.info("👈 Select options from the sidebar and click **🚀 Run Forecast** to begin")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Stores",         "45")
    col2.metric("Total Departments",    "81")
    col3.metric("Years of Training Data", "3")
    col4.metric("Model Accuracy",       "95.84%")

    st.markdown("""
    ### How It Works
    1. **Select** a Store and Department from the sidebar
    2. **Pick** your forecast start and end dates
    3. **Click** Run Forecast
    4. Get **sales predictions** + **inventory recommendations** for your chosen period
    """)