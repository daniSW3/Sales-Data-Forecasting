# sales_forecast_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import your existing analysis functions (we'll include them directly for now)
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import scipy.optimize as optimize

# Try to import optional packages
try:
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False

# Copy your existing task functions here
def task1_preprocess_explore(df):
    print("=== TASK 1: Data Preprocessing and Exploration ===")
    
    # Convert to datetime and create copy
    df_clean = df.copy()
    df_clean['Order_Conversion_Date'] = pd.to_datetime(df_clean['Order_Conversion_Date'])
    
    # Data aggregation for different timeframes
    df_clean.set_index('Order_Conversion_Date', inplace=True)
    
    # Weekly aggregation
    weekly_sales = df_clean.groupby('Product Type')['Volume'].resample('W').sum().reset_index()
    weekly_total = df_clean['Volume'].resample('W').sum().reset_index()
    weekly_total['Product Type'] = 'Total'
    
    # Monthly aggregation  
    monthly_sales = df_clean.groupby('Product Type')['Volume'].resample('M').sum().reset_index()
    monthly_total = df_clean['Volume'].resample('M').sum().reset_index()
    monthly_total['Product Type'] = 'Total'
    
    # Yearly aggregation
    yearly_sales = df_clean.groupby('Product Type')['Volume'].resample('Y').sum().reset_index()
    
    monthly_total_ts = monthly_total.set_index('Order_Conversion_Date')['Volume']
    monthly_total_ts_clean = monthly_total_ts.dropna()
    
    return {
        'weekly_sales': weekly_sales,
        'monthly_sales': monthly_sales, 
        'yearly_sales': yearly_sales,
        'monthly_total': monthly_total,
        'monthly_total_ts': monthly_total_ts_clean
    }

def task2_forecasting_models(monthly_total_ts):
    print("\n=== TASK 2: Time Series Forecasting Models ===")
    
    if len(monthly_total_ts) < 6:
        st.warning("Insufficient data for proper forecasting. Need at least 6 months of data.")
        return None
    
    # Prepare data for modeling
    data = monthly_total_ts.reset_index()
    data.columns = ['Date', 'Volume']
    
    # Use 80% for training, 20% for testing
    train_size = max(6, int(len(data) * 0.8))
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    
    if len(test_data) == 0:
        test_data = train_data.iloc[-2:]
        train_data = train_data.iloc[:-2]
    
    # ARIMA Model
    try:
        if PMDARIMA_AVAILABLE:
            auto_model = auto_arima(train_data['Volume'], seasonal=False, stepwise=True, 
                                   suppress_warnings=True, error_action='ignore',
                                   max_p=3, max_q=3, max_d=2)
            order = auto_model.order
        else:
            order = (1, 1, 1)
        
        arima_model = ARIMA(train_data['Volume'], order=order)
        arima_fit = arima_model.fit()
        arima_forecast = arima_fit.forecast(steps=len(test_data))
        arima_forecast_index = test_data['Date']
        
    except Exception as e:
        st.error(f"ARIMA modeling failed: {e}")
        arima_forecast = np.full(len(test_data), train_data['Volume'].mean())
        arima_forecast_index = test_data['Date']
        arima_fit = None
    
    # Simple Moving Average as baseline
    sma_forecast = np.full(len(test_data), train_data['Volume'].rolling(window=3).mean().iloc[-1])
    
    # Model Evaluation
    arima_mae = mean_absolute_error(test_data['Volume'], arima_forecast)
    arima_rmse = np.sqrt(mean_squared_error(test_data['Volume'], arima_forecast))
    arima_mape = np.mean(np.abs((test_data['Volume'] - arima_forecast) / test_data['Volume'])) * 100
    
    sma_mae = mean_absolute_error(test_data['Volume'], sma_forecast)
    sma_rmse = np.sqrt(mean_squared_error(test_data['Volume'], sma_forecast))
    sma_mape = np.mean(np.abs((test_data['Volume'] - sma_forecast) / test_data['Volume'])) * 100
    
    # Determine best model
    best_model = 'ARIMA' if arima_mape < sma_mape else 'SMA'
    
    return {
        'arima_model': arima_fit,
        'best_model': best_model,
        'test_data': test_data,
        'arima_forecast': arima_forecast,
        'sma_forecast': sma_forecast,
        'train_data': train_data,
        'full_data': data,
        'metrics': {
            'arima_mape': arima_mape,
            'sma_mape': sma_mape
        }
    }

def task3_forecast_future(monthly_total_ts, model_results, forecast_months=12):
    print("\n=== TASK 3: Future Sales Forecasting ===")
    
    if model_results is None:
        st.error("No model results available. Cannot generate future forecast.")
        return None
    
    best_model_type = model_results['best_model']
    
    # Future forecast dates
    last_date = monthly_total_ts.index[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                periods=forecast_months, freq='M')
    
    if best_model_type == 'ARIMA' and model_results['arima_model'] is not None:
        arima_model = model_results['arima_model']
        future_forecast = arima_model.forecast(steps=forecast_months)
        
        try:
            forecast_result = arima_model.get_forecast(steps=forecast_months)
            confidence_interval = forecast_result.conf_int()
            lower_ci = confidence_interval.iloc[:, 0]
            upper_ci = confidence_interval.iloc[:, 1]
        except:
            confidence_std = future_forecast * 0.15
            lower_ci = future_forecast - 1.96 * confidence_std
            upper_ci = future_forecast + 1.96 * confidence_std
    else:
        historical_avg = monthly_total_ts.mean()
        future_forecast = np.full(forecast_months, historical_avg)
        confidence_std = future_forecast * 0.2
        lower_ci = future_forecast - 1.96 * confidence_std
        upper_ci = future_forecast + 1.96 * confidence_std
    
    return {
        'future_dates': future_dates,
        'future_forecast': future_forecast,
        'confidence_lower': lower_ci,
        'confidence_upper': upper_ci,
        'best_model_used': best_model_type
    }

def task4_optimize_strategy(monthly_sales, future_forecast_results):
    print("\n=== TASK 4: Product Mix Optimization ===")
    
    # Prepare product data
    product_data = monthly_sales.pivot_table(
        index='Order_Conversion_Date', 
        columns='Product Type', 
        values='Volume', 
        aggfunc='sum'
    ).fillna(0)
    
    # Calculate key metrics
    product_metrics = pd.DataFrame({
        'Avg_Monthly_Sales': product_data.mean(),
        'Sales_Std': product_data.std(),
        'Total_Sales': product_data.sum(),
        'CV': (product_data.std() / product_data.mean())
    })
    
    # Define optimization constraints
    MAX_TOTAL_CAPACITY = 950000
    MIN_PRODUCT_ALLOCATION = 0.10
    LAYER_PROFIT_MARGIN = 0.25
    SASSO_PROFIT_MARGIN = 0.35
    
    # Forecasted demand
    if future_forecast_results is not None:
        layer_forecast = future_forecast_results['future_forecast'].mean()
    else:
        layer_forecast = product_metrics.loc['Layer', 'Avg_Monthly_Sales']
    
    sasso_historical = product_metrics.loc['Sasso', 'Avg_Monthly_Sales']
    
    # Optimization objective function
    def objective_function(allocation):
        layer_alloc, sasso_alloc = allocation
        layer_sales = min(layer_alloc * MAX_TOTAL_CAPACITY, layer_forecast)
        sasso_sales = min(sasso_alloc * MAX_TOTAL_CAPACITY, sasso_historical)
        total_profit = (layer_sales * LAYER_PROFIT_MARGIN + sasso_sales * SASSO_PROFIT_MARGIN)
        return -total_profit
    
    # Constraints and bounds
    constraints = [{'type': 'eq', 'fun': lambda x: x[0] + x[1] - 1}]
    bounds = [(MIN_PRODUCT_ALLOCATION, 1 - MIN_PRODUCT_ALLOCATION), 
              (MIN_PRODUCT_ALLOCATION, 1 - MIN_PRODUCT_ALLOCATION)]
    
    # Run optimization
    initial_guess = [0.5, 0.5]
    result = optimize.minimize(objective_function, initial_guess, 
                     method='SLSQP', bounds=bounds, constraints=constraints)
    
    optimal_allocation = result.x
    optimal_profit = -result.fun
    
    # Calculate optimal quantities
    layer_quantity = optimal_allocation[0] * MAX_TOTAL_CAPACITY
    sasso_quantity = optimal_allocation[1] * MAX_TOTAL_CAPACITY
    total_used = layer_quantity + sasso_quantity
    capacity_utilization = (total_used / MAX_TOTAL_CAPACITY) * 100
    
    return {
        'optimal_allocation': optimal_allocation,
        'optimal_quantities': [layer_quantity, sasso_quantity],
        'expected_profit': optimal_profit,
        'capacity_utilization': capacity_utilization,
        'product_metrics': product_metrics
    }

# Streamlit App Configuration
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .forecast-card {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">🚀 Sales Forecasting & Optimization Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar for file upload and controls
    with st.sidebar:
        st.header("📁 Data Input")
        uploaded_file = st.file_uploader("Upload Sales Data Excel File", type=['xlsx'])
        
        st.header("⚙️ Settings")
        forecast_months = st.slider("Forecast Period (Months)", 3, 24, 12)
        
        st.header("🎯 Analysis Options")
        run_forecasting = st.checkbox("Run Sales Forecasting", value=True)
        run_optimization = st.checkbox("Run Product Optimization", value=True)
        
        if st.button("🚀 Run Full Analysis", type="primary"):
            st.session_state.run_analysis = True
        else:
            if 'run_analysis' not in st.session_state:
                st.session_state.run_analysis = False
    
    # Main content area
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_excel(uploaded_file, engine='openpyxl')
            st.success(f"✅ Data loaded successfully! {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Display data preview
            with st.expander("📋 Data Preview", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("First 10 rows:")
                    st.dataframe(df.head(10), use_container_width=True)
                with col2:
                    st.write("Data Summary:")
                    st.dataframe(df.describe(), use_container_width=True)
            
            if st.session_state.run_analysis:
                # Initialize progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Task 1: Data Preprocessing
                status_text.text("🔄 Task 1: Preprocessing Data...")
                aggregated_data = task1_preprocess_explore(df)
                progress_bar.progress(25)
                
                # Display Task 1 Results
                st.header("📈 Data Overview & Trends")
                
                # Key Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_sales = df['Volume'].sum()
                    st.metric("Total Sales Volume", f"{total_sales:,.0f}")
                
                with col2:
                    avg_sales = df['Volume'].mean()
                    st.metric("Average Order Volume", f"{avg_sales:,.0f}")
                
                with col3:
                    product_mix = df['Product Type'].value_counts()
                    st.metric("Top Product", product_mix.index[0])
                
                with col4:
                    date_range = f"{df['Order_Conversion_Date'].min().strftime('%Y-%m-%d')} to {df['Order_Conversion_Date'].max().strftime('%Y-%m-%d')}"
                    st.metric("Data Period", date_range)
                
                # Sales Trends Visualization
                st.subheader("Sales Trends Over Time")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                
                # Monthly trend
                monthly_data = aggregated_data['monthly_total_ts']
                ax1.plot(monthly_data.index, monthly_data.values, marker='o', linewidth=2, color='#1f77b4')
                ax1.set_title('Monthly Sales Trend', fontsize=14, fontweight='bold')
                ax1.set_xlabel('Date')
                ax1.set_ylabel('Sales Volume')
                ax1.grid(True, alpha=0.3)
                ax1.tick_params(axis='x', rotation=45)
                
                # Product distribution
                product_totals = df.groupby('Product Type')['Volume'].sum()
                colors = ['#ff9999', '#66b3ff']
                ax2.pie(product_totals.values, labels=product_totals.index, autopct='%1.1f%%', 
                       startangle=90, colors=colors)
                ax2.set_title('Sales Distribution by Product Type', fontsize=14, fontweight='bold')
                
                st.pyplot(fig)
                
                if run_forecasting:
                    # Task 2 & 3: Forecasting
                    status_text.text("🔮 Task 2 & 3: Running Sales Forecasting...")
                    model_results = task2_forecasting_models(aggregated_data['monthly_total_ts'])
                    future_forecast = task3_forecast_future(
                        aggregated_data['monthly_total_ts'], 
                        model_results, 
                        forecast_months=forecast_months
                    )
                    progress_bar.progress(60)
                    
                    # Display Forecasting Results
                    if future_forecast is not None:
                        st.header("🔮 Sales Forecast")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Forecast Summary")
                            avg_forecast = future_forecast['future_forecast'].mean()
                            total_forecast = future_forecast['future_forecast'].sum()
                            
                            if len(future_forecast['future_forecast']) > 1:
                                growth_rate = ((future_forecast['future_forecast'][-1] - future_forecast['future_forecast'][0]) / future_forecast['future_forecast'][0]) * 100
                            else:
                                growth_rate = 0
                            
                            st.metric("Average Monthly Forecast", f"{avg_forecast:,.0f}")
                            st.metric("Total Forecast Period Sales", f"{total_forecast:,.0f}")
                            st.metric("Growth Rate", f"{growth_rate:+.1f}%")
                            st.metric("Model Used", future_forecast['best_model_used'])
                        
                        with col2:
                            st.subheader("Forecast Details")
                            forecast_df = pd.DataFrame({
                                'Month': [d.strftime('%Y-%m') for d in future_forecast['future_dates']],
                                'Forecast': future_forecast['future_forecast'],
                                'Lower CI': future_forecast['confidence_lower'],
                                'Upper CI': future_forecast['confidence_upper']
                            })
                            st.dataframe(forecast_df.style.format({
                                'Forecast': '{:,.0f}',
                                'Lower CI': '{:,.0f}', 
                                'Upper CI': '{:,.0f}'
                            }), use_container_width=True)
                        
                        # Forecast Visualization
                        st.subheader("Forecast Visualization")
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        # Historical data
                        historical_data = aggregated_data['monthly_total_ts']
                        ax.plot(historical_data.index, historical_data.values, 
                               label='Historical', color='blue', linewidth=2)
                        
                        # Forecast data
                        ax.plot(future_forecast['future_dates'], future_forecast['future_forecast'], 
                               label='Forecast', color='red', linestyle='--', linewidth=2)
                        
                        # Confidence interval
                        ax.fill_between(future_forecast['future_dates'], 
                                      future_forecast['confidence_lower'],
                                      future_forecast['confidence_upper'],
                                      alpha=0.3, color='red', label='95% Confidence Interval')
                        
                        ax.set_title('Sales Forecast with Confidence Intervals', fontsize=14, fontweight='bold')
                        ax.set_xlabel('Date')
                        ax.set_ylabel('Sales Volume')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        ax.tick_params(axis='x', rotation=45)
                        
                        st.pyplot(fig)
                
                if run_optimization and future_forecast is not None:
                    # Task 4: Optimization
                    status_text.text("⚖️ Task 4: Optimizing Product Mix...")
                    optimization_results = task4_optimize_strategy(
                        aggregated_data['monthly_sales'],
                        future_forecast
                    )
                    progress_bar.progress(85)
                    
                    # Display Optimization Results
                    st.header("⚖️ Product Mix Optimization")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Optimal Allocation")
                        allocation_df = pd.DataFrame({
                            'Product': ['Layer', 'Sasso'],
                            'Allocation %': [f"{optimization_results['optimal_allocation'][0]:.1%}", 
                                           f"{optimization_results['optimal_allocation'][1]:.1%}"],
                            'Quantity': optimization_results['optimal_quantities']
                        })
                        st.dataframe(allocation_df, use_container_width=True)
                        
                        st.metric("Expected Monthly Profit", f"${optimization_results['expected_profit']:,.2f}")
                        st.metric("Capacity Utilization", f"{optimization_results['capacity_utilization']:.1f}%")
                    
                    with col2:
                        st.subheader("Production Plan")
                        fig, ax = plt.subplots(figsize=(8, 6))
                        products = ['Layer', 'Sasso']
                        quantities = optimization_results['optimal_quantities']
                        
                        bars = ax.bar(products, quantities, color=['skyblue', 'lightcoral'])
                        ax.set_ylabel('Production Quantity')
                        ax.set_title('Optimal Production Quantities', fontweight='bold')
                        ax.grid(True, alpha=0.3)
                        
                        for bar, quantity in zip(bars, quantities):
                            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                                   f'{quantity:,.0f}', ha='center', va='bottom', fontweight='bold')
                        
                        st.pyplot(fig)
                    
                    # Risk-Return Profile
                    st.subheader("Product Risk-Return Profile")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    product_metrics = optimization_results['product_metrics']
                    ax.scatter(product_metrics['Avg_Monthly_Sales'], 
                              product_metrics['CV'],
                              s=200, alpha=0.6)
                    
                    for product in product_metrics.index:
                        ax.annotate(product, 
                                  (product_metrics.loc[product, 'Avg_Monthly_Sales'],
                                   product_metrics.loc[product, 'CV']),
                                  xytext=(5, 5), textcoords='offset points',
                                  fontweight='bold')
                    
                    ax.set_xlabel('Average Monthly Sales (Volume)')
                    ax.set_ylabel('Coefficient of Variation (Risk)')
                    ax.set_title('Product Risk-Return Profile', fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                
                # Complete analysis
                progress_bar.progress(100)
                status_text.text("✅ Analysis Complete!")
                
                # Download Results Section
                st.header("📥 Download Results")
                
                # Create summary report
                summary_data = {
                    'Analysis Date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                    'Total Historical Sales': [total_sales],
                    'Forecast Period (Months)': [forecast_months],
                    'Total Forecasted Sales': [total_forecast if 'total_forecast' in locals() else 0],
                    'Optimal Layer Allocation': [f"{optimization_results['optimal_allocation'][0]:.1%}" if 'optimization_results' in locals() else 'N/A'],
                    'Optimal Sasso Allocation': [f"{optimization_results['optimal_allocation'][1]:.1%}" if 'optimization_results' in locals() else 'N/A'],
                    'Expected Monthly Profit': [f"${optimization_results['expected_profit']:,.2f}" if 'optimization_results' in locals() else 'N/A'],
                }
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
                
                # Download buttons
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    csv = summary_df.to_csv(index=False)
                    st.download_button(
                        label="📄 Download Summary Report",
                        data=csv,
                        file_name="sales_forecast_summary.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    if 'forecast_df' in locals():
                        forecast_csv = forecast_df.to_csv(index=False)
                        st.download_button(
                            label="📈 Download Forecast Data",
                            data=forecast_csv,
                            file_name="sales_forecast_details.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                
                with col3:
                    if 'optimization_results' in locals():
                        optimization_data = pd.DataFrame({
                            'Product': ['Layer', 'Sasso'],
                            'Optimal_Allocation': optimization_results['optimal_allocation'],
                            'Production_Quantity': optimization_results['optimal_quantities']
                        })
                        opt_csv = optimization_data.to_csv(index=False)
                        st.download_button(
                            label="⚖️ Download Optimization Results",
                            data=opt_csv,
                            file_name="optimization_results.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
        
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            st.info("Please check that your Excel file has the correct format with columns: Order_Conversion_Date, Volume, Product Type")
    
    else:
        # Welcome screen when no file is uploaded
        st.info("👆 Please upload your Sales Data Excel file to get started!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 What This Dashboard Does:")
            st.markdown("""
            - **Sales Forecasting**: Predict future sales using advanced models
            - **Product Optimization**: Find optimal product mix for maximum profit
            - **Interactive Visualizations**: Explore trends and patterns
            - **Export Results**: Download reports for decision-making
            """)
        
        with col2:
            st.subheader("🚀 How to Use:")
            st.markdown("""
            1. **Upload** your Excel file with sales data
            2. **Adjust** settings in the sidebar
            3. **Click** 'Run Full Analysis'
            4. **Explore** results across different sections
            5. **Download** reports for stakeholders
            """)
            
            st.subheader("📊 Expected Data Format:")
            st.markdown("""
            Your Excel file should contain:
            - `Order_Conversion_Date`: Date of sales
            - `Volume`: Sales quantity
            - `Product Type`: Product category (Layer/Sasso)
            - `Order_Conversion_Status`: Order status
            """)

if __name__ == "__main__":
    main()