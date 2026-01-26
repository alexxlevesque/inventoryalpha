import streamlit as st
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.ingestor import DataIngestor
from src.brain.state_estimator import StateEstimator
from src.logic.inventory_manager import InventoryManager

st.set_page_config(page_title="Inventory Alpha Diagnostic", layout="wide")

# Custom Style
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_data():
    # Try to find the M5 data folder
    possible_paths = [
        "M5 Forecasting Accuracy",
        "../M5 Forecasting Accuracy",
        "/Users/alexlevesque/Desktop/Projects 2026/inventoryalpha/M5 Forecasting Accuracy"
    ]
    
    data_path = None
    for p in possible_paths:
        if os.path.exists(p) and os.path.exists(os.path.join(p, "sales_train_evaluation.csv")):
            data_path = p
            break
            
    if not data_path:
        raise FileNotFoundError("Could not find 'M5 Forecasting Accuracy' directory.")
        
    ingestor = DataIngestor(data_path)
    ingestor.load_raw()
    return ingestor

def main():
    st.title("Kalman Inventory Alpha Diagnostic ⚡")
    st.markdown("### Detect Phantom Demand & Reduce Dead Inventory")
    
    with st.spinner("Loading M5 Data..."):
        try:
            ingestor = load_data()
        except Exception as e:
            st.error(f"Data Load Error: {e}")
            return

    # --- Sidebar Configuration ---
    st.sidebar.header("Configuration")
    
    # Store/Item Selection
    # Load unique values (cached for performance)
    @st.cache_data
    def get_selectors(data_path):
        # Create a temp ingestor to just get the lists
        # Note: We reuse the main ingestor from load_data but we need to ensure it's loaded
        # Since we are inside main(), ingestor is available.
        pass

    with st.spinner("Fetching Store/Item lists..."):
        # We use the existing ingestor
        unique_stores = ingestor.get_unique_stores()
        
    store_id = st.sidebar.selectbox("Store ID", unique_stores, index=0)
    
    with st.spinner(f"Fetching Items for {store_id}..."):
        unique_items = ingestor.get_unique_items(store_id)
        
    # Default to HOBBIES_1_001 if available, else first item
    default_idx = 0
    if "HOBBIES_1_001" in unique_items:
        default_idx = unique_items.index("HOBBIES_1_001")
        
    item_id = st.sidebar.selectbox("Item ID", unique_items, index=default_idx)
    
    st.sidebar.markdown("---")
    st.sidebar.header("Inventory Parameters")
    
    lead_time = st.sidebar.slider("Lead Time (Days)", 1, 30, 7)
    service_level = st.sidebar.slider("Service Level (%)", 80.0, 99.9, 95.0, 0.1) / 100.0
    
    # Simulation
    st.sidebar.markdown("---")
    st.sidebar.header("Simulation")
    simulated_inventory = st.sidebar.number_input("Current Inventory (Units)", min_value=0, value=50)

    # --- Main Analysis ---
    try:
        # Fetch Data
        series = ingestor.get_clean_series(item_id, store_id)
        
        # Kalman Filter
        kf = StateEstimator()
        estimates, uncertainties = kf.run_filter(series)
        
        # Latest State
        latest_demand = estimates[-1]
        latest_uncertainty = uncertainties[-1]
        
        # Inventory Logic
        im = InventoryManager(lead_time=lead_time, service_level=service_level)
        
        analysis = im.detect_dead_inventory(
            sku=item_id,
            current_inventory=simulated_inventory,
            demand_estimate=latest_demand,
            demand_uncertainty=latest_uncertainty
        )
        
        # Display KPIs
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Target Inventory", f"{analysis['target_inventory']} units")
        kpi2.metric("Safety Stock", f"{analysis['safety_stock']} units")
        kpi3.metric("Excess Inventory", f"{analysis['excess_units']} units", 
                   delta=-analysis['excess_units'], delta_color="inverse")
        kpi4.metric("Reduction Potential", f"{analysis['reduction_potential_pct']}%")
        
        # Dynamic Explanation Sentence
        if analysis['excess_units'] > 0:
            st.markdown(f"**Insight:** Because your **Current Inventory** ({simulated_inventory} units) exceeds the **Target Inventory** ({analysis['target_inventory']} units), we've identified **{analysis['excess_units']} units** as 'Dead Inventory'. Reducing this could yield a **{analysis['reduction_potential_pct']}% reduction** in capital tied up, while still maintaining your **{analysis['safety_stock']} unit** safety stock buffer.")
        else:
            st.markdown(f"**Insight:** Your **Current Inventory** ({simulated_inventory} units) is lean relative to the **Target Inventory** ({analysis['target_inventory']} units). No excess inventory detected; your current levels are efficiently supporting the {service_level*100}% service level requirement.")

        st.divider()

        # --- Plots ---
        col_main, col_side = st.columns([2, 1])
        
        with col_main:
            st.subheader("Demand & Inventory Tracking")
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            
            # Plot Data
            lookback = 90
            start_idx = max(0, len(series) - lookback)
            time_axis = np.arange(start_idx, len(series))
            
            # Actuals
            ax1.scatter(time_axis, series[start_idx:], alpha=0.4, color='gray', s=15, label='Actual Sales')
            
            # Kalman Line
            ax1.plot(time_axis, estimates[start_idx:], color='#007acc', linewidth=2.5, label='Kalman Demand Estimate')
            
            # Confidence Interval
            std_dev = np.sqrt(uncertainties[start_idx:])
            ax1.fill_between(time_axis, 
                           estimates[start_idx:] - 1.96*std_dev, 
                           estimates[start_idx:] + 1.96*std_dev, 
                           color='#007acc', alpha=0.15, label='95% Confidence')
            
            ax1.set_ylabel("Units")
            ax1.set_xlabel("Time (Days)")
            ax1.legend(loc='upper left')
            ax1.grid(True, linestyle='--', alpha=0.3)
            ax1.set_title(f"True Demand Signal Extraction via Kalman Filter")
            
            st.pyplot(fig1)
            st.caption("**Demand vs. Estimate**: Raw sales (dots) are often noisy. The Kalman Filter extracts the 'True Demand' signal (bold line). The shaded blue area represents our 95% confidence interval—wider bands indicate higher volatility or less data certainty.")
            
        with col_side:
            st.subheader("Demand Probabilities")
            # PDF of Demand over Lead Time
            # Mean = Demand * LeadTime
            # Variance = Uncertainty * LeadTime (Random Walk assumption)
            mu_lt = latest_demand * lead_time
            sigma_lt = np.sqrt(max(latest_uncertainty, 1e-6) * lead_time)
            
            x = np.linspace(max(0, mu_lt - 4*sigma_lt), mu_lt + 4*sigma_lt, 200)
            y = stats.norm.pdf(x, mu_lt, sigma_lt)
            
            fig2, ax2 = plt.subplots(figsize=(5, 5))
            ax2.plot(x, y, color='green', linewidth=2)
            ax2.fill_between(x, y, where=(x <= analysis['target_inventory']), color='green', alpha=0.2, label='Service Level Met')
            ax2.fill_between(x, y, where=(x > analysis['target_inventory']), color='red', alpha=0.2, label='Stockout Risk')
            
            ax2.axvline(analysis['target_inventory'], color='black', linestyle='--', label='Target Level')
            ax2.set_title(f"Demand Distribution (Next {lead_time} Days)")
            ax2.legend(fontsize='small')
            ax2.get_yaxis().set_visible(False)
            
            st.pyplot(fig2)
            st.caption("**Risk Analysis**: This Bell curve represents the likelihood of different demand totals over the next lead time window. The red tail shows the probability of demand exceeding your target level (Stockout Risk).")
            
        # Inventory Bar Chart
        st.subheader("Inventory Snapshot")
        fig3, ax3 = plt.subplots(figsize=(8, 2))
        
        bars = ['Safety Stock', 'Cycle Stock', 'Target Level', 'Current Inventory']
        values = [analysis['safety_stock'], 
                  analysis['estimated_demand'] * lead_time, 
                  analysis['target_inventory'], 
                  simulated_inventory]
        colors = ['orange', 'lightblue', 'blue', 'purple']
        
        y_pos = np.arange(len(bars))
        ax3.barh(y_pos, values, color=colors)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(bars)
        ax3.set_xlabel("Units")
        
        # Add labels on bars
        for i, v in enumerate(values):
            ax3.text(v + 1, i, str(v), va='center')
            
        st.pyplot(fig3)
        st.caption("**Inventory Strategy**: Target inventory is the sum of Cycle Stock (average demand during lead time) and Safety Stock (the buffer for uncertainty). Comparing this to your 'Current Inventory' reveals potential 'Dead Inventory' or stockout risks.")

    except Exception as e:
        st.error(f"Analysis failed for {item_id} @ {store_id}: {str(e)}")
        st.info("Check if the Item ID and Store ID exist in the M5 dataset.")

if __name__ == "__main__":
    main()
