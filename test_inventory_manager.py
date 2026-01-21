import numpy as np
import polars as pl
from src.logic.inventory_manager import InventoryManager

def test_inventory_manager():
    # Setup
    manager = InventoryManager(lead_time=7, service_level=0.95)
    
    # Test Cases
    # 1. "Safe" SKU: Inventory matches demand
    # 2. "Excess" SKU: Inventory way too high
    # 3. "Volatile" SKU: High uncertainty requires higher safety stock
    
    skus = ["SAFE_SKU", "DEAD_SKU", "VOLATILE_SKU"]
    
    # Estimated Daily Demand
    demand_est = [10.0, 10.0, 10.0]
    
    # Inventory on Hand
    # SKU 1: 70 (Cycle) + ~10 (Safety) = ~80
    # SKU 2: 500 (Way too much)
    # SKU 3: 80 (Same as SKU 1)
    inventory = [85.0, 500.0, 85.0]
    
    # Uncertainty (Variance P)
    # SKU 1: Low noise (P=1)
    # SKU 2: Low noise (P=1)
    # SKU 3: High noise (P=16) -> StdDev=4 -> Safety Stock should assume 4x noise
    uncertainty = [1.0, 1.0, 16.0]
    
    print("Running Batch Analysis...")
    df = manager.batch_analysis(skus, inventory, demand_est, uncertainty)
    
    print("\nResults:")
    print(df)
    
    # Validation Logic
    # 1. DEAD_SKU should be top of list
    top_sku = df.row(0, named=True)["sku"]
    if top_sku == "DEAD_SKU":
        print("\nSUCCESS: Identified DEAD_SKU as top priority.")
    else:
        print(f"\nFAILURE: Expected DEAD_SKU, got {top_sku}")

    # 2. VOLATILE_SKU should have higher safety stock than SAFE_SKU
    safe_ss = df.filter(pl.col("sku") == "SAFE_SKU").select("safety_stock").item()
    vol_ss = df.filter(pl.col("sku") == "VOLATILE_SKU").select("safety_stock").item()
    
    print(f"\nSafety Stock Comparison:")
    print(f"SAFE_SKU (P=1): {safe_ss}")
    print(f"VOLATILE_SKU (P=16): {vol_ss}")
    
    if vol_ss > safe_ss:
        print("SUCCESS: Volatility correctly increased Safety Stock requirement.")
    else:
        print("FAILURE: Volatility did not increase Safety Stock.")
        
    # 3. Check Reduction Potential
    dead_reduction = df.filter(pl.col("sku") == "DEAD_SKU").select("reduction_potential_pct").item()
    print(f"\nDead Inventory Reduction Potential: {dead_reduction}%")
    if dead_reduction > 50:
        print("SUCCESS: Massive reduction identified.")

if __name__ == '__main__':
    test_inventory_manager()
