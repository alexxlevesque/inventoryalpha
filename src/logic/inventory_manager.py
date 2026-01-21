import numpy as np
import scipy.stats as stats
import polars as pl

class InventoryManager:
    def __init__(self, lead_time: int = 7, service_level: float = 0.95):
        """
        Args:
            lead_time (int): Days to replenish stock.
            service_level (float): Target probability of NOT stocking out (e.g., 95%).
        """
        self.lead_time = lead_time
        self.service_level = service_level
        
        # Calculate Z-score for service level (e.g., 1.645 for 95%)
        self.z_score = stats.norm.ppf(service_level)

    def calculate_safety_stock(self, uncertainty_series: np.ndarray) -> np.ndarray:
        """
        Calculates dynamic safety stock based on Kalman filter uncertainty.
        
        Formula: Safety Stock = Z * sqrt(P_t * LeadTime)
        Note: P_t is variance of the random walk process over 1 step. 
        Over `LeadTime` steps, variance scales linearly (assuming IID).
        """
        # Ensure non-negative uncertainty
        valid_uncertainty = np.maximum(uncertainty_series, 0)
        
        # Standard Deviation over the Lead Time window
        # P is variance, so std_dev = sqrt(P)
        # We assume the uncertainty scales with sqrt(time) for Random Walk
        std_dev_lead_time = np.sqrt(valid_uncertainty * self.lead_time)
        
        safety_stock = self.z_score * std_dev_lead_time
        return safety_stock

    def detect_dead_inventory(self, 
                            sku: str,
                            current_inventory: float,
                            demand_estimate: float,
                            demand_uncertainty: float) -> dict:
        """
        Analyzes a single SKU for excess inventory.
        """
        # 1. Calculate Target Inventory
        # Target = (Daily Demand * Lead Time) + Safety Stock
        # demand_uncertainty is P (Variance)
        
        safety_stock = self.z_score * np.sqrt(max(demand_uncertainty, 0) * self.lead_time)
        cycle_stock = demand_estimate * self.lead_time
        
        target_level = cycle_stock + safety_stock
        
        # 2. Calculate Excess
        excess = max(0, current_inventory - target_level)
        
        # 3. Confidence/Priority Score
        # If we have massive excess relative to target, it's high priority.
        # We can also use uncertainty: if uncertainty is LOW, we are confident it's dead.
        # Score = Excess / Target (Percentage overstocked)
        reduction_pct = (excess / current_inventory) if current_inventory > 0 else 0.0
        
        return {
            "sku": sku,
            "current_inventory": round(current_inventory, 2),
            "estimated_demand": round(demand_estimate, 2),
            "target_inventory": round(target_level, 2),
            "safety_stock": round(safety_stock, 2),
            "excess_units": round(excess, 2),
            "reduction_potential_pct": round(reduction_pct * 100, 1)
        }
    
    def batch_analysis(self, 
                       skus: list, 
                       inventory_levels: list, 
                       demand_estimates: list, 
                       uncertainties: list) -> pl.DataFrame:
        """
        Runs analysis on multiple SKUs and returns a sorted Polars DataFrame.
        """
        results = []
        for i, sku in enumerate(skus):
            res = self.detect_dead_inventory(
                sku, 
                inventory_levels[i], 
                demand_estimates[i], 
                uncertainties[i]
            )
            results.append(res)
            
        df = pl.DataFrame(results)
        
        # Sort by potential reduction percentage descending
        if not df.is_empty():
            df = df.sort("reduction_potential_pct", descending=True)
            
        return df
