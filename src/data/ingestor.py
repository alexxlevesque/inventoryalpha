import polars as pl
import numpy as np
import os

class DataIngestor:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.calendar_path = os.path.join(data_path, "calendar.csv")
        self.prices_path = os.path.join(data_path, "sell_prices.csv")
        self.sales_path = os.path.join(data_path, "sales_train_evaluation.csv")
    
    def load_raw(self):
        """
        Lazy load the datasets.
        """
        # We use scan_csv for lazy evaluation to handle large files efficiently
        self.calendar = pl.scan_csv(self.calendar_path)
        self.prices = pl.scan_csv(self.prices_path)
        self.sales = pl.scan_csv(self.sales_path)

    def merge_data(self, item_id: str, store_id: str) -> pl.DataFrame:
        """
        Merges sales, calendar, and prices for a specific SKU.
        Returns a sorted DataFrame with columns: date, sales, price, etc.
        """
        # 1. Filter Sales for the specific Item/Store
        sales_subset = (
            self.sales
            .filter((pl.col("item_id") == item_id) & (pl.col("store_id") == store_id))
            .collect() # Collect here to melt in memory, usually small enough for single SKU
        )
        
        if sales_subset.height == 0:
            raise ValueError(f"No data found for item_id={item_id} and store_id={store_id}")

        # 2. Melt to long format (d_1, d_2, ... -> d, sales)
        # We need to identify columns that start with 'd_'
        id_vars = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
        value_vars = [c for c in sales_subset.columns if c.startswith("d_")]
        
        melted_sales = sales_subset.unpivot(
            index=id_vars,
            on=value_vars,
            variable_name="d",
            value_name="sales"
        )

        # 3. Join with Calendar
        # Calendar has 'd' column like 'd_1', 'd_2'... perfect for join
        calendar = self.calendar.collect()
        
        with_calendar = melted_sales.join(
            calendar,
            on="d",
            how="left"
        )
        
        # 4. Join with Prices
        # Prices join on store_id, item_id, wm_yr_wk
        prices = self.prices.filter((pl.col("item_id") == item_id) & (pl.col("store_id") == store_id)).collect()
        
        final_df = with_calendar.join(
            prices,
            on=["store_id", "item_id", "wm_yr_wk"],
            how="left"
        )
        
        # 5. Type Conversions and Sorting
        final_df = final_df.with_columns([
            pl.col("date").str.to_date(),
            pl.col("sales").cast(pl.Float64),
            pl.col("sell_price").cast(pl.Float64)
        ]).sort("date")
        
        return final_df

    def get_clean_series(self, item_id: str, store_id: str) -> np.ndarray:
        """
        Returns a NumPy array of (sales, price) for the SKU.
        """
        df = self.merge_data(item_id, store_id)
        
        # Return relevant columns as numpy array
        # You might want strictly the demand (sales), or sales + price.
        # Returning sales for now as primary signal.
        return df.select("sales").to_numpy().flatten()

    def get_unique_stores(self) -> list:
        """
        Returns a sorted list of unique store_ids.
        """
        # Efficiently get unique values from LazyFrame
        return self.sales.select("store_id").unique().collect().get_column("store_id").sort().to_list()

    def get_unique_items(self, store_id: str = None) -> list:
        """
        Returns a sorted list of unique item_ids.
        If store_id is provided, filters for items present in that store (usually all items are in all stores in M5, but good practice).
        """
        if store_id:
            return (
                self.sales
                .filter(pl.col("store_id") == store_id)
                .select("item_id")
                .unique()
                .collect()
                .get_column("item_id")
                .sort()
                .to_list()
            )
        else:
            return self.sales.select("item_id").unique().collect().get_column("item_id").sort().to_list()
