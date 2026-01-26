## Kalman Inventory Alpha Diagnostic

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-FF4B4B.svg)](https://streamlit.io/)
[![Polars](https://img.shields.io/badge/Polars-0.19+-orange.svg)](https://pola.rs/)
[![Poetry](https://img.shields.io/badge/Poetry-Package_Manager-blueviolet.svg)](https://python-poetry.org/)

**Inventory Alpha** is a microeconomic diagnostic platform designed to identify "Phantom Demand" and "Dead Inventory" using advanced state estimation techniques. By applying a **Kalman Filter** to noisy sales data, the system extracts the true underlying demand signal, allowing for radical inventory reduction without compromising service levels.

## Key Value Proposition
*   **Detect Phantom Demand:** Separate random noise from actual demand trends.
*   **Reduce Dead Inventory:** Identify overstocked SKUs that tie up capital.
*   **Dynamic Safety Stock:** Calculate safety levels based on real-time signal uncertainty rather than static historical variance.
*   **Increase Capital Efficiency:** Optimized stock levels lead to better cash flow.

---

## Tech Stack
- **Dashboard:** [Streamlit](https://streamlit.io/)
- **Data Engine:** [Polars](https://pola.rs/) (High-performance DataFrame library)
- **Math & Stats:** [NumPy](https://numpy.org/), [SciPy](https://scipy.org/)
- **State Estimation:** Custom **Recursive Kalman Filter** implementation for 1D Random Walk signal extraction.
- **Visualization:** [Matplotlib](https://matplotlib.org/) & [Plotly](https://plotly.com/python/)

---

## Architecture & Methodology
The project was developed in a 5-sprint agile cycle:

1.  **Sprint 1: Infrastructure & Ingestion** - Scalable data loading using Polars and the M5 Forecasting dataset.
2.  **Sprint 2: Signal Processing** - Pre-processing filters to handle out-of-stock zeros and outliers.
3.  **Sprint 3: State Estimation** - Implementation of a Recursive Kalman Filter to estimate latent demand.
4.  **Sprint 4: Inventory Logic** - Dynamic safety stock and "Alpha" (excess) calculation logic.
5.  **Sprint 5: Visualization** - Interactive dashboard for real-time SKU diagnostics.

### The Kalman Advantage
Traditional inventory models often use simple moving averages or exponential smoothing. Inventory Alpha uses a custom **Recursive Kalman Filter**, which:
- Accounts for **Measurement Noise ($R$)**: Random sales fluctuations and measurement errors.
- Accounts for **Process Uncertainty ($Q$)**: How fast the underlying demand signal actually changes.
- Provides a **Dynamic Covariance ($P$)**: Used directly to calculate Z-score based safety stock buffers.

---

## Diagnostic Insights
The platform provides three main analytical views:
1.  **Demand & Inventory Tracking:** A 90-day lookback comparing actual sales to the Kalman-estimated "True Demand" signal with 95% confidence bands.
2.  **Demand Probabilities:** A probability density function (PDF) showing the distribution of demand over the replenishment lead time.
3.  **Inventory Snapshot:** A breakdown of current stock vs. Target Inventory (Cycle Stock + Safety Stock).