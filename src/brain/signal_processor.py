import numpy as np
from scipy.fft import rfft, rfftfreq
import plotly.graph_objects as go

class SignalProcessor:
    def __init__(self):
        pass

    def compute_fft(self, series: np.ndarray, sampling_rate: float = 1.0):
        # Detrend the series by removing the mean to avoid a large zero-frequency component
        # We might want to remove linear trend too, but start with mean removal.
        series_clean = series - np.mean(series)
        
        n = len(series_clean)
        yf = rfft(series_clean)
        xf = rfftfreq(n, 1 / sampling_rate)
        
        magnitude = np.abs(yf)
        
        return xf, magnitude

    def detect_seasonality(self, series: np.ndarray, top_k: int = 3):
        xf, magnitude = self.compute_fft(series)
        
        # Filter out zero frequency (if not fully removed by de-meaning) and very low frequencies
        # that roughly correspond to the length of the dataset
        mask = xf > 0
        xf = xf[mask]
        magnitude = magnitude[mask]
        
        # Get indices of top_k magnitudes
        # We use partition for efficiency if array is large, or just sort
        if len(magnitude) == 0:
            return []
            
        # Sort indices by magnitude descending
        sorted_indices = np.argsort(magnitude)[::-1]
        
        results = []
        for i in range(min(top_k, len(sorted_indices))):
            idx = sorted_indices[i]
            freq = xf[idx]
            mag = magnitude[idx]
            period = 1 / freq if freq != 0 else np.inf
            
            results.append({
                "period": round(period, 2),
                "frequency": freq,
                "magnitude": mag
            })
            
        return results

    def plot_frequency_spectrum(self, series: np.ndarray, title: str = "Frequency Spectrum"):
        xf, magnitude = self.compute_fft(series)
        
        # Calculate periods for tooltip (avoid division by zero)
        periods = np.where(xf > 0, 1/xf, np.inf)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=xf,
            y=magnitude,
            mode='lines',
            name='Magnitude',
            hovertemplate='Frequency: %{x:.4f}<br>Magnitude: %{y:.2f}<br>Period: %{text:.2f} days<extra></extra>',
            text=periods
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Frequency (1/day)',
            yaxis_title='Magnitude',
            template='plotly_dark'
        )
        
        return fig
