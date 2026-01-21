import numpy as np

class StateEstimator:
    def __init__(self, 
                 process_noise: float = 0.1, 
                 measurement_noise: float = 1.0):
        """
        Initialize the Kalman Filter State Estimator.
        
        Args:
            process_noise (float): Q - Variance of the "True Demand" (how much it wiggles).
            measurement_noise (float): R - Variance of the "Sales Data" (how noisy the observations are).
        """
        # System matrices for a 1D Random Walk model
        self.F = 1.0 # State Transition
        self.H = 1.0 # Observation Matrix
        
        self.Q = process_noise      # Process Noise Covariance
        self.R = measurement_noise  # Measurement Noise Covariance
        
        # State
        self.x = 0.0 # Estimated State (Mean)
        self.P = 1.0 # Estimated Covariance (Uncertainty)

    def initialize(self, initial_value: float, initial_uncertainty: float = 10.0):
        """Sets the initial state."""
        self.x = initial_value
        self.P = initial_uncertainty

    def predict(self):
        """
        Predict step: $x_{pred} = Fx$, $P_{pred} = FPF^T + Q$
        """
        self.x = self.F * self.x
        self.P = (self.F * self.P * self.F) + self.Q
        return self.x, self.P

    def update(self, measurement: float):
        """
        Update step: Correct prediction with actual data.
        
        Args:
            measurement (float): The observed sales value.
        """
        # Handle missing data (NaN) by skipping update (effectively "trusting prediction")
        if np.isnan(measurement):
            return self.x, self.P

        # Kalmain Gain: K = P H^T (H P H^T + R)^-1
        k_denominator = (self.H * self.P * self.H) + self.R
        K = (self.P * self.H) / k_denominator
        
        # State Update: x = x + K(z - Hx)
        residual = measurement - (self.H * self.x)
        self.x = self.x + (K * residual)
        
        # Covariance Update: P = (I - KH)P
        self.P = (1 - K * self.H) * self.P
        
        return self.x, self.P, K

    def run_filter(self, series: np.ndarray):
        """
        Runs the filter over an entire time series.
        
        Returns:
            estimated_states (np.ndarray): The denoised "True Demand".
            uncertainties (np.ndarray): The variance P at each step.
        """
        n = len(series)
        estimates = np.zeros(n)
        uncertainties = np.zeros(n)
        
        # Initialize with the first observation
        self.initialize(series[0] if not np.isnan(series[0]) else 0)
        
        for i in range(n):
            # 1. Predict
            self.predict()
            
            # 2. Update (if we have data)
            measurement = series[i]
            self.update(measurement)
            
            # Store results
            estimates[i] = self.x
            uncertainties[i] = self.P
            
        return estimates, uncertainties
    
    def smooth(self, observations):
        """
        Apply Kalman Smoothing (Rauch-Tung-Striebel Smoother).
        Not strictly necessary for the MVP "Live" Diagnostic, but good for historical analysis.
        Leaving as placeholder or simple forward pass for now.
        """
        return self.run_filter(observations)
