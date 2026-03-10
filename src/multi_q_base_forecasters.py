from typing import List, Callable, Tuple
import numpy as np

# ============================================================================
# Data Generation
# ============================================================================

def generate_quantile_data(n: int, m: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data where Y is the quantile of a mixture distribution.
    
    Parameters
    ----------
    n : int
        Number of samples
    m : int
        Number of discretized theta values
    seed : int
        Random seed
    
    Returns
    -------
    X : np.ndarray
        Shape (n,) - input features in [0, 1]
    Y : np.ndarray
        Shape (n,) - observations in [0, 1] representing quantiles
    """
    np.random.seed(seed)
    
    X = np.random.uniform(0, 1, size=n)
    
    # True conditional distribution: Y depends on X
    true_mean = 0.2 + 0.6 * X
    
    # Add noise and clip to [0, 1]
    noise = np.random.normal(0, 0.3, size=n)
    Y = np.clip(true_mean + noise, 0, 1)
    
    # Discretize Y to {0, 1/m, 2/m, ..., 1} for cleaner experiment
    Y = np.round(Y * m) / m
    
    return X, Y




# ============================================================================
# Base Forecast Class
# ============================================================================

class QuantileAwareForecaster:
    """
    A forecaster that predicts quantiles with sorted outputs.
    
    For quantile levels alpha in alpha_list, we compute:
    1. Multiple independent linear predictions
    2. Sort them to ensure monotonicity in alpha
    3. Return sorted predictions as quantile estimates
    
    This ensures:
    - Predictions are monotonically increasing in alpha (quantile property)
    - Different forecasters can specialize in different parts of distribution
    """
    
    def __init__(
        self,
        alpha_list: List[float],
        base_weights: np.ndarray = None,
        base_biases: np.ndarray = None,
        noise_level: float = 0.0,
        quantile_offset: str = 'linear',
    ):
        """
        Initialize quantile-aware forecaster.
        
        Parameters
        ----------
        alpha_list : List[float]
            Quantile levels to predict
        base_weights : np.ndarray, optional
            Shape (n_alphas,) - linear weights for each quantile
        base_biases : np.ndarray, optional
            Shape (n_alphas,) - linear intercepts for each quantile
        noise_level : float
            Standard deviation of noise added to sorted predictions
        quantile_offset : str
            'linear': offset proportional to alpha
            'sqrt': offset proportional to sqrt(alpha)
            'squared': offset proportional to alpha^2
            'custom': use provided weights/biases directly
        """
        self.alpha_list = np.asarray(alpha_list, dtype=np.float64)
        self.n_alphas = len(self.alpha_list)
        self.noise_level = noise_level
        self.quantile_offset = quantile_offset
        
        # Initialize weights and biases if not provided
        if base_weights is None:
            self.base_weights = np.random.randn(self.n_alphas)
        else:
            self.base_weights = np.asarray(base_weights, dtype=np.float64)
        
        if base_biases is None:
            self.base_biases = np.random.uniform(0.2, 0.8, size=self.n_alphas)
        else:
            self.base_biases = np.asarray(base_biases, dtype=np.float64)
    
    def __call__(self, x: float) -> np.ndarray:
        """
        Predict quantiles at all alpha levels.
        
        Parameters
        ----------
        x : float
            Input value
        
        Returns
        -------
        predictions : np.ndarray
            Shape (n_alphas,) with sorted quantile predictions
        """
        # Compute unsorted linear predictions
        linear_preds = self.base_weights * x + self.base_biases
        
        # Add quantile-specific offsets to encourage diversity
        if self.quantile_offset == 'linear':
            offsets = self.alpha_list * 0.3
        elif self.quantile_offset == 'sqrt':
            offsets = np.sqrt(self.alpha_list) * 0.3
        elif self.quantile_offset == 'squared':
            offsets = (self.alpha_list ** 2) * 0.3
        elif self.quantile_offset == 'custom':
            offsets = np.zeros_like(self.alpha_list)
        else:
            offsets = np.zeros_like(self.alpha_list)
        
        preds_with_offset = linear_preds + offsets
        
        # Add noise before sorting for diversity
        if self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level, size=self.n_alphas)
            preds_with_offset += noise
        
        # CRITICAL: Sort to ensure monotonicity in alpha
        # This guarantees pred[i] <= pred[i+1] for all i
        sorted_preds = np.sort(preds_with_offset)
        
        # Clip to [0, 1] (valid quantile range)
        clipped_preds = np.clip(sorted_preds, 0, 1)
        
        return clipped_preds


def create_quantile_aware_forecaster_class(
    alpha_list: List[float],
    F: int = 5,
    seed: int = 42,
) -> Tuple[List[Callable], List[str]]:
    """
    Create forecasters with good coverage across quantile range.
    
    Strategy:
    1. Create forecasters specializing in different Y regions
    2. Each uses sorted linear combinations (non-linear via sorting)
    3. Ensemble provides coverage from low to high quantiles
    
    Parameters
    ----------
    alpha_list : List[float]
        Quantile levels (e.g., [0.2, 0.4, 0.7])
    F : int
        Number of forecasters
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    forecasters : List[Callable]
        F forecaster functions
    names : List[str]
        Names describing each forecaster
    """
    np.random.seed(seed)
    alpha_list = np.asarray(alpha_list, dtype=np.float64)
    n_alphas = len(alpha_list)
    
    forecasters = []
    names = []
    
    # ========================================================================
    # Forecaster 1: Lower Tail Specialist
    # ========================================================================
    # Predicts systematically low values
    # Good for lower quantiles
    weights_low = np.random.uniform(-0.3, 0.1, size=n_alphas)
    biases_low = np.random.uniform(0.1, 0.3, size=n_alphas)
    
    forecasters.append(
        QuantileAwareForecaster(
            alpha_list=alpha_list,
            base_weights=weights_low,
            base_biases=biases_low,
            noise_level=0.05,
            quantile_offset='linear',
        )
    )
    names.append("Lower Tail Specialist")
    
    # ========================================================================
    # Forecaster 2: Central Tendency
    # ========================================================================
    # Predicts around the mean
    # Good for median and central quantiles
    weights_central = np.random.uniform(-0.2, 0.3, size=n_alphas)
    biases_central = np.random.uniform(0.3, 0.5, size=n_alphas)
    
    forecasters.append(
        QuantileAwareForecaster(
            alpha_list=alpha_list,
            base_weights=weights_central,
            base_biases=biases_central,
            noise_level=0.08,
            quantile_offset='sqrt',
        )
    )
    names.append("Central Tendency")
    
    # ========================================================================
    # Forecaster 3: Upper Tail Specialist
    # ========================================================================
    # Predicts systematically high values
    # Good for upper quantiles
    weights_high = np.random.uniform(0.2, 0.5, size=n_alphas)
    biases_high = np.random.uniform(0.6, 0.8, size=n_alphas)
    
    forecasters.append(
        QuantileAwareForecaster(
            alpha_list=alpha_list,
            base_weights=weights_high,
            base_biases=biases_high,
            noise_level=0.05,
            quantile_offset='linear',
        )
    )
    names.append("Upper Tail Specialist")
    
    # ========================================================================
    # Forecaster 4: Conservative (Low Variance)
    # ========================================================================
    # Low noise, stable predictions
    weights_cons = np.random.uniform(-0.1, 0.2, size=n_alphas)
    biases_cons = np.random.uniform(0.35, 0.55, size=n_alphas)
    
    forecasters.append(
        QuantileAwareForecaster(
            alpha_list=alpha_list,
            base_weights=weights_cons,
            base_biases=biases_cons,
            noise_level=0.02,
            quantile_offset='linear',
        )
    )
    names.append("Conservative (Low Variance)")
    
    # ========================================================================
    # Forecaster 5: Aggressive (High Variance)
    # ========================================================================
    # High noise, captures variability
    weights_agg = np.random.uniform(-0.4, 0.4, size=n_alphas)
    biases_agg = np.random.uniform(0.2, 0.7, size=n_alphas)
    
    forecasters.append(
        QuantileAwareForecaster(
            alpha_list=alpha_list,
            base_weights=weights_agg,
            base_biases=biases_agg,
            noise_level=0.15,
            quantile_offset='squared',
        )
    )
    names.append("Aggressive (High Variance)")
    
    return forecasters, names

