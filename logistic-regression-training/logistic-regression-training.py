import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here

    samples, features = X.shape
    
    w = np.zeros(features)
    b = 0
    
    for i in range(steps):
        z = X @ w + b
        p = _sigmoid(z)

        dL_dw = (X.T @ (p-y))/samples
        dL_db = np.sum(p-y)/samples

        w = w - lr*dL_dw
        b = b - lr*dL_db

    return (w,b)

        
        