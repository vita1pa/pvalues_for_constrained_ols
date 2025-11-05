from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.optimize import lsq_linear

# Load your real data here
# X should be (144, 20) matrix
# y should be (144,) vector
# Replace these with your actual data loading:
# X = pd.read_csv('your_X_data.csv').values  # or however you load X
# y = pd.read_csv('your_y_data.csv').values.ravel()  # or however you load y

# For demonstration, using placeholder data structure:
np.random.seed(42)
n = 144
p = 20
X = np.random.normal(0, 1, (n, p))  # Replace with your real X data
y = np.random.normal(0, 1, n)       # Replace with your real y data

# Constraints: 
# Variables 0-9: positive (>= 0)
# Variables 10-14: negative (<= 0) 
# Variables 15-19: unconstrained
lb = np.array([0] * 10 + [-np.inf] * 5 + [-np.inf] * 5)  # First 10 >= 0, rest unconstrained
ub = np.array([np.inf] * 10 + [0] * 5 + [np.inf] * 5)    # Variables 10-14 <= 0, rest unconstrained

print("Lower bounds:", lb)
print("Upper bounds:", ub)

# Original constrained fit
res = lsq_linear(X, y, bounds=(lb, ub))
beta_hat = res.x
print(f"Constrained beta estimates (20 variables): {beta_hat}")


# Time-series bootstrap configuration
# Adjust block_size based on your data frequency:
# - Daily data with weekly patterns: block_size = 5-7
# - Weekly data with monthly patterns: block_size = 4-5  
# - Monthly data with yearly patterns: block_size = 12
# - For mild autocorrelation: block_size = 3-5
BLOCK_SIZE = 10  # Adjust this based on your time structure

def block_bootstrap_sample(n, block_size):
    """Generate time-series aware bootstrap indices using block bootstrap"""
    if block_size >= n:
        # If block size is too large, use all data in order
        return np.arange(n)
    
    n_blocks_needed = int(np.ceil(n / block_size))
    max_start = n - block_size + 1
    
    # Sample starting positions for blocks
    block_starts = np.random.choice(max_start, n_blocks_needed, replace=True)
    
    indices = []
    for start in block_starts:
        block_indices = np.arange(start, min(start + block_size, n))
        indices.extend(block_indices)
    
    # Trim to exact length needed
    return np.array(indices[:n])

# Bootstrap function (to be parallelized) - Time Series Aware
def bootstrap_fit(seed, X, y, lb, ub, block_size=BLOCK_SIZE):
    np.random.seed(seed)  # Ensure reproducibility per process
    indices = block_bootstrap_sample(len(y), block_size)
    X_b = X[indices]
    y_b = y[indices]
    res_b = lsq_linear(X_b, y_b, bounds=(lb, ub))
    return res_b.x


# Parallel bootstrapping
B = 1000  # Adjust as needed; higher for better precision
partial_fit = partial(bootstrap_fit, X=X, y=y, lb=lb, ub=ub)
with Pool(processes=cpu_count()) as pool:
    boot_betas = np.array(pool.map(partial_fit, range(B)))

# Approximate standard errors from bootstrap
boot_se = np.std(boot_betas, axis=0)

# Approximate p-values (two-sided, vs 0) assuming normality for large B
from scipy.stats import norm

z_scores = beta_hat / boot_se
p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))

print(f"Approximate p-values (20 variables): {p_values}")
print(f"Bootstrap SEs: {boot_se}")

# 95% bootstrap confidence intervals (percentile method)
ci_low = np.percentile(boot_betas, 2.5, axis=0)
ci_high = np.percentile(boot_betas, 97.5, axis=0)

print("\n=== RESULTS SUMMARY ===")
for i in range(20):
    constraint_type = "positive" if i < 10 else "negative" if i < 15 else "unconstrained"
    print(f"Variable {i:2d} ({constraint_type:12s}): β={beta_hat[i]:8.4f}, p={p_values[i]:8.4f}, CI=[{ci_low[i]:8.4f}, {ci_high[i]:8.4f}]")
