import numpy as np
from scipy.optimize import lsq_linear
from functools import partial
from multiprocessing import Pool, cpu_count
from scipy.stats import norm

# Load your real data here
# X should be (144, 20) matrix
# y should be (144,) vector
# Replace these with your actual data loading:
# X = pd.read_csv('your_X_data.csv').values  # or however you load X
# y = pd.read_csv('your_y_data.csv').values.ravel()  # or however you load y

# For demonstration, using placeholder data structure:
if __name__ == '__main__':
    # number of variables in each constraint category
    n_positive = 10
    n_negative = 10
    n_unconstrained = 10
    p = n_positive + n_negative + n_unconstrained
    
    np.random.seed(42)
    n = 144
    X = np.random.normal(0, 1, (n, p))  # Replace with your real X data
    y = np.random.normal(0, 1, n)       # Replace with your real y data

    # Constraints: 
    # Variables 0-9: positive (>= 0)
    # Variables 10-14: negative (<= 0) 
    # Variables 15-19: unconstrained
    lb = np.array([0] * n_positive + [-np.inf] * n_negative + [-np.inf] * n_unconstrained)
    ub = np.array([np.inf] * n_positive + [0] * n_negative + [np.inf] * n_unconstrained)

    print("Lower bounds:", lb)
    print("Upper bounds:", ub)

    # Original constrained fit
    res = lsq_linear(X, y, bounds=(lb, ub))
    beta_hat = res.x
    print(f"Constrained beta estimates ({p} variables): {beta_hat}")

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
def bootstrap_fit(seed, X, y, lb, ub, block_size=10):
    np.random.seed(seed)  # Ensure reproducibility per process
    indices = block_bootstrap_sample(len(y), block_size)
    X_b = X[indices]
    y_b = y[indices]
    res_b = lsq_linear(X_b, y_b, bounds=(lb, ub))
    return res_b.x

    # ================== DIFFERENT BOOTSTRAP METHODS ==================
    print("\n=== RUNNING DIFFERENT BOOTSTRAP METHODS ===")

# 1. Standard Bootstrap (with replacement, no time structure)
def standard_bootstrap_worker(args):
    seed, X, y, lb, ub = args
    np.random.seed(seed)
    indices = np.random.choice(len(y), len(y), replace=True)  # Standard bootstrap
    X_b = X[indices]
    y_b = y[indices]
    res_b = lsq_linear(X_b, y_b, bounds=(lb, ub))
    return res_b.x

# 2. Block Bootstrap (time-series aware)
def block_bootstrap_worker(args):
    seed, X, y, lb, ub, block_size = args
    np.random.seed(seed)
    indices = block_bootstrap_sample(len(y), block_size)
    X_b = X[indices]
    y_b = y[indices]
    res_b = lsq_linear(X_b, y_b, bounds=(lb, ub))
    return res_b.x

    # 3. Theoretical (asymptotic) - using OLS standard errors as approximation
    print("Computing theoretical standard errors...")
    X_reg = X.copy()
    try:
        # Simple OLS approximation for theoretical SE
        XtX_inv = np.linalg.inv(X_reg.T @ X_reg)
        residuals = y - X_reg @ beta_hat
        mse = np.sum(residuals**2) / (len(y) - len(beta_hat))
        se_theoretical = np.sqrt(mse * np.diag(XtX_inv))
    except:
        # Fallback if matrix is singular
        se_theoretical = np.ones(len(beta_hat)) * 0.1

    B = 500  # Reduced for faster execution
    print(f"Running {B} iterations for each bootstrap method...")

    # Standard Bootstrap
    print("1. Standard Bootstrap...")
    args_standard = [(i, X, y, lb, ub) for i in range(B)]
    with Pool(processes=cpu_count()) as pool:
        boot_betas_standard = np.array(pool.map(standard_bootstrap_worker, args_standard))

    # Block Bootstrap  
    print("2. Block Bootstrap...")
    args_block = [(i, X, y, lb, ub, BLOCK_SIZE) for i in range(B)]
    with Pool(processes=cpu_count()) as pool:
        boot_betas_block = np.array(pool.map(block_bootstrap_worker, args_block))

    print("All bootstrap methods completed!")

    # Calculate different standard errors and p-values
    se_standard = np.std(boot_betas_standard, axis=0)
    se_block = np.std(boot_betas_block, axis=0)

    # P-values for each method
    z_theoretical = beta_hat / se_theoretical
    z_standard = beta_hat / se_standard  
    z_block = beta_hat / se_block

    p_values_theoretical = 2 * (1 - norm.cdf(np.abs(z_theoretical)))
    p_values_standard = 2 * (1 - norm.cdf(np.abs(z_standard)))
    p_values_block = 2 * (1 - norm.cdf(np.abs(z_block)))

    print(f"\nStandard errors comparison:")
    print(f"Theoretical SE: {se_theoretical}")
    print(f"Standard Bootstrap SE: {se_standard}")
    print(f"Block Bootstrap SE: {se_block}")

    print("\n=== RESULTS SUMMARY ===")
    for i in range(p):
        constraint_type = "positive" if i < n_positive else "negative" if i < n_positive + n_negative else "unconstrained"
        print(f"Variable {i:2d} ({constraint_type:12s}): β={beta_hat[i]:8.4f}")
        print(f"    Theoretical: p={p_values_theoretical[i]:8.4f}, SE={se_theoretical[i]:8.4f}")
        print(f"    Standard:    p={p_values_standard[i]:8.4f}, SE={se_standard[i]:8.4f}")
        print(f"    Block:       p={p_values_block[i]:8.4f}, SE={se_block[i]:8.4f}")

    # ================== CREATE PLOTS ==================
    print("\n=== CREATING VISUALIZATION PLOTS ===")
    try:
        from visualization import create_bootstrap_comparison_plots
        
        create_bootstrap_comparison_plots(
            beta_hat=beta_hat,
            se_theoretical=se_theoretical,  
            se_standard=se_standard, 
            se_block=se_block,
            p_values_theoretical=p_values_theoretical,  
            p_values_standard=p_values_standard, 
            p_values_block=p_values_block,
            boot_betas_theoretical=None,
            boot_betas_standard=boot_betas_standard,
            boot_betas_block=boot_betas_block,
            save_path="",
            n_positive=n_positive,
            n_negative=n_negative, 
            n_unconstrained=n_unconstrained
        )
        print("✓ Plots saved successfully with DIFFERENT methods!")
        
    except Exception as e:
        print(f"✗ Error creating plots: {e}")
