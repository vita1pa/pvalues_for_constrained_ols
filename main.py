from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.optimize import lsq_linear

# Generate synthetic data (replace with your real X, y)
np.random.seed(42)
n = 100
p = 3
X = np.random.normal(0, 1, (n, p))
true_beta = np.array([1.5, -2.0, 0.5])  # Example true values satisfying constraints
y = X @ true_beta + np.random.normal(0, 0.5, n)

# Bounds: var1 (beta[0]) >= 0, var2 (beta[1]) <= 0, var3 (beta[2]) unconstrained
lb = np.array([0, -np.inf, -np.inf])
ub = np.array([np.inf, 0, np.inf])

# Original constrained fit
res = lsq_linear(X, y, bounds=(lb, ub))
beta_hat = res.x
print("Constrained beta estimates (var1, var2, var3):", beta_hat)


# Bootstrap function (to be parallelized)
def bootstrap_fit(seed, X, y, lb, ub):
    np.random.seed(seed)  # Ensure reproducibility per process
    indices = np.random.choice(len(y), len(y), replace=True)
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

print("Approximate p-values (var1, var2, var3):", p_values)
print("Bootstrap SEs:", boot_se)

# 95% bootstrap confidence intervals (percentile method)
ci_low = np.percentile(boot_betas, 2.5, axis=0)
ci_high = np.percentile(boot_betas, 97.5, axis=0)
print("95% CIs:", list(zip(ci_low, ci_high)))
