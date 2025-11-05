import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_bootstrap_comparison_plots(beta_hat, se_theoretical, se_standard, se_block, 
                                    p_values_theoretical, p_values_standard, p_values_block,
                                    boot_betas_theoretical, boot_betas_standard, boot_betas_block,
                                    save_path=""):
    """
    Create comprehensive visualization comparing bootstrap methods
    
    Parameters:
    -----------
    beta_hat : array-like
        Coefficient estimates (constrained)
    se_theoretical, se_standard, se_block : array-like
        Standard errors from each method
    p_values_theoretical, p_values_standard, p_values_block : array-like
        P-values from each method
    boot_betas_theoretical, boot_betas_standard, boot_betas_block : array-like
        Alternative coefficient estimates from each method (for comparison)
    save_path : str
        Path prefix for saving plots
    """
    
    # Define constraint groups and colors
    n_vars = len(beta_hat)
    constraint_types = ['positive'] * 10 + ['negative'] * 5 + ['unconstrained'] * 5
    constraint_colors = {'positive': '#FF6B6B', 'negative': '#4ECDC4', 'unconstrained': '#45B7D1'}
    method_colors = ['#FFB347', '#87CEEB', '#98D8C8']
    method_labels = ['Theoretical (Naive)', 'Standard Bootstrap', 'Block Bootstrap']
    
    # Create variable labels
    var_labels = [f'Var{i+1}' for i in range(n_vars)]
    
    # PLOT 1: IMPROVED P-VALUES COMPARISON
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Since p-values are hard to see, let's use a different approach
    # Convert to -log10(p-values) for better visualization
    log_p_theoretical = -np.log10(np.maximum(p_values_theoretical, 1e-10))
    log_p_standard = -np.log10(np.maximum(p_values_standard, 1e-10))
    log_p_block = -np.log10(np.maximum(p_values_block, 1e-10))
    
    x_pos = np.arange(n_vars)
    width = 0.25
    
    # Create grouped bars
    bars1 = ax.bar(x_pos - width, log_p_theoretical, width, 
                   label=method_labels[0], color=method_colors[0], alpha=0.8)
    bars2 = ax.bar(x_pos, log_p_standard, width, 
                   label=method_labels[1], color=method_colors[1], alpha=0.8)
    bars3 = ax.bar(x_pos + width, log_p_block, width, 
                   label=method_labels[2], color=method_colors[2], alpha=0.8)
    
    # Add constraint group separators
    group_boundaries = [9.5, 14.5]  # Between groups
    for boundary in group_boundaries:
        ax.axvline(x=boundary, color='black', linestyle='--', alpha=0.5, linewidth=2)
    
    # Color code constraint regions
    constraint_regions = [(0, 9, 'Positive Constraints\n(β ≥ 0)', 'positive'), 
                         (10, 14, 'Negative Constraints\n(β ≤ 0)', 'negative'), 
                         (15, 19, 'Unconstrained\n(No limits)', 'unconstrained')]
    
    for start, end, label, ctype in constraint_regions:
        ax.axvspan(start-0.5, end+0.5, alpha=0.1, color=constraint_colors[ctype])
        # Place labels at top
        ax.text((start+end)/2, ax.get_ylim()[1]*0.9, label, ha='center', va='center',
                fontsize=11, fontweight='bold', 
                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9, edgecolor=constraint_colors[ctype]))
    
    # Add significance threshold
    significance_line = -np.log10(0.05)
    ax.axhline(y=significance_line, color='red', linestyle='-', alpha=0.8, linewidth=2, 
               label=f'α = 0.05 threshold (-log₁₀ = {significance_line:.1f})')
    
    # Customize axes
    ax.set_xlabel('Variables', fontsize=14, fontweight='bold')
    ax.set_ylabel('-log₁₀(p-value)', fontsize=14, fontweight='bold')
    ax.set_title('Statistical Significance Comparison: Bootstrap Methods\n'
                'Higher bars = More significant (stronger evidence against H₀: β=0)', 
                fontsize=16, fontweight='bold', pad=25)
    
    # Set x-axis labels with rotation
    ax.set_xticks(x_pos)
    ax.set_xticklabels(var_labels, rotation=45, ha='right', fontsize=10)
    
    # Legend positioned to avoid overlap
    ax.legend(loc='upper left', bbox_to_anchor=(0, 0.85), fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}pvalues_comparison_improved.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # PLOT 2: COEFFICIENTS WITH BARS AND BOXPLOTS
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Top subplot: Coefficient values with bars - using bootstrap means if available
    x_pos = np.arange(n_vars)
    width = 0.25
    
    # Use bootstrap means for different estimates, or fallback to same values
    beta_theoretical = beta_hat  # theoretical uses the original constrained estimate
    beta_standard_mean = np.mean(boot_betas_standard, axis=0) if boot_betas_standard is not None else beta_hat
    beta_block_mean = np.mean(boot_betas_block, axis=0) if boot_betas_block is not None else beta_hat
    
    bars1 = ax1.bar(x_pos - width, beta_theoretical, width, 
                   label=method_labels[0], color=method_colors[0], alpha=0.8)
    bars2 = ax1.bar(x_pos, beta_standard_mean, width, 
                   label=method_labels[1], color=method_colors[1], alpha=0.8)
    bars3 = ax1.bar(x_pos + width, beta_block_mean, width, 
                   label=method_labels[2], color=method_colors[2], alpha=0.8)
    
    # Add constraint group separators
    group_boundaries = [9.5, 14.5]  # Between groups
    for boundary in group_boundaries:
        ax1.axvline(x=boundary, color='black', linestyle='--', alpha=0.5, linewidth=2)
    
    # Color code constraint regions
    constraint_regions = [(0, 9, 'Positive Constraints\n(β ≥ 0)', 'positive'), 
                         (10, 14, 'Negative Constraints\n(β ≤ 0)', 'negative'), 
                         (15, 19, 'Unconstrained\n(No limits)', 'unconstrained')]
    
    for start, end, label, ctype in constraint_regions:
        ax1.axvspan(start-0.5, end+0.5, alpha=0.1, color=constraint_colors[ctype])
        ax1.text((start+end)/2, ax1.get_ylim()[1]*0.9, label, ha='center', va='center',
                fontsize=11, fontweight='bold', 
                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9, edgecolor=constraint_colors[ctype]))
    
    # Add zero reference line
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.7, linewidth=1)
    
    # Customize axes
    ax1.set_xlabel('Variables', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Coefficient Values (β)', fontsize=12, fontweight='bold')
    ax1.set_title('Mean Coefficient Estimates by Bootstrap Method', fontsize=14, fontweight='bold')
    
    # Set x-axis labels with rotation
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(var_labels, rotation=45, ha='right', fontsize=10)
    
    # Legend positioned to avoid overlap
    ax1.legend(loc='upper left', bbox_to_anchor=(0, 0.85), fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Bottom subplot: SE BOXPLOTS by constraint groups (for BETA standard errors)
    group_names = ['Positive\\n(Vars 1-10)', 'Negative\\n(Vars 11-15)', 'Unconstrained\\n(Vars 16-20)']
    group_ranges = [(0, 10), (10, 15), (15, 20)]
    
    # Prepare data for grouped boxplots of SE values
    se_data_by_group = []
    positions = []
    colors_list = []
    
    pos = 1
    for i, (start, end) in enumerate(group_ranges):
        # Get SE values for this constraint group
        group_se_theoretical = se_theoretical[start:end]
        group_se_standard = se_standard[start:end]
        group_se_block = se_block[start:end]
        
        se_data_by_group.extend([group_se_theoretical, group_se_standard, group_se_block])
        positions.extend([pos, pos+1, pos+2])
        colors_list.extend(method_colors)
        pos += 4  # Space between groups
    
    bp = ax2.boxplot(se_data_by_group, positions=positions, patch_artist=True, widths=0.8)
    
    # Color the boxplots
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Customize the boxplot
    ax2.set_xlabel('Method by Constraint Group', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Standard Errors (SE)', fontsize=12, fontweight='bold')
    ax2.set_title('Standard Error Distribution: Methods × Constraint Groups', fontsize=14, fontweight='bold')
    
    # Set x-tick labels
    group_centers = [2, 6, 10]
    ax2.set_xticks(group_centers)
    ax2.set_xticklabels(group_names, fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add method legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, alpha=0.7, label=label) 
                      for color, label in zip(method_colors, method_labels)]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}coefficients_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # PLOT 3: SEPARATE BOXPLOTS FOR SE COMPARISON
    
    print(f"Saved improved plots:")
    print(f"1. '{save_path}pvalues_comparison_improved.png' - P-values comparison using -log10 scale")
    print(f"2. '{save_path}coefficients_comparison.png' - Coefficients (top) + SE boxplots (bottom)")
    
    return True

def create_simple_comparison_plot(beta_hat, se_methods_dict, p_values_methods_dict, save_path=""):
    """
    Simple interface for quick plotting
    
    Parameters:
    -----------
    beta_hat : array-like
        Coefficient estimates
    se_methods_dict : dict
        {'method_name': se_array, ...}
    p_values_methods_dict : dict  
        {'method_name': p_values_array, ...}
    """
    
    methods = list(se_methods_dict.keys())
    se_arrays = list(se_methods_dict.values())
    p_arrays = list(p_values_methods_dict.values())
    
    if len(methods) == 3:
        return create_bootstrap_comparison_plots(
            beta_hat, se_arrays[0], se_arrays[1], se_arrays[2],
            p_arrays[0], p_arrays[1], p_arrays[2], save_path
        )
    else:
        print("This function expects exactly 3 methods for comparison")
        return False