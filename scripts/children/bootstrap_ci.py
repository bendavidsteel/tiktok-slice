import numpy as np
from sklearn.utils import resample
from tqdm import tqdm

def bootstrap_classifier_percentage_ci(k, N, precision, recall, n_bootstraps=1000, confidence=0.95):
    """
    Calculate confidence interval for the percentage of true positives using bootstrapping.
    
    Parameters:
    - k: Number of items classified as positive
    - N: Total population size
    - precision: Classifier precision
    - recall: Classifier recall
    - n_bootstraps: Number of bootstrap samples (default: 10000)
    - confidence: Confidence level (default: 0.95 for 95% CI)
    
    Returns:
    - point_estimate, lower_bound, upper_bound: Point estimate and confidence interval
    """
    # Generate synthetic dataset based on classifier results
    # 1 = classified as positive, 0 = classified as negative
    synthetic_data = np.zeros(N)
    synthetic_data[:k] = 1
    
    # Bootstrap resampling
    bootstrap_estimates = []
    
    for _ in tqdm(range(n_bootstraps)):
        # Resample with replacement
        bootstrap_sample = resample(synthetic_data, replace=True, n_samples=N)
        
        # Count positives in this bootstrap sample
        k_bootstrap = np.sum(bootstrap_sample)
        
        # Apply precision/recall correction to estimate true positives
        T_est = k_bootstrap * precision / recall if recall > 0 else 0
        
        # Calculate percentage
        percentage_est = (T_est / N) * 100
        bootstrap_estimates.append(percentage_est)
    
    # Calculate confidence interval
    alpha = 1 - confidence
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = max(0, np.percentile(bootstrap_estimates, lower_percentile))
    upper_bound = min(100, np.percentile(bootstrap_estimates, upper_percentile))
    
    # Calculate point estimate
    point_estimate = k * precision / recall / N * 100 if recall > 0 else 0
    
    return point_estimate, lower_bound, upper_bound

# Example usage
if __name__ == "__main__":
    # Example values
    k = 156963  # Items classified as positive
    N = 940372  # Total population
    precision = 0.88  # 80% precision
    recall = 0.81  # 70% recall
    
    # Estimate using bootstrapping
    est, lower, upper = bootstrap_classifier_percentage_ci(k, N, precision, recall)
    
    print(f"Estimated percentage: {est:.2f}%")
    print(f"95% Confidence Interval: [{lower:.2f}%, {upper:.2f}%]")