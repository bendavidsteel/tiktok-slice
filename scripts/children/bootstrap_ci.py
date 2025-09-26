import numpy as np
from sklearn.utils import resample
from tqdm import tqdm

# Alternative approach using Beta distribution for additional uncertainty
def bootstrap_classifier_percentage_ci(k, N, precision, recall, n_bootstraps=1000, confidence=0.95, test_set_size=300):
    """
    Alternative approach using Beta distribution to model uncertainty in precision and recall.
    """
    
    # Estimate base rate
    estimated_true_positives = k * precision / recall if recall > 0 else 0
    estimated_proportion = estimated_true_positives / N
    
    bootstrap_estimates = []
    
    for _ in tqdm(range(n_bootstraps)):
        # Add uncertainty to precision and recall using Beta distribution
        # Using a concentration parameter based on sample size assumption
        concentration = test_set_size  # Adjust based on how much data was used to estimate precision/recall
        
        # Sample precision and recall from Beta distributions centered on observed values
        sampled_precision = np.random.beta(
            precision * concentration, 
            (1 - precision) * concentration
        )
        sampled_recall = np.random.beta(
            recall * concentration,
            (1 - recall) * concentration
        )
        
        # Sample the observed positives with some noise
        k_sampled = np.random.binomial(N, k/N)
        
        # Calculate estimate with sampled parameters
        T_est = k_sampled * sampled_precision / sampled_recall if sampled_recall > 0 else 0
        percentage_est = (T_est / N) * 100
        bootstrap_estimates.append(percentage_est)
    
    # Calculate confidence interval
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = max(0, np.percentile(bootstrap_estimates, lower_percentile))
    upper_bound = min(100, np.percentile(bootstrap_estimates, upper_percentile))
    
    # Calculate point estimate
    point_estimate = (estimated_true_positives / N) * 100
    
    return point_estimate, lower_bound, upper_bound


# Example usage
if __name__ == "__main__":
    # Example values
    k = 156963  # Items classified as positive
    N = 940372  # Total population
    precision = 0.88  # 80% precision
    recall = 0.81  # 70% recall
    test_set_sample_size = 100
    
    # Estimate using bootstrapping
    est, lower, upper = bootstrap_classifier_percentage_ci(k, N, precision, recall, test_set_size=test_set_sample_size)
    
    print(f"Estimated percentage: {est:.2f}%")
    print(f"95% Confidence Interval: [{lower:.2f}%, {upper:.2f}%]")