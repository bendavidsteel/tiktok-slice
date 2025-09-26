import numpy as np
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt

class RepresentativenessAdjustedEstimator:
    """
    Estimator that accounts for non-representative test sets through
    systematic uncertainty quantification.
    """
    
    def __init__(self, k, N, precision, recall, test_set_size, 
                 representativeness_confidence=0.7):
        """
        Parameters:
        - k: Number classified as positive
        - N: Total population size
        - precision, recall: Point estimates from test set
        - test_set_size: Size of test set used for precision/recall
        - representativeness_confidence: How confident you are that the test set
                                        represents the true population (0-1 scale)
                                        1.0 = perfectly representative
                                        0.5 = moderately representative
                                        0.1 = barely representative
        """
        self.k = k
        self.N = N
        self.precision = precision
        self.recall = recall
        self.test_set_size = test_set_size
        self.repr_conf = representativeness_confidence
        
    def calculate_effective_sample_size(self):
        """
        Adjust effective sample size based on representativeness.
        Key insight: A large but unrepresentative sample provides
        less information than its size suggests.
        """
        # Effective sample size decreases with lower representativeness
        # This is based on the concept from survey statistics where
        # design effect adjusts sample size for clustering/stratification
        design_effect = 1 / self.repr_conf**2
        effective_n = self.test_set_size / design_effect
        return max(10, effective_n)  # Minimum of 10 to avoid numerical issues
    
    def get_parameter_distributions(self):
        """
        Create distributions for precision and recall that incorporate
        both sampling uncertainty and representativeness uncertainty.
        """
        eff_n = self.calculate_effective_sample_size()
        
        # Method 1: Beta distributions with effective sample size
        # The concentration parameter reflects our "effective" information
        alpha_prec = self.precision * eff_n
        beta_prec = (1 - self.precision) * eff_n
        
        alpha_rec = self.recall * eff_n
        beta_rec = (1 - self.recall) * eff_n
        
        return {
            'precision': {'alpha': alpha_prec, 'beta': beta_prec},
            'recall': {'alpha': alpha_rec, 'beta': beta_rec}
        }
    
    def hierarchical_bootstrap(self, n_bootstrap=2000, confidence=0.95):
        """
        Hierarchical bootstrap that samples at multiple levels:
        1. Sample possible "true" precision/recall from wide distributions
        2. Sample the classification process
        3. Account for possible systematic biases
        """
        
        dists = self.get_parameter_distributions()
        bootstrap_estimates = []
        
        for _ in tqdm(range(n_bootstrap), desc="Hierarchical bootstrap"):
            # Level 1: Sample from parameter uncertainty
            prec_sample = np.random.beta(
                dists['precision']['alpha'],
                dists['precision']['beta']
            )
            rec_sample = np.random.beta(
                dists['recall']['alpha'],
                dists['recall']['beta']
            )
            
            # Level 2: Add systematic bias possibility
            # When test set is unrepresentative, there might be systematic bias
            if self.repr_conf < 0.9:
                # Possible systematic bias increases with lower representativeness
                bias_scale = (1 - self.repr_conf) * 0.2  # Up to ±20% bias
                
                # Sample potential bias
                prec_bias = np.random.normal(0, bias_scale)
                rec_bias = np.random.normal(0, bias_scale)
                
                # Apply bias with bounds
                prec_sample = np.clip(prec_sample + prec_bias, 0.01, 0.99)
                rec_sample = np.clip(rec_sample + rec_bias, 0.01, 0.99)
            
            # Level 3: Simulate classification process
            estimated_true_prop = (self.k * prec_sample / rec_sample) / self.N
            estimated_true_prop = np.clip(estimated_true_prop, 0, 1)
            
            # Add population-level uncertainty
            n_actual_positives = np.random.binomial(self.N, estimated_true_prop)
            n_actual_negatives = self.N - n_actual_positives
            
            # Simulate classification
            true_positives = np.random.binomial(n_actual_positives, rec_sample)
            
            if prec_sample > 0 and prec_sample < 1:
                expected_fp = true_positives * (1 - prec_sample) / prec_sample
                fp_rate = expected_fp / n_actual_negatives if n_actual_negatives > 0 else 0
                fp_rate = np.clip(fp_rate, 0, 1)
                false_positives = np.random.binomial(n_actual_negatives, fp_rate)
            else:
                false_positives = 0
            
            k_sim = true_positives + false_positives
            
            # Estimate
            if rec_sample > 0:
                est = (k_sim * prec_sample / rec_sample) / self.N * 100
            else:
                est = 0
            
            bootstrap_estimates.append(est)
        
        # Calculate intervals
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_estimates, alpha/2 * 100)
        upper = np.percentile(bootstrap_estimates, (1-alpha/2) * 100)
        point = np.median(bootstrap_estimates)
        
        return point, lower, upper, bootstrap_estimates
    

def demonstrate_approach():
    """
    Demonstrate the complete approach with your example.
    """
    # Your parameters
    k = 156963
    N = 940372
    precision = 0.88
    recall = 0.81
    test_set_size = 6110  # Example test set size
    
    print("="*70)
    print("REPRESENTATIVENESS-ADJUSTED CONFIDENCE INTERVALS")
    print("="*70)
    
    # Scenario 1: High confidence in representativeness
    print("\n1. HIGH CONFIDENCE in test set representativeness (0.9):")
    estimator_high = RepresentativenessAdjustedEstimator(
        k, N, precision, recall, test_set_size, 
        representativeness_confidence=0.9
    )
    point_h, lower_h, upper_h, _ = estimator_high.hierarchical_bootstrap()
    print(f"   Estimate: {point_h:.2f}%")
    print(f"   95% CI: [{lower_h:.2f}%, {upper_h:.2f}%]")
    print(f"   Width: {upper_h - lower_h:.2f}%")
    
    # Scenario 2: Moderate confidence
    print("\n2. MODERATE CONFIDENCE in test set representativeness (0.5):")
    estimator_mod = RepresentativenessAdjustedEstimator(
        k, N, precision, recall, test_set_size,
        representativeness_confidence=0.5
    )
    point_m, lower_m, upper_m, _ = estimator_mod.hierarchical_bootstrap()
    print(f"   Estimate: {point_m:.2f}%")
    print(f"   95% CI: [{lower_m:.2f}%, {upper_m:.2f}%]")
    print(f"   Width: {upper_m - lower_m:.2f}%")
    
    # Scenario 3: Low confidence
    print("\n3. LOW CONFIDENCE in test set representativeness (0.2):")
    estimator_low = RepresentativenessAdjustedEstimator(
        k, N, precision, recall, test_set_size,
        representativeness_confidence=0.2
    )
    point_l, lower_l, upper_l, dists = estimator_low.hierarchical_bootstrap()
    print(f"   Estimate: {point_l:.2f}%")
    print(f"   95% CI: [{lower_l:.2f}%, {upper_l:.2f}%]")
    print(f"   Width: {upper_l - lower_l:.2f}%")
    

if __name__ == "__main__":
    demonstrate_approach()