import numpy as np
from scipy import stats
from scipy.sparse import csr_array
from itertools import combinations

def check_block_frequency(n_videos=5157488, n_commenters=13500000,
                        mean_comments_per_video=4.92, view_weighted=True,
                        block_size=10, video_count=5, n_trials=10):
    """
    Check frequency of comment blocks using the generative model.
    
    Parameters:
    -----------
    n_videos, n_commenters, mean_comments_per_video, view_weighted : same as generate_comment_model
    block_size : int
        Number of commenters required in a block
    video_count : int
        Number of videos required in a block
    n_trials : int
        Number of simulations to run
    """
    block_counts = []
    
    for trial in range(n_trials):
        # Generate view counts and probabilities
        view_counts = np.random.lognormal(mean=7.5, sigma=1.8, size=n_videos)
        if view_weighted:
            comment_probs = view_counts / view_counts.sum()
        else:
            comment_probs = np.ones(n_videos) / n_videos
            
        # Generate comments
        total_comments = np.random.negative_binomial(
            n=2, 
            p=2/(2 + mean_comments_per_video),
            size=n_videos
        )
        
        # Generate commenters with preferential attachment
        total_comment_count = total_comments.sum()
        commenter_weights = np.power(np.ones(n_commenters), 0.1)
        commenter_probs = commenter_weights / commenter_weights.sum()
        
        all_commenters = np.random.choice(
            n_commenters,
            size=total_comment_count,
            p=commenter_probs
        )
        
        # Create sparse matrix
        video_indices = np.repeat(np.arange(n_videos), total_comments)
        data = np.ones(total_comment_count, dtype=np.uint8)
        
        comment_matrix = csr_array(
            (data, (all_commenters, video_indices)),
            shape=(n_commenters, n_videos),
            dtype=np.uint8
        )
        
        # Find blocks
        # First get videos with enough commenters
        video_commenter_counts = comment_matrix.sum(axis=0)
        candidate_videos = np.where(video_commenter_counts >= block_size)[0]
        
        # Sample a subset of videos if too many candidates
        if len(candidate_videos) > 1000:
            candidate_videos = np.random.choice(candidate_videos, 1000, replace=False)
        
        # Count blocks
        blocks = 0
        for videos in combinations(candidate_videos, video_count):
            # Get commenters who commented on all these videos
            video_slice = comment_matrix[:, videos].tocsc()
            commenter_counts = video_slice.sum(axis=1)
            common_commenters = (commenter_counts >= video_count).sum()
            
            if common_commenters >= block_size:
                blocks += 1
        
        block_counts.append(blocks)
        print(f"Trial {trial+1}: Found {blocks} blocks")
    
    print("\nSummary Statistics:")
    print(f"Mean blocks per trial: {np.mean(block_counts):.2f}")
    print(f"Standard deviation: {np.std(block_counts):.2f}")
    print(f"Range: {min(block_counts)} - {max(block_counts)}")
    
    # Calculate approximate probability
    total_possible = np.math.comb(n_commenters, block_size) * np.math.comb(n_videos, video_count)
    avg_probability = np.mean(block_counts) / total_possible
    print(f"\nApproximate probability of such a block: {avg_probability:.2e}")
    
    return block_counts, avg_probability

if __name__ == "__main__":
    # Run with smaller numbers first for testing
    counts, prob = check_block_frequency(
        n_videos=100000,  # smaller for testing
        n_commenters=200000,  # smaller for testing
        block_size=10,
        video_count=5,
        n_trials=5
    )