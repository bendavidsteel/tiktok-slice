import os
import numpy as np
import polars as pl
from tqdm import tqdm
import gc
import warnings
warnings.filterwarnings('ignore')

# GPU acceleration with cuML
import cudf
import cuml
from cuml.linear_model import LinearRegression as cuLinearRegression
from cuml.preprocessing import StandardScaler as cuStandardScaler
from cuml.metrics import r2_score as cu_r2_score

print("Using cuML GPU acceleration")

def convert_to_gpu(df):
    """Convert Polars DataFrame to cuDF for GPU processing"""
    return cudf.from_pandas(df.to_pandas())

def do_regression_gpu(indep_df, dep_df, check_collinearity=True, gpu_memory_limit=8):
    """GPU-accelerated regression using cuML"""
    dep_vars = ['playCount']
    indep_cols = indep_df.columns
    
    # Combine dataframes
    df = pl.concat([indep_df, dep_df], how='horizontal')
    df = df.drop_nulls().drop_nans()
    
    print(f"Processing {len(df)} rows with {len(indep_cols)} features")
    
    # Check if dataset fits in GPU memory (rough estimate)
    estimated_memory_gb = (len(df) * len(indep_cols) * 8) / (1024**3)  # 8 bytes per float64
    
    if estimated_memory_gb > gpu_memory_limit:
        print(f"Dataset too large for GPU memory ({estimated_memory_gb:.2f}GB > {gpu_memory_limit}GB)")
        print("Using chunked processing...")
        return do_regression_chunked_gpu(indep_df, dep_df, chunk_size=50000)
    
    # Convert to GPU dataframes
    gpu_df = convert_to_gpu(df)
    X = gpu_df[indep_cols]
    print("Data loaded to GPU")
    
    # Efficient collinearity check on GPU
    if check_collinearity and len(indep_cols) < 1000:
        print("Checking collinearity...")
        # Use cuML's correlation
        corr_matrix = X.corr().to_pandas().values
            
        high_corr_pairs = np.where(np.abs(corr_matrix) > 0.8)
        corr_count = 0
        for i, j in zip(high_corr_pairs[0], high_corr_pairs[1]):
            if i != j and i < j:
                print(f"High correlation between {indep_cols[i]} and {indep_cols[j]}: {corr_matrix[i, j]:.3f}")
                corr_count += 1
                if corr_count > 10:  # Limit output
                    print("... (showing first 10 high correlations)")
                    break
    
    results = {}
    
    for dep_var in dep_vars:
        print(f"\nProcessing {dep_var}...")
        
        y = gpu_df[dep_var]
        
        # Use cuML LinearRegression
        model = cuLinearRegression(fit_intercept=True, normalize=False)
        model.fit(X, y)
        
        # Get predictions and calculate R²
        predictions = model.predict(X)
        r2 = cu_r2_score(y, predictions)
        
        # Get coefficients
        coefs = model.coef_.to_pandas() if hasattr(model.coef_, 'to_pandas') else model.coef_
        intercept = model.intercept_
        
        print(f"GPU Results for {dep_var}, R²: {r2:.4f}")
        
        # Create results DataFrame
        results_df = pl.DataFrame({
            'feature': indep_cols,
            'coefficient': coefs.flatten() if hasattr(coefs, 'flatten') else coefs,
        })
        
        # Filter and display significant results
        significant_results = results_df.filter(pl.col('coefficient').abs() > 0.01)
        
        if len(significant_results) > 0:
            print(f"Found {len(significant_results)} significant coefficients:")
            if len(significant_results) > 10:
                print("Top 5 positive coefficients:")
                print(significant_results.sort('coefficient', descending=True).head(5))
                print("Top 5 negative coefficients:")
                print(significant_results.sort('coefficient').head(5))
            else:
                print(significant_results)
        
        results[dep_var] = results_df
        
    
    return results

def do_regression_chunked_gpu(indep_df, dep_df, chunk_size=50000):
    """Process large datasets in chunks with GPU acceleration"""
    dep_vars = ['playCount']
    indep_cols = indep_df.columns
    
    # Combine dataframes
    df = pl.concat([indep_df, dep_df], how='horizontal')
    df = df.drop_nulls().drop_nans()
    
    n_rows = len(df)
    n_chunks = (n_rows + chunk_size - 1) // chunk_size
    
    print(f"Processing {n_rows} rows in {n_chunks} chunks of size {chunk_size}")
    
    for dep_var in dep_vars:
        print(f"\nProcessing {dep_var} in chunks...")
        
        all_coefs = []
        all_r2s = []
        
        for chunk_idx in tqdm(range(n_chunks), desc="Processing chunks"):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, n_rows)
            
            # Get chunk
            chunk = df.slice(start_idx, end_idx - start_idx)
            
            gpu_chunk = convert_to_gpu(chunk)
            X_chunk = gpu_chunk[indep_cols]
            y_chunk = gpu_chunk[dep_var]
            
            # Fit model on GPU
            model = cuLinearRegression(fit_intercept=True)
            model.fit(X_chunk, y_chunk)
            
            # Get results
            predictions = model.predict(X_chunk)
            r2 = cu_r2_score(y_chunk, predictions)
            coefs = model.coef_.to_pandas() if hasattr(model.coef_, 'to_pandas') else model.coef_
            
            all_coefs.append(coefs.flatten() if hasattr(coefs, 'flatten') else coefs)
            all_r2s.append(r2)
            
            # Clean up GPU memory
            del gpu_chunk, model, predictions
            cudf.core.memory.get_global_memory_manager().deallocate_all()
        
        # Combine results (weighted average by chunk size)
        final_coefs = np.mean(all_coefs, axis=0)
        avg_r2 = np.mean(all_r2s)
        
        print(f"Final results for {dep_var}:")
        print(f"Average R²: {avg_r2:.4f}")
        
        # Create results DataFrame
        results_df = pl.DataFrame({
            'feature': indep_cols,
            'coefficient': final_coefs,
        })
        
        # Show significant coefficients
        significant_results = results_df.filter(pl.col('coefficient').abs() > 0.01)
        
        if len(significant_results) > 0:
            print(f"Found {len(significant_results)} significant coefficients:")
            if len(significant_results) > 10:
                print("Top 5 positive:")
                print(significant_results.sort('coefficient', descending=True).head(5))
                print("Top 5 negative:")
                print(significant_results.sort('coefficient').head(5))
            else:
                print(significant_results)

def drop_low_sum_cols_gpu(df, min_sum=100):
    """GPU-accelerated version of dropping low sum columns"""
    gpu_df = convert_to_gpu(df)
    # Calculate sums on GPU
    col_sums = gpu_df.sum()
    
    # Find columns to keep
    cols_to_keep = []
    for col in gpu_df.columns:
        if col_sums[col] >= min_sum:
            cols_to_keep.append(col)
    
    # Return as Polars DataFrame
    result = gpu_df[cols_to_keep].to_pandas()
    return pl.from_pandas(result)

def process_hashtags_gpu(video_df, min_count=50):
    """GPU-accelerated hashtag processing"""
    print("Extracting hashtags...")
    
    # Extract hashtags (CPU operation for string processing)
    hashtag_df = (video_df
                  .select(['video_id', 'desc'])
                  .with_columns(
                      pl.col('desc').str.extract_all('#[a-zA-Z0-9_]+').alias('hashtags')
                  )
                  .select(['video_id', 'hashtags'])
                  .explode('hashtags')
                  .filter(pl.col('hashtags').is_not_null())
                  )
    
    print(f"Found {len(hashtag_df)} hashtag instances")
    
    # Get hashtag counts
    hashtag_counts = (hashtag_df
                     .group_by('hashtags')
                     .len()
                     .filter(pl.col('len') > min_count)
                     .sort('len', descending=True)
                     )
    
    print(f"Keeping {len(hashtag_counts)} hashtags with >{min_count} occurrences")
    
    # Create dummy variables
    hashtag_dummies_df = (video_df
                          .select('video_id')
                          .join(
                              hashtag_df
                              .join(hashtag_counts.select('hashtags'), on='hashtags', how='semi')
                              .to_dummies('hashtags')
                              .group_by('video_id')
                              .sum(),
                              on='video_id',
                              how='left'
                          )
                          .drop('video_id')
                          .fill_null(0)
                          )
    
    return drop_low_sum_cols_gpu(hashtag_dummies_df, min_sum=100)

def prepare_dependent_variables_gpu(video_df):
    """Prepare and normalize dependent variables on GPU"""
    print("Preparing dependent variables...")
    
    # Log transform
    dep_df = (video_df
              .select(['diggCount', 'commentCount', 'shareCount', 'playCount'])
              .with_columns([
                  pl.col('diggCount').log1p(),
                  pl.col('commentCount').log1p(),
                  pl.col('shareCount').log1p(),
                  pl.col('playCount').log1p()
              ])
              )
    
    # Use GPU for standardization
    gpu_dep_df = convert_to_gpu(dep_df)
    scaler = cuStandardScaler()
    
    # Fit and transform on GPU
    scaled_data = scaler.fit_transform(gpu_dep_df)
    
    # Convert back to Polars
    scaled_df = pl.from_pandas(scaled_data.to_pandas())
    scaled_df.columns = dep_df.columns
    
    return scaled_df

def correlations_gpu(video_df: pl.DataFrame, sample_size=None):
    """GPU-accelerated correlation analysis"""
    print(f"Starting GPU correlation analysis on {len(video_df)} videos")
    
    # Sample data if specified
    if sample_size and len(video_df) > sample_size:
        print(f"Sampling {sample_size} videos...")
        video_df = video_df.sample(n=sample_size, seed=42)
    
    # Prepare dependent variables
    dep_df = prepare_dependent_variables_gpu(video_df)
    
    # Process hashtags
    print("\n" + "="*50)
    print("HASHTAG ANALYSIS")
    print("="*50)
    hashtag_dummies_df = process_hashtags_gpu(video_df, min_count=50)
    print(f"Hashtag features shape: {hashtag_dummies_df.shape}")
    
    print("Running hashtag regression...")
    hashtag_results = do_regression_gpu(hashtag_dummies_df, dep_df)
    
    # Process topics
    print("\n" + "="*50)
    print("TOPIC ANALYSIS")
    print("="*50)
    
    topic_dummies_df = (video_df['topic_layer_0']
                        .to_dummies()
                        .drop('topic_layer_0_null', strict=False)
                        )
    
    print(f"Topic features shape: {topic_dummies_df.shape}")
    
    if topic_dummies_df.shape[1] > 0:
        print("Running topic regression...")
        topic_results = do_regression_gpu(topic_dummies_df, dep_df)
    else:
        print("No topic features to analyze")
    

def main():
    """Main function with GPU acceleration"""
    print("GPU-Accelerated TikTok Correlation Analysis")
    print("="*50)
    
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    
    # Load data
    hour_engagement_path = './data/stats/1hour/engagement.parquet.zstd'
    day_engagement_path = './data/stats/24hour/engagement.parquet.zstd'
    
    if os.path.exists(hour_engagement_path) and os.path.exists(day_engagement_path):
        print("Loading pre-processed engagement data...")
        hour_video_df = pl.read_parquet(hour_engagement_path)
        day_video_df = pl.read_parquet(day_engagement_path)
    else:
        print("Processing raw video data...")
        hour_video_df = pl.DataFrame()
        day_video_df = pl.DataFrame()
        
        video_dir_path = os.path.join('.', 'data', 'results', '2024_04_10', 'hours')
        video_pbar = tqdm(total=60*60 + 23 * 60, desc='Reading videos')
        
        for root, dirs, files in os.walk(video_dir_path):
            for file in files:
                if file == 'videos.parquet.zstd':
                    root_sections = root.split('/')
                    hour, minute, second = root_sections[-3], root_sections[-2], root_sections[-1]
                    video_pbar.update(1)
                    
                    result_path = os.path.join(root, file)
                    batch_video_df = pl.read_parquet(result_path)
                    
                    # Select relevant columns
                    batch_video_df = batch_video_df.select([
                        pl.col('video_id'),
                        pl.col('authorVerified'),
                        pl.col('author').struct.field('nickname').alias('authorNickname'),
                        pl.col('author').struct.field('signature').alias('authorSignature'),
                        pl.col('musicOriginal'),
                        pl.col('videoDuration'),
                        pl.col('videoQuality'),
                        pl.col('locationCreated'),
                        pl.col('desc'),
                        pl.col('shareCount'),
                        pl.col('diggCount'),
                        pl.col('commentCount'),
                        pl.col('playCount'),
                        pl.col('diversificationLabels'),
                        pl.col('aigcLabelType')
                    ])
                    
                    if minute == '42':
                        day_video_df = pl.concat([day_video_df, batch_video_df], how='diagonal_relaxed')
                    if hour == '19':
                        hour_video_df = pl.concat([hour_video_df, batch_video_df], how='diagonal_relaxed')
        
        video_pbar.close()
    
    # Join topic data
    hour_topic_path = './data/topic_model_videos_toponymy/video_topics.parquet.gzip'
    day_topic_path = './data/topic_model_videos_toponymy_24hour/video_topics.parquet.gzip'
    
    if os.path.exists(hour_topic_path):
        print("Loading topic data...")
        hour_topic_df = pl.read_parquet(hour_topic_path, columns=['id', 'topic_layer_0'])
        hour_video_df = hour_video_df.join(hour_topic_df, left_on='video_id', right_on='id', how='left')
    
    if os.path.exists(day_topic_path):
        day_topic_df = pl.read_parquet(day_topic_path, columns=['id', 'topic_layer_0'])
        day_video_df = day_video_df.join(day_topic_df, left_on='video_id', right_on='id', how='left')
    
    # Run analysis
    print("\n" + "="*60)
    print("HOUR-BASED ANALYSIS")
    print("="*60)
    correlations_gpu(hour_video_df, sample_size=None)  # Limit sample size for GPU memory
    
    print("\n" + "="*60)
    print("DAY-BASED ANALYSIS")
    print("="*60)
    correlations_gpu(day_video_df, sample_size=None)  # Limit sample size for GPU memory
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()