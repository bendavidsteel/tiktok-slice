import configparser
import os
import time
import joblib  # Added for model saving

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import sklearn
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    HistGradientBoostingClassifier, ExtraTreesClassifier,
    StackingClassifier
)
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
import tqdm
import torch
import transformers


def get_videos_embeddings(embeddings_dir_path):
    embeddings = None
    img_features = None
    video_df = None
    filenames = os.listdir(embeddings_dir_path)
    if 'video_embeddings.npy' in filenames and 'videos.parquet.gzip' in filenames and 'img_features.npy' in filenames:
        batch_embeddings = np.load(os.path.join(embeddings_dir_path, 'video_embeddings.npy'), allow_pickle=True)
        batch_img_features = np.load(os.path.join(embeddings_dir_path, 'img_features.npy'), allow_pickle=True)
        if not batch_embeddings.shape:
            raise ValueError(f"Embeddings shape is empty")

        batch_video_df = pl.read_parquet(os.path.join(embeddings_dir_path, 'videos.parquet.gzip'))

        if batch_embeddings.shape[0] != len(batch_video_df):
            raise ValueError(f"Embeddings shape {batch_embeddings.shape[0]} does not match number of videos {len(batch_video_df)}")
        
        assert batch_embeddings.shape[0] == len(batch_video_df)

        if embeddings is None:
            embeddings = batch_embeddings
        else:
            embeddings = np.concatenate([embeddings, batch_embeddings])

        if img_features is None:
            img_features = batch_img_features
        else:
            img_features = np.concatenate([img_features, batch_img_features])

        if video_df is None:
            video_df = batch_video_df
        else:
            video_df = pl.concat([video_df, batch_video_df], how='diagonal_relaxed')
    else:
        raise ValueError(f"Missing files in {embeddings_dir_path}")

    return embeddings, img_features, video_df


class LogisticRegressionClassifier:
    def __init__(self, random_state=42):
        self.model = LogisticRegression(
            max_iter=1000, 
            random_state=random_state,
            C=1.0,
            class_weight='balanced'  # Helps with imbalanced datasets
        )
        self.is_trained = False
        self.name = "Logistic Regression"
        
    def train(self, X_train, y_train):
        """Train the logistic regression model"""
        print(f"Training {self.name} on {X_train.shape[0]} samples...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("Training complete.")
        
    def predict(self, X):
        """Predict class labels for X"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities for X"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)[:, 1]  # Return probability of positive class


class HistGradientBoostingClassifier:
    def __init__(self, random_state=42):
        self.model = sklearn.ensemble.HistGradientBoostingClassifier(
            max_iter=100,  # Number of boosting iterations
            learning_rate=0.1,
            max_depth=None,  # Let the model decide the depth
            l2_regularization=1.0,
            random_state=random_state
        )
        self.is_trained = False
        self.name = "HistGradient Boosting"
        
    def train(self, X_train, y_train):
        """Train the histogram-based gradient boosting model"""
        print(f"Training {self.name} on {X_train.shape[0]} samples...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("Training complete.")
        
    def predict(self, X):
        """Predict class labels for X"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities for X"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)[:, 1]  # Return probability of positive class


class ExtraTreesClassifier:
    def __init__(self, random_state=42):
        self.model = sklearn.ensemble.ExtraTreesClassifier(
            n_estimators=100,
            criterion='gini',
            max_features='sqrt',  # Use sqrt(n_features) features per split
            bootstrap=True,  # Use bootstrap samples
            class_weight='balanced',  # Automatically adjust weights
            n_jobs=-1,  # Use all available cores
            random_state=random_state
        )
        self.is_trained = False
        self.name = "Extra Trees"
        
    def train(self, X_train, y_train):
        """Train the Extra Trees model"""
        print(f"Training {self.name} on {X_train.shape[0]} samples...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("Training complete.")
        
    def predict(self, X):
        """Predict class labels for X"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities for X"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)[:, 1]  # Return probability of positive class


class StackingEnsembleClassifier:
    def __init__(self, random_state=42):
        # Define base classifiers
        estimators = [
            ('logistic', LogisticRegression(
                max_iter=1000, 
                random_state=random_state,
                C=1.0
            )),
            ('hist_gb', sklearn.ensemble.HistGradientBoostingClassifier(
                max_iter=100, 
                random_state=random_state
            )),
            ('extra_trees', sklearn.ensemble.ExtraTreesClassifier(
                n_estimators=100, 
                random_state=random_state,
                n_jobs=-1
            ))
        ]
        
        # Define final estimator
        final_estimator = LogisticRegression(max_iter=1000, random_state=random_state)
        
        # Create stacking ensemble
        self.model = sklearn.ensemble.StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=5,  # 5-fold cross-validation for stacking
            stack_method='predict_proba',
            n_jobs=-1  # Use all available cores
        )
        self.is_trained = False
        self.name = "Stacking Ensemble"
        
    def train(self, X_train, y_train):
        """Train the stacking ensemble model"""
        print(f"Training {self.name} on {X_train.shape[0]} samples...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("Training complete.")
        
    def predict(self, X):
        """Predict class labels for X"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities for X"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)[:, 1]  # Return probability of positive class


def get_dir_probs(dir_path):
    embeddings, img_features, video_df = get_videos_embeddings(dir_path)
    return embeddings, img_features, video_df


def perform_cross_validation(models, X, y, cv=5):
    """Perform cross-validation for multiple models and return results"""
    cv_results = {}
    
    # Define the stratified k-fold cross-validator
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    for model_name, model_instance in models.items():
        print(f"\n--- Cross-validation for {model_name} ---")
        
        fold_accuracies = []
        fold_precisions = []
        fold_recalls = []
        fold_f1s = []
        fold_times = []  # Track inference time for each fold
        
        # Manually perform cross-validation to get more detailed metrics
        for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
            X_train_fold, X_test_fold = X[train_index], X[test_index]
            y_train_fold, y_test_fold = y[train_index], y[test_index]
            
            # Train the model
            model_instance.train(X_train_fold, y_train_fold)
            
            # Measure inference time
            start_time = time.time()
            y_pred = model_instance.predict(X_test_fold)
            inference_time = time.time() - start_time
            
            # Calculate metrics
            accuracy = accuracy_score(y_test_fold, y_pred)
            precision = precision_score(y_test_fold, y_pred)
            recall = recall_score(y_test_fold, y_pred)
            f1 = f1_score(y_test_fold, y_pred)
            samples_per_second = len(X_test_fold) / inference_time
            
            # Store metrics
            fold_accuracies.append(accuracy)
            fold_precisions.append(precision)
            fold_recalls.append(recall)
            fold_f1s.append(f1)
            fold_times.append(samples_per_second)
            
            print(f"Fold {fold+1}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, Speed={samples_per_second:.1f} samples/sec")
        
        # Calculate average metrics
        avg_accuracy = np.mean(fold_accuracies)
        avg_precision = np.mean(fold_precisions)
        avg_recall = np.mean(fold_recalls)
        avg_f1 = np.mean(fold_f1s)
        avg_speed = np.mean(fold_times)
        
        # Calculate standard deviations
        std_accuracy = np.std(fold_accuracies)
        std_precision = np.std(fold_precisions)
        std_recall = np.std(fold_recalls)
        std_f1 = np.std(fold_f1s)
        std_speed = np.std(fold_times)
        
        # Store results
        cv_results[model_name] = {
            'accuracies': fold_accuracies,
            'precisions': fold_precisions,
            'recalls': fold_recalls,
            'f1s': fold_f1s,
            'speeds': fold_times,
            'avg_accuracy': avg_accuracy,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_f1': avg_f1,
            'avg_speed': avg_speed,
            'std_accuracy': std_accuracy,
            'std_precision': std_precision,
            'std_recall': std_recall,
            'std_f1': std_f1,
            'std_speed': std_speed
        }
        
        print(f"Average: Accuracy={avg_accuracy:.4f}±{std_accuracy:.4f}, "
              f"Precision={avg_precision:.4f}±{std_precision:.4f}, "
              f"Recall={avg_recall:.4f}±{std_recall:.4f}, "
              f"F1={avg_f1:.4f}±{std_f1:.4f}, "
              f"Speed={avg_speed:.1f}±{std_speed:.1f} samples/sec")
    
    return cv_results


def plot_cross_validation_results(cv_results, save_path='./figs'):
    """Plot cross-validation results for comparison"""
    os.makedirs(save_path, exist_ok=True)
    
    # Convert results to a format suitable for plotting
    model_names = list(cv_results.keys())
    
    # Create a grouped bar chart for average metrics
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(model_names))
    width = 0.2
    
    metrics = ['avg_accuracy', 'avg_precision', 'avg_recall', 'avg_f1']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Plot bars for each metric
    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        values = [cv_results[model][metric] for model in model_names]
        errors = [cv_results[model][f'std_{metric[4:]}'] for model in model_names]
        
        ax.bar(x + i*width - width*1.5, values, width, label=label, 
               color=color, alpha=0.7, yerr=errors, capsize=5)
    
    ax.set_ylabel('Score')
    ax.set_title('Cross-Validation Results Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(save_path, 'cross_validation_comparison.png'))
    
    # Create a box plot for F1 scores across folds
    fig, ax = plt.subplots(figsize=(10, 6))
    
    f1_data = [cv_results[model]['f1s'] for model in model_names]
    
    ax.boxplot(f1_data, labels=model_names)
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Scores Distribution Across Cross-Validation Folds')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(save_path, 'f1_distribution_boxplot.png'))
    
    # Plot speed vs F1 score
    fig, ax = plt.subplots(figsize=(10, 6))
    
    f1_scores = [cv_results[model]['avg_f1'] for model in model_names]
    f1_stds = [cv_results[model]['std_f1'] for model in model_names]
    speeds = [cv_results[model]['avg_speed'] for model in model_names]
    speed_stds = [cv_results[model]['std_speed'] for model in model_names]
    
    # Create scatter plot with error bars
    ax.errorbar(speeds, f1_scores, xerr=speed_stds, yerr=f1_stds, 
                fmt='o', ecolor='black', capsize=5, markersize=8)
    
    # Add model names as annotations
    for i, model in enumerate(model_names):
        ax.annotate(model, (speeds[i], f1_scores[i]), 
                   xytext=(7, 0), textcoords='offset points',
                   fontsize=10, va='center')
    
    ax.set_xlabel('Inference Speed (samples/second)')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score vs Inference Speed')
    ax.grid(alpha=0.3)
    
    # Add a line connecting the pareto-optimal points
    # Sort by speed
    sorted_indices = np.argsort(speeds)
    sorted_speeds = [speeds[i] for i in sorted_indices]
    sorted_f1s = [f1_scores[i] for i in sorted_indices]
    
    # Find Pareto-optimal points (non-dominated solutions)
    pareto_indices = []
    max_f1 = -float('inf')
    
    for i in range(len(sorted_speeds)):
        if sorted_f1s[i] > max_f1:
            pareto_indices.append(i)
            max_f1 = sorted_f1s[i]
    
    pareto_speeds = [sorted_speeds[i] for i in pareto_indices]
    pareto_f1s = [sorted_f1s[i] for i in pareto_indices]
    
    # Plot Pareto frontier
    ax.plot(pareto_speeds, pareto_f1s, 'r--', alpha=0.7, label='Pareto Frontier')
    ax.legend()
    
    plt.tight_layout()
    fig.savefig(os.path.join(save_path, 'f1_vs_speed.png'))


def evaluate_model(y_true, y_pred, y_prob=None, X_test=None, model=None):
    """Evaluate model performance with timing"""
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }
    
    # Measure inference time if model and test data provided
    if model is not None and X_test is not None:
        # Run multiple times to get stable timing
        num_runs = 5
        timing_results = []
        
        for _ in range(num_runs):
            start_time = time.time()
            _ = model.predict(X_test)
            end_time = time.time()
            timing_results.append(end_time - start_time)
        
        # Calculate average time
        avg_time = np.mean(timing_results)
        samples_per_second = len(X_test) / avg_time
        
        results["inference_time"] = avg_time
        results["samples_per_second"] = samples_per_second
    
    # If probabilities are provided, calculate F1 for different thresholds
    if y_prob is not None:
        thresholds = np.linspace(0.3, 0.9, 20)
        f1s = []
        for threshold in thresholds:
            threshold_preds = y_prob > threshold
            f1s.append(f1_score(y_true, threshold_preds))
        
        # Find best threshold
        best_idx = np.argmax(f1s)
        results["best_threshold"] = thresholds[best_idx]
        results["best_f1"] = f1s[best_idx]
        results["thresholds"] = thresholds
        results["f1_scores"] = f1s
    
    return results


def plot_evaluation_results(results, save_path='./figs'):
    """Plot evaluation results"""
    os.makedirs(save_path, exist_ok=True)
    
    # Plot F1 scores vs thresholds
    if "thresholds" in results and "f1_scores" in results:
        fig, ax = plt.subplots()
        ax.plot(results["thresholds"], results["f1_scores"])
        ax.axvline(x=results["best_threshold"], color='r', linestyle='--', 
                  label=f'Best threshold: {results["best_threshold"]:.2f}')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 Score vs Threshold')
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(save_path, 'model_thres_f1.png'))
    
    # Create a bar chart for metrics
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    if all(metric in results for metric in metrics):
        fig, ax = plt.subplots()
        values = [results[metric] for metric in metrics]
        ax.bar(metrics, values)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Metrics')
        for i, v in enumerate(values):
            ax.text(i, v + 0.01, f'{v:.2f}', ha='center')
        plt.tight_layout()
        fig.savefig(os.path.join(save_path, 'model_metrics.png'))



def save_trained_models(models, save_dir='./models'):
    """Save trained models to disk"""
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n--- Saving Trained Models to {save_dir} ---")
    
    for model_name, model_instance in models.items():
        if model_instance.is_trained:
            # Create filename based on model name
            filename = f"{model_name.lower().replace(' ', '_')}.joblib"
            filepath = os.path.join(save_dir, filename)
            
            # Save the model
            joblib.dump(model_instance, filepath)
            print(f"Saved {model_name} to {filepath}")
        else:
            print(f"Warning: {model_name} is not trained and will not be saved")


def main():
    config = configparser.ConfigParser()
    config.read('./config/config.ini')

    # Get data from directories
    dir_path = './data/aif_aigc'
    
    # Process data once and store results for both models
    print("\n--- Processing Child Videos ---")
    embeddings, img_features, id_df = get_dir_probs(dir_path)
    video_df = pl.read_excel('./data/annotations-tk-20250626-from-results.xlsx')
    video_df = id_df.join(video_df.unique('id'), left_on='aweme_id', right_on='id', how='left', maintain_order='left')

    # Prepare dataset for classifiers
    print("\n--- Preparing Dataset for Classification Models ---")
    X = embeddings
    y = video_df.select(pl.when(pl.col('choice').is_in(['GenAI', 'Partial GenAI'])).then(1).otherwise(0).alias('label'))['label'].to_numpy()
    print(f"Dataset prepared: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Initialize all models - Removed VotingEnsembleClassifier as requested
    models = {
        "Logistic Regression": LogisticRegressionClassifier(),
        "HistGradient Boosting": HistGradientBoostingClassifier(),
        "Extra Trees": ExtraTreesClassifier(),
        "Stacking Ensemble": StackingEnsembleClassifier()
    }
    
    # Store XClip timing and F1 results for later comparison
    model_timings = {}
    
    # Perform cross-validation
    print("\n--- Performing 5-Fold Cross-Validation ---")
    cv_results = perform_cross_validation(models, X, y, cv=5)
    
    # Plot cross-validation results
    plot_cross_validation_results(cv_results)
    
    # For the main evaluation, still split 50/50 to be consistent with original code
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train and evaluate each model on the fixed split
    model_results = {}
    
    for model_name, model_instance in models.items():
        print(f"\n--- Training and Evaluating {model_name} ---")
        model_instance.train(X_train, y_train)
        
        # Measure inference time
        print(f"Measuring inference speed for {model_name}...")
        start_time = time.time()
        y_pred = model_instance.predict(X_test)
        inference_time = time.time() - start_time
        samples_per_second = len(X_test) / inference_time
        
        print(f"{model_name} inference speed: {samples_per_second:.1f} samples/second")
        print(f"{model_name} inference time: {inference_time:.4f} seconds for {len(X_test)} samples")
        
        # Get probability predictions
        y_prob = model_instance.predict_proba(X_test)
        
        # Evaluate model
        results = evaluate_model(y_test, y_pred, y_prob, X_test, model_instance)
        model_results[model_name] = results
        
        # Store timing results for later comparison
        model_timings[model_name] = {
            "f1": results["best_f1"],
            "samples_per_second": results["samples_per_second"]
        }
        
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1 Score: {results['f1']:.4f}")
        print(f"Best threshold: {results['best_threshold']:.2f} with F1 score: {results['best_f1']:.4f}")
        
        # Plot evaluation results
        plot_evaluation_results(results, save_path=f'./figs/{model_name.lower().replace(" ", "_")}')
    
    # Compare all models with XClip
    print("\n--- Comparing All Models ---")
    
    # Create comparison figure for all models
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for model_name, results in model_results.items():
        print(f"{model_name} Best F1: {results['best_f1']:.4f}")
        
        # Add model's threshold curve to plot
        ax.plot(results["thresholds"], results["f1_scores"], label=model_name, linewidth=2)
    
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1 Score')
    ax.set_title('Model Comparison: F1 Score vs Threshold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig('./figs/all_models_comparison.png')
    
    # Create a bar chart for final F1 scores
    fig, ax = plt.subplots(figsize=(12, 8))
    model_names = list(model_results.keys())
    f1_scores = [results['best_f1'] for results in model_results.values()]
    
    # Sort by F1 score
    f1_sorted_indices = np.argsort(f1_scores)[::-1]  # Sort in descending order
    f1_sorted_names = [model_names[i] for i in f1_sorted_indices]
    f1_sorted_scores = [f1_scores[i] for i in f1_sorted_indices]
    
    # Add colors - make the highest score green
    colors = ['#1f77b4'] * len(f1_sorted_scores)
    colors[0] = '#2ca02c'  # Green for the best model
    
    bars = ax.bar(f1_sorted_names, f1_sorted_scores, alpha=0.7, color=colors)
    
    ax.set_ylabel('Best F1 Score')
    ax.set_title('Best F1 Score Comparison Across Models (Ranked)')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    fig.savefig('./figs/best_f1_comparison.png')
    
    # Plot F1 Score vs Inference Speed
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Extract data from model_timings
    all_model_names = list(model_timings.keys())
    all_f1_scores = [model_timings[name]['f1'] for name in all_model_names]
    all_speeds = [model_timings[name]['samples_per_second'] for name in all_model_names]
    
    # Create scatter plot
    ax.scatter(all_speeds, all_f1_scores, s=100, alpha=0.7)
    
    # Add model names as annotations
    for i, model in enumerate(all_model_names):
        ax.annotate(model, (all_speeds[i], all_f1_scores[i]), 
                   xytext=(7, 0), textcoords='offset points',
                   fontsize=12, va='center')
    
    ax.set_xlabel('Inference Speed (samples/second)', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('F1 Score vs Inference Speed', fontsize=14)
    ax.grid(alpha=0.3)
    
    # Find and highlight Pareto-optimal models
    # Sort by speed
    speed_sorted_indices = np.argsort(all_speeds)
    speed_sorted_speeds = [all_speeds[i] for i in speed_sorted_indices]
    speed_sorted_f1s = [all_f1_scores[i] for i in speed_sorted_indices]
    speed_sorted_names = [all_model_names[i] for i in speed_sorted_indices]
    
    # Find Pareto-optimal points (non-dominated solutions: higher F1, higher speed)
    pareto_indices = []
    max_f1 = -float('inf')
    
    for i in range(len(speed_sorted_speeds)):
        if speed_sorted_f1s[i] > max_f1:
            pareto_indices.append(i)
            max_f1 = speed_sorted_f1s[i]
    
    pareto_speeds = [speed_sorted_speeds[i] for i in pareto_indices]
    pareto_f1s = [speed_sorted_f1s[i] for i in pareto_indices]
    pareto_names = [speed_sorted_names[i] for i in pareto_indices]
    
    # Highlight Pareto-optimal models
    ax.scatter(pareto_speeds, pareto_f1s, s=200, facecolors='none', edgecolors='r', linewidth=2, 
              label='Pareto-optimal Models')
    
    # Draw Pareto frontier
    ax.plot(pareto_speeds, pareto_f1s, 'r--', alpha=0.7)
    
    # Add a legend
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    fig.savefig('./figs/f1_vs_speed.png')
    
    # Print a summary table of results
    print("\n--- Summary of Model Performance ---")
    print(f"{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'Best F1':<10} {'Speed (samples/s)':<15}")
    print("-" * 90)
    
    # Add other models - use f1_sorted_names to maintain consistency
    for name in f1_sorted_names:  
        results = model_results[name]
        print(f"{name:<25} {results['accuracy']:.4f}{'':<5} {results['precision']:.4f}{'':<5} {results['recall']:.4f}{'':<5} {results['f1']:.4f}{'':<5} {results['best_f1']:.4f}{'':<5} {results['samples_per_second']:.1f}")
    
    # Identify the best model by F1 score
    best_model_name = f1_sorted_names[0]
    print(f"\nBest model for F1 score: {best_model_name} with F1 score of {model_timings[best_model_name]['f1']:.4f}")
    
    # Identify the fastest model
    speeds = [model_timings[name]['samples_per_second'] for name in all_model_names]
    fastest_model_index = np.argmax(speeds)
    fastest_model = all_model_names[fastest_model_index]
    fastest_speed = speeds[fastest_model_index]
    
    print(f"Fastest model: {fastest_model} with {fastest_speed:.1f} samples/second")
    
    # Identify Pareto-optimal models (the ones that are on the frontier)
    print("\n--- Pareto-Optimal Models (best tradeoff between F1 and speed) ---")
    for name in pareto_names:
        f1 = model_timings[name]['f1']
        speed = model_timings[name]['samples_per_second']
        print(f"{name:<25} F1: {f1:.4f}, Speed: {speed:.1f} samples/second")
    
    # Provide recommendations based on use case
    print("\n--- Recommendations Based on Use Case ---")
    print("For best accuracy: Use", f1_sorted_names[0])
    print("For fastest inference: Use", fastest_model)
    
    # Find best balanced model (product of normalized F1 and speed)
    normalized_f1 = [(f1 - min(all_f1_scores)) / (max(all_f1_scores) - min(all_f1_scores)) for f1 in all_f1_scores]
    normalized_speed = [(speed - min(all_speeds)) / (max(all_speeds) - min(all_speeds)) for speed in all_speeds]
    balanced_score = [0.7 * f1 + 0.3 * speed for f1, speed in zip(normalized_f1, normalized_speed)]
    
    best_balanced_index = np.argmax(balanced_score)
    best_balanced_model = all_model_names[best_balanced_index]
    
    print(f"For balanced performance (F1 and speed): Use {best_balanced_model}")
    
    # Save trained models
    save_trained_models(models)
    
if __name__ == '__main__':
    main()