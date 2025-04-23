"""
Analysis of batch evaluation experiment results.
Includes metrics for uniformity bias and accuracy.
"""
import json
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from src.evaluation import load_sample_data

def load_results(file_path="data/raw/evaluation_results.json"):
    """Load evaluation results from JSON file."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def convert_ground_truth_to_numeric(sample_data):
    """Convert ground truth scores from text to numeric (0.0 or 1.0)."""
    numeric_gt = {}
    
    for sample in sample_data["samples"]:
        sample_id = str(sample["id"])
        numeric_scores = []
        
        for dim in ["LC", "CC", "FL", "SA"]:
            if sample["scores"][dim] == 0:
                numeric_scores.append(0.0)
            else:  # "High"
                numeric_scores.append(1.0)
        
        numeric_gt[sample_id] = numeric_scores
    
    return numeric_gt

def calculate_uniformity_bias(scores):
    """
    Calculate uniformity bias in evaluation scores.
    Higher values indicate more uniformity (greater bias).
    
    Args:
        scores: Dictionary mapping example IDs to score arrays
        
    Returns:
        Dictionary with uniformity metrics
    """
    # Extract scores into a numpy array
    score_arrays = []
    for example_id in sorted(scores.keys(), key=int):
        if isinstance(scores[example_id], list) and len(scores[example_id]) == 4:
            score_arrays.append(scores[example_id])
    
    if not score_arrays:
        return {
            "mean_adjacent_correlation": None,
            "column_variance": None,
            "unique_score_ratio": None
        }
        
    score_matrix = np.array(score_arrays)
    
    # Calculate adjacent example correlation (average correlation between rows)
    n_examples = score_matrix.shape[0]
    adjacent_correlations = []
    
    for i in range(n_examples - 1):
        for j in range(i + 1, min(i + 3, n_examples)):  # Look at 2 adjacent examples
            if np.std(score_matrix[i]) > 0 and np.std(score_matrix[j]) > 0:
                corr, _ = pearsonr(score_matrix[i], score_matrix[j])
                adjacent_correlations.append(corr)
    
    mean_adjacent_correlation = np.mean(adjacent_correlations) if adjacent_correlations else None
    
    # Calculate average column variance (how much scores vary within a dimension)
    column_variance = np.mean(np.var(score_matrix, axis=0))
    
    # Calculate unique score ratio (measure of diversity)
    unique_scores = len(np.unique(score_matrix.round(1)))
    max_possible_unique = min(score_matrix.size, 11)  # Max unique scores (0.0-1.0 in 0.1 increments)
    unique_score_ratio = unique_scores / max_possible_unique
    
    return {
        "mean_adjacent_correlation": mean_adjacent_correlation,
        "column_variance": column_variance,
        "unique_score_ratio": unique_score_ratio
    }

def calculate_accuracy_metrics(predicted_scores, ground_truth_scores):
    """
    Calculate accuracy metrics compared to ground truth.
    
    Args:
        predicted_scores: Dictionary mapping example IDs to score arrays
        ground_truth_scores: Dictionary mapping example IDs to ground truth score arrays
        
    Returns:
        Dictionary with accuracy metrics
    """
    all_pred = []
    all_truth = []
    
    for example_id in sorted(ground_truth_scores.keys(), key=int):
        if example_id in predicted_scores and isinstance(predicted_scores[example_id], list):
            all_pred.extend(predicted_scores[example_id])
            all_truth.extend(ground_truth_scores[example_id])
    
    if not all_pred:
        return {
            "mse": None,
            "accuracy": None,
            "correlation": None
        }
    
    # Calculate mean squared error
    mse = np.mean([(p - t)**2 for p, t in zip(all_pred, all_truth)])
    
    # Calculate binary accuracy (rounding to nearest 0 or 1)
    binary_pred = [round(p) for p in all_pred]
    binary_truth = [round(t) for t in all_truth]
    accuracy = sum(p == t for p, t in zip(binary_pred, binary_truth)) / len(binary_truth)
    
    # Calculate correlation
    correlation, _ = pearsonr(all_pred, all_truth)
    
    return {
        "mse": mse,
        "accuracy": accuracy,
        "correlation": correlation
    }

def analyze_results(results_file="data/raw/evaluation_results.json", sample_file="data/sample_data.yaml"):
    """
    Analyze results from all evaluation approaches.
    
    Args:
        results_file: Path to results JSON file
        sample_file: Path to sample data YAML file
        
    Returns:
        Dictionary with analysis results
    """
    # Load data
    results = load_results(results_file)
    sample_data = load_sample_data(sample_file)
    
    # Convert ground truth to numeric
    ground_truth = convert_ground_truth_to_numeric(sample_data)
    
    # Initialize analysis results
    analysis = {
        "uniformity_bias": {
            "baseline_batch": [],
            "independent": [],
            "filler_token_batch": []
        },
        "accuracy": {
            "baseline_batch": [],
            "independent": [],
            "filler_token_batch": []
        },
        "summary": {
            "baseline_batch": {},
            "independent": {},
            "filler_token_batch": {}
        }
    }
    
    # Analyze each method across trials
    for method in ["baseline_batch", "independent", "filler_token_batch"]:
        all_ub_metrics = []
        all_acc_metrics = []
        
        for trial_data in results[method]:
            if "results" in trial_data and trial_data["results"]:
                # Calculate uniformity bias
                ub_metrics = calculate_uniformity_bias(trial_data["results"])
                all_ub_metrics.append(ub_metrics)
                analysis["uniformity_bias"][method].append(ub_metrics)
                
                # Calculate accuracy metrics
                acc_metrics = calculate_accuracy_metrics(trial_data["results"], ground_truth)
                all_acc_metrics.append(acc_metrics)
                analysis["accuracy"][method].append(acc_metrics)
        
        # Calculate summary statistics
        if all_ub_metrics and all(m["mean_adjacent_correlation"] is not None for m in all_ub_metrics):
            analysis["summary"][method]["mean_adjacent_correlation"] = np.mean([m["mean_adjacent_correlation"] for m in all_ub_metrics])
            analysis["summary"][method]["column_variance"] = np.mean([m["column_variance"] for m in all_ub_metrics])
            analysis["summary"][method]["unique_score_ratio"] = np.mean([m["unique_score_ratio"] for m in all_ub_metrics])
        
        if all_acc_metrics and all(m["mse"] is not None for m in all_acc_metrics):
            analysis["summary"][method]["mse"] = np.mean([m["mse"] for m in all_acc_metrics])
            analysis["summary"][method]["accuracy"] = np.mean([m["accuracy"] for m in all_acc_metrics])
            analysis["summary"][method]["correlation"] = np.mean([m["correlation"] for m in all_acc_metrics])
    
    return analysis

def save_analysis(analysis, filename="data/processed/analysis_results.json"):
    """Save analysis results to a JSON file."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Convert numpy values to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        else:
            return obj
    
    analysis_serializable = convert_numpy(analysis)
    
    with open(filename, 'w') as f:
        json.dump(analysis_serializable, f, indent=2)
    
    print(f"Analysis results saved to {filename}")
    return filename

def create_summary_dataframe(analysis):
    """Create a summary DataFrame for easy comparison of methods."""
    data = []
    
    for method in ["baseline_batch", "independent", "filler_token_batch"]:
        if method in analysis["summary"]:
            row = {
                "method": method,
                **analysis["summary"][method]
            }
            data.append(row)
    
    df = pd.DataFrame(data)
    
    # Calculate normalized scores (0-100 where higher is better)
    if "mean_adjacent_correlation" in df.columns:
        # For uniformity metrics, lower is better
        df["uniformity_score"] = 100 * (1 - (df["mean_adjacent_correlation"] - df["mean_adjacent_correlation"].min()) / 
                                     (df["mean_adjacent_correlation"].max() - df["mean_adjacent_correlation"].min() + 1e-10))
    
    if "mse" in df.columns:
        # For accuracy metrics, higher is better (except MSE)
        df["accuracy_score"] = 100 * (1 - (df["mse"] - df["mse"].min()) / 
                                   (df["mse"].max() - df["mse"].min() + 1e-10))
    
    # Overall score combining uniformity and accuracy
    if "uniformity_score" in df.columns and "accuracy_score" in df.columns:
        df["overall_score"] = (df["uniformity_score"] + df["accuracy_score"]) / 2
    
    return df

def print_summary_table(df):
    """Print a summary table of the results."""
    print("\n===== EXPERIMENT RESULTS =====\n")
    print("Uniformity Bias Metrics (lower is better):")
    print(df[["method", "mean_adjacent_correlation", "column_variance", "unique_score_ratio"]].to_string(index=False))
    
    print("\nAccuracy Metrics (higher is better except MSE):")
    print(df[["method", "mse", "accuracy", "correlation"]].to_string(index=False))
    
    if "overall_score" in df.columns:
        print("\nNormalized Scores (0-100, higher is better):")
        print(df[["method", "uniformity_score", "accuracy_score", "overall_score"]].to_string(index=False))
    
    # Print the best method
    if "overall_score" in df.columns:
        best_method = df.loc[df["overall_score"].idxmax(), "method"]
        print(f"\nBest method: {best_method}")

def main():
    """Run the analysis."""
    print("Analyzing experiment results...")
    analysis = analyze_results()
    
    # Save analysis results
    save_analysis(analysis)
    
    # Create and print summary table
    df = create_summary_dataframe(analysis)
    print_summary_table(df)
    
    # Save summary table
    summary_file = "results/tables/summary_table.csv"
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    df.to_csv(summary_file, index=False)
    print(f"Summary table saved to {summary_file}")
    
    return 0

if __name__ == "__main__":
    exit(main())