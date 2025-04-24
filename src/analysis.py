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
    try:
        results = load_results(results_file)
        if not results:
            print(f"Warning: No data found in {results_file}")
            return {"error": "No results data found"}
    except Exception as e:
        print(f"Error loading results from {results_file}: {e}")
        return {"error": f"Failed to load results: {str(e)}"}
    
    try:
        sample_data = load_sample_data(sample_file)
    except Exception as e:
        print(f"Error loading sample data from {sample_file}: {e}")
        return {"error": f"Failed to load sample data: {str(e)}"}
    
    # Convert ground truth to numeric
    try:
        ground_truth = convert_ground_truth_to_numeric(sample_data)
    except Exception as e:
        print(f"Error converting ground truth to numeric: {e}")
        return {"error": f"Failed to convert ground truth: {str(e)}"}
    
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
        if method not in results or not results[method]:
            print(f"Warning: No data for method {method}")
            continue
            
        all_ub_metrics = []
        all_acc_metrics = []
        
        for trial_data in results[method]:
            if "results" in trial_data and trial_data["results"]:
                # Calculate uniformity bias
                try:
                    ub_metrics = calculate_uniformity_bias(trial_data["results"])
                    all_ub_metrics.append(ub_metrics)
                    analysis["uniformity_bias"][method].append(ub_metrics)
                except Exception as e:
                    print(f"Error calculating uniformity bias for {method}: {e}")
                
                # Calculate accuracy metrics
                try:
                    acc_metrics = calculate_accuracy_metrics(trial_data["results"], ground_truth)
                    all_acc_metrics.append(acc_metrics)
                    analysis["accuracy"][method].append(acc_metrics)
                except Exception as e:
                    print(f"Error calculating accuracy metrics for {method}: {e}")
        
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
    
    # Check if analysis has valid summary data
    if not analysis or "summary" not in analysis:
        print("Warning: No valid summary data found in analysis results")
        return pd.DataFrame({"method": ["No valid data available"]})
    
    for method in ["baseline_batch", "independent", "filler_token_batch"]:
        if method in analysis["summary"] and analysis["summary"][method]:
            row = {
                "method": method,
                **analysis["summary"][method]
            }
            data.append(row)
    
    if not data:
        print("Warning: No summary data available for any method")
        return pd.DataFrame({"method": ["No data available"]})
    
    df = pd.DataFrame(data)
    
    # Calculate normalized scores (0-100 where higher is better) if we have the metrics
    if "mean_adjacent_correlation" in df.columns and len(df) > 1:
        # For uniformity metrics, lower is better
        min_val = df["mean_adjacent_correlation"].min()
        max_val = df["mean_adjacent_correlation"].max()
        if min_val != max_val:  # Avoid division by zero
            df["uniformity_score"] = 100 * (1 - (df["mean_adjacent_correlation"] - min_val) / 
                                        (max_val - min_val))
    
    if "mse" in df.columns and len(df) > 1:
        # For accuracy metrics, higher is better (except MSE)
        min_val = df["mse"].min()
        max_val = df["mse"].max() 
        if min_val != max_val:  # Avoid division by zero
            df["accuracy_score"] = 100 * (1 - (df["mse"] - min_val) / 
                                    (max_val - min_val))
    
    return df

def run_significance_tests(results):
    """
    Run statistical significance tests between methods.
    
    Args:
        results: Dictionary with evaluation results
        
    Returns:
        Dictionary with p-values for key metrics
    """
    from scipy.stats import ttest_ind
    
    # Check if results is valid
    if results is None or not isinstance(results, dict):
        print("Warning: No valid results data for significance testing")
        return {}
    
    methods = ["baseline_batch", "independent", "filler_token_batch"]
    metrics = {
        "mean_adjacent_correlation": [],
        "mse": [],
        "accuracy": [],
        "correlation": []
    }
    
    # Extract metrics from all trials for each method
    for method in methods:
        if method not in results or not results[method]:
            continue
            
        method_metrics = {metric: [] for metric in metrics.keys()}
        
        for trial_data in results[method]:
            # Get uniformity bias metrics from basic trial data
            if isinstance(trial_data, dict) and "results" in trial_data and trial_data["results"]:
                # Calculate uniformity bias for this trial
                ub_metrics = calculate_uniformity_bias(trial_data["results"])
                if ub_metrics["mean_adjacent_correlation"] is not None:
                    method_metrics["mean_adjacent_correlation"].append(ub_metrics["mean_adjacent_correlation"])
                
                # Calculate accuracy metrics for this trial
                gt = convert_ground_truth_to_numeric(load_sample_data())
                acc_metrics = calculate_accuracy_metrics(trial_data["results"], gt)
                if acc_metrics["mse"] is not None:
                    method_metrics["mse"].append(acc_metrics["mse"])
                if acc_metrics["accuracy"] is not None:
                    method_metrics["accuracy"].append(acc_metrics["accuracy"])
                if acc_metrics["correlation"] is not None:
                    method_metrics["correlation"].append(acc_metrics["correlation"])
        
        # Add to metrics dict
        for metric, values in method_metrics.items():
            if values:  # Only if we have values
                metrics[metric].append((method, values))
    
    # Run t-tests between methods for each metric
    significance_results = {}
    
    for metric, method_values in metrics.items():
        if len(method_values) < 2:
            continue
            
        significance_results[metric] = {}
        
        # Run t-test for each pair of methods
        for i, (method1, values1) in enumerate(method_values):
            for method2, values2 in method_values[i+1:]:
                if len(values1) > 1 and len(values2) > 1:  # Need at least 2 values for t-test
                    try:
                        t_stat, p_value = ttest_ind(values1, values2)
                        key = f"{method1}_vs_{method2}"
                        significance_results[metric][key] = {
                            "t_statistic": float(t_stat),
                            "p_value": float(p_value),
                            "significant": bool(p_value < 0.05)
                        }
                    except Exception as e:
                        print(f"Error running t-test for {method1} vs {method2} on {metric}: {e}")
    
    return significance_results

def print_summary_table(df):
    """Print a summary table of the results."""
    print("\n===== EXPERIMENT RESULTS =====\n")
    print("Uniformity Bias Metrics (lower is better):")
    print(df[["method", "mean_adjacent_correlation", "column_variance", "unique_score_ratio"]].to_string(index=False))
    
    print("\nAccuracy Metrics (higher is better except MSE):")
    print(df[["method", "mse", "accuracy", "correlation"]].to_string(index=False))
    
    if "uniformity_score" in df.columns and "accuracy_score" in df.columns:
        print("\nNormalized Scores (0-100, higher is better):")
        print(df[["method", "uniformity_score", "accuracy_score"]].to_string(index=False))
        
def print_significance_tests(significance_results):
    """Print results of significance tests."""
    print("\n===== STATISTICAL SIGNIFICANCE TESTS =====\n")
    
    for metric, tests in significance_results.items():
        print(f"{metric}:")
        
        for comparison, results in tests.items():
            significant = "SIGNIFICANT" if results["significant"] else "not significant"
            print(f"  {comparison}: p={results['p_value']:.4f} ({significant})")
        
        print()

def main():
    """Run the analysis."""
    print("Analyzing experiment results...")
    
    # Load raw results for significance testing
    try:
        results = load_results()
        print("Loaded raw evaluation results.")
    except Exception as e:
        print(f"Warning: Could not load raw results: {e}")
        results = None
    
    # Run significance tests if possible
    if results:
        significance_results = run_significance_tests(results)
        print("Completed significance tests.")
    else:
        significance_results = {}
        print("Skipping significance tests due to missing data.")
    
    # Run regular analysis
    try:
        analysis = analyze_results()
        if "error" in analysis:
            print(f"Analysis error: {analysis['error']}")
            # Still proceed with what we have
        else:
            print("Completed analysis of results.")
    except Exception as e:
        print(f"Error in analysis: {e}")
        # Create a minimal analysis result to continue
        analysis = {
            "summary": {
                "baseline_batch": {},
                "independent": {},
                "filler_token_batch": {}
            }
        }
    
    # Add significance results to analysis
    if significance_results:
        analysis["significance_tests"] = significance_results
    
    # Save analysis results if we have meaningful data
    if analysis and "summary" in analysis and any(analysis["summary"].values()):
        try:
            save_analysis(analysis)
            print("Saved analysis results.")
        except Exception as e:
            print(f"Error saving analysis: {e}")
    else:
        print("Not saving analysis due to insufficient data.")
    
    # Create and print summary table
    try:
        df = create_summary_dataframe(analysis)
        print_summary_table(df)
    except Exception as e:
        print(f"Error creating summary table: {e}")
        df = None
    
    # Print significance test results if available
    if significance_results:
        try:
            print_significance_tests(significance_results)
        except Exception as e:
            print(f"Error printing significance tests: {e}")
    
    # Save summary table if available
    if df is not None:
        try:
            summary_file = "results/tables/summary_table.csv"
            os.makedirs(os.path.dirname(summary_file), exist_ok=True)
            df.to_csv(summary_file, index=False)
            print(f"Summary table saved to {summary_file}")
        except Exception as e:
            print(f"Error saving summary table: {e}")
    
    print("Analysis complete.")
    return 0

if __name__ == "__main__":
    exit(main())