"""
Visualization of batch evaluation experiment results.
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.evaluation import load_sample_data
from src.analysis import load_results, convert_ground_truth_to_numeric

# Define consistent colors for methods
METHOD_COLORS = {
    "Baseline Batch": "#1f77b4",    # Blue
    "Independent": "#ff7f0e",       # Orange
    "Explicit Reasoning": "#2ca02c", # Green
    "Ground Truth": "#d62728"       # Red
}

def method_display_name(method_name):
    """Return display-friendly method names."""
    mapping = {
        "baseline_batch": "Baseline Batch",
        "independent": "Independent",
        "filler_token_batch": "Explicit Reasoning" 
    }
    return mapping.get(method_name, method_name)

def load_analysis(file_path="data/processed/analysis_results.json"):
    """Load analysis results from JSON file."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def create_heatmap_data(results, method, trial_idx=0):
    """
    Create data for score heatmap.
    
    Args:
        results: Dictionary with evaluation results
        method: Evaluation method name
        trial_idx: Index of trial to visualize
        
    Returns:
        DataFrame suitable for heatmap visualization
    """
    if method not in results or not results[method] or trial_idx >= len(results[method]):
        return None
    
    trial_data = results[method][trial_idx]
    if "results" not in trial_data or not trial_data["results"]:
        return None
    
    # Extract scores
    scores = {}
    for example_id in sorted(trial_data["results"].keys(), key=int):
        if isinstance(trial_data["results"][example_id], list) and len(trial_data["results"][example_id]) == 4:
            scores[example_id] = trial_data["results"][example_id]
    
    # Create DataFrame
    df = pd.DataFrame.from_dict(scores, orient='index', columns=["LC", "CC", "FL", "SA"])
    df.index.name = "Example"
    
    return df

def plot_score_heatmap(df, title, filename):
    """
    Plot a heatmap of evaluation scores.
    
    Args:
        df: DataFrame with scores
        title: Plot title
        filename: Output file path
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, cmap="coolwarm", vmin=0, vmax=1, linewidths=.5)
    plt.title(title)
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()
    
    print(f"Heatmap saved to {filename}")

def plot_method_comparison(analysis, filename="results/figures/method_comparison.png"):
    """
    Plot comparison of evaluation methods.
    
    Args:
        analysis: Dictionary with analysis results
        filename: Output file path
    """
    # Create summary data
    data = []
    metrics = {
        "mean_adjacent_correlation": "Adjacent\nCorrelation\n(lower better)",
        "column_variance": "Column\nVariance\n(higher better)",
        "unique_score_ratio": "Unique Score\nRatio\n(higher better)",
        "mse": "Mean Squared\nError\n(lower better)",
        "accuracy": "Binary\nAccuracy\n(higher better)",
        "correlation": "Correlation with\nGround Truth\n(higher better)"
    }
    
    for method in ["baseline_batch", "independent", "filler_token_batch"]:
        if method in analysis["summary"]:
            for metric, label in metrics.items():
                if metric in analysis["summary"][method]:
                    data.append({
                        "Method": method_display_name(method),
                        "Metric": label,
                        "Value": analysis["summary"][method][metric]
                    })
    
    if not data:
        print("No data available for method comparison plot")
        return
    
    df = pd.DataFrame(data)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plotting each metric in its own subplot
    for i, (metric, label) in enumerate(metrics.items()):
        values = df[df["Metric"] == label]
        if len(values) > 0:
            ax = plt.subplot(2, 3, i+1)
            
            # Sort values based on metric direction
            if "lower better" in label:
                values = values.sort_values("Value")
            else:
                values = values.sort_values("Value", ascending=False)
            
            # Create color map based on method names
            colors = [METHOD_COLORS[method] for method in values["Method"]]
            
            # Plot the bars
            bars = sns.barplot(x="Method", y="Value", data=values, palette=colors, ax=ax)
            ax.set_title(label)
            ax.set_ylabel("")
            ax.set_xlabel("")
            
            # Add value labels
            for j, bar in enumerate(bars.patches):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height * 0.5,
                    f"{values.iloc[j]['Value']:.4f}",
                    ha='center',
                    va='center',
                    color='white',
                    fontweight='bold'
                )
    
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()
    
    print(f"Method comparison plot saved to {filename}")

def plot_score_distribution(results, ground_truth, filename="results/figures/score_distribution.png"):
    """
    Plot distribution of scores across methods.
    
    Args:
        results: Dictionary with evaluation results
        ground_truth: Dictionary with ground truth scores
        filename: Output file path
    """
    # Collect all scores
    all_scores = []
    
    methods = ["baseline_batch", "independent", "filler_token_batch"]
    
    for i, method in enumerate(methods):
        if method in results and results[method]:
            for trial_idx, trial_data in enumerate(results[method]):
                if "results" in trial_data and trial_data["results"]:
                    for example_id, scores in trial_data["results"].items():
                        if isinstance(scores, list) and len(scores) == 4:
                            for dim_idx, score in enumerate(scores):
                                dimension = ["LC", "CC", "FL", "SA"][dim_idx]
                                all_scores.append({
                                    "Method": method_display_name(method),
                                    "Dimension": dimension,
                                    "Example": example_id,
                                    "Score": score,
                                    "Trial": trial_idx + 1
                                })
    
    # Add ground truth
    for example_id, scores in ground_truth.items():
        for dim_idx, score in enumerate(scores):
            dimension = ["LC", "CC", "FL", "SA"][dim_idx]
            all_scores.append({
                "Method": "Ground Truth",
                "Dimension": dimension,
                "Example": example_id,
                "Score": score,
                "Trial": 0
            })
    
    if not all_scores:
        print("No data available for score distribution plot")
        return
    
    df = pd.DataFrame(all_scores)
    
    # Create plot
    plt.figure(figsize=(15, 10))
    
    # Plot score distribution for each dimension
    dimensions = ["LC", "CC", "FL", "SA"]
    dimension_names = ["Lexical Complexity", "Construction Complexity", "Formality Level", "Socratic Approach"]
    
    # Create custom color palette that matches our METHOD_COLORS
    methods_in_data = df["Method"].unique()
    custom_palette = {method: METHOD_COLORS.get(method, "#777777") for method in methods_in_data}
    
    for i, (dim, name) in enumerate(zip(dimensions, dimension_names)):
        ax = plt.subplot(2, 2, i+1)
        
        # Violin plot with consistent colors
        sns.violinplot(
            x="Method", 
            y="Score", 
            data=df[df["Dimension"] == dim], 
            palette=custom_palette,
            ax=ax, 
            inner="box"
        )
        
        ax.set_title(name)
        ax.set_ylim(-0.1, 1.1)
        ax.set_ylabel("Score (0-1)")
        
        if i >= 2:  # Only show x labels for bottom row
            ax.set_xlabel("Method")
        else:
            ax.set_xlabel("")
    
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()
    
    print(f"Score distribution plot saved to {filename}")

def main():
    """Create visualizations for the experiment results."""
    print("Creating visualizations...")
    
    # Load data
    results = load_results()
    analysis = load_analysis()
    sample_data = load_sample_data()
    ground_truth = convert_ground_truth_to_numeric(sample_data)
    
    # Create ground truth heatmap for reference
    gt_df = pd.DataFrame.from_dict(ground_truth, orient='index', columns=["LC", "CC", "FL", "SA"])
    gt_df.index.name = "Example"
    plot_score_heatmap(gt_df, "Ground Truth Scores", "results/figures/ground_truth_heatmap.png")
    
    # Create method comparison plot - most important visualization
    plot_method_comparison(analysis)
    
    # Create score distribution plot - shows distribution across dimensions
    plot_score_distribution(results, ground_truth)
    
    print("Visualizations complete.")
    return 0

if __name__ == "__main__":
    exit(main())