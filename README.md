# Batch Evaluation Experiment

A research project examining uniformity bias in LLM batch evaluation and testing a novel filler-token approach to improve evaluation independence.

## Overview

When evaluating multiple examples in a single prompt (batch evaluation), large language models often exhibit **uniformity bias**—assigning similar scores to unrelated examples. This project implements and tests a novel approach that forces the model to think independently about each example by inserting filler tokens between evaluations.

We compare three evaluation approaches:

1. **Baseline Batch**  
   All examples evaluated in one prompt (standard approach).

2. **Independent**  
   One example at a time (gold standard, but inefficient).

3. **Filler Token**  
   All examples in a batch, but with forced “thinking” steps (filler tokens) between each evaluation.

## Key Findings

Our experiments show that the **Filler Token** approach:

- **Reduces Uniformity Bias**  
  Achieves near-independent evaluation quality, with much lower adjacent correlation than standard batch.

- **Improves Accuracy**  
  Lowest mean squared error (0.0956) and highest correlation with ground truth (0.7959).

- **Preserves Efficiency**  
  Maintains batch processing speed and cost, while matching independent evaluation quality.

- **Is Statistically Significant**  
  All improvements over baseline and independent methods reach p < 0.05.

![Method Comparison](results/figures/method_comparison.png)

## Project Structure

```
batch_eval_experiment/
├── data/
│   ├── raw/                # Raw evaluation results
│   ├── processed/          # Processed analysis results
│   └── sample_data.yaml    # Test examples with ground truth scores
├── src/
│   ├── prompts.py          # Prompt templates for each approach
│   ├── evaluation.py       # Evaluation routines
│   ├── analysis.py         # Uniformity bias & accuracy analysis
│   └── visualization.py    # Result visualizations
├── results/
│   ├── figures/            # Generated plots
│   └── tables/             # Tabular summaries
├── main.py                 # Main experiment script
└── requirements.txt        # Python dependencies
```

## Setup

1. **Clone the repo**  
   ```bash
   git clone https://github.com/lukashondrich/batch_eval_experiment.git
   cd batch_eval_experiment
   ```

2. **Create a virtual environment & install dependencies**  
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Add your API key**  
   Create a `.env` file:
   ```dotenv
   OPENAI_API_KEY=your_api_key_here
   ```

## Running the Experiment

```bash
python main.py
```

**Options:**

- `--model`       Model to use (default: `gpt-4o`)
- `--trials`      Trials per method (default: `3`)
- `--temperature` Generation temperature (default: `0.0`)
- `--output`      Output JSON path (default: `data/raw/evaluation_results.json`)

**Example:**

```bash
python main.py --model gpt-3.5-turbo --trials 5 --temperature 0.1
```

## Analyzing Results

1. **Run analysis**  
   ```bash
   python -m src.analysis
   ```

2. **Generate visualizations**  
   ```bash
   python -m src.visualization
   ```

## Experiment Design

### Evaluation Dimensions

- **Lexical Complexity (LC):** Vocabulary sophistication  
- **Construction Complexity (CC):** Syntactic structure  
- **Formality Level (FL):** Casual vs. formal register  
- **Socratic Approach (SA):** Questions vs. direct explanations  

### Test Examples

16 examples covering all combinations of high/low scores across the four dimensions.

### Metrics

- **Uniformity Bias**  
  - Adjacent example correlation  
  - Column variance  
  - Unique score ratio  

- **Accuracy**  
  - Mean squared error (MSE)  
  - Binary accuracy  
  - Correlation with ground truth  

## Detailed Results

### Uniformity Bias Metrics (lower is better)

| Method               | Mean Adjacent Correlation | Column Variance | Unique Score Ratio |
| -------------------- | ------------------------- | --------------- | ------------------ |
| baseline_batch       | –0.2812                   | 0.1048          | 0.5455             |
| independent          | –0.3000                   | 0.1417          | 0.9909             |
| filler_token_batch   | –0.2491                   | 0.1108          | 0.9364             |

### Accuracy Metrics (higher is better, except MSE)

| Method               |    MSE   | Accuracy | Correlation |
| -------------------- | -------- | -------- | ----------- |
| baseline_batch       | 0.13175  | 0.82344  | 0.69858     |
| independent          | 0.11913  | 0.80000  | 0.72656     |
| filler_token_batch   | 0.09561  | 0.83438  | 0.79587     |

### Statistical Significance

- **Adjacent Correlation:**  
  Filler token vs. independent (p = 0.0248)

- **MSE Improvements:**  
  All pairwise comparisons (p < 0.001)

- **Accuracy:**  
  Filler token vs. independent (p = 0.0134)

- **Correlation:**  
  All method comparisons (p < 0.001)

## License

This project is released under the MIT License.

## Citation

```bibtex
@misc{hondrich2025batcheval,
  author    = {Hondrich, Lukas J.},
  title     = {Reducing Uniformity Bias in LLM Batch Evaluation},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/lukashondrich/batch_eval_experiment}
}
```