# Batch Evaluation Experiment

A research project examining uniformity bias in LLM batch evaluation and testing a novel filler token approach to improve evaluation independence.

## Overview

When evaluating multiple examples in a single prompt (batch evaluation), large language models often exhibit "uniformity bias" - assigning similar scores to unrelated examples. This project implements and tests a novel approach that forces the model to think independently about each example by inserting filler tokens between evaluations.

The experiment compares three evaluation approaches:
1. **Baseline Batch**: All examples evaluated in a single prompt (standard approach)
2. **Independent**: One example at a time (gold standard, but inefficient)
3. **Filler Token**: All examples in a batch, but with forced thinking steps between evaluations

## Project Structure

```
├── data/
│   ├── raw/                  # Raw evaluation results
│   ├── processed/            # Processed analysis results
│   └── sample_data.yaml      # Test examples with ground truth scores
├── src/
│   ├── prompts.py            # Prompt templates for different approaches
│   ├── evaluation.py         # Functions to run evaluations
│   ├── analysis.py           # Analysis of uniformity bias and accuracy
│   └── visualization.py      # Visualization of results
├── results/
│   ├── figures/              # Generated visualizations
│   └── tables/               # Tabular results
├── main.py                   # Main experiment script
└── requirements.txt          # Dependencies
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/lukashondrich/batch_eval_experiment.git
cd batch_eval_experiment
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Running the Experiment

Run the complete experiment:
```bash
python main.py
```

Options:
- `--model`: Model to use (default: gpt-4o)
- `--trials`: Number of trials per method (default: 3)
- `--temperature`: Generation temperature (default: 0.0)
- `--output`: Output file path (default: data/raw/evaluation_results.json)

Example with custom settings:
```bash
python main.py --model gpt-3.5-turbo --trials 5 --temperature 0.1
```

## Analyzing Results

After running the experiment, analyze the results:
```bash
python -m src.analysis
```

Generate visualizations:
```bash
python -m src.visualization
```

## Experiment Design

### Evaluation Dimensions

The experiment evaluates language tutoring messages on four dimensions:
1. **Lexical Complexity** (LC): Vocabulary sophistication
2. **Construction Complexity** (CC): Syntactic structure
3. **Formality Level** (FL): Casual vs. formal register
4. **Socratic Approach** (SA): Questions vs. direct explanations

### Test Examples

The test set includes 16 examples carefully designed to cover all possible combinations of high/low scores across the four dimensions.

### Metrics

The experiment measures:
1. **Uniformity Bias**:
   - Adjacent example correlation
   - Column variance
   - Unique score ratio

2. **Accuracy**:
   - Mean squared error
   - Binary accuracy
   - Correlation with ground truth

## Results


## License

MIT

## Citation

If you use this code or methodology in your research, please cite:
```
@misc{hondrich2025batcheval,
  author = {Hondrich, Lukas J.},
  title = {Reducing Uniformity Bias in LLM Batch Evaluation},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/lukashondrich/batch_eval_experiment}
}
```