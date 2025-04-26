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