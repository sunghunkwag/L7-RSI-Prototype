# L7-RSI: Lightweight Recursive Self-Improvement Prototype

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sunghunkwag/L7-RSI-Prototype/blob/master/L7_RSI_Colab.ipynb)

This repository contains a single-file implementation of a Recursive Self-Improvement (RSI) system based on Genetic Programming (GP).

## ⚠️ Disclaimer
This is an **experimental research prototype**. It demonstrates the *concept* of an AI system modifying its own learning parameters and code structure. It is **not** an AGI and requires significant time to converge on complex algorithms.

## Overview
L7-RSI is a Python script that combines:
1.  **Pure Atomic Genetic Programming**: Synthesizes Python code line-by-line (loops, conditionals, assignments) without pre-defined templates.
2.  **Meta-Learning (L1 Policy)**: Dynamically adjusts mutation rates and search budgets based on performance feedback.
3.  **Transfer Learning (Memory)**: Archives successful code snippets ("shards") to reuse in future generations.
4.  **Pareto Optimization**: Uses Multi-Objective optimization (Performance vs. Stability vs. Complexity) to select survivors.

## Features
- **Zero External Dependencies**: Runs with standard Python libraries + `numpy`.
- **Self-Contained**: Logic for synthesized code execution, safety sandboxing, and evolution is all in `L7_RSI_FINAL.py`.
- **Safety**: Uses `sys.settrace` to prevent infinite loops in generated code.

## Architecture
- **L0 (Worker)**: Generates and evaluates candidate code (e.g., sorting algorithms, symbolic regression).
- **L1 (Improver)**: Tunes the hyperparameters of L0 (e.g., probability of inserting a `while` loop).
- **L2 (Evaluator)**: Manages test cases and fitness functions.

> [!IMPORTANT]
> **BETA TESTING PHASE (V14)**
> This version introduces **Robust Paired Evaluation** and **Sequential Acceptance Gates** to eliminate "lucky" code injections. It is currently undergoing rigorous stabilization testing.

## 🚀 Latest Updates (v14.0 - Robustness Overhaul)
**Key Upgrades Applied (2025-12-25):**

1.  **Paired Evaluation Protocol**: Candidates are now evaluated against the incumbent on the *exact same* data points (Common Random Numbers) to drastically reduce variance.
2.  **Sequential Acceptance Gate**: Candidates must pass a 4-stage gauntlet:
    - **Suite**: High-confidence improvement on a deterministic test suite.
    - **CI (Confidence Interval)**: 95% Lower Confidence Bound > 0.
    - **OOD (Out-of-Distribution)**: Must not regress on unseen tasks.
    - **Adversarial**: Must withstand replay of historical hard inputs.
3.  **Meta-Meta Strategy V2**: Stagnation triggers a structured "Panic Mode" that intelligently relaxes complexity penalties before boosting mutation.
4.  **Target Function Standardization**: Centralized ground-truth definition (`hybrid_hard`, `sinexp`) prevents task drift.
5.  **Fixes**: Solved `KnowledgeArchive` usage counting bug and `run_seed_process` signature mismatch.

## Results (Validated)
- **Symbolic Regression**: Can rediscover polynomial and trigonometric functions.
- **Sorting**: Can evolve logic that partially sorts arrays (e.g., shifting elements mechanisms) from scratch.
- **Self-Adaptation**: Observed to increase mutation rates during stagnation and decrease them during convergence.

## Usage

### 1. Installation
Requires Python 3.8+ and NumPy.

```bash
pip install numpy
```

### 2. Run
```bash
python L7_RSI_FINAL.py --meta_rounds 1000 --workers 4
```
- Outputs will be saved to `runs/` directory (JSON logs) and `discovered_algorithms/` (synthesized Python files).

## License
MIT License
