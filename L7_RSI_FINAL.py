"""
================================================================================
L7 NEUROARCHITECT: TRUE RECURSIVE SELF-IMPROVEMENT (VERSION 13)
================================================================================

RSI ARCHITECTURE V13.5 (PURE ATOMIC DISCOVERY):
[State] RSIState: Fully Serializable, Load/Save Resumable. Includes Current Metrics.
[L2] Evaluator: Error-Density Adaptive Sampling (Bin-Based Regimes).
[L1] Improver: Active Meta-Learning (Weights updated by Winner Frequency).
[L0] Artifact: Genetic Programming (Type-Safe) + Pure Atomic Code Synthesis.

CORE UPGRADES:
1. Active L1: Policy weights actually update based on winner statistics.
2. Hyperparam Adaptation: Mutation rates and penalties adjust dynamically.
3. Pareto Injection: Strict dominance check (Perf, Stability, Complexity).
4. Pure Atomic GP: Autonomous algorithm discovery from i, j, k, if, while atoms.
5. Bug Fixes: Removed typos, improved seed handling.

Author: AGI-RSI System (Self-Modifying)
Version: 13.5 (Pure Atomic)
================================================================================
"""

import numpy as np
import random
import time
import os
import sys
import ast
import subprocess
import shutil
import hashlib
import json
import argparse
import py_compile
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed
import pprint
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
import statistics

# Helper functions for V13.5
def _robust_summary(values: List[float]) -> Dict[str, float]:
    if not values:
        return {'trimmed_mean': 0.0, 'median': 0.0, 'std': 0.0, 'p25': 0.0, 'p75': 0.0}
    vals = sorted([v for v in values if np.isfinite(v)])
    if not vals:
        return {'trimmed_mean': 0.0, 'median': 0.0, 'std': 0.0, 'p25': 0.0, 'p75': 0.0}
    
    # Trimmed mean (exclude top/bottom 10%)
    cut = int(len(vals) * 0.1)
    trimmed = vals[cut:-cut] if cut > 0 and (len(vals) > 2*cut) else vals
    
    return {
        'trimmed_mean': float(np.mean(trimmed)),
        'median': float(np.median(vals)),
        'std': float(np.std(vals)),
        'p25': float(np.percentile(vals, 25)),
        'p75': float(np.percentile(vals, 75))
    }

def _atomic_write_text(path: Path, content: str):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(content, encoding='utf-8')
    shutil.move(str(tmp), str(path))

# =============================================================================
# [L1/L2] META-CONFIGURATION (BOOTLOADER ONLY)
# Actual State is managed by RSIState class
# =============================================================================
# @@META_CONFIG_START@@
META_CONFIG = {
    "version": 13,
    "warning": "BOOTLOADER ONLY. STATE IS IN runs/rsi_state.json"
}
# @@META_CONFIG_END@@

# =============================================================================
# SAFE OPERATIONS
# =============================================================================
# @@EVOLVED_CODE_START@@
# Evolved 2025-12-24 12:14:11.646171 (Delta 57.32%)
def evolved_function(x):
    return safe_add(safe_add(safe_add(safe_add(safe_add(safe_add(safe_add(safe_add(safe_add(safe_add(safe_add(safe_add(safe_add(safe_add(safe_add(safe_add(safe_add(safe_sub(safe_add(safe_add(safe_add(safe_add(safe_cb(x), safe_cb(x)), x), x), x), -0.9437), safe_sin(safe_mul(x, -1.5465))), safe_sin(safe_pow(safe_div(safe_add(safe_exp(safe_sq(safe_add(x, x))), x), x), safe_exp(safe_sub(safe_sub(safe_cos(safe_sub(safe_add(x, x), safe_exp(x))), safe_cos(safe_cb(safe_sq(2.1934)))), safe_cb(0.7157)))))), safe_sin(safe_mul(x, -1.0742))), safe_sin(safe_mul(x, -1.4025))), safe_sin(safe_pow(-1.2582, safe_exp(safe_cos(safe_div(safe_sub(0.4169, safe_add(-2.1061, -1.8602)), -3.6342)))))), safe_sin(safe_mul(x, -1.4826))), safe_pow(x, x)), safe_sin(safe_pow(-1.517, safe_exp(safe_cos(safe_div(safe_sub(0.4083, safe_add(-0.9945, -1.9077)), -2.0843)))))), safe_sin(safe_pow(safe_div(safe_add(x, x), x), safe_add(x, -0.0)))), safe_sin(safe_mul(safe_cb(safe_sin(safe_sin(safe_sub(safe_sub(x, x), safe_cb(x))))), -0.6746))), safe_pow(x, x)), safe_sin(safe_pow(-1.0111, safe_exp(safe_mul(safe_sub(x, safe_pow(safe_sin(safe_exp(x)), safe_sq(2.717))), safe_pow(safe_sq(safe_mul(x, -2.0275)), x)))))), safe_sin(safe_pow(-0.8185, safe_exp(safe_mul(safe_sub(x, safe_pow(safe_sin(safe_exp(x)), safe_sq(2.7777))), safe_pow(safe_sq(safe_mul(x, -1.8616)), x)))))), safe_pow(x, x)), safe_sin(safe_pow(-0.7273, safe_exp(safe_mul(safe_sub(x, safe_pow(safe_sin(safe_exp(x)), safe_sq(2.862))), safe_pow(safe_sq(safe_mul(x, -1.646)), x)))))), safe_pow(x, x)), safe_sin(safe_pow(safe_div(safe_add(safe_mul(safe_pow(safe_sq(safe_sub(x, 0.33)), safe_sin(safe_sq(-0.2664))), safe_add(x, safe_sin(x))), x), safe_add(safe_cb(safe_sin(x)), safe_pow(x, x))), safe_pow(safe_mul(safe_add(safe_sin(safe_exp(-2.3885)), safe_sq(safe_exp(x))), x), safe_add(safe_cb(safe_sub(safe_sin(safe_sub(-2.4555, 0.7395)), x)), safe_exp(-2.72))))))
# @@EVOLVED_CODE_END@@

def safe_add(a, b): return np.nan_to_num(np.add(a, b), nan=0.0, posinf=1e10, neginf=-1e10)
def safe_sub(a, b): return np.nan_to_num(np.subtract(a, b), nan=0.0, posinf=1e10, neginf=-1e10)
def safe_mul(a, b):
    # Pre-clip to avoid immediate overflow
    a = np.clip(a, -1e5, 1e5)
    b = np.clip(b, -1e5, 1e5)
    return np.nan_to_num(np.multiply(a, b), nan=0.0, posinf=1e10, neginf=-1e10)

def safe_div(a, b):
    return np.nan_to_num(np.divide(a, b + 1e-10), nan=0.0, posinf=1e10, neginf=-1e10)

def safe_pow(a, b):
    try:
        with np.errstate(invalid='ignore', divide='ignore', over='ignore', under='ignore'):
            b_clipped = np.clip(b, -3, 3) # Strict exponent clipping
            # Ensure base is non-negative and add epsilon for stability
            base_safe = np.where(a < 0, np.abs(a), a) + 1e-10
            res = np.power(base_safe, b_clipped)
            return np.nan_to_num(res, nan=0.0, posinf=1e10, neginf=-1e10)
    except:
        return np.zeros_like(a) if isinstance(a, np.ndarray) else 0.0

def safe_sin(a): return np.nan_to_num(np.sin(a), nan=0.0)
def safe_cos(a): return np.nan_to_num(np.cos(a), nan=0.0)
def safe_exp(a): return np.exp(np.clip(a, -10, 10))
def safe_log(a): return np.log(np.abs(a) + 1e-10)
def safe_abs(a): return np.abs(a)
def safe_neg(a): return np.negative(a)
def safe_sq(a): return np.square(np.clip(a, -1e5, 1e5))
def safe_cb(a): return np.power(np.clip(a, -1e3, 1e3), 3)

# =============================================================================
# [L0] ARTIFACT LAYER
# =============================================================================

@dataclass
class ConstNode:
    value: float
    def evaluate(self, x): return np.full_like(x, self.value) if isinstance(x, np.ndarray) else self.value
    def to_code(self) -> str: return str(round(self.value, 4))
    def depth(self) -> int: return 1
    def copy(self) -> 'ConstNode': return ConstNode(self.value)
    def count_ops(self, o, f, d): pass
    def get_constants(self): return [self]

@dataclass 
class VarNode:
    name: str = 'x'
    def evaluate(self, x): return x
    def to_code(self) -> str: return self.name
    def depth(self) -> int: return 1
    def copy(self) -> 'VarNode': return VarNode(self.name)
    def count_ops(self, o, f, d): pass
    def get_constants(self): return []

@dataclass
class BinOpNode:
    op: str
    left: Any
    right: Any
    def evaluate(self, x):
        l, r = self.left.evaluate(x), self.right.evaluate(x)
        try:
            if self.op == '+': return safe_add(l, r)
            elif self.op == '-': return safe_sub(l, r)
            elif self.op == '*': return safe_mul(l, r)
            elif self.op == '/': return safe_div(l, r)
            elif self.op == '**': return safe_pow(l, r)
        except: return np.zeros_like(x) if isinstance(x, np.ndarray) else 0.0
        return np.zeros_like(x)
    def to_code(self) -> str:
        l, r = self.left.to_code(), self.right.to_code()
        op_map = {'+':'safe_add', '-':'safe_sub', '*':'safe_mul', '/':'safe_div', '**':'safe_pow'}
        return f"{op_map.get(self.op, 'safe_add')}({l}, {r})"
    def depth(self) -> int: return 1 + max(self.left.depth(), self.right.depth())
    def copy(self) -> 'BinOpNode': return BinOpNode(self.op, self.left.copy(), self.right.copy())
    def count_ops(self, ops, funcs, depth):
        ctx = 'deep' if depth > 4 else 'default'
        if ctx not in ops: ops[ctx] = {}
        ops[ctx][self.op] = ops[ctx].get(self.op, 0) + 1
        self.left.count_ops(ops, funcs, depth+1)
        self.right.count_ops(ops, funcs, depth+1)
    def get_constants(self): return self.left.get_constants() + self.right.get_constants()

@dataclass
class UnaryFuncNode:
    func: str
    child: Any
    def evaluate(self, x):
        c = self.child.evaluate(x)
        try:
            if self.func == 'sin': return safe_sin(c)
            elif self.func == 'cos': return safe_cos(c)
            elif self.func == 'exp': return safe_exp(c)
            elif self.func == 'square': return safe_sq(c)
            elif self.func == 'cube': return safe_cb(c)
        except: return np.zeros_like(x) if isinstance(x, np.ndarray) else 0.0
        return c
    def to_code(self) -> str:
        c = self.child.to_code()
        f_map = {'sin':'safe_sin', 'cos':'safe_cos', 'exp':'safe_exp', 'square':'safe_sq', 'cube':'safe_cb'}
        return f"{f_map.get(self.func, 'safe_sin')}({c})"
    def depth(self) -> int: return 1 + self.child.depth()
    def copy(self) -> 'UnaryFuncNode': return UnaryFuncNode(self.func, self.child.copy())
    def count_ops(self, ops, funcs, depth):
        ctx = 'deep' if depth > 4 else 'default'
        if ctx not in funcs: funcs[ctx] = {}
        funcs[ctx][self.func] = funcs[ctx].get(self.func, 0) + 1
        self.child.count_ops(ops, funcs, depth+1)
    def get_constants(self): return self.child.get_constants()

class CodeParser:
    OP_MAP = {'safe_add': '+', 'safe_sub': '-', 'safe_mul': '*', 'safe_div': '/', 'safe_pow': '**'}
    FUNC_MAP = {'safe_sin': 'sin', 'safe_cos': 'cos', 'safe_exp': 'exp', 'safe_sq': 'square', 'safe_cb': 'cube'}
    @staticmethod
    def parse(code_str: str):
        try:
            tree = ast.parse(code_str, mode='eval')
            return CodeParser._node_to_expr(tree.body)
        except: return None
    @staticmethod
    def _node_to_expr(node):
        if isinstance(node, ast.Call):
            func = node.func.id if isinstance(node.func, ast.Name) else node.func.attr
            if func in CodeParser.OP_MAP: return BinOpNode(CodeParser.OP_MAP[func], CodeParser._node_to_expr(node.args[0]), CodeParser._node_to_expr(node.args[1]))
            elif func in CodeParser.FUNC_MAP: return UnaryFuncNode(CodeParser.FUNC_MAP[func], CodeParser._node_to_expr(node.args[0]))
        elif isinstance(node, ast.Name): return VarNode(node.id)
        elif isinstance(node, ast.Constant): return ConstNode(node.value)
        elif isinstance(node, ast.Num): return ConstNode(node.n)
        return VarNode('x')

# =============================================================================
# DATA STRUCTURES (STATE)
# =============================================================================

@dataclass
class L1Policy:
    op_weights: Dict[str, Dict[str, float]] # 'default', 'deep'
    func_weights: Dict[str, Dict[str, float]]
    mutation_rate: float
    crossover_rate: float
    tournament_size: int
    optimizer_steps: int
    complexity_penalty: float
    dynamic_rules: Dict[str, str] = field(default_factory=dict) # e.g. "mutation_rate": "0.1 * stagnation + 0.05"

@dataclass
class L2State:
    adversarial_buffer: List[Dict] # {x, error, seed, round}
    bin_edges: List[float] # [-3.0, -2.4, ..., 3.0]
    bin_weights: List[float] # Weights for sampling each bin

@dataclass
class RSIMetrics:
    perf_delta: float
    stability_succ: float
    complexity_depth: float
    # Extended metrics (V13.5)
    perf_median: float = 0.0
    perf_p25: float = 0.0
    perf_p75: float = 0.0
    perf_std: float = 0.0
    complexity_code_len: float = 0.0

@dataclass
class RSIState:
    round_idx: int
    total_seeds: int
    l1_policy: L1Policy
    l2_state: L2State
    evolved_seeds: List[str]
    pareto_archive: List[Dict]  # archive records (see _archive_record())
    current_metrics: RSIMetrics

    # Deployed incumbent (used for paired evaluation to reduce variance)
    incumbent_code: str = ""
    incumbent_hash: str = ""

    # Deterministic evaluation suite seeds (for sequential testing / CI checks)
    eval_suite_seeds: List[int] = field(default_factory=list)
    eval_suite_cursor: int = 0  # number of suite seeds currently in use

    # Evaluation protocol knobs (variance reduction + stronger holdout)
    eval_points: int = 200
    eval_train_frac: float = 0.60
    eval_valid_frac: float = 0.20  # holdout = 1 - train - valid

    # Code Synthesis State (Pure GP)
    code_synth_active: bool = False
    code_stagnation_count: int = 0
    last_eval_score: float = -1e9
    gp_fail_count: int = 0  # Track consecutive GP failures
    perf_fail_count: int = 0  # Track consecutive catastrophic perf failures (meta-meta)

    @staticmethod
    def default():
        return RSIState(
            round_idx=0,
            total_seeds=0,
            l1_policy=L1Policy(
                op_weights={
                    "default": {"+": 2, "-": 3, "*": 3, "/": 0.5, "**": 4},
                    "deep": {"+": 5, "-": 5, "*": 2, "/": 0.5, "**": 1},
                },
                func_weights={
                    "default": {"sin": 6, "cos": 1, "exp": 3, "square": 1, "cube": 4},
                    "deep": {"sin": 2, "cos": 1, "exp": 1, "square": 2, "cube": 1},
                },
                mutation_rate=0.3,
                crossover_rate=0.6,
                tournament_size=7,
                optimizer_steps=5,

                complexity_penalty=0.00005,
                dynamic_rules={},
            ),
            l2_state=L2State(
                adversarial_buffer=[],
                bin_edges=np.linspace(-3, 3, 11).tolist(),
                bin_weights=[1.0] * 10,
            ),
            evolved_seeds=[],
            pareto_archive=[],
            current_metrics=RSIMetrics(
                perf_delta=0.0, stability_succ=0.0, complexity_depth=10.0,
                perf_median=0.0, perf_p25=0.0, perf_p75=0.0, perf_std=0.0, complexity_code_len=0.0
            ),
            incumbent_code="",
            incumbent_hash="",
            eval_suite_seeds=[],
            eval_suite_cursor=0,
            eval_points=200,
            eval_train_frac=0.60,
            eval_valid_frac=0.20,
            code_synth_active=True,
            code_stagnation_count=0,
            last_eval_score=-1e9,
            gp_fail_count=0,
            perf_fail_count=0,
        )

# =============================================================================
# EVALUATION + ARCHIVE HELPERS (VARIANCE REDUCTION / HOLDOUT HARDENING)
# =============================================================================

_ARCHIVE_DIR = Path("runs/archive")
_GENEALOGY_PATH = Path("runs/genealogy.json")

# Deterministic evaluation suite for sequential testing (reduce acceptance noise)
_EVAL_SUITE_RNG_SEED = 1337
_EVAL_SUITE_MIN = 8
_EVAL_SUITE_MAX = 32

# Acceptance thresholds (tuned to reduce false negatives while guarding regressions)
_ACCEPT_MIN_MEAN_IMPROVEMENT = 0.5   # %
_ACCEPT_MIN_POS_RATE = 0.60         # fraction of suite seeds with positive improvement
_ACCEPT_Z = 1.645                   # ~90% one-sided lower CI bound
_ACCEPT_MAX_DEPTH = 12              # guardrail for pathological complexity

def code_hash(code: str) -> str:
    """Stable code hash for archive/genealogy management."""
    h = hashlib.sha256(code.encode("utf-8")).hexdigest()
    return h[:16]

def neutral_l2_state() -> L2State:
    """Holdout evaluator state (fixed distribution, no adversarial buffer leakage)."""
    return L2State(
        adversarial_buffer=[],
        bin_edges=np.linspace(-3, 3, 11).tolist(),
        bin_weights=[1.0] * 10,
    )

def safe_parse_expr(code: str):
    """Parse an expression string into the internal DSL, returning None on failure."""
    try:
        return CodeParser.parse(code)
    except Exception:
        return None

def _mse(expr, x: np.ndarray, y: np.ndarray) -> float:
    try:
        yp = np.clip(expr.evaluate(x), -1000, 1000)
        mse = float(np.mean((yp - y) ** 2))
        if not np.isfinite(mse):
            return float("inf")
        return mse
    except Exception:
        return float("inf")

def paired_improvement_pct(expr_cand, expr_inc, x: np.ndarray, y: np.ndarray) -> float:
    """Percent improvement of candidate vs incumbent on the same samples."""
    inc = _mse(expr_inc, x, y)
    cand = _mse(expr_cand, x, y)
    den = max(abs(inc), 1e-12)
    if not np.isfinite(inc) or not np.isfinite(cand):
        return -1e9
    return (inc - cand) / den * 100.0

def summarize_improvements(imprs: List[float]) -> Dict[str, float]:
    if not imprs:
        return {"mean": -1e9, "std": 1e9, "stderr": 1e9, "pos_rate": 0.0, "n": 0}
    arr = np.array(imprs, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": -1e9, "std": 1e9, "stderr": 1e9, "pos_rate": 0.0, "n": 0}
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    n = int(arr.size)
    stderr = std / math.sqrt(max(n, 1))
    pos_rate = float(np.mean(arr > 0.0))
    return {"mean": mean, "std": std, "stderr": float(stderr), "pos_rate": pos_rate, "n": n}

def lower_confidence_bound(mean: float, stderr: float, z: float = _ACCEPT_Z) -> float:
    return float(mean - z * stderr)

def extract_evolved_return_expr(file_text: str) -> Optional[str]:
    """Extract the current deployed expression from the @@EVOLVED_CODE@@ region."""
    try:
        s = file_text.find("def evolved_function")
        if s < 0:
            return None
        snippet = file_text[s:file_text.find("# @@EVOLVED_CODE_END@@", s)]
        for line in snippet.splitlines():
            line = line.strip()
            if line.startswith("return "):
                return line[len("return "):].strip()
        return None
    except Exception:
        return None

# =============================================================================
# TARGET FUNCTION (GROUND TRUTH)
# =============================================================================

def target_function(x):
    """Ground-truth function for the symbolic regression task.

    Centralized here so that *holdout*, *suite*, and *adversarial* evaluations
    are always computed against the exact same target definition.
    """
    x = np.asarray(x)
    return 2.5 * (x ** 3) - np.sin(5 * x) + np.exp(0.1 * x)

# =============================================================================
# [L2] TASK ENGINE (ADAPTIVE SAMPLING)
# =============================================================================

class TaskEngine:
    def __init__(self, seed: int, l2_state: L2State):
        if seed is not None: 
            np.random.seed(seed)
            random.seed(seed)
        self.state = l2_state
        
    def generate_data(self, task_name: str, n_points=100):
        # Target Function Registry
        def hybrid_hard(x): return 2.5*(x**3) - np.sin(5*x) + np.exp(0.1*x)
        def sinexp(x): return np.sin(2*x) + np.exp(np.abs(x)/2)
        def piecewise(x): return np.where(x > 0, np.sin(x), x**2)
        def simple_poly(x): return x**2 + x
        
        funcs = {
            'hybrid_hard': target_function,
            'sinexp': sinexp,
            'piecewise': piecewise,
            'simple_poly': simple_poly
        }
        target_f = funcs.get(task_name, target_function)
        def sinexp(x): return np.sin(2*x) + np.exp(np.abs(x)/2)
        def piecewise(x): return np.where(x > 0, np.sin(x), x**2)
        def simple_poly(x): return x**2 + x
        
        funcs = {
            'hybrid_hard': hybrid_hard,
            'sinexp': sinexp,
            'piecewise': piecewise,
            'simple_poly': simple_poly
        }
        target_f = funcs.get(task_name, hybrid_hard)
        
        # 1. Bin-based Sampling (70%)
        # Sample bins according to weights
        nbins = len(self.state.bin_weights)
        w_sum = sum(self.state.bin_weights)
        if w_sum <= 0 or not np.isfinite(w_sum):
             probs = np.ones(nbins) / nbins
        else:
             probs = np.array(self.state.bin_weights) / w_sum
        chosen_bins = np.random.choice(nbins, size=int(n_points*0.7), p=probs)
        
        x_adaptive = []
        for b_idx in chosen_bins:
            low, high = self.state.bin_edges[b_idx], self.state.bin_edges[b_idx+1]
            x_adaptive.append(random.uniform(low, high))
            
        # 2. Adversarial Buffer Replay (30%)
        x_adv = []
        if self.state.adversarial_buffer:
            adv_samples = random.sample(self.state.adversarial_buffer, k=min(len(self.state.adversarial_buffer), int(n_points*0.3)))
            x_adv = [r['x'] + random.gauss(0, 0.05) for r in adv_samples] # Jitter
        else:
             x_adv = np.random.uniform(-3, 3, int(n_points*0.3)).tolist()
        
        x_full = np.array(x_adaptive + x_adv)
        x_full = np.clip(x_full, -3.5, 3.5)
        np.random.shuffle(x_full)
        y_full = target_f(x_full)
        
        return x_full, y_full

# =============================================================================
# [L1] EVOLUTION ENGINE (DEEP/DEFAULT CONTEXT)
# =============================================================================

class ExpressionGenerator:
    def __init__(self, max_depth, policy: L1Policy):
        self.max_depth = max_depth
        self.policy = policy
    
    def generate(self, depth=0):
        if depth >= self.max_depth:
            return VarNode('x') if random.random() < 0.6 else ConstNode(round(random.uniform(-3,3),2))
        
        choice = random.choices(['const', 'var', 'bin', 'un'], weights=[1, 3, 4, 4], k=1)[0]
        if choice == 'const': return ConstNode(round(random.uniform(-3,3),2))
        if choice == 'var': return VarNode('x')
        
        ctx = 'deep' if depth > 4 else 'default'
        ops = self.policy.op_weights[ctx]
        funcs = self.policy.func_weights[ctx]
        
        if choice == 'bin':
            op = random.choices(list(ops.keys()), weights=list(ops.values()))[0]
            return BinOpNode(op, self.generate(depth+1), self.generate(depth+1))
        
        func = random.choices(list(funcs.keys()), weights=list(funcs.values()))[0]
        return UnaryFuncNode(func, self.generate(depth+1))

    def mutate(self, expr, prob=None):
        if prob is None: prob = self.policy.mutation_rate
        if random.random() < prob: return self.generate(depth=2)
        
        expr_c = expr.copy()
        if isinstance(expr_c, BinOpNode):
            expr_c.left = self.mutate(expr_c.left, prob*0.5)
            expr_c.right = self.mutate(expr_c.right, prob*0.5)
        elif isinstance(expr_c, UnaryFuncNode):
            expr_c.child = self.mutate(expr_c.child, prob*0.5)
        elif isinstance(expr_c, ConstNode):
            if random.random() < 0.5: expr_c.value *= random.uniform(0.8, 1.2)
        return expr_c
    
    def crossover(self, p1, p2):
        child = p1.copy()
        if isinstance(child, BinOpNode):
            if random.random() < 0.5: child.left = p2.copy()
            else: child.right = p2.copy()
        elif isinstance(child, UnaryFuncNode):
            child.child = p2.copy()
        return child

class EvolutionEngine:
    def __init__(self, evaluator, policy: L1Policy):
        self.evaluator = evaluator
        self.policy = policy
        self.max_depth = 8
        self.generator = ExpressionGenerator(self.max_depth, policy)
        self.population = []
        self.best_expr = None
        self.best_fitness = 0.0

    def initialize(self, seeds: List[str]):
        parsed = [CodeParser.parse(s) for s in seeds if CodeParser.parse(s)]
        for expr in parsed:
            self._add(expr, self.evaluator.evaluate(expr))
            for _ in range(5): 
                m = self.generator.mutate(expr)
                self._add(m, self.evaluator.evaluate(m))
        while len(self.population) < 500:
            e = self.generator.generate()
            self._add(e, self.evaluator.evaluate(e))
        self._update_best()

    def _add(self, expr, fit):
        self.population.append((expr, fit))

    def _update_best(self):
        for expr, fit in self.population:
            if fit > self.best_fitness:
                self.best_fitness = fit
                self.best_expr = expr.copy()

    def run(self, generations=50):
        for _ in range(generations):
            # Selection
            k = self.policy.tournament_size
            def select(): # Tournament
                c = random.sample(self.population, min(k, len(self.population)))
                return max(c, key=lambda x: x[1])[0].copy()
            
            new_pop = []
            # Elitism (+ Optimization)
            elites = sorted(self.population, key=lambda x: x[1], reverse=True)[:5]
            for e, f in elites:
                opt = self.optimize_constants(e, self.policy.optimizer_steps)
                new_pop.append((opt, self.evaluator.evaluate(opt)))
            
            # Reproduction
            while len(new_pop) < 500:
                if random.random() < self.policy.crossover_rate:
                    child = self.generator.crossover(select(), select())
                else:
                    child = self.generator.mutate(select())
                new_pop.append((child, self.evaluator.evaluate(child)))
            
            if (_ + 1) % 10 == 0:
                 print(f"  [L0-SymReg] Gen {_+1}/{generations} | Best Fit: {self.best_fitness:.4f}", flush=True)
            self.population = new_pop
            self._update_best()
        return self.best_expr

    def optimize_constants(self, expr, steps):
        vals = expr.get_constants()
        if not vals: return expr
        curr = expr.copy()
        curr_fit = self.evaluator.evaluate(curr)
        for _ in range(steps):
            bk = [c.value for c in vals]
            for c in curr.get_constants(): c.value += random.gauss(0, 0.1)
            new_fit = self.evaluator.evaluate(curr)
            if new_fit <= curr_fit:
                for c, v in zip(curr.get_constants(), bk): c.value = v
            else:
                curr_fit = new_fit
        return curr

# =============================================================================
# [L0+] CODE SYNTHESIS ENGINE (PURE ATOMIC GP)
# =============================================================================

# Allowed Atoms (No pre-written algorithm templates)
_ATOM_VARS = ["i", "j", "k", "temp", "arr", "n"]
_ATOM_CMP_OPS = ["<", ">", "<=", ">=", "==", "!="]
_ATOM_NUMS = ["0", "1", "2"]

@dataclass
class AtomicLine:
    """A single line of code evolved from atomic primitives."""
    indent: int
    content: str  # The actual code string, e.g., "if arr[i] > arr[j]:"

    def to_code(self) -> str:
        return "    " * self.indent + self.content

    def to_dict(self) -> Dict[str, Any]:
        return {"indent": self.indent, "content": self.content}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "AtomicLine":
        return AtomicLine(indent=int(d["indent"]), content=str(d["content"]))

    def clone(self) -> "AtomicLine":
        return AtomicLine(self.indent, self.content)

@dataclass
class CodeGenome:
    """A complete algorithm made of atomic lines."""
    lines: List[AtomicLine]
    fitness: float = float("-inf")
    valid: bool = False
    output: str = ""
    line_count: int = 0
    generation: int = 0

    def clone(self) -> "CodeGenome":
        return CodeGenome(
            lines=[l.clone() for l in self.lines],
            fitness=self.fitness,
            valid=self.valid,
            output=self.output,
            line_count=self.line_count,
            generation=self.generation
        )

    def to_python(self, func_name: str = "synthesized_sort") -> str:
        code_lines = [
            f"def {func_name}(arr):",
            "    '''Algorithm evolved from pure atoms.'''",
            "    arr = list(arr)  # copy input",
            "    n = len(arr)",
            "    if n <= 1: return arr",
            "    i = 0; j = 0; k = 0; temp = 0; _loop_count = 0",
        ]
        for line in self.lines:
            code_lines.append(line.to_code())
        code_lines.append("    return arr")
        self.line_count = len(code_lines)
        return "\n".join(code_lines)

# ATOMIC HELPERS 
def _random_var(rng): return rng.choice(_ATOM_VARS)
def _random_array_access(rng):
    base = rng.choice(["i", "j", "k"])
    if rng.random() < 0.3: idx = base
    elif rng.random() < 0.6: idx = f"{base} {rng.choice(['+', '-'])} 1"
    else: idx = rng.choice(["n - 1", "0", "n - i"])
    return f"arr[{idx}]"

def _random_condition(rng):
    r = rng.random()
    if r < 0.6: return f"{_random_array_access(rng)} {rng.choice(_ATOM_CMP_OPS)} {_random_array_access(rng)}"
    else: return f"{rng.choice(['i', 'j', 'k', 'n'])} {rng.choice(_ATOM_CMP_OPS)} {rng.choice(['i', 'j', 'n', '0', '1'])}"

def _random_assignment(rng):
    r = rng.random()
    if r < 0.3: return f"temp = {_random_array_access(rng)}"
    elif r < 0.5: return f"{_random_array_access(rng)} = {_random_array_access(rng)}"
    elif r < 0.65: return f"{_random_array_access(rng)} = temp"
    elif r < 0.8: return f"{rng.choice(['i','j'])} += 1"
    else: return f"{rng.choice(['i','j'])} = {rng.choice(['0', '1', 'n-1'])}"

def _generate_random_lines(rng, depth=1, restricted=False):
    lines = []
    t = rng.random()
    # RESTRICTED MODE: No nested loops, No while, Max depth 2
    if restricted and depth > 2:
         lines.append(AtomicLine(depth, _random_assignment(rng)))
         return lines

    if t < 0.3: lines.append(AtomicLine(depth, _random_assignment(rng)))
    elif t < 0.6:
        # Loop
        if restricted:
             # Only simple range in restricted
             v = "i" 
             lines.append(AtomicLine(depth, f"for {v} in range(n):"))
        else:
             v = rng.choice(["i", "j"])
             lines.append(AtomicLine(depth, f"for {v} in range({rng.choice(['0','1'])}, {rng.choice(['n','n-1'])}):"))
        
        lines.append(AtomicLine(depth+1, _random_assignment(rng)))
    elif t < 0.8:
        if restricted: # No While in restricted
             lines.append(AtomicLine(depth, f"if {_random_condition(rng)}:"))
             lines.append(AtomicLine(depth+1, _random_assignment(rng)))
        else:
             lines.append(AtomicLine(depth, f"while {_random_condition(rng)} and _loop_count < 200:"))
             lines.append(AtomicLine(depth+1, "_loop_count += 1"))
             lines.append(AtomicLine(depth+1, _random_assignment(rng)))
    else:
        lines.append(AtomicLine(depth, f"if {_random_condition(rng)}:"))
        lines.append(AtomicLine(depth+1, _random_assignment(rng)))
    return lines

def random_code_genome(rng, restricted=False):
    lines = []
    # In restricted, smaller genome
    cnt = rng.randint(1, 3) if restricted else rng.randint(2, 5)
    for _ in range(cnt): lines.extend(_generate_random_lines(rng, restricted=restricted))
    return CodeGenome(lines=lines)

# =============================================================================
# [L0+] TRANSFER LEARNING MEMORY
# =============================================================================

@dataclass
class KnowledgeShard:
    """A reusable piece of knowledge (code snippet) with meta-data."""
    source_task: str
    code_snippet: str
    fitness_impact: float
    tags: List[str] # e.g. ["loop", "math", "optimization"]
    uses: int = 0

class KnowledgeArchive:
    """
    Transfer Learning System:
    Stores successful genes/patterns from previous runs to speed up future tasks.
    """
    def __init__(self):
        self.long_term_memory: List[KnowledgeShard] = []
        
    def absorb(self, task_name: str, genome: CodeGenome, fitness: float):
        """Extract reusable patterns from valid, high-fitness code."""
        if fitness < 0: return
        
        # Simple pattern extraction: atomic lines are already shards
        for line in genome.lines:
            # We filter for 'interesting' lines (control flow, assignments)
            tags = []
            if "for " in line.content: tags.append("loop")
            if "if " in line.content: tags.append("condition")
            if "temp" in line.content: tags.append("swap")
            
            if tags:
                shard = KnowledgeShard(
                    source_task=task_name,
                    code_snippet=line.content,
                    fitness_impact=fitness,
                    tags=tags
                )
                
                # Check duplicates
                if not any(s.code_snippet == shard.code_snippet for s in self.long_term_memory):
                    self.long_term_memory.append(shard)
                    
    def retrieve(self, tags: List[str] = None, top_k: int = 5) -> List[str]:
        """Recall relevant snippets for the current context."""
        candidates = self.long_term_memory
        if tags:
            candidates = [c for c in candidates if any(t in c.tags for t in tags)]
            
        candidates.sort(key=lambda x: x.fitness_impact + (x.uses * 0.1), reverse=True)
        return [c.code_snippet for c in candidates[:top_k]]

    def _bump_use(self, snippet: str) -> None:
        """Increment usage count for a recalled snippet (robust to retrieval ordering)."""
        for shard in self.long_term_memory:
            if shard.code_snippet == snippet:
                shard.uses += 1
                return

    def promote_to_grammar(self, top_n=3) -> List[Tuple[str, str]]:
        """Identify high-impact patterns and promote them to grammar macros."""
        # Ensure uniqueness and quality
        self.long_term_memory.sort(key=lambda x: x.uses, reverse=True)
        promoted = []
        for c in self.long_term_memory[:top_n]:
            if c.uses > 3: 
                # Create a safe function name hash
                import hashlib
                h = hashlib.md5(c.code_snippet.encode()).hexdigest()[:6]
                macro_name = f"_macro_{h}"
                promoted.append((macro_name, c.code_snippet))
        return promoted

# Global Memory
_TRANSFER_MEMORY = KnowledgeArchive()

def mutate_with_transfer(rng, genome, rate=0.3):
    """Mutation that utilizes Transfer Learning memory."""
    new_lines = []
    
    # Retrieve snippets from memory (Transfer)
    memories = _TRANSFER_MEMORY.retrieve(top_k=5)
    
    for line in genome.lines:
        if rng.random() < 0.05: continue 
        if rng.random() < rate:
            # Attempt transfer injection
            if memories and rng.random() < 0.4:
                snippet = rng.choice(memories)
                _TRANSFER_MEMORY._bump_use(snippet) # Reinforce
                new_lines.append(AtomicLine(line.indent, snippet))
            else:
                # Fallback to standard atomic mutation
                if "for " in line.content: new_lines.append(AtomicLine(line.indent, f"for {rng.choice(['i','j'])} in range(n):"))
                elif "while " in line.content: new_lines.append(AtomicLine(line.indent, f"while {_random_condition(rng)} and _loop_count < 200:"))
                elif "if " in line.content: new_lines.append(AtomicLine(line.indent, f"if {_random_condition(rng)}:"))
                elif "=" in line.content: new_lines.append(AtomicLine(line.indent, _random_assignment(rng)))
                else: new_lines.append(line.clone())
        else:
            new_lines.append(line.clone())
            
    if rng.random() < 0.2:
        idx = rng.randint(0, len(new_lines)) if new_lines else 0
        new_lines.insert(idx, AtomicLine(1, _random_assignment(rng)))
    return CodeGenome(lines=new_lines)


def eval_code_genome(genome):
    code = genome.to_python()
    
    # Trace function to kill infinite loops
    def trace_lines(frame, event, arg):
        if event == 'line':
            frame.f_locals['_ops'] = frame.f_locals.get('_ops', 0) + 1
            if frame.f_locals['_ops'] > 5000: # Max OPS limit
                raise TimeoutError("Instruction limit exceeded")
        return trace_lines

    try:
        exec_globals = {}
        exec(code, exec_globals)
        fn = exec_globals.get("synthesized_sort")
        if not fn: return -9999
        score = 0
        
        # [HARDCORE] No easy tests. All unsorted.
        tests = [[3,1,2], [5,4,3,2,1], [10,2,8,4,6], [9,7,5,3,1]]
        
        import sys
        
        for t in tests:
            target = sorted(t)
            res = None # Initialize result
            
            if "while" in code:
                # Dangerous: Use Trace
                ops_counter = [0]
                def trace_lines(frame, event, arg):
                    if event == 'line':
                        ops_counter[0] += 1
                        if ops_counter[0] > 5000: raise TimeoutError
                    return trace_lines
                
                sys.settrace(trace_lines)
                try:
                    res = fn(list(t))
                except TimeoutError:
                    score -= 100
                    continue
                finally:
                    sys.settrace(None)
            else:
                # Safe: Run raw (Fast Mode)
                try:
                    res = fn(list(t))
                except Exception:
                    pass # Normal error

            # [ANTI-CHEAT] Identity Penalty
            # Check if res is valid first
            if res is None:
                 score -= 10 # Crash penalty
                 continue

            if res == t: 
                score -= 50 # Doing nothing is NOT sorting
                continue

            if res == target: 
                score += 100
            elif len(res) == len(t) and sorted(res) == target: 
                score += 20 # Partial credit for keeping elements
            else:
                score -= 10 # Wrong usage penalization
        return score
    except: return -9999

def evolve_code_genome(rng, gens=50, restricted=False):
    pop = [random_code_genome(rng, restricted=restricted) for _ in range(50)]
    best = pop[0]
    for _ in range(gens):
        for p in pop: p.fitness = eval_code_genome(p)
        pop.sort(key=lambda x: x.fitness, reverse=True)
        if pop[0].fitness > best.fitness: best = pop[0].clone()
        new_pop = [p.clone() for p in pop[:10]]
        while len(new_pop) < 50:
            parent = rng.choice(pop[:20])
            child = mutate_with_transfer(rng, parent) # Use Transfer Learning!!
            new_pop.append(child)
        if (_ + 1) % 5 == 0:
            print(f"  [GP] Gen {_+1}/{gens} | Best Fit: {best.fitness} | Pop Avg: {sum(p.fitness for p in pop)/len(pop):.2f}", flush=True)
        pop = new_pop
    
    # Absorb knowledge if successful
    if best.fitness > 0:
        _TRANSFER_MEMORY.absorb("sorting", best, best.fitness)
        
    return best

class FitnessEvaluator:
    def __init__(self, x, y, penalty):
        self.x = x
        self.y = y
        self.penalty = penalty
        try:
             p = np.poly1d(np.polyfit(x, y, 3))
             self.base_mse = float(np.mean((p(x) - y)**2))
        except: 
             # Fallback to mean model if polyfit fails
             y0 = float(np.mean(y))
             self.base_mse = float(np.mean((y0 - y)**2))

    def evaluate(self, expr):
        try:
            yp = np.clip(expr.evaluate(self.x), -1000, 1000)
            mse = np.mean((yp - self.y)**2)
            if not np.isfinite(mse): return 0.0
            return 1.0 / (1.0 + mse + expr.depth()*self.penalty)
        except: return 0.0
    
    def get_raw_delta(self, expr):
        try:
            yp = np.clip(expr.evaluate(self.x), -1000, 1000)
            mse = np.mean((yp - self.y)**2)
            if self.base_mse < 1e-9: return 0.0
            return (self.base_mse - mse) / self.base_mse * 100.0
        except: return -100.0

# =============================================================================
# [L1+] META-META LEARNING (GRAMMAR-GUIDED STRUCTURAL EVOLUTION)
# =============================================================================

class ProductionRule:
    def __init__(self, lhs, rhs, weight=1.0):
        self.lhs = lhs; self.rhs = rhs; self.weight = weight

class PythonGrammar:
    """Context-Free Grammar for evolving Python structure."""
    def __init__(self):
        self.rules = {}
        self._build_grammar()
    
    def _build_grammar(self):
        self._add("Program", ["FunctionDef"])
        self._add("FunctionDef", ["DEF", "IDENTIFIER", "LPAREN", "Params", "RPAREN", "COLON", "Block"])
        self._add("Block", ["Statement"], weight=0.6)
        self._add("Block", ["Statement", "Statement"], weight=0.4)
        self._add("Statement", ["Return"])
        self._add("Statement", ["IfStatement"])
        self._add("Return", ["RETURN", "Expr"])
        self._add("Expr", ["BinOp"], weight=2.0)
        self._add("Expr", ["Call"], weight=1.0)
        self._add("Expr", ["IDENTIFIER"], weight=3.0)
        self._add("Expr", ["LITERAL"], weight=1.0)
        self._add("BinOp", ["Expr", "OP", "Expr"])
        self._add("Call", ["IDENTIFIER", "LPAREN", "Expr", "RPAREN"])
        self._add("IfStatement", ["IF", "Expr", "COLON", "Block", "ELSE", "COLON", "Block"])
        
    def _add(self, lhs, rhs, weight=1.0):
        if lhs not in self.rules: self.rules[lhs] = []
        self.rules[lhs].append(ProductionRule(lhs, rhs, weight))

    def inject_rule(self, lhs, rhs, weight=5.0):
        """Dynamic Grammar Injection"""
        self._add(lhs, rhs, weight)
        print(f"  > [GRAMMAR] Injected Rule: {lhs} -> {rhs}")

    def update_weights(self, strategy: str):
        """Meta-Meta Update: Evolve the grammar itself based on high-level strategy."""
        if strategy == "recursion":
            # Boost function calls (recursion probability)
            for r in self.rules.get("Expr", []): 
                if "Call" in r.rhs: r.weight *= 2.0
        elif strategy == "complexity":
             # Boost block size and nested logic
             for r in self.rules.get("Block", []):
                 if len(r.rhs) > 1: r.weight *= 1.5

# Global Grammar State
_META_GRAMMAR = PythonGrammar()

# =============================================================================
# [L1+] META-META LEARNING (SYMBOLIC POLICY EVOLUTION)
# =============================================================================

@dataclass
class StateVector:
    """Normalized System State Vector [0.0 - 1.0]"""
    perf_delta: float       # Normalized performance (0.0=fail, 1.0=super)
    stability: float        # Normalized stability (0.0-1.0)
    diversity: float        # Population diversity proxy
    stagnation: float       # Stagnation level (0.0 - 1.0)
    complexity: float       # Current complexity level

    @staticmethod
    def from_state(state: RSIState, metrics: RSIMetrics) -> 'StateVector':
        # Normalize inputs
        p = np.clip((metrics.perf_delta + 100) / 200, 0.0, 1.0)
        s = np.clip(metrics.stability_succ / 100.0, 0.0, 1.0)
        d = 0.5 # Placeholder for diversity
        st = np.clip(state.code_stagnation_count / 10.0, 0.0, 1.0)
        c = np.clip(metrics.complexity_depth / 20.0, 0.0, 1.0)
        return StateVector(p, s, d, st, c)
    
    def to_dict(self):
        return asdict(self)

class PolicyEvaluator:
    """Evaluates symbolic learning policies."""
    
    @staticmethod
    def evaluate(rule: str, state_vec: StateVector) -> float:
        """
        Evaluate a rule string against the state vector.
        Supported vars: P(perf), S(stability), D(diversity), ST(stagnation), C(complexity)
        Max length precaution + simple math only.
        """
        try:
            # Safe Context
            ctx = {
                "P": state_vec.perf_delta,
                "S": state_vec.stability,
                "D": state_vec.diversity,
                "ST": state_vec.stagnation,
                "C": state_vec.complexity,
                "min": min, "max": max, "abs": abs
            }
            # dangerous but controlled by self-evolution
            val = float(eval(rule, {"__builtins__":{}}, ctx))
            return val
        except:
            return 0.0

def run_meta_meta_update(state: RSIState, cand: RSIMetrics):
    """Evolve the system's learning structure (L1 parameters & Grammar)."""
    
    # 1. Update State History (for meta-reward in future)
    s_vec = StateVector.from_state(state, cand)
    
    # Check stagnation for emergency overrides (Legacy Safety Net)
    if cand.perf_delta < -100.0:
        state.perf_fail_count += 1
    elif cand.perf_delta > 50.0:
        state.perf_fail_count = 0 
    
    print(f"[META-META] Stagnation: {state.perf_fail_count} | State: P={s_vec.perf_delta:.2f} ST={s_vec.stagnation:.2f}")

    # 2. Symbolic Policy Execution
    # If dynamic rules exist, use them to set hyperparameters
    if state.l1_policy.dynamic_rules:
        # Mutation Rate Policy
        if "mutation_rate" in state.l1_policy.dynamic_rules:
            rule = state.l1_policy.dynamic_rules["mutation_rate"]
            new_mut = PolicyEvaluator.evaluate(rule, s_vec)
            new_mut = np.clip(new_mut, 0.05, 0.95)
            state.l1_policy.mutation_rate = new_mut
            print(f"  > [POLICY] Dynamic Mutation Rate: {new_mut:.3f} (Rule: {rule})")

        # Penalty Policy
        if "penalty" in state.l1_policy.dynamic_rules:
            rule = state.l1_policy.dynamic_rules["penalty"]
            new_pen = PolicyEvaluator.evaluate(rule, s_vec)
            new_pen = np.clip(new_pen, 0.00001, 0.01)
            state.l1_policy.complexity_penalty = new_pen
            print(f"  > [POLICY] Dynamic Penalty: {new_pen:.5f} (Rule: {rule})")
            
    else:
        # [BOOTSTRAP] Initialize Default Symbolic Policies if empty
        # P = Perf, ST = Stagnation
        # Heuristic: If stagnated (ST high), boost mutation. If Perf high, lower mutation.
        state.l1_policy.dynamic_rules = {
            "mutation_rate": "0.1 + 0.8 * ST - 0.2 * P",
            "penalty": "0.00005 * (1.0 + P)"
        }
        print("  > [POLICY] Bootstrapped Symbolic Policies.")

    # 3. Policy Evolution (Meta-Optimization)
    # Randomly mutate the policy strings occasionally
    if random.random() < 0.2: # 20% chance per round
        target = random.choice(["mutation_rate", "penalty"])
        curr_rule = state.l1_policy.dynamic_rules[target]
        
        # Simple string mutation (append/replace terms)
        mutations = [
            lambda r: r + " + 0.1 * ST",
            lambda r: r + " - 0.1 * P",
            lambda r: r + " * 1.1",
            lambda r: f"({r}) * 0.9",
        ]
        
        # Fork a test? No, we do online modification with rollback if needed in future
        # For now, simple drift.
        new_rule = random.choice(mutations)(curr_rule)
        print(f"  > [META-EVO] Mutating Policy [{target}]: {curr_rule} -> {new_rule}")
        state.l1_policy.dynamic_rules[target] = new_rule

    # 4. Crisis Management (Override)
    if state.perf_fail_count >= 5:
        print("[META-META] EMERGENCY RANDOM RESTART TRIGGERED!")
        state.evolved_seeds = []
        state.code_stagnation_count = 0
        state.l1_policy.dynamic_rules["mutation_rate"] = "0.8" # Reset to high constant
        
    # 5. Evolve Grammar (Structural Preferences)
    if state.current_metrics.complexity_depth > 5.0:
        _META_GRAMMAR.update_weights("simplicity")
    # 6. Dynamic Grammar Expansion (Phase 2)
    # Check if we have useful patterns in memory to promote
    new_macros = _TRANSFER_MEMORY.promote_to_grammar()
    for name, code in new_macros:
        # Avoid duplicate injection
        if any(r.rhs == [name] for r in _META_GRAMMAR.rules.get("Expr", [])):
            continue
            
        print(f"  > [GRAMMAR] Discovered Macro Candidate: {name} <= {code.strip()}")
        # NOTE: Full integration requires declaring the macro function in global scope.
        # This will be handled by the Self-Modification (Phase 3) engine in future rounds.
        # For now, we just signal availability.
        
    else:
        _META_GRAMMAR.update_weights("complexity")

# =============================================================================
# [Phase 3] REFLECTIVE SANDBOX (ENGINE SAFETY)
# =============================================================================

class ReflectionSandbox:
    """Verifies that modifications to the EvolutionEngine do not break the system."""
    
    @staticmethod
    def verify_engine_update(new_source_code: str) -> bool:
        print("  > [SANDBOX] Verifying new Engine code...")
        try:
            # 1. Syntax Check
            ast.parse(new_source_code)
            
            # 2. Functional Simulation
            # Create a temporary exec environment
            # We mix globals() to allow access to system types (L1Policy, etc)
            # Unified scope ensures classes defined in the string are visible to each other
            sandbox_scope = globals().copy()
            
            # Execute in a discrete namespace
            exec(new_source_code, sandbox_scope, sandbox_scope)
            
            # Check results
            if 'EvolutionEngine' not in sandbox_scope:
                 print("  > [SANDBOX] FAIL: EvolutionEngine class not found.")
                 return False
                 
            # Instantiate and run mini-test
            # Mock dependencies
            mock_evaluator = type('MockEval', (), {'evaluate': lambda s,x: 1.0})() 
            mock_policy = L1Policy({}, {}, 0.1, 0.5, 2, 1, 0.001, {})
            
            EngineClass = sandbox_scope['EvolutionEngine']
            eng = EngineClass(mock_evaluator, mock_policy)
            eng.initialize(["x"])
            best = eng.run(generations=2)
            
            if best is None: 
                print("  > [SANDBOX] FAIL: Engine returned None.")
                return False
                
            print("  > [SANDBOX] PASS: Engine functional.")
            return True
            
        except Exception as e:
            print(f"  > [SANDBOX] FAIL: {e}")
            return False

# =============================================================================
# WORKER
# =============================================================================

def run_seed_process(
    seed: int,
    task_name: str,
    l1_dict: Dict,
    l2_dict: Dict,
    seeds: List[str],
    incumbent_code: str,
    eval_points: int = 200,
    train_frac: float = 0.60,
    valid_frac: float = 0.20,
):
    """Worker process: evolve a candidate, then report paired generalization metrics.

    Key changes vs older worker:
    - Larger dataset default (eval_points) to shrink estimator variance.
    - Three-way split: train / validation / holdout.
    - Paired evaluation vs incumbent on the exact same samples (common random numbers).
    """
    # Reconstruct Dataclasses from Dicts (Pickle safety)
    l1 = L1Policy(**l1_dict)
    l2 = L2State(**l2_dict)

    task_eng = TaskEngine(seed, l2)
    x, y = task_eng.generate_data(task_name, n_points=eval_points)

    n = len(x)
    t_end = int(n * train_frac)
    v_end = int(n * (train_frac + valid_frac))
    xt, xv, xh = x[:t_end], x[t_end:v_end], x[v_end:]
    yt, yv, yh = y[:t_end], y[t_end:v_end], y[v_end:]

    eval_train = FitnessEvaluator(xt, yt, l1.complexity_penalty)
    eng = EvolutionEngine(
        eval_train,
        l1
    )
    # Patch EvolutionEngine to accept seeds (it does already in initialize)
    # The user code calls eng.run(generations=40, pop_size=120, seeds=seeds)
    # But EvolutionEngine.run only accepts generations. .initialize accepts seeds. 
    # I must check EvolutionEngine.run signature again.
    
    eng.initialize(seeds)
    best_expr = eng.run(generations=40) 
    
    # User provided: best_expr, best_fit = eng.run(generations=40, pop_size=120, seeds=seeds)
    # My engine returns just best_expr. 
    # I will stick to my engine's signature for now OR update EvolutionEngine as well?
    # User's patch doesn't include EvolutionEngine updates. 
    # So I will adapt the call.
    
    if best_expr is None:
        return {
            "seed": seed,
            "success": False,
            "valid_impr": -1e9,
            "holdout_impr": -1e9,
            "delta_valid": -100.0,
            "delta_holdout": -100.0,
            "depth": 999,
            "code": "x",
            "hard_records": [],
        }

    cand_code = best_expr.to_code()
    cand_depth = int(best_expr.depth())

    # Incumbent parse (paired evaluation to reduce noise)
    inc_expr = safe_parse_expr(incumbent_code) or safe_parse_expr("x")

    # Paired improvement vs incumbent
    valid_impr = paired_improvement_pct(best_expr, inc_expr, xv, yv) if len(xv) else -1e9
    holdout_impr = paired_improvement_pct(best_expr, inc_expr, xh, yh) if len(xh) else -1e9

    # Baseline (polyfit) deltas preserved for logging/debugging
    delta_valid = FitnessEvaluator(xv, yv, l1.complexity_penalty).get_raw_delta(best_expr) if len(xv) else -100.0
    delta_holdout = FitnessEvaluator(xh, yh, l1.complexity_penalty).get_raw_delta(best_expr) if len(xh) else -100.0

    # Hard samples from holdout (feed L2 adversarial buffer)
    hard_records = []
    try:
        yp = np.clip(best_expr.evaluate(xh), -1000, 1000)
        abs_err = np.abs(yp - yh)
        if len(abs_err):
            k = min(12, len(abs_err))
            idx = np.argsort(abs_err)[-k:]
            for i in idx:
                hard_records.append({"x": float(xh[i]), "error": float(abs_err[i])})
    except Exception:
        pass

    success = (holdout_impr > 0.25) and (valid_impr > 0.0) and (cand_depth <= _ACCEPT_MAX_DEPTH)

    return {
        "seed": seed,
        "success": bool(success),
        "valid_impr": float(valid_impr),
        "holdout_impr": float(holdout_impr),
        "delta_valid": float(delta_valid),
        "delta_holdout": float(delta_holdout),
        "depth": cand_depth,
        "code": cand_code,
        "hard_records": hard_records,
    }

# =============================================================================
# ORCHESTRATOR
# =============================================================================

class NeuroArchitect:
    def __init__(self, path):
        self.path = Path(path)
        self.state_path = Path("runs/rsi_state.json")

    def load_state(self) -> RSIState:
        state: RSIState
        if self.state_path.exists():
            try:
                with open(self.state_path, "r") as f:
                    d = json.load(f)

                l1 = L1Policy(**d["l1_policy"])
                l2 = L2State(**d["l2_state"])
                metrics = RSIMetrics(**d["current_metrics"])

                state = RSIState(
                    round_idx=d["round_idx"],
                    total_seeds=d.get("total_seeds", 0),
                    l1_policy=l1,
                    l2_state=l2,
                    evolved_seeds=d.get("evolved_seeds", []),
                    pareto_archive=d.get("pareto_archive", []),
                    current_metrics=metrics,
                    incumbent_code=d.get("incumbent_code", ""),
                    incumbent_hash=d.get("incumbent_hash", ""),
                    eval_suite_seeds=d.get("eval_suite_seeds", []),
                    eval_suite_cursor=d.get("eval_suite_cursor", 0),
                    eval_points=d.get("eval_points", 200),
                    eval_train_frac=d.get("eval_train_frac", 0.60),
                    eval_valid_frac=d.get("eval_valid_frac", 0.20),
                    # Persist code synthesis state
                    code_synth_active=d.get("code_synth_active", False),
                    code_stagnation_count=d.get("code_stagnation_count", 0),
                    last_eval_score=d.get("last_eval_score", -1e9),
                    gp_fail_count=d.get("gp_fail_count", 0),
                    perf_fail_count=d.get("perf_fail_count", 0),
                )
            except Exception as e:
                print(f"[RSI] State load failed, using defaults: {e}")
                state = RSIState.default()
        else:
            state = RSIState.default()

        # Bootstrap incumbent & suite seeds even if older state files are missing these fields.
        self._bootstrap_incumbent(state)
        self._ensure_eval_suite(state, max(int(state.eval_suite_cursor or 0), int(_EVAL_SUITE_MIN)))

        # Ensure incumbent is available as a seed (reduces regression search)
        if state.incumbent_code and state.incumbent_code not in state.evolved_seeds:
            state.evolved_seeds = [state.incumbent_code] + state.evolved_seeds

        # Keep seed pool compact
        state.evolved_seeds = list(dict.fromkeys(state.evolved_seeds))[:30]
        return state

    def _bootstrap_incumbent(self, state: RSIState):
        if not state.incumbent_code:
            # Fallback to evolved function found in file or default
            deployed = extract_evolved_return_expr(self.path.read_text("utf-8") if self.path.exists() else "")
            state.incumbent_code = deployed or "x"
            state.incumbent_hash = code_hash(state.incumbent_code)

    def _ensure_eval_suite(self, state: RSIState, n_needed: int):
        current_n = len(state.eval_suite_seeds)
        if current_n < n_needed:
            # Deterministic expansion
            rng = random.Random(_EVAL_SUITE_RNG_SEED + current_n)
            needed = n_needed - current_n
            new_seeds = [rng.randint(10000, 999999) for _ in range(needed)]
            state.eval_suite_seeds.extend(new_seeds)

    def _sequential_acceptance_eval(self, code: str, state: RSIState) -> Dict[str, Any]:
        """Run robust paired evaluation against incumbent on the evaluation suite."""
        cand_expr = safe_parse_expr(code)
        inc_expr = safe_parse_expr(state.incumbent_code)
        if not cand_expr: return {}
        
        imprs = []
        depth = cand_expr.depth()
        
        # 1. Deterministic Suite Evaluation (Paired)
        suite_seeds = state.eval_suite_seeds[:state.eval_suite_cursor+8] # Use a subset dynamically? User code uses cursor.
        # Check user code logic: _ensure_eval_suite sets cursor? 
        # Actually user code only passed 'state'. 
        # I will evaluate on ALL suite seeds currently available.
        suite_seeds = state.eval_suite_seeds
        
        te = TaskEngine(0, state.l2_state) # L2 state doesn't matter for suite if we force target
        
        for s in suite_seeds:
            # Generate suite data (Task: hybrid_hard ground truth)
            # We must force 'hybrid_hard' which maps to target_function
            t_rng = TaskEngine(s, neutral_l2_state())
            x, y = t_rng.generate_data("hybrid_hard", n_points=state.eval_points)
            imp = paired_improvement_pct(cand_expr, inc_expr, x, y)
            imprs.append(imp)
            
        summary = summarize_improvements(imprs)
        suite_lcb = lower_confidence_bound(summary['mean'], summary['stderr'])
        
        # 2. Out-of-Distribution (OOD) - 'sinexp'
        te_ood = TaskEngine(12345, neutral_l2_state())
        x_ood, y_ood = te_ood.generate_data("sinexp", n_points=state.eval_points)
        ood_imp = paired_improvement_pct(cand_expr, inc_expr, x_ood, y_ood)
        ood_lcb = ood_imp # Single shot, no CI
        
        # 3. Adversarial Replay
        # Evaluate on buffer
        adv_impr = 0.0
        if state.l2_state.adversarial_buffer:
             # Reconstruct buffer samples
             x_adv = np.array([r['x'] for r in state.l2_state.adversarial_buffer])
             # y_adv = target_function(x_adv) # Need target function
             # We assume 'hybrid_hard' is target
             y_adv = target_function(x_adv)
             adv_impr = paired_improvement_pct(cand_expr, inc_expr, x_adv, y_adv)

        return {
            "suite_summary": summary,
            "suite_lcb": suite_lcb,
            "ood_lcb": ood_lcb,
            "adv_impr": adv_impr,
            "depth": depth,
            "suite_n": len(suite_seeds)
        }

    def _archive_candidate(self, state, code, metrics, report, parent_hash):
        return {
            'ts': datetime.now().isoformat(timespec='seconds'),
            'round': state.round_idx,
            'code_hash': code_hash(code),
            'parent_hash': parent_hash,
            'metrics': asdict(metrics),
            'report': report,
            'code': code
        }

    def save_state(self, state: RSIState):
        self.state_path.parent.mkdir(exist_ok=True)
        with open(self.state_path, 'w') as f:
            # dataclass to dict
            json.dump(asdict(state), f, indent=2)

    def optimize_l1(self, successes: List[Dict], state: RSIState):
        if not successes: return
        print(f"  > [L1] Analyzing {len(successes)} winners for active policy optimization...")
        
        # 1. Parse Winners to Count Ops per Context
        # Since `count_ops` is recursive and modifies dict, lets create a fresh one
        # `ops` passed to count_ops is expected to be Dict[str, Dict[str, int]]
        agg_ops = {} 
        agg_funcs = {}
        for r in successes:
             expr = CodeParser.parse(r['code'])
             if expr: 
                 # We must pass empty dicts that will be populated
                 # The count_ops impl: `if ctx not in ops: ops[ctx] = {}`
                 expr.count_ops(agg_ops, agg_funcs, 1)

        # 2. Update Weights (Multiplicative)
        # alpha = 0.1 (Learning Rate)
        alpha = 0.1
        
        for ctx in ['default', 'deep']:
            # Ops
            if ctx in agg_ops:
                total_ops = sum(agg_ops[ctx].values())
                if total_ops > 0:
                    for op, count in agg_ops[ctx].items():
                        freq = count / total_ops
                        # Update: w = w * (1 + alpha * (freq - (1/N_ops))) roughly
                        # Simplified: w = w * (1.0 + alpha * freq) then normalize
                        if op in state.l1_policy.op_weights[ctx]:
                             state.l1_policy.op_weights[ctx][op] *= (1.0 + alpha * freq * 5.0) 
            
            # Funcs
            if ctx in agg_funcs:
                total_funcs = sum(agg_funcs[ctx].values())
                if total_funcs > 0:
                     for func, count in agg_funcs[ctx].items():
                         freq = count / total_funcs
                         if func in state.l1_policy.func_weights[ctx]:
                             state.l1_policy.func_weights[ctx][func] *= (1.0 + alpha * freq * 5.0)

        # Normalize Weights
        for ctx in ['default', 'deep']:
            for w_dict in [state.l1_policy.op_weights[ctx], state.l1_policy.func_weights[ctx]]:
                 total = sum(w_dict.values())
                 if total > 0:
                     for k in w_dict: w_dict[k] = max(0.1, min(10.0, (w_dict[k] / total) * 10.0))

        # 3. Hyperparameter Adaptation
        avg_depth = sum(total_depths) / len(total_depths) if total_depths else 1.0
        success_rate = len(successes) / state.total_seeds if state.total_seeds > 0 else 0.0
        
        updates = []
        # If complexity is too high, increase penalty
        if avg_depth > 6:
            state.l1_policy.complexity_penalty *= 1.2
            updates.append("Penalty ↑")
        elif avg_depth < 3:
             state.l1_policy.complexity_penalty *= 0.8
             updates.append("Penalty ↓")

        # If strict optimization is stalling (low success), increase mutation
        # If finding many solutions, decrease mutation to fine tune
        if success_rate < 0.1:
             state.l1_policy.mutation_rate = min(0.8, state.l1_policy.mutation_rate * 1.1)
             updates.append("Mut Rate ↑")
        elif success_rate > 0.6:
             state.l1_policy.mutation_rate = max(0.05, state.l1_policy.mutation_rate * 0.9)
             updates.append("Mut Rate ↓")
        
        print(f"  > [L1] Optimized Weights. Hyperparams: {', '.join(updates)}")
        print(f"  > [L1] Current MutRate: {state.l1_policy.mutation_rate:.3f}, Penalty: {state.l1_policy.complexity_penalty:.5f}")


    def meta_round(self, n_seeds: int, seed_base: int, workers: int):
        state = self.load_state()
        state.round_idx += 1
        state.total_seeds = int(n_seeds)
        print(f"\n[RSI] Round {state.round_idx} Start (Seeds {n_seeds}, Base {seed_base})...")

        # Prepare inputs for workers
        l1_d = asdict(state.l1_policy)
        l2_d = asdict(state.l2_state)
        seeds = list(state.evolved_seeds)

        # Ensure incumbent is part of the warm-start pool (stability)
        if state.incumbent_code and state.incumbent_code not in seeds:
            seeds = [state.incumbent_code] + seeds
        seeds = list(dict.fromkeys(seeds))[:30]

        results: List[Dict] = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=int(workers)) as exe:
            futs = {
                exe.submit(
                    run_seed_process,
                    int(seed_base + i),
                    "hybrid_hard",
                    l1_d,
                    l2_d,
                    seeds,
                    state.incumbent_code,
                    int(state.eval_points),
                    float(state.eval_train_frac),
                    float(state.eval_valid_frac),
                ): i
                for i in range(int(n_seeds))
            }
            for f in concurrent.futures.as_completed(futs):
                try:
                    r = f.result()
                    results.append(r)
                except Exception as e:
                    print(f"  ! Worker failed: {e}")

        if not results:
            print("[RSI] No results; saving state and exiting.")
            state.code_stagnation_count += 1
            self.save_state(state)
            return

        # ---------------------------------------------------------------------
        # [L2] Update adversarial buffer + sampling weights (variance reduced)
        # ---------------------------------------------------------------------
        all_hard: List[Dict] = []
        for r in results:
            all_hard.extend(r.get("hard_records", []))

        state.l2_state.adversarial_buffer.extend(all_hard)
        state.l2_state.adversarial_buffer.sort(key=lambda x: x.get("error", 0.0), reverse=True)
        state.l2_state.adversarial_buffer = state.l2_state.adversarial_buffer[:200]

        # Stabilize L2 weights (log -> zscore -> softmax -> clip)
        errors_in_bins = np.array([0.1] * 10, dtype=float)
        for h in state.l2_state.adversarial_buffer:
            bx = int((np.clip(float(h.get("x", 0.0)), -3.0, 3.0) + 3.0) / 0.6)
            bx = min(9, max(0, bx))
            errors_in_bins[bx] += float(h.get("error", 0.0))

        log_weights = np.log1p(errors_in_bins)
        mean_w = float(np.mean(log_weights))
        std_w = float(np.std(log_weights))
        normalized = (log_weights - mean_w) / (std_w + 1e-8)

        z = normalized / 0.5
        exp_z = np.exp(z - np.max(z))
        final_weights = exp_z / np.sum(exp_z)
        final_weights = np.clip(final_weights * 1000.0, 0.0, 1000.0)
        state.l2_state.bin_weights = final_weights.tolist()

        # ---------------------------------------------------------------------
        # [L1] Policy update using only generalizing winners
        # ---------------------------------------------------------------------
        successes = [r for r in results if r.get("success", False)]
        print(f"[RSI] Seed Successes: {len(successes)}/{len(results)}")

        if len(successes) < 5:
            print("  > [L1] Skipping Optimization (insufficient generalizing winners); boosting exploration.")
            state.l1_policy.mutation_rate = min(0.9, state.l1_policy.mutation_rate + 0.05)
        else:
            self.optimize_l1(successes, state)

        # ---------------------------------------------------------------------
        # Candidate selection (use holdout-improvement to reduce overfit)
        # ---------------------------------------------------------------------
        pool = sorted(
            results,
            key=lambda r: (r.get("holdout_impr", -1e9), r.get("valid_impr", -1e9)),
            reverse=True,
        )

        uniq = []
        seen = set()
        for r in pool:
            ch = code_hash(r.get("code", "x"))
            if ch in seen:
                continue
            seen.add(ch)
            uniq.append(r)
            if len(uniq) >= 5:
                break

        if not uniq:
            print("[RSI] No viable candidates; saving state.")
            state.code_stagnation_count += 1
            self.save_state(state)
            return

        # ---------------------------------------------------------------------
        # Robust acceptance gate: paired suite + CI + adversarial replay + OOD
        # ---------------------------------------------------------------------
        top_reports = []
        for r in uniq[:3]:
            rep = self._sequential_acceptance_eval(r["code"], state)
            # Score: prioritize statistical confidence + OOD + replay robustness
            score = float(rep.get("suite_lcb", -1e9)) + 0.5 * float(rep.get("ood_lcb", -1e9)) + 0.05 * float(rep.get("adv_impr", 0.0))
            top_reports.append((score, r, rep))

        top_reports.sort(key=lambda x: x[0], reverse=True)
        best_score, best_r, best_rep = top_reports[0]

        summ = best_rep["suite_summary"]
        suite_mean = float(summ["mean"])
        suite_pos = float(summ["pos_rate"])
        suite_lcb = float(best_rep["suite_lcb"])
        ood_lcb = float(best_rep["ood_lcb"])
        adv_impr = float(best_rep["adv_impr"])
        depth = int(best_rep["depth"])
        suite_n = int(best_rep["suite_n"])

        print(
            f"[EVAL] Best candidate: mean {suite_mean:.3f}% | LCB {suite_lcb:.3f}% | pos {suite_pos*100:.1f}% | "
            f"OOD LCB {ood_lcb:.3f}% | replay {adv_impr:.3f}% | depth {depth} | suite n={suite_n}"
        )

        cand_metrics = RSIMetrics(
            perf_delta=suite_mean,
            stability_succ=suite_pos * 100.0,
            complexity_depth=float(depth),
        )
        baseline_metrics = state.current_metrics

        accept_gate = (
            depth <= _ACCEPT_MAX_DEPTH
            and suite_mean >= _ACCEPT_MIN_MEAN_IMPROVEMENT
            and suite_pos >= _ACCEPT_MIN_POS_RATE
            and suite_lcb > 0.0
            and ood_lcb > -1.0
            and adv_impr > -5.0
        )

        accept = bool(accept_gate and self.check_pareto_dominance(cand_metrics, baseline_metrics))

        if accept:
            parent_hash = state.incumbent_hash or "bootstrap"
            rec = self._archive_candidate(state, best_r["code"], cand_metrics, best_rep, parent_hash)
            state.pareto_archive.append(rec)
            state.pareto_archive = state.pareto_archive[-200:]

            state.current_metrics = cand_metrics
            state.incumbent_code = best_r["code"]
            state.incumbent_hash = code_hash(best_r["code"])

            # Seed pool refresh
            state.evolved_seeds = [state.incumbent_code] + [s for s in state.evolved_seeds if s != state.incumbent_code]
            state.evolved_seeds = list(dict.fromkeys(state.evolved_seeds))[:30]

            state.code_stagnation_count = 0
            state.last_eval_score = suite_mean
            state.perf_fail_count = 0

            print(f"[RSI] ACCEPTED: deploying candidate (+{suite_mean:.3f}% mean paired improvement).")

            # Deploy (self-modify)
            self.inject(best_r["code"], cand_metrics, suite_mean)
        else:
            print("[RSI] REJECTED: candidate failed robustness/Pareto gate.")
            # Update stagnation tracking based on suite mean
            if suite_mean <= float(state.last_eval_score) + 0.05:
                state.code_stagnation_count += 1
            else:
                state.code_stagnation_count = 0
                state.last_eval_score = suite_mean

        # Optional: Trigger pure atomic GP when stagnated (legacy feature)
        if state.code_stagnation_count >= 5 and state.code_synth_active:
            print(f"\n[CODE SYNTHESIS] Stagnation ({state.code_stagnation_count}) detected. Running Pure Atomic GP...")
            try:
                is_restricted = (state.gp_fail_count >= 3)
                if is_restricted:
                    print("  > [GP] Restricted Mode Active (Simplification Strategy)")

                synth_rng = random.Random(state.total_seeds)
                best_algo = evolve_code_genome(synth_rng, gens=20, restricted=is_restricted)
                print(f"  > Synthesized Algo Fitness: {best_algo.fitness}")

                if best_algo.fitness <= 0.001:
                    state.gp_fail_count += 1
                else:
                    state.gp_fail_count = 0

                if best_algo.fitness >= 900.0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    with open(f"discovered_algo_{timestamp}.py", "w") as f:
                        f.write(best_algo.to_python(f"sort_{timestamp}"))
                    print("  > [GP] High-fitness algorithm saved.")
            except Exception as e:
                print(f"  > [GP] Synthesis error: {e}")

        self.save_state(state)
        print(f"[RSI] Round {state.round_idx} Complete.")

    def check_pareto_dominance(self, cand: RSIMetrics, base: RSIMetrics):
        # [FIX] Realistic Dominance with Robust Negative Baseline Handling
        
        def is_better(new_val, old_val, threshold=0.05):
            # If old_val is 0 or near 0, use absolute diff
            if abs(old_val) < 1e-9:
                return new_val > 1e-9 # Any positive is better
            
            # If old_val is negative (e.g. -100), new_val (-90) is better
            # (new - old) / abs(old) > threshold
            gain = (new_val - old_val)
            rel_gain = gain / (abs(old_val) + 1e-9)
            return rel_gain > threshold

        # Improvement check
        p_better = is_better(cand.perf_delta, base.perf_delta, 0.05)
        s_better = is_better(cand.stability_succ, base.stability_succ, 0.05)
        
        # Complexity check (Acceptable range 3-20 for atomic code)
        # Using 3-20 because atomic code depth is usually 4-8.
        c_acceptable = (3.0 <= cand.complexity_depth <= 20.0)
        c_better = cand.complexity_depth < base.complexity_depth
        
        # Dominance Logic
        if p_better and s_better and c_acceptable: return True
        if p_better and c_better and c_acceptable: return True
        
        # Strict fallback
        if p_better and s_better and c_better: return True
        
        # Huge gain override
        if cand.perf_delta > base.perf_delta + 20.0: return True

        return False

    def inject(self, code, metrics, delta):
        print(f"[INJECTOR] Deploying Improvement (+{delta:.2f}%)...")
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            txt = self.path.read_text('utf-8')
            c_s = txt.find('# @@EVOLVED_CODE_START@@')
            c_e = txt.find('# @@EVOLVED_CODE_END@@')
            evolved = f'# Evolved {datetime.now()} (Delta {delta:.2f}%)\ndef evolved_function(x):\n    return {code}'
            
            # Phase 3: Engine Modification Check
            # If the code seems to contain class definitions (Subject-Level RSI), handle differently
            if "class EvolutionEngine" in code:
                print("[INJECTOR] Detected ENGINE modification request.")
                if ReflectionSandbox.verify_engine_update(code):
                    # We need to replace the whole EvolutionEngine block
                    # This requires markers. For now, we just log success.
                    print("[INJECTOR] Engine update verified! (Markers not yet enforced for safety)")
                else:
                    print("[INJECTOR] Engine update REJECTED by Sandbox.")
                    return

            if c_s >= 0 and c_e >= 0 and c_s < c_e:
                txt = txt[:c_s] + '# @@EVOLVED_CODE_START@@\n' + evolved + '\n' + txt[c_e:]
            else:
                raise RuntimeError("MARKERS NOT FOUND or INVALID")
            
            tmp.write_text(txt, 'utf-8')
            py_compile.compile(str(tmp), doraise=True)
            shutil.move(str(tmp), str(self.path))
            print("[SUCCESS] Self-Modification Complete.")
        except Exception as e:
            print(f"[FAIL] Injection Rolled Back: {e}")
            if tmp.exists(): tmp.unlink()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_rounds", type=int, default=1)
    parser.add_argument("--validate", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed_base", type=int, default=42)
    parser.add_argument("--smoke", action='store_true')
    parser.add_argument("--resume", action='store_true')
    args = parser.parse_args()
    
    arch = NeuroArchitect(__file__)
    
    if not args.resume and not args.smoke:
        if Path("runs/rsi_state.json").exists(): Path("runs/rsi_state.json").unlink()

    if args.smoke:
        if Path("runs/rsi_state.json").exists(): Path("runs/rsi_state.json").unlink()
        arch.meta_round(5, 42, 2)
        arch.meta_round(5, 42, 2)
    else:
        for i in range(args.meta_rounds):
            arch.meta_round(args.validate, args.seed_base + i*1000, args.workers)
