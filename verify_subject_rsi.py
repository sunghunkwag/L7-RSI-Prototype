import logging
import sys
from pathlib import Path
import numpy as np

# Setup proper paths to import L7_RSI_FINAL
sys.path.append(str(Path(".").resolve()))

from L7_RSI_FINAL import (
    RSIState, RSIMetrics, L1Policy, StateVector, PolicyEvaluator,
    _META_GRAMMAR, _TRANSFER_MEMORY, KnowledgeShard, ReflectionSandbox
)

def test_symbolic_policy():
    print("\n[TEST] Symbolic Policy Evolution")
    
    # 1. Setup State
    l1 = L1Policy({}, {}, 0.5, 0.5, 5, 1, 0.001, {
        "mutation_rate": "0.1 + 0.5 * ST" # Simple rule: Mut = 0.1 + 0.5 * Stagnation
    })
    
    # Case A: Low Stagnation (S=0.0) -> Mut should be 0.1
    state = RSIState.default()
    state.l1_policy = l1
    state.code_stagnation_count = 0 
    cand = RSIMetrics(0,0,0)
    
    s_vec = StateVector.from_state(state, cand)
    val = PolicyEvaluator.evaluate(l1.dynamic_rules["mutation_rate"], s_vec)
    print(f"  > Rule: {l1.dynamic_rules['mutation_rate']}")
    print(f"  > [Stagnation=0] Calcluated Mut: {val} (Expected ~0.1)")
    
    if abs(val - 0.1) < 0.01: print("  > PASS")
    else: print("  > FAIL")

    # Case B: High Stagnation (S=10 => ST=1.0) -> Mut should be 0.6
    state.code_stagnation_count = 10
    s_vec = StateVector.from_state(state, cand)
    val = PolicyEvaluator.evaluate(l1.dynamic_rules["mutation_rate"], s_vec)
    print(f"  > [Stagnation=10] Calcluated Mut: {val} (Expected ~0.6)")
    
    if abs(val - 0.6) < 0.01: print("  > PASS")
    else: print("  > FAIL")

def test_grammar_expansion():
    print("\n[TEST] Dynamic Grammar Expansion")
    
    # Inject high usage pattern
    shard = KnowledgeShard("test", "temp = arr[i]", 1.0, ["swap"], uses=10)
    _TRANSFER_MEMORY.long_term_memory.append(shard)
    
    # Check promotion
    promoted = _TRANSFER_MEMORY.promote_to_grammar()
    print(f"  > Promoted: {promoted}")
    
    if len(promoted) > 0 and "_macro_" in promoted[0][0]:
        print("  > PASS: Macro identified")
    else:
        print("  > FAIL: No promotion")

def test_reflection_sandbox():
    print("\n[TEST] Reflective Sandbox")
    
    # 1. Valid Engine
    valid_code = """
class EvolutionEngine:
    def __init__(self, evaluator, policy):
        self.evaluator = evaluator
    def initialize(self, seeds): pass
    def run(self, generations): 
        return None # Simplified return for test
"""
    print("  > Testing Valid Code...")
    res = ReflectionSandbox.verify_engine_update(valid_code) # Should fail logically in run() test because my mock checks return
    # Wait, my sandbox code checks "if best is None: FAIL". 
    # So the above code will actually FAIL because it returns None.
    # Let's write code that passes the sandbox check.
    
    valid_passing_code = """
class MockGenome: # Explicit Rename
    def __init__(self, lines=None): 
        self.fitness = 100.0
        self.lines = lines or []

class EvolutionEngine:
    def __init__(self, evaluator, policy):
        pass
    def initialize(self, seeds): pass
    def run(self, generations): 
        return MockGenome()
"""
    res = ReflectionSandbox.verify_engine_update(valid_passing_code)
    if res: print("  > PASS (Valid code accepted)")
    else: print("  > FAIL (Valid code rejected)")
    
    # 2. Invalid Code (Syntax Error)
    print("  > Testing Syntax Error...")
    invalid_syntax = "class EvolutionEngine: def broken("
    res = ReflectionSandbox.verify_engine_update(invalid_syntax)
    if not res: print("  > PASS (Invalid code rejected)")
    else: print("  > FAIL (Invalid code accepted)")
    
    # 3. Malicious/Broken Logic (Crash)
    print("  > Testing Crash Code...")
    crash_code = """
class EvolutionEngine:
    def __init__(self, e, p): pass
    def initialize(self, s): pass
    def run(self, g): raise RuntimeError("Die")
"""
    res = ReflectionSandbox.verify_engine_update(crash_code)
    if not res: print("  > PASS (Crash code rejected)")
    else: print("  > FAIL (Crash code accepted)")


if __name__ == "__main__":
    test_symbolic_policy()
    test_grammar_expansion()
    test_reflection_sandbox()
