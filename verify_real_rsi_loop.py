import shutil
import subprocess
import time
import os
import re
from pathlib import Path

# Config
ORIGINAL_SCRIPT = "L7_RSI_FINAL.py"
TEST_SCRIPT = "rsi_test_sandbox.py"
STATE_FILE = "runs/rsi_state.json"
MAX_ROUNDS = 5

def main():
    print("🚀 STARTING REAL RSI LOOP VERIFICATION")
    print("========================================")

    # 1. Sandbox setup
    if os.path.exists(TEST_SCRIPT): os.remove(TEST_SCRIPT)
    if os.path.exists(STATE_FILE): os.remove(STATE_FILE)
    
    print(f"[1] Copying {ORIGINAL_SCRIPT} -> {TEST_SCRIPT}...")
    shutil.copy(ORIGINAL_SCRIPT, TEST_SCRIPT)

    # 2. Patch to use 'simple_poly' task (x**2 + x) to guarantee fast evolution
    # The default in main is 'hybrid_hard'. We will override via command line args if possible,
    # OR we patch the default arg in the file if the script hardcodes it.
    # Looking at L7_RSI_FINAL.py, meta_round takes 'task_name'.
    # And __main__ calls arch.meta_round(..., 'hybrid_hard', ...) ?
    # Let's check __main__ block of L7_RSI_FINAL.py (lines 1276-1278 from previous views).
    # It calls: arch.meta_round(args.validate, args.seed_base + i*1000, args.workers) (It uses default task_name='hybrid_hard')
    
    # We MUST patch the __main__ block to pass 'simple_poly' OR patch the default in meta_round definition.
    # Let's patch the default value in the definition.
    print("[2] Patching task to 'simple_poly' for fast evolution...")
    content = Path(TEST_SCRIPT).read_text("utf-8")
    
    # Patch default arg: task_name: str = 'hybrid_hard' -> 'sinexp'
    # Use strict replace to ensure we only change the default
    new_content = content.replace("task_name: str = 'hybrid_hard'", "task_name: str = 'sinexp'")
    
    if content == new_content:
        print("    [WARNING] Could not patch default task name via string replace. Trying regex...")
        new_content = re.sub(r"task_name:\s*str\s*=\s*'hybrid_hard'", "task_name: str = 'sinexp'", content)
    
    if content == new_content:
         print("    [ERROR] Failed to patch task name. Evolution might be too slow for test.")
    else:
         print("    [OK] Task patched to 'sinexp'.")

    # [NEW] Patch target_function to be 'sinexp' so that suite/holdout/adv all use sinexp
    # Search for: def target_function(x):
    #                 ...
    #                 return 2.5 * (x ** 3) - np.sin(5 * x) + np.exp(0.1 * x)
    # Replace return line with: return np.sin(2*x) + np.exp(np.abs(x)/2)
    # [NEW] Patch target_function to be 'simple_poly' (x*x) for GUARANTEED FAST EVOLUTION
    # Search for: def target_function(x):
    #                 ...
    #                 return 2.5 * (x ** 3) - np.sin(5 * x) + np.exp(0.1 * x)
    # Replace return line with: return safe_sq(x)
    print("[2.b] Patching target_function to 'simple_poly' (x*x)...")
    content_v14 = Path(TEST_SCRIPT).read_text("utf-8")
    content_v14 = re.sub(
        r"return 2\.5 \* \(x \*\* 3\) - np\.sin\(5 \* x\) \+ np\.exp\(0\.1 \* x\)",
        r"return np.clip(x * x, -1e4, 1e4)",
        content_v14
    )
    # Also patch acceptable threshold to be lower to allow early winners
    content_v14 = content_v14.replace("_ACCEPT_MIN_MEAN_IMPROVEMENT = 0.5", "_ACCEPT_MIN_MEAN_IMPROVEMENT = 0.01")
    Path(TEST_SCRIPT).write_text(content_v14, "utf-8")

    # 3. Snapshot original 'Evolved' block
    original_evolved_match = re.search(r"# Evolved (.*?) \(Delta (.*?)%\)", content)
    original_ts = original_evolved_match.group(1) if original_evolved_match else "None"
    print(f"[3] Original Code Timestamp: {original_ts}")

    # 4. Run the Sandbox
    print(f"[4] Running {TEST_SCRIPT} for {MAX_ROUNDS} rounds (Timeout: 600s)...")
    cmd = ["python", TEST_SCRIPT, "--meta_rounds", str(MAX_ROUNDS), "--workers", "4", "--validate", "50"]
    
    start_time = time.time()
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Stream output
    success_detected = False
    try:
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(f"    [SUB] {line.strip()}")
                if "[INJECTOR] Deploying Improvement" in line:
                    print("    🎯 INJECTION EVENT DETECTED!")
                    success_detected = True
                if "[SUCCESS] Self-Modification Complete" in line:
                    print("    ✅ SELF-MODIFICATION CONFIRMED!")
            
            if time.time() - start_time > 600:
                print("    ⏰ TIMEOUT detected. Killing process.")
                process.kill()
                break
    except KeyboardInterrupt:
        process.kill()

    # 5. Verify File Change
    print("[5] Verifying Physical File Change...")
    final_content = Path(TEST_SCRIPT).read_text("utf-8")
    new_evolved_match = re.search(r"# Evolved (.*?) \(Delta (.*?)%\)", final_content)
    new_ts = new_evolved_match.group(1) if new_evolved_match else "None"
    
    print(f"    Original TS: {original_ts}")
    print(f"    New TS:      {new_ts}")

    if new_ts != original_ts and new_ts != "None":
        print("\n🏆 VERIFICATION SUCCESS: SYSTEM HAS EVOLVED!")
        print(f"    - Identity Shift: {original_ts} -> {new_ts}")
        print("    - Real code modification has occurred on disk.")
    else:
        print("\n❌ VERIFICATION FAILED: CODE DID NOT CHANGE.")
        print("    - Possible reasons: GP didn't find better solution, or injection disabled.")

    # Cleanup
    # os.remove(TEST_SCRIPT) # Keep for inspection
    # if os.path.exists(STATE_FILE): os.remove(STATE_FILE)

if __name__ == "__main__":
    main()
