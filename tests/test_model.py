import sys
import os
import torch
import torch.optim as optim
import numpy as np
import time
import matplotlib.pyplot as plt

# --- UNIVERSAL PATH SETUP (Colab & Local) ---
try:
    # Case 1: Running as a script (!python tests/test_model.py)
    current_file = os.path.abspath(__file__)
    test_dir = os.path.dirname(current_file)      # .../project/tests
    project_root = os.path.dirname(test_dir)      # .../project
except NameError:
    # Case 2: Running in a Jupyter/Colab Cell
    # Assumes your notebooks are at the root or you mount the drive correctly
    print("⚠️ Detected Interactive Mode (Colab Cell). Assuming Root Path.")
    test_dir = os.path.join(os.getcwd(), 'tests')
    project_root = os.getcwd()

# Add project root to system path so we can import src
if project_root not in sys.path:
    sys.path.append(project_root)

# Debug Print
print(f"📂 Project Root: {project_root}")
print(f"📂 Src Path:    {os.path.join(project_root, 'src')}")

try:
    from src.model import DifferentiableTrader
except ImportError as e:
    print(f"❌ Critical Error: {e}")
    print("   -> Did you create the 'src' folder and 'model.py'?")
    sys.exit(1)

# --- CONFIGURATION ---
ROUNDS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_MODEL = 32  
INPUT_DIM = 6 # Retail V1.1 Input

def run_stability_test():
    print(f"🔥 STARTING MODEL STABILITY TEST (Device: {DEVICE})")
    print(f"   (Input Dim: {INPUT_DIM} | D_Model: {D_MODEL})")
    print("="*50)
    
    passing_rounds = 0
    results = []
    
    seq_len = 200
    input_tensor = torch.randn(1, seq_len, INPUT_DIM).to(DEVICE)

    for i in range(ROUNDS):
        print(f"\n--- Round {i+1}/{ROUNDS} ---")
        try:
            start_time = time.time()
            model = DifferentiableTrader(input_dim=INPUT_DIM, d_model=D_MODEL).to(DEVICE)
            output = model(input_tensor)
            
            if output.shape[-1] != 1:
                raise ValueError(f"Output shape mismatch. Got {output.shape}")
            
            optimizer = optim.AdamW(model.parameters(), lr=0.001)
            loss = output.mean()
            loss.backward()
            optimizer.step()
            
            elapsed = time.time() - start_time
            print(f"   ✅ Round {i+1} Passed ({elapsed:.4f}s)")
            passing_rounds += 1
            results.append(elapsed)

            if i == ROUNDS - 1:
                print("   📊 Generating Debug Plot...")
                signal = output.detach().cpu().numpy().squeeze()
                
                plt.figure(figsize=(10, 4))
                plt.plot(signal, label='Model Output (Tanh)', color='cyan')
                plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
                plt.ylim(-1.1, 1.1)
                plt.title(f"Model Reaction to Noise (Round {i+1})")
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Save to test_dir (so it stays organized)
                if not os.path.exists(test_dir): os.makedirs(test_dir)
                save_path = os.path.join(test_dir, "test_debug_signal.png")
                plt.savefig(save_path)
                print(f"      Saved to '{save_path}'")

        except Exception as e:
            print(f"   ❌ Round {i+1} FAILED: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*50)
    if passing_rounds == ROUNDS:
        print(f"🏆 TEST PASSED: {passing_rounds}/{ROUNDS} rounds successful.")
        sys.exit(0)
    else:
        print(f"💥 TEST FAILED: Only {passing_rounds}/{ROUNDS} rounds successful.")
        sys.exit(1)

if __name__ == "__main__":
    run_stability_test()
