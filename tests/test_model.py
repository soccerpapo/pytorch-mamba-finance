import sys
import os
import torch
import torch.optim as optim
import numpy as np
import time

# --- PATH SETUP ---
# Add 'src' to path so we can import from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

try:
    from model import DifferentiableTrader
    # 👇 NEW: Import the generator instead of redefining it
    from data import generate_synthetic_data 
except ImportError:
    print("❌ Critical Error: Could not import modules from 'src/'.")
    sys.exit(1)

# --- CONFIGURATION ---
ROUNDS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_MODEL = 32

def run_stability_test():
    print(f"🔥 STARTING MODEL STABILITY TEST (Device: {DEVICE})")
    print("="*50)
    
    passing_rounds = 0
    results = []

    # 1. Get Data from src/data.py
    # This returns a Numpy array, so we must convert it to a Tensor here.
    raw_data = generate_synthetic_data(n_samples=1000)
    
    # Convert to Tensor [Batch=1, Seq_Len=1000, Features=2]
    input_tensor = torch.tensor(raw_data, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    for i in range(ROUNDS):
        print(f"\n--- Round {i+1}/{ROUNDS} ---")
        
        try:
            # A. Initialize Model (Check for weight initialization errors)
            start_time = time.time()
            model = DifferentiableTrader(input_dim=2, d_model=D_MODEL).to(DEVICE)
            
            # B. Forward Pass (Check for shape mismatches)
            output = model(input_tensor)
            
            # C. Check Output Shape
            if output.shape[-1] != 1:
                raise ValueError(f"Output shape mismatch. Got {output.shape}, expected ending in 1.")
            
            # D. Mock Optimization Step (Check for backward pass errors)
            optimizer = optim.AdamW(model.parameters(), lr=0.001)
            loss = output.mean()
            loss.backward()
            optimizer.step()
            
            elapsed = time.time() - start_time
            print(f"   ✅ Round {i+1} Passed ({elapsed:.4f}s)")
            passing_rounds += 1
            results.append(elapsed)

        except Exception as e:
            print(f"   ❌ Round {i+1} FAILED: {str(e)}")

    print("\n" + "="*50)
    if passing_rounds == ROUNDS:
        print(f"🏆 TEST PASSED: {passing_rounds}/{ROUNDS} rounds successful.")
        print(f"   Avg Init+Forward Time: {np.mean(results):.4f}s")
        sys.exit(0)
    else:
        print(f"💥 TEST FAILED: Only {passing_rounds}/{ROUNDS} rounds successful.")
        sys.exit(1)

if __name__ == "__main__":
    run_stability_test()
    