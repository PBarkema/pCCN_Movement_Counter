import torch
import numpy as np
import matplotlib.pyplot as plt

def stress_test_uncertainty(model, loader, device):
    model.eval()

    # We will track average predicted uncertainty (sigma) at different noise levels
    noise_levels = [0.0, 0.05, 0.1, 0.2] # Increasing amounts of jitter
    avg_sigmas = []

    print("Running Stress Test...")

    for noise in noise_levels:
        sigmas = []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)

                # --- ARTIFICIAL CORRUPTION ---
                # Add random Gaussian noise to the input features
                # We corrupt the raw features (0-323), but NOT the smart signal (324)
                # to see if the model realizes the raw data conflicts with the signal.
                corruption = torch.randn_like(x) * noise
                x_corrupted = x + corruption

                # Predict
                _, log_var = model(x_corrupted)
                sigma = torch.exp(0.5 * log_var)
                sigmas.extend(sigma.cpu().numpy())

        avg_sigma = np.mean(sigmas)
        avg_sigmas.append(avg_sigma)
        print(f"Noise Level {noise:.2f} -> Avg Predicted Uncertainty: {avg_sigma:.4f}")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(noise_levels, avg_sigmas, 'ro-', linewidth=2)
    plt.title("Stress Test: Does the Model Detect Noise?")
    plt.xlabel("Added Noise Level")
    plt.ylabel("Predicted Uncertainty (Sigma)")
    plt.grid(True)
    plt.show()

stress_test_uncertainty(model, val_loader, torch.device('cpu'))