

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr

def evaluate_probabilistic_model(model, loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    trues, preds, stds, nlls = [], [], [], []

    print("Running Evaluation with Correlation Analysis...")

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            mean, log_var = model(x)
            sigma = torch.exp(0.5 * log_var)

            # NLL Calculation
            var = torch.exp(log_var)
            nll = 0.5 * (log_var + (y - mean)**2 / var).mean()

            trues.extend(y.cpu().numpy())
            preds.extend(mean.cpu().numpy())
            stds.extend(sigma.cpu().numpy())
            nlls.append(nll.item())

    trues = np.array(trues)
    preds = np.array(preds)
    stds = np.array(stds)

    # 1. Calculate Absolute Errors
    errors = np.abs(trues - preds)

    # 2. Compute Correlation (The "Self-Awareness" Score)
    # We use Spearman because the relationship is monotonic but likely not linear
    corr, _ = spearmanr(stds, errors)

    # Standard Metrics
    mae = mean_absolute_error(trues, preds)
    r2 = r2_score(trues, preds)
    lower = preds - 1.96 * stds
    upper = preds + 1.96 * stds
    inside_bounds = (trues >= lower) & (trues <= upper)
    picp = inside_bounds.mean()
    mpiw = np.mean(upper - lower)

    print(f"\n{'='*40}")
    print(f"HYBRID MODEL REPORT")
    print(f"{'='*40}")
    print(f"1. PERFORMANCE")
    print(f"   MAE (Error):      {mae:.4f}")
    print(f"   R2 Score:         {r2:.4f}")
    print(f"\n2. SAFETY & AWARENESS")
    print(f"   PICP (Target 0.95): {picp:.2%}")
    print(f"   MPIW (Avg Width):   {mpiw:.2f}")
    print(f"   Error-Uncertainty Correlation: {corr:.4f}")

    if corr > 0.4:
        print("   ✅ Strong Awareness: Model knows when it's wrong.")
    elif corr > 0.1:
        print("   ⚠️ Weak Awareness: Model guesses somewhat randomly.")
    else:
        print("   ❌ No Awareness: Uncertainty is uncorrelated with error.")
    print(f"{'='*40}")

    # PLOTTING
    plt.figure(figsize=(14, 5))

    # Plot 1: Prediction vs Truth
    sorted_idx = np.argsort(trues)
    plt.subplot(1, 2, 1)
    plt.plot(trues[sorted_idx], 'k-', linewidth=2, label='True')
    plt.plot(preds[sorted_idx], 'b--', alpha=0.7, label='Pred')
    plt.fill_between(range(len(trues)), lower[sorted_idx], upper[sorted_idx], color='blue', alpha=0.2)
    plt.title(f"Predictions (PICP: {picp:.0%})")
    plt.legend()

    # Plot 2: The Correlation Plot
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=stds, y=errors, alpha=0.6)

    # Simple linear fit for visualization
    m, b = np.polyfit(stds, errors, 1)
    plt.plot(stds, m*stds + b, 'r--', label=f'Corr: {corr:.2f}')

    plt.title("Do Errors trigger High Uncertainty?")
    plt.xlabel("Predicted Uncertainty (Sigma)")
    plt.ylabel("Absolute Error")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout
