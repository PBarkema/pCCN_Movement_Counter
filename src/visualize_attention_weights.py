import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np

def visualize_attention_robust(model, val_loader, device):
    model.eval()
    attention_weights = []

    # 1. Define the hook
    def hook_fn(module, input, output):
        # output shape: (Batch, Channels, 1) or (Batch, Channels)
        weights = output.detach().cpu().numpy()
        # Average across the batch
        weights = weights.mean(axis=0).flatten()
        attention_weights.append(weights)

    # 2. Find the Sigmoid layer dynamically
    # We look inside model.attention for the Sigmoid layer
    target_layer = None
    for name, module in model.attention.named_modules():
        if isinstance(module, nn.Sigmoid):
            target_layer = module
            break

    if target_layer is None:
        print("Error: Could not find a Sigmoid layer in model.attention!")
        return

    # 3. Register the hook on the found layer
    handle = target_layer.register_forward_hook(hook_fn)
    print(f"Hook attached to: {target_layer}")

    # 4. Run inference
    print("Extracting attention weights...")
    with torch.no_grad():
        for i, (x, _) in enumerate(val_loader):
            x = x.to(device)
            _ = model(x)
            if i >= 5: break # Only need a few batches

    # 5. Cleanup
    handle.remove()

    # 6. Plotting
    if len(attention_weights) > 0:
        avg_weights = np.mean(attention_weights, axis=0)

        plt.figure(figsize=(12, 6))
        plt.bar(range(len(avg_weights)), avg_weights, color='teal', alpha=0.7)
        plt.title("Feature Importance (Channel Attention Weights)")
        plt.xlabel(f"Feature Index (0-{len(avg_weights)-1})")
        plt.ylabel("Importance (0=Ignore, 1=Focus)")

        # Highlight top features
        top_indices = avg_weights.argsort()[-5:][::-1]
        print("\nTop 5 Most Important Features:")
        for i in top_indices:
            print(f"Feature {i}: Score {avg_weights[i]:.4f}")
            plt.text(i, avg_weights[i], f"{i}", ha='center', va='bottom', fontweight='bold')

        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        print("No weights captured. Check if the model is running correctly.")

# --- RUN THE FIX ---
visualize_attention_robust(model, val_loader, device)