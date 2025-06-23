import matplotlib.pyplot as plt
import os

# Create plots directory if it doesn't exist
plots_dir = os.path.join('..', 'plots')
os.makedirs(plots_dir, exist_ok=True)

def save_plot(plt, filename):
    """Save a matplotlib plot to the plots directory."""
    plt.savefig(os.path.join(plots_dir, filename))
    plt.close()