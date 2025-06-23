import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create plots directory if it doesn't exist
plots_dir = os.path.join('..', 'plots')
os.makedirs(plots_dir, exist_ok=True)

def save_plot(plt, filename):
    """Save a matplotlib plot to the plots directory."""
    plt.savefig(os.path.join(plots_dir, filename))
    plt.close()

def analyze_data(csv_path='../data/Mall_Customers.csv' ):
    # Read the data
    df = pd.read_csv(csv_path)

    # Basic summary statistics
    print("\n=== Basic Summary Statistics ===")
    print(df.describe())

    # Check data info (dtypes and null values)
    print("\n=== Data Info ===")
    print(df.info())

    # Distribution of categorical variables
    print("\n=== Gender Distribution ===")
    print(df['Gender'].value_counts())

    # Correlation matrix for numerical columns
    numerical_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    correlation = df[numerical_cols].corr()
    print("\n=== Correlation Matrix ===")
    print(correlation)

    # Create correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    save_plot(plt, 'correlation_heatmap.png')

    return df

if __name__ == "__main__":
    df = analyze_data()
