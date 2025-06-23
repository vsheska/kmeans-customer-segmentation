import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_analyze_data(csv_path='../data/Mall_Customers.csv' ):
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
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.show()

    return df

if __name__ == "__main__":
    df = load_and_analyze_data()
