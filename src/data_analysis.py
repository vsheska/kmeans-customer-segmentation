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


def plot_gender_averages(df):
    # Calculate means for each gender
    gender_means = df.groupby('Gender')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    gender_means.plot(kind='bar', width=0.8)

    # Customize the plot
    plt.title('Average Metrics by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Average Value')
    plt.legend(title='Metrics', loc='upper right')
    plt.ylim(0, 100)

    # Add value labels on top of each bar
    x_offset = -0.3  # Starting position for the first bar in each group
    for i in range(len(gender_means.columns)):
        for j in range(len(gender_means.index)):
            value = gender_means.iloc[j, i]
            x_pos = j + x_offset + (i * 0.3)  # Adjust x position for each bar
            plt.text(x_pos, value, f'{value:.1f}',
                     horizontalalignment='center',
                     verticalalignment='bottom')

    # Adjust layout and save
    plt.tight_layout()
    save_plot(plt, 'gender_averages.png')


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

    plot_gender_averages(df)


    return df

if __name__ == "__main__":
    df = analyze_data()
