import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from clustering import find_optimal_k, perform_kmeans, predict_clusters, plot_elbow_curve
from utils import save_plot


def plot_gender_averages(df):
    # Calculate means for each gender
    gender_means = df.groupby('gender')[['age', 'annual_income', 'spending_score']].mean()

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

def plot_gender_pie(df):
    # Calculate gender distribution
    gender_counts = df['gender'].value_counts()

    # Create pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%',
            colors=['tab:blue', 'tab:orange'], startangle=90)

    # Add title
    plt.title('Gender Distribution')

    # Save plot
    save_plot(plt, 'gender_pie.png')

def plot_distributions(df):
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    # Age distribution
    ax1.hist(df['age'], bins=10, edgecolor='black', color='tab:blue')
    ax1.set_title('Age Distribution')
    ax1.set_xlabel('Age')
    ax1.set_ylabel('Frequency')

    # Annual Income distribution
    ax2.hist(df['annual_income'], bins=10, edgecolor='black', color='tab:orange')
    ax2.set_title('Annual Income Distribution')
    ax2.set_xlabel('Annual Income (k$)')
    ax2.set_ylabel('Frequency')

    # Adjust layout and save
    plt.tight_layout()
    save_plot(plt, 'age_income_distributions.png')

def display_cluster_info(df, labels):
    # Add cluster labels to the dataframe
    df['cluster'] = labels

    # Print cluster summary
    print("\n=== Cluster Analysis ===")
    print(f"Number of clusters: {len(set(labels))}")
    print("\nCluster Sizes:")
    cluster_sizes = df['cluster'].value_counts().sort_index()
    for cluster, size in cluster_sizes.items():
        print(f"Cluster {cluster}: {size} customers ({size / len(df) * 100:.1f}%)")

    # Calculate and print cluster characteristics
    print("\nCluster Characteristics (Mean Values):")
    cluster_means = df.groupby('cluster')[['age', 'annual_income', 'spending_score']].mean()

    # Format and display the characteristics
    for cluster in cluster_means.index:
        print(f"\nCluster {cluster}:")
        print(f"  Age: {cluster_means.loc[cluster, 'age']:.1f}")
        print(f"  Annual Income: ${cluster_means.loc[cluster, 'annual_income']:.1f}k")
        print(f"  Spending Score: {cluster_means.loc[cluster, 'spending_score']:.1f}/100")


def analyze_data(csv_path='../data/Mall_Customers.csv' ):
    # Read the data
    df = pd.read_csv(csv_path, header=0,
                     names=['id', 'gender', 'age', 'annual_income', 'spending_score'],
                     index_col='id')

    # Basic summary statistics
    print("\n=== Basic Summary Statistics ===")
    print(df.describe())

    # Check data info (dtypes and null values)
    print("\n=== Data Info ===")
    print(df.info())

    # Distribution of categorical variables
    print("\n=== Gender Distribution ===")
    print(df['gender'].value_counts())

    # Correlation matrix for numerical columns
    numerical_cols = ['age', 'annual_income', 'spending_score']
    correlation = df[numerical_cols].corr()
    print("\n=== Correlation Matrix ===")
    print(correlation)

    # Create correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    save_plot(plt, 'correlation_heatmap.png')

    plot_gender_averages(df)
    plot_gender_pie(df)
    plot_distributions(df)

    k_range = range(1, 11)
    inertias = find_optimal_k(df, numerical_cols, k_range)

    # Plot elbow curve
    plot_elbow_curve(k_range, inertias)

    print("\n=== Elbow Method Results ===")
    for k, inertia in zip(k_range, inertias):
        print(f"k={k}: inertia={inertia:.2f}")

    optimal_k = 5  # Based on data from the elbow curve
    pipeline, inertia = perform_kmeans(df, numerical_cols, n_clusters=optimal_k)
    labels = predict_clusters(df, pipeline)

    # Display clustering results
    display_cluster_info(df, labels)

    return df

if __name__ == "__main__":
    df = analyze_data()
