# K-means Customer Segmentation

This project demonstrates customer segmentation using the K-means clustering algorithm on mall customer data. The analysis helps identify distinct customer groups based on their annual income and spending patterns.

## Project Overview

The project analyzes mall customer data to identify natural groupings of customers based on their:
- Annual Income
- Spending Score
- Age
- Gender

## Dataset

The dataset (`Mall_Customers.csv`) contains the following features:
- CustomerID: Unique identifier for each customer
- Gender: Customer's gender (Male/Female)
- Age: Customer's age
- Annual Income (k$): Customer's annual income in thousands of dollars
- Spending Score (1-100): Score assigned to customer based on spending behavior

## Setup

1. Create a virtual environment:

```bash
python -m venv .venv
```

2. Activate the virtual environment:

```bash
source .venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Project Structure
```
kmeans-customer-segmentation/
├── data
│   └── Mall_Customers.csv
├── plots
│   ├── age_income_distributions.png
│   ├── cluster_scatter_age_spending_score.png
│   ├── cluster_scatter_annual_income_spending_score.png
│   ├── correlation_heatmap.png
│   ├── elbow_curve.png
│   ├── gender_averages.png
│   └── gender_pie.png
├── src
│   ├── __init__.py
│   ├── clustering.py
│   ├── data_analysis.py
│   └── utils.py
├── README.md
└── requirements.txt
```

## Generated Visualizations

The project creates several visualizations:
- Elbow curve for optimal cluster selection
- Customer segments scatter plot
- Age and income distributions
- Gender-based analysis
- Correlation heatmap

## Dependencies

- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

## Usage

Run the analysis script from the project root:
```bash
python src/data_analysis.py
```
## Results
The analysis segments customers into distinct groups based on their characteristics, helping to:
- Identify customer patterns
- Target marketing strategies
- Optimize resource allocation
- Improve customer service

The visualizations can be found in the `plots/` directory after running the analysis.
