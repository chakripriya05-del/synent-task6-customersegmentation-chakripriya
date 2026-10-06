# Synent Task 6 – Customer Segmentation (K-Means)

## Problem Statement
Group mall customers by behaviour (annual income and spending score) so the business can target each group differently.

## Dataset
Mall Customer Dataset (Kaggle): 200 customers, 5 columns – CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100). No missing values or duplicates. File: `data/Mall_Customers.csv`.

## Approach
1. Preprocessing: check duplicates and missing values, rename columns, standardize income and spending score.
2. Choose k using the elbow method and silhouette score (tested k = 2 to 10).
3. Fit K-Means with the best k (5).
4. Visualize the clusters with centroids and profile each cluster (size, age, income, spending, gender mix).

Run: `pip install -r requirements.txt && python customer_segmentation.py`
Outputs: `plots/`, `segment_profile.csv`, `data/customers_with_clusters.csv`.

## Results
K-Means with **k = 5** gave a silhouette score of **0.555**.

| Segment | Customers | Avg age | Avg income (k$) | Avg spending |
|---|---|---|---|---|
| Average (mid income, mid spend) | 81 | 42.7 | 55.3 | 49.5 |
| Premium (high income, high spend) | 39 | 32.7 | 86.5 | 82.1 |
| Impulsive (low income, high spend) | 22 | 25.3 | 25.7 | 79.4 |
| Cautious (high income, low spend) | 35 | 41.1 | 88.2 | 17.1 |
| Budget (low income, low spend) | 23 | 45.2 | 26.3 | 20.9 |

## Business Insights
- **Premium (39):** younger, high income and high spending. Best targets for loyalty programs and new-arrival previews.
- **Cautious (35):** earn as much as Premium but spend very little (17 vs 82). The biggest untapped opportunity, so try personalized offers.
- **Impulsive (22):** the youngest group, low income but high spending. Respond well to promotions and flash sales.
- **Average (81):** the largest group (40% of customers) with moderate income and spend. Stable base worth retaining.
- **Budget (23):** older, low income, low spend. Keep costs low; discounts may help.
- Income alone does not predict spending: both high-income segments differ hugely in spend.

## Author
Chakri Priya
