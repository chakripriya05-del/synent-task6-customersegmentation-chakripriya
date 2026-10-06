"""Synent Task 6 - Customer Segmentation (Mall Customer Dataset) with K-Means
Run: python customer_segmentation.py
Put the Kaggle file at data/Mall_Customers.csv for real results. If it is missing,
a SYNTHETIC dataset with the same schema is generated so the pipeline can be tested.
"""
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

sns.set_theme(style="whitegrid")
PATH = "data/Mall_Customers.csv"

def synthetic(n=200, seed=42):
    rng = np.random.default_rng(seed)
    # (income mean, spend mean, age mean, share) - typical mall-customer structure
    groups = [(26, 20, 45, .21), (26, 78, 25, .12), (55, 50, 42, .24), (88, 17, 40, .21), (88, 82, 32, .22)]
    rows = []
    for inc, sp, age, share in groups:
        for _ in range(int(n * share)):
            rows.append([np.clip(rng.normal(inc, 8), 15, 137), np.clip(rng.normal(sp, 9), 1, 99),
                         int(np.clip(rng.normal(age, 8), 18, 70))])
    d = pd.DataFrame(rows, columns=["Annual Income (k$)", "Spending Score (1-100)", "Age"]).round(0)
    d.insert(0, "CustomerID", range(1, len(d) + 1))
    d.insert(1, "Genre", rng.choice(["Male", "Female"], len(d), p=[.44, .56]))
    return d[["CustomerID", "Genre", "Age", "Annual Income (k$)", "Spending Score (1-100)"]]

if os.path.exists(PATH):
    df, SOURCE = pd.read_csv(PATH), "REAL Kaggle data"
else:
    df, SOURCE = synthetic(), "SYNTHETIC data (Kaggle CSV not found)"
    df.to_csv("data/Mall_Customers_synthetic.csv", index=False)
print("Data source:", SOURCE, "| shape:", df.shape)

# 1. Preprocessing ---------------------------------------------------------
df = df.drop_duplicates().rename(columns={"Genre": "gender", "Gender": "gender", "Annual Income (k$)": "income",
        "Spending Score (1-100)": "spending", "Age": "age", "CustomerID": "id"})
print("Missing values:\n", df.isna().sum().to_string())
X = df[["income", "spending"]]                     # behaviour features
Xs = StandardScaler().fit_transform(X)

# 2. Choose k: elbow + silhouette -----------------------------------------
ks = range(2, 11); inertia, sil = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
    inertia.append(km.inertia_); sil.append(silhouette_score(Xs, km.labels_))
best_k = list(ks)[int(np.argmax(sil))]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(ks, inertia, "o-"); ax[0].set(title="Elbow Method", xlabel="k", ylabel="Inertia")
ax[1].plot(ks, sil, "o-", color="green"); ax[1].axvline(best_k, ls="--", color="red")
ax[1].set(title=f"Silhouette Score (best k={best_k})", xlabel="k", ylabel="Score")
plt.tight_layout(); plt.savefig("plots/01_elbow_silhouette.png", dpi=150); plt.close()

# 3. Final model -----------------------------------------------------------
km = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(Xs)
df["cluster"] = km.labels_
print(f"k={best_k}, silhouette={silhouette_score(Xs, km.labels_):.3f}")

# 4. Visualise -------------------------------------------------------------
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="income", y="spending", hue="cluster", palette="tab10", s=70)
cent = StandardScaler().fit(X).inverse_transform(km.cluster_centers_)
plt.scatter(cent[:, 0], cent[:, 1], c="black", marker="X", s=250, label="Centroids")
plt.title("Customer Segments: Income vs Spending"); plt.legend(); plt.tight_layout()
plt.savefig("plots/02_clusters.png", dpi=150); plt.close()

# 5. Profile & label the segments -----------------------------------------
prof = df.groupby("cluster").agg(customers=("id", "count"), avg_age=("age", "mean"),
        avg_income=("income", "mean"), avg_spending=("spending", "mean"),
        pct_female=("gender", lambda s: (s == "Female").mean() * 100)).round(1)
mi, ms = df["income"].median(), df["spending"].median()
si, ss = df["income"].std(), df["spending"].std()
def label(r):
    di, ds = (r.avg_income - mi) / si, (r.avg_spending - ms) / ss
    if abs(di) < 0.5 and abs(ds) < 0.5: return "Average - mid income, mid spend"
    hi, hs = di > 0, ds > 0
    return {(1, 1): "Premium - high income, high spend", (1, 0): "Cautious - high income, low spend",
            (0, 1): "Impulsive - low income, high spend", (0, 0): "Budget - low income, low spend"}[(hi, hs)]
prof["segment"] = prof.apply(label, axis=1)
print("\nSegment profile:\n", prof.to_string())
prof.to_csv("segment_profile.csv")
df.to_csv("data/customers_with_clusters.csv", index=False)

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for a, c in zip(ax, ["age", "income", "spending"]):
    sns.boxplot(data=df, x="cluster", y=c, hue="cluster", palette="tab10", legend=False, ax=a); a.set_title(f"{c} by cluster")
plt.tight_layout(); plt.savefig("plots/03_cluster_profiles.png", dpi=150); plt.close()
