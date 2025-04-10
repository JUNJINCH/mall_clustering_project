import seaborn as sns
import matplotlib.pyplot as plt

def plot_clusters_with_new(df, new_income, new_score, prediction):
    fig, ax = plt.subplots()
    sns.scatterplot(
        x="Annual_Income",
        y="Spending_Score",
        hue="Cluster",
        data=df,
        palette="Set2",
        alpha=0.6,
        ax=ax
    )
    ax.scatter(new_income, new_score, c='black', s=100, edgecolor='white', label="New Customer")
    ax.legend()
    ax.set_title("Customer Segments")
    ax.set_xlabel("Annual Income (in 1000s)")
    ax.set_ylabel("Spending Score")
    return fig