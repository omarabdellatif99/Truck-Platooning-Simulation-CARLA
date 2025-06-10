import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from itertools import combinations
from collections import Counter
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import time 
start_time = time.time()
# Load data
my_data = pd.read_csv(r"C:\Users\ImmersiveRealityLab\Desktop\Apriori\Groceries_dataset.csv\Groceries_dataset.csv")

# Inspecting dataset
print(my_data.size)
print(my_data.columns)
print(my_data.head())
print(my_data.tail())
print("\nStatistical Summary:")
print(my_data.describe())
print("----- Top 15 Most Selling Items -----")
print(my_data.itemDescription.value_counts().head(15))
print("----- Bottom 15 Least Selling Items -----")
print(my_data.itemDescription.value_counts().tail(15).sort_values())

# Top 15 sold items bar chart
Item_distr = my_data.groupby(by="itemDescription").size().reset_index(name='Frequency')
Item_distr = Item_distr.sort_values(by='Frequency', ascending=False).head(15)
bars = Item_distr["itemDescription"]
height = Item_distr["Frequency"]
x_pos = np.arange(len(bars))

plt.figure(figsize=(14, 7))
plt.style.use('fivethirtyeight')
plt.bar(x_pos, height, color="#4682B2", edgecolor="black", linewidth=1.5, alpha=0.8)
plt.title("Top 15 Sold Items", fontsize=18, fontweight='bold', color="#333333")
plt.xlabel("Item Name", fontsize=14, fontweight='bold', color="#555555")
plt.ylabel("Number of Quantity Sold", fontsize=14, fontweight='bold', color="#555555")
plt.xticks(x_pos, bars, rotation=30, ha='right', fontsize=12, color="#444444")
for i, v in enumerate(height):
    plt.text(i, v + 2, str(v), ha='center', fontsize=15, fontweight='bold', color="#222222")
plt.grid(axis='y', linestyle='--', alpha=0.8)
plt.tight_layout()
plt.show()

# ------------------------------------------
# Heatmap of Co-occurrence for Top 15 Items
# ------------------------------------------

basket_data = my_data.groupby("Member_number")["itemDescription"].apply(list)
item_pairs = []
for items in basket_data:
    item_pairs.extend(combinations(items, 2))

pair_counts = Counter(item_pairs)
item_counts = Counter(my_data["itemDescription"])

n = 15
top_n_items = [item for item, _ in item_counts.most_common(n)]

co_occurrence_matrix = pd.DataFrame(index=top_n_items, columns=top_n_items).fillna(0)
for (item1, item2), count in pair_counts.items():
    if item1 in top_n_items and item2 in top_n_items:
        co_occurrence_matrix.loc[item1, item2] = count
        co_occurrence_matrix.loc[item2, item1] = count

co_occurrence_matrix = co_occurrence_matrix.astype("int64")

plt.figure(figsize=(10, 10))
sns.heatmap(co_occurrence_matrix, annot=True, fmt="d", linewidths=0.5, cbar=False, cmap="coolwarm", square=True)
plt.title("Co-occurrence of Top 15 Items")
plt.tight_layout()
plt.show()

# ------------------------------------------
# Network Graph of Top 60 Co-occurring Pairs
# ------------------------------------------

top_pairs = dict(pair_counts.most_common(60))
G = nx.Graph()
items_set = dict()
for item1, item2 in top_pairs.keys():
    items_set[item1] = 0
    items_set[item2] = 0

for item in items_set.keys():
    items_set[item] = item_counts[item]

for node, freq in items_set.items():
    G.add_node(node, frequency=freq)

for (item1, item2), weight in top_pairs.items():
    if item1 != item2:
        G.add_edge(item1, item2, weight=weight)
plt.figure(figsize=(10, 10))
pos = nx.spring_layout(G, seed=42)
node_size = [items_set[node] * 1.5 for node in G.nodes()]
node_colors = [items_set[node] for node in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=plt.cm.viridis, node_size=node_size, alpha=0.95)
edges = list(G.edges())
weights = [G[u][v]['weight'] for u, v in edges]
max_weight = max(weights) if weights else 1
nx.draw_networkx_edges(G, pos, edgelist=edges, width=[w / max_weight * 1.5 for w in weights], edge_color="gray", alpha=0.9)
label_pos = {node: (x, y + 0.09) for node, (x, y) in pos.items()}
nx.draw_networkx_labels(G, label_pos, font_size=8.5, font_weight="bold", font_color="black")
plt.title("Network Graph of Top 60 Item Pairs", fontsize=13)
plt.tight_layout()
plt.show()

# ------------------------------------------
# Preparing for Apriori
# ------------------------------------------

my_data["singleTransaction"] = my_data["Member_number"].astype(str) + '_' + my_data["Date"].astype(str)
my_data.head(10)
transactions = my_data.groupby("singleTransaction")["itemDescription"].apply(list).tolist()
print(transactions[:5])
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
frequent_itemsets = apriori(df_encoded, min_support=0.001, use_colnames=True)
frequent_itemsets["count"] = frequent_itemsets["itemsets"].apply(lambda x: len(x))
print("----- Frequent Itemsets -----")
print(frequent_itemsets.head())
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules["jaccard"] = rules["support"] / (rules["antecedent support"] + rules["consequent support"] - rules["support"])
rules["certainty"] = rules["confidence"] - rules["consequent support"]
rules["kulczynski"] = 0.5 * (rules["confidence"] + rules["support"] / rules["consequent support"])

cols = ['antecedents', 'consequents', 'antecedent support',
        'consequent support', 'support', 'confidence', 'lift', 'conviction',
        'jaccard', 'certainty', 'kulczynski']

rules = rules[cols]
print("----- Association Rules -----")
print(rules.head())


# ------------------------------------------
# Checking runtime
# ------------------------------------------
end_time = time.time()
runtime = end_time - start_time
print(f"\n--- Total Runtime: {runtime:.2f} seconds ---")
