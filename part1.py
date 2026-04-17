import pandas as pd
import networkx as nx
from itertools import combinations
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"/ "part1"


def build_graph(file_name):
    file_path = DATA_DIR / file_name
    df = pd.read_csv(file_path)

    df = df[['page_name', 'thread_subject', 'username']].dropna().drop_duplicates()

    G = nx.Graph()
    G.add_nodes_from(df['username'].unique())

    grouped = df.groupby(['page_name', 'thread_subject'])

    for (page, thread), group in grouped:
        users = group['username'].unique().tolist()
        if len(users) >= 2:
            edges = combinations(users, 2)
            G.add_edges_from(edges)

    return df, G


def basic_summary(df, G, file_name):
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    num_components = nx.number_connected_components(G)
    num_isolates = len(list(nx.isolates(G)))
    density = nx.density(G)
    largest_cc = max(nx.connected_components(G), key=len)
    largest_cc_size = len(largest_cc)

    print(f"\n=== {file_name} ===")
    print("Number of nodes:", num_nodes)
    print("Number of edges:", num_edges)
    print("Number of connected components:", num_components)
    print("Number of isolates:", num_isolates)
    print("Density:", density)
    print("Largest connected component size:", largest_cc_size)

    group_sizes = (
        df.groupby(['page_name', 'thread_subject'])['username']
        .nunique()
        .reset_index(name='num_users')
        .sort_values('num_users', ascending=False)
    )

    print("\nTop 10 discussion units by number of users:")
    print(group_sizes.head(10))

    sample_page = group_sizes.iloc[0]['page_name']
    sample_thread = group_sizes.iloc[0]['thread_subject']

    sample_users = (
        df[(df['page_name'] == sample_page) & (df['thread_subject'] == sample_thread)]
        ['username']
        .unique()
        .tolist()
    )

    all_edges_exist = all(G.has_edge(u, v) for u, v in combinations(sample_users, 2))

    print("\nSample page:", sample_page)
    print("Sample thread:", sample_thread)
    print("Number of users in sample:", len(sample_users))
    print("All expected edges exist in the sample discussion unit:", all_edges_exist)

    return {
        "file": file_name,
        "nodes": num_nodes,
        "edges": num_edges,
        "connected_components": num_components,
        "isolates": num_isolates,
        "density": density,
        "largest_cc_size": largest_cc_size
    }


def network_metrics(G, file_name):
    largest_cc_nodes = max(nx.connected_components(G), key=len)
    G_lcc = G.subgraph(largest_cc_nodes).copy()

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    avg_degree = sum(dict(G.degree()).values()) / num_nodes
    avg_clustering = nx.average_clustering(G)
    giant_component_ratio = G_lcc.number_of_nodes() / num_nodes

    avg_shortest_path = nx.average_shortest_path_length(G_lcc)
    diameter = nx.diameter(G_lcc)

    metrics = {
        "file": file_name,
        "nodes": num_nodes,
        "edges": num_edges,
        "avg_degree": avg_degree,
        "avg_clustering": avg_clustering,
        "giant_component_ratio": giant_component_ratio,
        "largest_cc_size": G_lcc.number_of_nodes(),
        "avg_shortest_path_lcc": avg_shortest_path,
        "diameter_lcc": diameter
    }

    print(f"\n=== Network metrics: {file_name} ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    return metrics


def random_graph_comparison(G, file_name, seed=42):
    n = G.number_of_nodes()
    m = G.number_of_edges()

    G_rand = nx.gnm_random_graph(n, m, seed=seed)

    largest_cc_real = max(nx.connected_components(G), key=len)
    G_real_lcc = G.subgraph(largest_cc_real).copy()

    largest_cc_rand = max(nx.connected_components(G_rand), key=len)
    G_rand_lcc = G_rand.subgraph(largest_cc_rand).copy()

    comparison = {
        "file": file_name,
        "real_clustering": nx.average_clustering(G),
        "random_clustering": nx.average_clustering(G_rand),
        "real_avg_shortest_path_lcc": nx.average_shortest_path_length(G_real_lcc),
        "random_avg_shortest_path_lcc": nx.average_shortest_path_length(G_rand_lcc),
        "real_diameter_lcc": nx.diameter(G_real_lcc),
        "random_diameter_lcc": nx.diameter(G_rand_lcc),
        "real_giant_component_ratio": G_real_lcc.number_of_nodes() / n,
        "random_giant_component_ratio": G_rand_lcc.number_of_nodes() / n
    }

    print(f"\n=== Random comparison: {file_name} ===")
    for k, v in comparison.items():
        print(f"{k}: {v}")

    return comparison


df_bot, G_bot = build_graph("BOT_REQUESTS.csv")
df_users, G_users = build_graph("USERS.csv")
df_rfd, G_rfd = build_graph("REQUEST_FOR_DELETION.csv")

summary_bot = basic_summary(df_bot, G_bot, "BOT_REQUESTS.csv")
summary_users = basic_summary(df_users, G_users, "USERS.csv")
summary_rfd = basic_summary(df_rfd, G_rfd, "REQUEST_FOR_DELETION.csv")

summary_df = pd.DataFrame([summary_bot, summary_users, summary_rfd])
print("\n=== Summary table ===")
print(summary_df)


metrics_bot = network_metrics(G_bot, "BOT_REQUESTS.csv")
metrics_users = network_metrics(G_users, "USERS.csv")
metrics_rfd = network_metrics(G_rfd, "REQUEST_FOR_DELETION.csv")

metrics_df = pd.DataFrame([metrics_bot, metrics_users, metrics_rfd])

print("\n=== Metrics table ===")
print(metrics_df)

metrics_output_path = DATA_DIR.parent / "outputs" / "part1_metrics_table.csv"
metrics_df.to_csv(metrics_output_path, index=False)
print(f"\nMetrics table saved to: {metrics_output_path}")


comp_bot = random_graph_comparison(G_bot, "BOT_REQUESTS.csv")
comp_users = random_graph_comparison(G_users, "USERS.csv")
comp_rfd = random_graph_comparison(G_rfd, "REQUEST_FOR_DELETION.csv")

comparison_df = pd.DataFrame([comp_bot, comp_users, comp_rfd])
print("\n=== Random comparison table ===")
print(comparison_df)