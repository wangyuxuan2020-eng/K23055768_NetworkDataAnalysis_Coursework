from pathlib import Path
import math

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import osmnx as ox
import networkx as nx

from shapely.geometry import Point, box, LineString


# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data" / "part2"
OUTPUT_DIR = BASE_DIR.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# =========================
# Helpers
# =========================
def find_first_existing_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def edge_circuity(row):
    geom = row.geometry
    actual = row.length

    start = geom.coords[0]
    end = geom.coords[-1]
    straight = math.dist(start, end)

    if straight == 0:
        return None
    return actual / straight


# =========================
# 1. Read and standardise accident data
# =========================
files = sorted(DATA_DIR.glob("*.csv"))
standardised_dfs = []
used_files = []
skipped_files = []

print("Files found:")
for f in files:
    print("-", f.name)

for f in files:
    print(f"\nReading {f.name} ...")

    try:
        df = pd.read_csv(f, encoding="utf-8", low_memory=False)
    except Exception:
        try:
            df = pd.read_csv(f, encoding="latin1", low_memory=False)
        except Exception:
            df = pd.read_csv(f, encoding="latin1", low_memory=False, on_bad_lines="skip")

    print("Shape:", df.shape)

    easting_col = find_first_existing_column(df, [
        "Grid Ref: Easting", "Easting", "GridRefEasting"
    ])
    northing_col = find_first_existing_column(df, [
        "Grid Ref: Northing", "Northing", "GridRefNorthing"
    ])
    date_col = find_first_existing_column(df, [
        "Accident Date", "Date"
    ])
    time_col = find_first_existing_column(df, [
        "Time (24hr)", "Time"
    ])
    severity_col = find_first_existing_column(df, [
        "Casualty Severity", "Severity", "Accident Severity"
    ])

    print("Easting column:", easting_col)
    print("Northing column:", northing_col)
    print("Date column:", date_col)
    print("Time column:", time_col)
    print("Severity column:", severity_col)

    required = {
        "easting": easting_col,
        "northing": northing_col,
        "date": date_col,
        "time": time_col,
        "severity": severity_col
    }

    missing_required = [k for k, v in required.items() if v is None]

    if missing_required:
        print(f"Skipping {f.name} because these required columns are missing: {missing_required}")
        skipped_files.append(f.name)
        continue

    cleaned = df[[
        easting_col,
        northing_col,
        date_col,
        time_col,
        severity_col
    ]].copy()

    cleaned = cleaned.rename(columns={
        easting_col: "easting",
        northing_col: "northing",
        date_col: "date",
        time_col: "time",
        severity_col: "severity"
    })

    cleaned["source_file"] = f.name
    standardised_dfs.append(cleaned)
    used_files.append(f.name)

print("\nUsed files:", used_files)
print("Skipped files:", skipped_files)

accidents_clean = pd.concat(standardised_dfs, ignore_index=True)

print("\n=== Combined standardised accidents data ===")
print("Shape:", accidents_clean.shape)
print("\nPreview:")
print(accidents_clean.head())

print("\nMissing values:")
print(accidents_clean.isna().sum())


# =========================
# 2. Convert to GeoDataFrame
# =========================
accidents_gdf = gpd.GeoDataFrame(
    accidents_clean,
    geometry=gpd.points_from_xy(accidents_clean["easting"], accidents_clean["northing"]),
    crs="EPSG:27700"
)

print("\n=== Accident GeoDataFrame ===")
print(accidents_gdf.shape)
print(accidents_gdf.head())

print("\nCoordinate bounds:")
print("min easting:", accidents_gdf["easting"].min())
print("max easting:", accidents_gdf["easting"].max())
print("min northing:", accidents_gdf["northing"].min())
print("max northing:", accidents_gdf["northing"].max())


# =========================
# 3. Find candidate 1km x 1km cells
# =========================
accidents_gdf["grid_x"] = (accidents_gdf["easting"] // 1000) * 1000
accidents_gdf["grid_y"] = (accidents_gdf["northing"] // 1000) * 1000

grid_counts = (
    accidents_gdf.groupby(["grid_x", "grid_y"])
    .size()
    .reset_index(name="accident_count")
    .sort_values("accident_count", ascending=False)
)

print("\n=== Top candidate 1km x 1km cells ===")
print(grid_counts.head(20))

top_cell = grid_counts.iloc[0]
xmin = top_cell["grid_x"]
ymin = top_cell["grid_y"]
xmax = xmin + 1000
ymax = ymin + 1000

print("\n=== Selected top cell ===")
print("xmin:", xmin)
print("xmax:", xmax)
print("ymin:", ymin)
print("ymax:", ymax)
print("accident_count:", top_cell["accident_count"])

cell_accidents = accidents_gdf[
    (accidents_gdf["easting"] >= xmin) & (accidents_gdf["easting"] < xmax) &
    (accidents_gdf["northing"] >= ymin) & (accidents_gdf["northing"] < ymax)
].copy()

print("\nAccidents in selected cell:", len(cell_accidents))
print(cell_accidents.head())


# =========================
# 4. Shifted window search for best 1km x 1km area
# =========================
best_window = None
best_count = -1

for dx in range(-500, 501, 100):
    for dy in range(-500, 501, 100):
        test_xmin = xmin + dx
        test_xmax = test_xmin + 1000
        test_ymin = ymin + dy
        test_ymax = test_ymin + 1000

        count = (
            (accidents_gdf["easting"] >= test_xmin) & (accidents_gdf["easting"] < test_xmax) &
            (accidents_gdf["northing"] >= test_ymin) & (accidents_gdf["northing"] < test_ymax)
        ).sum()

        if count > best_count:
            best_count = count
            best_window = (test_xmin, test_xmax, test_ymin, test_ymax)

print("\n=== Best shifted 1km x 1km window ===")
print("xmin:", best_window[0])
print("xmax:", best_window[1])
print("ymin:", best_window[2])
print("ymax:", best_window[3])
print("accident_count:", best_count)

best_cell_accidents = accidents_gdf[
    (accidents_gdf["easting"] >= best_window[0]) & (accidents_gdf["easting"] < best_window[1]) &
    (accidents_gdf["northing"] >= best_window[2]) & (accidents_gdf["northing"] < best_window[3])
].copy()

print("\nAccidents in best shifted window:", len(best_cell_accidents))
print(best_cell_accidents.head())


# =========================
# 5. Define study area
# =========================
study_xmin = best_window[0]
study_xmax = best_window[1]
study_ymin = best_window[2]
study_ymax = best_window[3]

study_poly_bng = gpd.GeoDataFrame(
    geometry=[box(study_xmin, study_ymin, study_xmax, study_ymax)],
    crs="EPSG:27700"
)

study_poly_wgs84 = study_poly_bng.to_crs(epsg=4326)
polygon_wgs84 = study_poly_wgs84.geometry.iloc[0]

print("\n=== Study area polygon (WGS84) ===")
print(polygon_wgs84)


# =========================
# 6. Download OSM road network
# =========================
G_drive = ox.graph_from_polygon(polygon_wgs84, network_type="drive")

print("\n=== OSM road network downloaded ===")
print("Number of nodes:", G_drive.number_of_nodes())
print("Number of edges:", G_drive.number_of_edges())


# =========================
# 7. Project road network and compute Task A metrics
# =========================
G_drive_proj = ox.project_graph(G_drive, to_crs="EPSG:27700")
nodes_gdf, edges_gdf = ox.graph_to_gdfs(G_drive_proj)

print("\n=== Projected road network ===")
print("Projected nodes:", len(nodes_gdf))
print("Projected edges:", len(edges_gdf))

area_m2 = 1000 * 1000
area_km2 = area_m2 / 1_000_000

avg_street_length = edges_gdf["length"].mean()
node_density = len(nodes_gdf) / area_km2

node_degrees = dict(G_drive_proj.degree())
intersection_nodes = [n for n, d in node_degrees.items() if d >= 2]
intersection_density = len(intersection_nodes) / area_km2

total_edge_length_m = edges_gdf["length"].sum()
edge_density_m_per_km2 = total_edge_length_m / area_km2

largest_wcc = max(nx.weakly_connected_components(G_drive_proj), key=len)
G_largest = G_drive_proj.subgraph(largest_wcc).copy()
spatial_diameter = nx.diameter(nx.to_undirected(G_largest), weight="length")

edges_gdf["circuity"] = edges_gdf.apply(edge_circuity, axis=1)
avg_circuity = edges_gdf["circuity"].dropna().mean()

num_self_loops = nx.number_of_selfloops(G_drive_proj)
is_planar_undirected, _ = nx.check_planarity(nx.Graph(G_largest))

print("\n=== Task A road network metrics ===")
print("Average street length (m):", avg_street_length)
print("Node density (nodes/km²):", node_density)
print("Intersection density (intersections/km²):", intersection_density)
print("Edge density (m/km²):", edge_density_m_per_km2)
print("Spatial diameter (m):", spatial_diameter)
print("Average circuity:", avg_circuity)
print("Number of self-loops:", num_self_loops)
print("Planar as undirected graph?:", is_planar_undirected)


# =========================
# 8. Plot study area road network + accident points
# =========================
study_accidents = best_cell_accidents.copy()

edges_in_window = edges_gdf.cx[study_xmin:study_xmax, study_ymin:study_ymax].copy()

fig, ax = plt.subplots(figsize=(10, 10))
edges_in_window.plot(ax=ax, linewidth=0.8)
study_accidents.plot(ax=ax, markersize=8, alpha=0.7)

ax.set_title("Leeds study area: road network and accident points")
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")
ax.set_aspect("equal")

plt.tight_layout()

output_path = OUTPUT_DIR / "part2_road_network_accidents.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print("\nFigure saved to:", output_path)
plt.show()


# =========================
# 9. ORIGINAL distance to nearest intersection
# =========================
intersection_node_ids = [n for n, d in node_degrees.items() if d >= 2]
intersection_points = nodes_gdf.loc[intersection_node_ids].copy()

print("\n=== Intersection points ===")
print("Number of intersection nodes:", len(intersection_points))

intersection_points_window = intersection_points.cx[study_xmin:study_xmax, study_ymin:study_ymax].copy()
print("Intersection nodes in study window:", len(intersection_points_window))

study_accidents = best_cell_accidents.copy()
accident_to_intersection_dist = study_accidents.geometry.apply(
    lambda p: intersection_points_window.distance(p).min()
)

study_accidents = study_accidents.copy()
study_accidents["dist_to_nearest_intersection_m"] = accident_to_intersection_dist

print("\n=== Distance to nearest intersection ===")
print(study_accidents["dist_to_nearest_intersection_m"].describe())

distance_summary = study_accidents["dist_to_nearest_intersection_m"].describe()
print("\nMedian distance to nearest intersection (m):", distance_summary["50%"])
print("Mean distance to nearest intersection (m):", distance_summary["mean"])
print("Max distance to nearest intersection (m):", distance_summary["max"])


# =========================
# 10. Histogram
# =========================
fig, ax = plt.subplots(figsize=(8, 5))
study_accidents["dist_to_nearest_intersection_m"].hist(ax=ax, bins=20)

ax.set_title("Distance from accidents to nearest intersection")
ax.set_xlabel("Distance to nearest intersection (m)")
ax.set_ylabel("Number of accidents")

plt.tight_layout()

output_path = OUTPUT_DIR / "part2_distance_to_intersection_hist.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print("\nHistogram saved to:", output_path)
plt.show()


# =========================
# 11. Compare distance to intersections by severity
# =========================
print("\n=== Severity counts ===")
print(study_accidents["severity"].value_counts(dropna=False))

severity_distance_summary = (
    study_accidents.groupby("severity")["dist_to_nearest_intersection_m"]
    .describe()
)

print("\n=== Distance to nearest intersection by severity ===")
print(severity_distance_summary)

severity_medians = (
    study_accidents.groupby("severity")["dist_to_nearest_intersection_m"]
    .median()
    .sort_values()
)

severity_means = (
    study_accidents.groupby("severity")["dist_to_nearest_intersection_m"]
    .mean()
    .sort_values()
)

print("\nMedian distance by severity:")
print(severity_medians)

print("\nMean distance by severity:")
print(severity_means)


# =========================
# 12. Boxplot: distance to nearest intersection by severity
# =========================
fig, ax = plt.subplots(figsize=(8, 5))

study_accidents.boxplot(
    column="dist_to_nearest_intersection_m",
    by="severity",
    ax=ax
)

ax.set_title("Distance to nearest intersection by severity")
ax.set_xlabel("Severity")
ax.set_ylabel("Distance to nearest intersection (m)")

plt.suptitle("")  # 去掉 pandas 默认副标题
plt.tight_layout()

output_path = OUTPUT_DIR / "part2_distance_by_severity_boxplot.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print("\nBoxplot saved to:", output_path)

plt.show()


# =========================
# 13. Match each accident to nearest road segment
# =========================
from scipy.spatial import cKDTree
import numpy as np

# 研究窗口内的边
edges_in_window = edges_gdf.cx[study_xmin:study_xmax, study_ymin:study_ymax].copy()

print("\n=== Road segments in study window ===")
print("Number of edges in window:", len(edges_in_window))

# 用每条边的中点做最近邻近似匹配
edges_in_window = edges_in_window.copy()
edges_in_window["midpoint"] = edges_in_window.geometry.interpolate(0.5, normalized=True)

edge_mid_coords = np.array([(geom.x, geom.y) for geom in edges_in_window["midpoint"]])
acc_coords = np.array([(geom.x, geom.y) for geom in study_accidents.geometry])

tree = cKDTree(edge_mid_coords)
distances_to_edge_mid, nearest_edge_idx = tree.query(acc_coords, k=1)

# 给事故点附上最近边段信息
study_accidents = study_accidents.copy().reset_index(drop=True)
edges_in_window = edges_in_window.reset_index(drop=True)

study_accidents["nearest_edge_idx"] = nearest_edge_idx
study_accidents["dist_to_nearest_edge_mid_m"] = distances_to_edge_mid
study_accidents["nearest_edge_length_m"] = study_accidents["nearest_edge_idx"].map(
    edges_in_window["length"].to_dict()
)

print("\n=== Accident to nearest road segment summary ===")
print(study_accidents["nearest_edge_length_m"].describe())

print("\nMedian nearest edge length (m):", study_accidents["nearest_edge_length_m"].median())
print("Mean nearest edge length (m):", study_accidents["nearest_edge_length_m"].mean())
print("Max nearest edge length (m):", study_accidents["nearest_edge_length_m"].max())


# =========================
# 14. Histogram of nearest road segment length
# =========================
fig, ax = plt.subplots(figsize=(8, 5))

study_accidents["nearest_edge_length_m"].hist(ax=ax, bins=20)

ax.set_title("Length of nearest road segment for accidents")
ax.set_xlabel("Nearest road segment length (m)")
ax.set_ylabel("Number of accidents")

plt.tight_layout()

output_path = OUTPUT_DIR / "part2_nearest_edge_length_hist.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print("\nNearest edge length histogram saved to:", output_path)

plt.close()


# =========================
# 15. Compare nearest road segment length by severity
# =========================
edge_length_by_severity = (
    study_accidents.groupby("severity")["nearest_edge_length_m"]
    .describe()
)

print("\n=== Nearest road segment length by severity ===")
print(edge_length_by_severity)

edge_length_medians = (
    study_accidents.groupby("severity")["nearest_edge_length_m"]
    .median()
    .sort_values()
)

edge_length_means = (
    study_accidents.groupby("severity")["nearest_edge_length_m"]
    .mean()
    .sort_values()
)

print("\nMedian nearest edge length by severity:")
print(edge_length_medians)

print("\nMean nearest edge length by severity:")
print(edge_length_means)


# =========================
# 16. Boxplot of nearest road segment length by severity
# =========================
fig, ax = plt.subplots(figsize=(8, 5))

study_accidents.boxplot(
    column="nearest_edge_length_m",
    by="severity",
    ax=ax
)

ax.set_title("Nearest road segment length by severity")
ax.set_xlabel("Severity")
ax.set_ylabel("Nearest road segment length (m)")

plt.suptitle("")
plt.tight_layout()

output_path = OUTPUT_DIR / "part2_nearest_edge_length_by_severity_boxplot.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print("\nNearest edge length boxplot saved to:", output_path)

plt.close()

# =========================
# 16B. OFFICIAL Task B additions:
#      exact edge matching + fraction along road
#      Moran's I on edge accident counts
#      network K-function with Monte Carlo envelope
# =========================
import numpy as np
from collections import defaultdict

# -------------------------
# Helper functions
# -------------------------
def as_simple_line(geom):
    """
    Ensure we work with a single LineString.
    If geometry is MultiLineString, keep the longest part.
    """
    if geom.geom_type == "LineString":
        return geom
    elif geom.geom_type == "MultiLineString":
        return max(list(geom.geoms), key=lambda g: g.length)
    else:
        return geom

def find_exact_nearest_edge(point, edges_df):
    """
    Find the exact nearest edge by geometry distance (not midpoint approximation).
    Returns (edge_id, distance).
    """
    dists = edges_df["geometry"].apply(lambda g: as_simple_line(g).distance(point))
    idx = dists.idxmin()
    return int(edges_df.loc[idx, "edge_id"]), float(dists.loc[idx])

def morans_i_from_neighbors(x, neighbors):
    """
    Manual Moran's I for edge-level values x and binary neighbor structure.
    neighbors: dict(edge_id -> set(neighbor_edge_ids))
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    x_bar = x.mean()

    S0 = sum(len(v) for v in neighbors.values())
    if S0 == 0:
        return np.nan

    den = np.sum((x - x_bar) ** 2)
    if den == 0:
        return np.nan

    num = 0.0
    for i, nbrs in neighbors.items():
        for j in nbrs:
            num += (x[i] - x_bar) * (x[j] - x_bar)

    I = (n / S0) * (num / den)
    return I

def morans_i_permutation_test(x, neighbors, n_perm=999, seed=42):
    """
    Permutation test for Moran's I.
    Returns observed I, permutation mean, upper-tail p-value, z-score.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)

    observed = morans_i_from_neighbors(x, neighbors)
    sims = []

    for _ in range(n_perm):
        x_perm = rng.permutation(x)
        sims.append(morans_i_from_neighbors(x_perm, neighbors))

    sims = np.asarray(sims, dtype=float)
    sim_mean = float(np.nanmean(sims))
    sim_std = float(np.nanstd(sims, ddof=1))

    # one-sided p-value for positive autocorrelation
    p_upper = (np.sum(sims >= observed) + 1) / (n_perm + 1)

    if sim_std == 0:
        z_score = np.nan
    else:
        z_score = (observed - sim_mean) / sim_std

    return observed, sim_mean, p_upper, z_score, sims

def snap_point_to_edge_record(point, edge_row):
    """
    Snap a point to an edge and compute distances along the edge.
    Returns a dictionary describing the snapped point on the network.
    """
    geom = as_simple_line(edge_row["geometry"])
    length = float(edge_row["length"])

    proj = float(geom.project(point))   # distance from start vertex along line
    proj = max(0.0, min(proj, length))

    return {
        "edge_id": int(edge_row["edge_id"]),
        "u": edge_row["u"],
        "v": edge_row["v"],
        "edge_length": length,
        "proj_m": proj,
        "to_u_m": proj,
        "to_v_m": length - proj,
        "frac_from_start": proj / length if length > 0 else np.nan,
        "frac_from_nearest_intersection": min(proj, length - proj) / length if length > 0 else np.nan
    }

def network_distance_between_snapped_points(p1, p2, node_dist):
    """
    Network distance between two snapped points on a road network.
    p1, p2 are dicts returned by snap_point_to_edge_record().
    node_dist is all-pairs shortest path length dict among network nodes.
    """
    # same edge: direct along-edge distance is possible
    candidates = []
    if p1["edge_id"] == p2["edge_id"]:
        candidates.append(abs(p1["proj_m"] - p2["proj_m"]))

    endpoint_options_1 = [
        (p1["u"], p1["to_u_m"]),
        (p1["v"], p1["to_v_m"])
    ]
    endpoint_options_2 = [
        (p2["u"], p2["to_u_m"]),
        (p2["v"], p2["to_v_m"])
    ]

    for n1, d1 in endpoint_options_1:
        for n2, d2 in endpoint_options_2:
            sp = node_dist.get(n1, {}).get(n2, np.inf)
            candidates.append(d1 + sp + d2)

    return float(min(candidates))

def compute_network_pairwise_distances(snapped_points, node_dist):
    """
    Compute all upper-triangle pairwise network distances.
    """
    n = len(snapped_points)
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = network_distance_between_snapped_points(snapped_points[i], snapped_points[j], node_dist)
            dists.append(d)
    return np.asarray(dists, dtype=float)

def sample_random_points_on_network(n_points, edges_df, rng):
    """
    CSR on network: sample points uniformly by edge length, then uniformly along chosen edge.
    """
    lengths = edges_df["length"].to_numpy(dtype=float)
    probs = lengths / lengths.sum()

    chosen_ids = rng.choice(edges_df["edge_id"].to_numpy(), size=n_points, p=probs, replace=True)

    sampled = []
    edges_lookup = edges_df.set_index("edge_id")

    for eid in chosen_ids:
        row = edges_lookup.loc[eid]
        length = float(row["length"])
        proj = float(rng.uniform(0, length))

        sampled.append({
            "edge_id": int(eid),
            "u": row["u"],
            "v": row["v"],
            "edge_length": length,
            "proj_m": proj,
            "to_u_m": proj,
            "to_v_m": length - proj,
            "frac_from_start": proj / length if length > 0 else np.nan,
            "frac_from_nearest_intersection": min(proj, length - proj) / length if length > 0 else np.nan
        })

    return sampled

def network_k_function(pairwise_dists, radii, n_points, network_length):
    """
    Simple network K-function estimate:
    K(r) = L / (n * (n - 1)) * number_of_ordered_pairs_with_distance<=r
    Since pairwise_dists are upper-triangle only, multiply count by 2.
    """
    results = []
    for r in radii:
        unordered_count = np.sum(pairwise_dists <= r)
        ordered_count = 2 * unordered_count
        k_r = (network_length / (n_points * (n_points - 1))) * ordered_count
        results.append(k_r)
    return np.asarray(results, dtype=float)


# -------------------------
# Rebuild edge table with exact IDs and endpoints
# -------------------------
edges_task_b = edges_gdf.cx[study_xmin:study_xmax, study_ymin:study_ymax].copy()
edges_task_b = edges_task_b.reset_index()   # keep u, v, key
edges_task_b["geometry"] = edges_task_b["geometry"].apply(as_simple_line)
edges_task_b["edge_id"] = np.arange(len(edges_task_b))

print("\n=== Task B exact edge table ===")
print("Edges available for Task B:", len(edges_task_b))
print(edges_task_b[["edge_id", "u", "v", "length"]].head())


# -------------------------
# 16C. Exact nearest-edge snapping for each accident
# -------------------------
study_accidents = study_accidents.copy().reset_index(drop=True)

nearest_edge_records = []
for p in study_accidents.geometry:
    eid, dist_exact = find_exact_nearest_edge(p, edges_task_b)
    edge_row = edges_task_b.loc[edges_task_b["edge_id"] == eid].iloc[0]
    rec = snap_point_to_edge_record(p, edge_row)
    rec["dist_to_edge_exact_m"] = dist_exact
    nearest_edge_records.append(rec)

nearest_edge_df = pd.DataFrame(nearest_edge_records)

study_accidents["edge_id_exact"] = nearest_edge_df["edge_id"]
study_accidents["dist_to_edge_exact_m"] = nearest_edge_df["dist_to_edge_exact_m"]
study_accidents["frac_from_start"] = nearest_edge_df["frac_from_start"]
study_accidents["frac_from_nearest_intersection"] = nearest_edge_df["frac_from_nearest_intersection"]

print("\n=== Task B fraction along road from nearest intersection ===")
print(study_accidents["frac_from_nearest_intersection"].describe())

print("\nMedian fraction from nearest intersection:")
print(study_accidents["frac_from_nearest_intersection"].median())

print("\nMean fraction from nearest intersection:")
print(study_accidents["frac_from_nearest_intersection"].mean())

fraction_by_severity = (
    study_accidents.groupby("severity")["frac_from_nearest_intersection"]
    .describe()
)
print("\n=== Fraction from nearest intersection by severity ===")
print(fraction_by_severity)

fraction_medians = (
    study_accidents.groupby("severity")["frac_from_nearest_intersection"]
    .median()
    .sort_values()
)
fraction_means = (
    study_accidents.groupby("severity")["frac_from_nearest_intersection"]
    .mean()
    .sort_values()
)

print("\nMedian fraction by severity:")
print(fraction_medians)

print("\nMean fraction by severity:")
print(fraction_means)

# Histogram
fig, ax = plt.subplots(figsize=(8, 5))
study_accidents["frac_from_nearest_intersection"].hist(ax=ax, bins=20)
ax.set_title("Fraction of road length from nearest intersection")
ax.set_xlabel("Fraction from nearest intersection (0 = at junction, 0.5 = middle)")
ax.set_ylabel("Number of accidents")
plt.tight_layout()

output_path = OUTPUT_DIR / "part2_fraction_from_intersection_hist.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print("\nFraction histogram saved to:", output_path)
plt.close()

# Boxplot by severity
fig, ax = plt.subplots(figsize=(8, 5))
study_accidents.boxplot(
    column="frac_from_nearest_intersection",
    by="severity",
    ax=ax
)
ax.set_title("Fraction from nearest intersection by severity")
ax.set_xlabel("Severity")
ax.set_ylabel("Fraction from nearest intersection")
plt.suptitle("")
plt.tight_layout()

output_path = OUTPUT_DIR / "part2_fraction_from_intersection_by_severity_boxplot.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print("Fraction boxplot saved to:", output_path)
plt.close()


# -------------------------
# 16D. Aggregate accidents by road segment and compute Moran's I
# -------------------------
edge_accident_counts = (
    study_accidents["edge_id_exact"]
    .value_counts()
    .rename_axis("edge_id")
    .reset_index(name="accident_count")
)

edges_task_b = edges_task_b.merge(edge_accident_counts, on="edge_id", how="left")
edges_task_b["accident_count"] = edges_task_b["accident_count"].fillna(0).astype(int)

print("\n=== Accident counts per edge summary ===")
print(edges_task_b["accident_count"].describe())
print("\nTop 10 edges by accident count:")
print(
    edges_task_b[["edge_id", "u", "v", "length", "accident_count"]]
    .sort_values("accident_count", ascending=False)
    .head(10)
)

# Build edge adjacency based on shared endpoints
endpoint_to_edges = defaultdict(set)
for _, row in edges_task_b.iterrows():
    endpoint_to_edges[row["u"]].add(int(row["edge_id"]))
    endpoint_to_edges[row["v"]].add(int(row["edge_id"]))

neighbors = {}
for eid in edges_task_b["edge_id"]:
    row = edges_task_b.loc[edges_task_b["edge_id"] == eid].iloc[0]
    nbrs = set(endpoint_to_edges[row["u"]]) | set(endpoint_to_edges[row["v"]])
    nbrs.discard(int(eid))
    neighbors[int(eid)] = nbrs

x = edges_task_b.sort_values("edge_id")["accident_count"].to_numpy(dtype=float)

moran_I, moran_mean, moran_p, moran_z, moran_sims = morans_i_permutation_test(
    x, neighbors, n_perm=999, seed=42
)

print("\n=== Moran's I on edge accident counts ===")
print("Observed Moran's I:", moran_I)
print("Permutation mean:", moran_mean)
print("Upper-tail p-value:", moran_p)
print("Z-score:", moran_z)

# Optional quick map of accident counts by edge
fig, ax = plt.subplots(figsize=(10, 10))
edges_task_b.plot(
    ax=ax,
    column="accident_count",
    linewidth=2.0,
    legend=True
)
ax.set_title("Road segments coloured by accident count")
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")
ax.set_aspect("equal")
plt.tight_layout()

output_path = OUTPUT_DIR / "part2_edge_accident_counts_map.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print("Edge accident count map saved to:", output_path)
plt.close()


# -------------------------
# 16E. Network K-function on accident locations
# -------------------------
# Build a simple undirected weighted graph for shortest path distances
G_metric = nx.Graph()
for _, row in edges_task_b.iterrows():
    u = row["u"]
    v = row["v"]
    w = float(row["length"])

    if G_metric.has_edge(u, v):
        if w < G_metric[u][v]["weight"]:
            G_metric[u][v]["weight"] = w
    else:
        G_metric.add_edge(u, v, weight=w)

node_dist = dict(nx.all_pairs_dijkstra_path_length(G_metric, weight="weight"))

# Observed snapped points
observed_snapped_points = nearest_edge_df.to_dict("records")
observed_pairwise = compute_network_pairwise_distances(observed_snapped_points, node_dist)

network_length_m = float(edges_task_b["length"].sum())
n_points = len(observed_snapped_points)

# Radii (in metres) for the K-function
radii = np.arange(25, 525, 25)   # 25m to 500m
observed_k = network_k_function(observed_pairwise, radii, n_points, network_length_m)

print("\n=== Network K-function (observed) ===")
for r, kval in zip(radii, observed_k):
    print(f"r = {r:>3} m  |  K(r) = {kval:.4f}")

# Monte Carlo CSR envelope on the network
rng = np.random.default_rng(42)
n_sim = 39   # if slow on your machine, reduce to 19

sim_ks = []
for sim in range(n_sim):
    sim_points = sample_random_points_on_network(n_points, edges_task_b, rng)
    sim_pairwise = compute_network_pairwise_distances(sim_points, node_dist)
    sim_k = network_k_function(sim_pairwise, radii, n_points, network_length_m)
    sim_ks.append(sim_k)

sim_ks = np.asarray(sim_ks, dtype=float)
k_mean = sim_ks.mean(axis=0)
k_low = np.percentile(sim_ks, 2.5, axis=0)
k_high = np.percentile(sim_ks, 97.5, axis=0)

print("\n=== Network K-function comparison with CSR envelope ===")
for i, r in enumerate(radii):
    if observed_k[i] > k_high[i]:
        status = "clustered at this scale"
    elif observed_k[i] < k_low[i]:
        status = "dispersed at this scale"
    else:
        status = "close to CSR at this scale"

    print(
        f"r = {r:>3} m | observed = {observed_k[i]:.4f} | "
        f"CSR mean = {k_mean[i]:.4f} | 95% envelope = [{k_low[i]:.4f}, {k_high[i]:.4f}] | {status}"
    )

# Plot K-function
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(radii, observed_k, label="Observed K(r)")
ax.plot(radii, k_mean, linestyle="--", label="CSR mean")
ax.fill_between(radii, k_low, k_high, alpha=0.2, label="95% CSR envelope")

ax.set_title("Network K-function of accident locations")
ax.set_xlabel("Network distance r (m)")
ax.set_ylabel("K(r)")
ax.legend()

plt.tight_layout()
output_path = OUTPUT_DIR / "part2_network_k_function.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print("\nNetwork K-function plot saved to:", output_path)
plt.close()


# =========================
# 17. Prepare representative nodes for Task C
# =========================

# 只保留研究窗口内的节点
nodes_in_window = nodes_gdf.cx[study_xmin:study_xmax, study_ymin:study_ymax].copy()

print("\n=== Nodes in study window ===")
print("Number of nodes in window:", len(nodes_in_window))

# 为了让 Voronoi 图更清楚，只抽样一部分节点
# 这里先每隔 10 个取 1 个
sample_nodes = nodes_in_window.iloc[::10].copy()

print("Sample nodes for Voronoi:", len(sample_nodes))
print(sample_nodes.head())


# =========================
# 18. Plot sampled nodes on road network
# =========================
fig, ax = plt.subplots(figsize=(10, 10))

edges_in_window.plot(ax=ax, linewidth=0.8)
sample_nodes.plot(ax=ax, markersize=20, alpha=0.8)

ax.set_title("Sampled road nodes for Voronoi analysis")
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")
ax.set_aspect("equal")

plt.tight_layout()

output_path = OUTPUT_DIR / "part2_sample_nodes_for_voronoi.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print("\nSample node figure saved to:", output_path)

plt.close()


# =========================
# 19. Build Voronoi polygons from sampled nodes
# =========================
from shapely.ops import voronoi_diagram

# 用样本节点做 Voronoi
sample_union = sample_nodes.geometry.unary_union

# 生成原始 Voronoi 图
vor = voronoi_diagram(sample_union, envelope=study_poly_bng.geometry.iloc[0])

# 拆成单个 polygon
vor_polys = list(vor.geoms)

vor_gdf = gpd.GeoDataFrame(
    geometry=vor_polys,
    crs="EPSG:27700"
)

print("\n=== Raw Voronoi polygons ===")
print("Number of raw polygons:", len(vor_gdf))

# 裁剪到研究区
study_area_polygon = study_poly_bng.geometry.iloc[0]
vor_gdf["geometry"] = vor_gdf.geometry.intersection(study_area_polygon)

# 去掉空几何
vor_gdf = vor_gdf[~vor_gdf.geometry.is_empty].copy()
vor_gdf = vor_gdf.reset_index(drop=True)

print("Number of clipped Voronoi polygons:", len(vor_gdf))
print(vor_gdf.head())


# =========================
# 20. Voronoi cell area summary
# =========================
vor_gdf["area_m2"] = vor_gdf.geometry.area

print("\n=== Voronoi cell area summary ===")
print(vor_gdf["area_m2"].describe())

print("\nLargest Voronoi cells:")
print(vor_gdf["area_m2"].sort_values(ascending=False).head())

print("\nSmallest Voronoi cells:")
print(vor_gdf["area_m2"].sort_values(ascending=True).head())


# =========================
# 21. Plot Voronoi polygons with road network and sampled nodes
# =========================
fig, ax = plt.subplots(figsize=(10, 10))

vor_gdf.plot(ax=ax, alpha=0.4, edgecolor="black")
edges_in_window.plot(ax=ax, linewidth=0.8)
sample_nodes.plot(ax=ax, markersize=20)

ax.set_title("Voronoi polygons of sampled road nodes")
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")
ax.set_aspect("equal")

plt.tight_layout()

output_path = OUTPUT_DIR / "part2_voronoi_polygons.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print("\nVoronoi figure saved to:", output_path)

plt.close()


# =========================
# 22. Formal Task C: choose N=4 seed points
# =========================

# 用研究区四个象限中心，选最近的 road node，保证 seeds 均匀分布
quadrant_targets = [
    (study_xmin + 250, study_ymin + 250),  # SW
    (study_xmin + 250, study_ymax - 250),  # NW
    (study_xmax - 250, study_ymin + 250),  # SE
    (study_xmax - 250, study_ymax - 250),  # NE
]

nodes_in_window = nodes_gdf.cx[study_xmin:study_xmax, study_ymin:study_ymax].copy()

seed_indices = []
used_node_ids = set()

for tx, ty in quadrant_targets:
    temp = nodes_in_window.copy()
    temp["target_dist"] = temp.geometry.distance(Point(tx, ty))
    temp = temp.sort_values("target_dist")

    for idx in temp.index:
        if idx not in used_node_ids:
            seed_indices.append(idx)
            used_node_ids.add(idx)
            break

# 直接把索引保存在一列里，避免 reset_index/rename 出错
seed_nodes = nodes_in_window.loc[seed_indices].copy()
seed_nodes["seed_node_id"] = seed_nodes.index.astype(str)

print("\n=== Formal Task C seeds (N=4) ===")
print(seed_nodes[["seed_node_id", "geometry"]])


# =========================
# 23. Voronoi diagram for the 4 formal seeds
# =========================
seed_union = seed_nodes.geometry.union_all()

vor4 = voronoi_diagram(seed_union, envelope=study_poly_bng.geometry.iloc[0])
vor4_polys = list(vor4.geoms)

vor4_gdf = gpd.GeoDataFrame(
    geometry=vor4_polys,
    crs="EPSG:27700"
)

study_area_polygon = study_poly_bng.geometry.iloc[0]
vor4_gdf["geometry"] = vor4_gdf.geometry.intersection(study_area_polygon)
vor4_gdf = vor4_gdf[~vor4_gdf.geometry.is_empty].copy().reset_index(drop=True)

print("\n=== Formal Voronoi cells (N=4) ===")
print("Number of cells:", len(vor4_gdf))
print(vor4_gdf.head())


# =========================
# 24. Match each Voronoi cell to its nearest seed
# =========================
# 给每个 cell 分配最近的 seed，方便后面逐 cell 处理
vor4_gdf["seed_id"] = None

for i, cell in vor4_gdf.iterrows():
    dists = seed_nodes.geometry.distance(cell.geometry.centroid)
    nearest_seed_idx = dists.idxmin()
    vor4_gdf.at[i, "seed_id"] = seed_nodes.loc[nearest_seed_idx, "seed_node_id"]

print("\nCell to seed mapping:")
print(vor4_gdf[["seed_id"]])


# =========================
# 25. Plot formal N=4 Voronoi cells
# =========================
fig, ax = plt.subplots(figsize=(10, 10))

vor4_gdf.plot(ax=ax, alpha=0.35, edgecolor="black")
edges_in_window.plot(ax=ax, linewidth=0.8)
seed_nodes.plot(ax=ax, markersize=40)

ax.set_title("Formal Voronoi diagram with 4 seed points")
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")
ax.set_aspect("equal")

plt.tight_layout()

output_path = OUTPUT_DIR / "part2_voronoi_4cells.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print("\nFormal 4-cell Voronoi figure saved to:", output_path)

plt.close()


# =========================
# 26. Build subgraph for each Voronoi cell
# =========================
cell_subgraphs = {}

# 用无向图更方便找 cycle
G_und = nx.Graph(G_drive_proj)

for i, cell in vor4_gdf.iterrows():
    poly = cell.geometry

    # 选出落在 cell 内的节点
    nodes_in_cell = nodes_gdf[nodes_gdf.geometry.within(poly)].copy()

    node_ids = set(nodes_in_cell.index.tolist())

    # 取 induced subgraph
    subG = G_und.subgraph(node_ids).copy()

    cell_subgraphs[i] = subG

    print(f"\nCell {i}:")
    print(" seed_id:", vor4_gdf.loc[i, 'seed_id'])
    print(" nodes:", subG.number_of_nodes())
    print(" edges:", subG.number_of_edges())


# =========================
# 27. Find approximate 42 km closed route in each cell
# =========================
TARGET_LEN = 42000  # metres

def cycle_length(subG, cycle_nodes):
    total = 0.0
    for a, b in zip(cycle_nodes, cycle_nodes[1:] + [cycle_nodes[0]]):
        if subG.has_edge(a, b):
            edge_data = subG.get_edge_data(a, b)
            if isinstance(edge_data, dict) and "length" in edge_data:
                total += edge_data["length"]
            else:
                # Multi-edge / fallback
                if isinstance(edge_data, dict):
                    vals = []
                    for _, v in edge_data.items():
                        if isinstance(v, dict) and "length" in v:
                            vals.append(v["length"])
                    if vals:
                        total += min(vals)
    return total

cell_route_results = []

for cell_id, subG in cell_subgraphs.items():
    result = {
        "cell_id": cell_id,
        "num_nodes": subG.number_of_nodes(),
        "num_edges": subG.number_of_edges(),
        "best_cycle_len_m": None,
        "repetitions": None,
        "route_len_m": None,
        "abs_error_m": None,
        "cycle_nodes": None
    }

    if subG.number_of_nodes() < 3 or subG.number_of_edges() < 3:
        cell_route_results.append(result)
        continue

    cycles = nx.cycle_basis(subG)

    if len(cycles) == 0:
        cell_route_results.append(result)
        continue

    best = None
    best_error = float("inf")

    for cyc in cycles:
        L = cycle_length(subG, cyc)
        if L <= 0:
            continue

        reps = max(1, round(TARGET_LEN / L))
        total_len = reps * L
        err = abs(total_len - TARGET_LEN)

        if err < best_error:
            best_error = err
            best = (L, reps, total_len, err, cyc)

    if best is not None:
        result["best_cycle_len_m"] = best[0]
        result["repetitions"] = best[1]
        result["route_len_m"] = best[2]
        result["abs_error_m"] = best[3]
        result["cycle_nodes"] = best[4]

    cell_route_results.append(result)

route_results_df = pd.DataFrame(cell_route_results)

print("\n=== Approximate 42 km closed routes by cell ===")
print(route_results_df[[
    "cell_id", "num_nodes", "num_edges",
    "best_cycle_len_m", "repetitions",
    "route_len_m", "abs_error_m"
]])



# =========================
# 28. Keep best 2-3 cells
# =========================
feasible_routes = route_results_df.dropna(subset=["route_len_m"]).copy()
feasible_routes = feasible_routes.sort_values("abs_error_m")

print("\n=== Best candidate cells for marathon routes ===")
print(feasible_routes.head(3)[[
    "cell_id", "best_cycle_len_m", "repetitions",
    "route_len_m", "abs_error_m"
]])



# =========================
# 29. Plot one example closed route
# =========================
if len(feasible_routes) > 0:
    best_cell_id = int(feasible_routes.iloc[0]["cell_id"])
    best_cycle_nodes = feasible_routes.iloc[0]["cycle_nodes"]

    poly = vor4_gdf.loc[best_cell_id, "geometry"]
    subG = cell_subgraphs[best_cell_id]

    fig, ax = plt.subplots(figsize=(10, 10))

    # 画 cell
    gpd.GeoSeries([poly], crs="EPSG:27700").plot(ax=ax, alpha=0.25, edgecolor="black")
    edges_in_window.plot(ax=ax, linewidth=0.8)

    # 画 cycle 的边
    route_edges = []
    for a, b in zip(best_cycle_nodes, best_cycle_nodes[1:] + [best_cycle_nodes[0]]):
        if subG.has_edge(a, b):
            edge_data = subG.get_edge_data(a, b)
            if "geometry" in edge_data:
                route_edges.append(edge_data["geometry"])
            else:
                # fallback
                pa = nodes_gdf.loc[a].geometry
                pb = nodes_gdf.loc[b].geometry
                route_edges.append(LineString([pa, pb]))

    route_gdf = gpd.GeoDataFrame(geometry=route_edges, crs="EPSG:27700")
    route_gdf.plot(ax=ax, linewidth=2.5)

    ax.set_title(f"Example approximate 42 km closed route in cell {best_cell_id}")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal")

    plt.tight_layout()

    output_path = OUTPUT_DIR / "part2_example_closed_route.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print("\nExample closed route figure saved to:", output_path)

    plt.close()