"""
File
----
corridor_rail_mapping.py

Inputs
------
• Amtrak_Routes.shp
• North_American_Rail_Network_Lines.shp
• corridors_top100.geojson

Outputs
-------
• estimated_corridor_travel_times.csv
• corridor_routes.geojson
• unreachable_corridors.csv
• graph_cache.pkl
• routed_corridors_cache.pkl
• corridor_routes_map.html
"""
import geopandas as gpd, pandas as pd, networkx as nx
from shapely.geometry import LineString, MultiLineString, Point
from scipy.spatial import cKDTree
from tqdm import tqdm
import folium, pickle, os
from math import acos, degrees, sqrt
from concurrent.futures import ThreadPoolExecutor

# ─── Config ───────────────────────────────────────────────────────
proj_crs, wgs_crs = "EPSG:5070", "EPSG:4326"
force_reroute     = False
DENSIFY_M         = 2_000          # ≤ 2 km sub-edges
K_CAND            = 10             # number of NON-Amtrak nodes to add
INIT_R, MAX_R     = 4_000, 15_000  # candidate search radii (m)
MAX_ANGLE         = 120
N_WORKERS         = 6

# ─── Paths ────────────────────────────────────────────────────────
amtrak_path  = "Data/NTAD_Amtrak_Routes_7440257972717285207/Amtrak_Routes.shp"
freight_path = "Data/NTAD_North_American_Rail_Network_Lines_4887242439196784421/North_American_Rail_Network_Lines.shp"
corridor_path = "Data/corridors_top100.geojson"

graph_cache    = "Output/graph_cache.pkl"
routed_cache_p = "Output/routed_corridors_cache.pkl"
csv_out        = "Output/estimated_corridor_travel_times.csv"
geojson_out    = "Output/corridor_routes.geojson"
unreach_out    = "Output/unreachable_corridors.csv"
map_out        = "Output/corridor_routes_map.html"

# ─── Helpers ──────────────────────────────────────────────────────
def round_pt(pt, p=5): return (round(pt.x, p), round(pt.y, p))

def densify_part(part):
    if part.length == 0: return []
    n = max(int(part.length // DENSIFY_M), 1)
    return [part.interpolate(i / n, normalized=True).coords[0] for i in range(n + 1)]

def time_heuristic(u, v):
    dx, dy = u[0]-v[0], u[1]-v[1]
    return (sqrt(dx*dx + dy*dy) / 1609.34) / 110

# ─── Build / load graph ───────────────────────────────────────────
rebuild = force_reroute or not os.path.exists(graph_cache)
if not rebuild:
    try:
        G, G_nodes, tree, comp_id, is_amtrak = pickle.load(open(graph_cache, "rb"))
    except Exception:
        rebuild = True

if rebuild:
    print("Building routing graph (fresh cache)…")
    amtrak  = gpd.read_file(amtrak_path).to_crs(proj_crs)
    freight = gpd.read_file(freight_path).to_crs(proj_crs)
    amtrak["speed_mph"],  amtrak["type"]  = 79, "amtrak"
    freight["speed_mph"], freight["type"] = 79, "freight"   # freight lines to be used as passenger rail
    rail = pd.concat([amtrak, freight], ignore_index=True)

    G, node_types = nx.Graph(), {}
    for _, r in tqdm(rail.iterrows(), total=len(rail)):
        speed, seg_type = r.speed_mph, r["type"]
        parts = [r.geometry] if isinstance(r.geometry, LineString) else r.geometry.geoms
        for part in parts:
            coords = densify_part(part)
            for a, b in zip(coords[:-1], coords[1:]):
                a, b = round_pt(Point(a)), round_pt(Point(b))
                if a == b: continue
                dx, dy  = a[0]-b[0], a[1]-b[1]
                dist_mi = sqrt(dx*dx + dy*dy) / 1609.34
                G.add_edge(a, b,
                           weight    = dist_mi / speed,
                           length_mi = dist_mi,
                           speed     = speed,
                           type      = seg_type,
                           geometry  = part)
                node_types.setdefault(a, set()).add(seg_type)
                node_types.setdefault(b, set()).add(seg_type)

    G_nodes = list(G.nodes)
    tree    = cKDTree(G_nodes)
    comp_id = {n: i for i, comp in enumerate(nx.connected_components(G)) for n in comp}
    is_amtrak = {n: ("amtrak" in node_types.get(n, ())) for n in G_nodes}
    pickle.dump((G, G_nodes, tree, comp_id, is_amtrak), open(graph_cache, "wb"))
    print("Graph cached.")
else:
    print("Loaded routing graph from cache.")

# ─── Candidate utilities ─────────────────────────────────────────
def vec(p, q):
    dx, dy = q[0]-p[0], q[1]-p[1]; n = sqrt(dx*dx + dy*dy)
    return (dx/n, dy/n) if n else (0,0)

def heading_ok(p, cand, q, ang=MAX_ANGLE):
    v1, v2 = vec((p.x,p.y), cand), vec((p.x,p.y), (q.x,q.y))
    dot = max(min(v1[0]*v2[0] + v1[1]*v2[1], 1), -1)
    return degrees(acos(dot)) <= ang

def snap(pt):
    _, idx = tree.query([pt.x, pt.y]); return G_nodes[idx]

def all_candidates(pt):
    """All Amtrak nodes + up to K_CAND nearest non-Amtrak nodes within radius."""
    r = INIT_R
    while True:
        idxs = tree.query_ball_point([pt.x, pt.y], r=r)
        if idxs or r >= MAX_R: break
        r *= 2
    idxs = sorted(idxs, key=lambda i: pt.distance(Point(G_nodes[i])))
    amtrak = [G_nodes[i] for i in idxs if is_amtrak[G_nodes[i]]]
    others = [G_nodes[i] for i in idxs if not is_amtrak[G_nodes[i]]][:K_CAND]
    return (amtrak + others) or [snap(pt)]

ESCALATE_STEPS = (K_CAND, K_CAND*2, K_CAND*4)  # 10 → 20 → 40

def first_viable_pair(starts, ends):
    best = None
    for s in starts:
        for e in ends:
            if comp_id[s] != comp_id[e]: continue
            lb = time_heuristic(s, e)
            if best and lb >= best[0]: continue
            try:
                path = nx.astar_path(G, s, e, weight="weight", heuristic=time_heuristic)
            except nx.NetworkXNoPath:
                continue
            t = sum(G[u][v]['weight'] for u,v in zip(path[:-1], path[1:]))
            if best and t >= best[0]: continue
            d = sum(G[u][v]['length_mi'] for u,v in zip(path[:-1], path[1:]))
            best = (t, d, d/t if t else None, LineString(path))
    return best

def route_one(line, name):
    """Two-stage search:
       1. Heading-constrained (≤120°) — fast, usually Amtrak.
       2. If that fails, no heading filter — allows CSX detours."""
    sp, ep = Point(line.coords[0]), Point(line.coords[-1])

    # --- gather once ---
    cand_start = all_candidates(sp)   # Amtrak + up-to-K_CAND freight
    cand_end   = all_candidates(ep)

    # --- stage 1: with heading filter ---
    S_head = [n for n in cand_start if heading_ok(sp, n, ep, MAX_ANGLE)]
    E_head = [n for n in cand_end   if heading_ok(ep, n, sp, MAX_ANGLE)]

    for k in ESCALATE_STEPS:               # e.g. 10 → 20 → 40
        best = first_viable_pair(S_head[:k], E_head[:k])
        if best:
            return best                    # success on Amtrak / aligned tracks

    # --- stage 2:  no heading filter, same escalation ladder ---
    for k in ESCALATE_STEPS:
        best = first_viable_pair(cand_start[:k], cand_end[:k])
        if best:
            return best                    # fallback (may use CSX component)

    print(f"❌ No viable rail path for {name}")
    return None

# ─── Load corridors & cache ──────────────────────────────────────
corridors = gpd.read_file(corridor_path).to_crs(proj_crs)
if os.path.exists(routed_cache_p) and not force_reroute:
    routed_cache = pickle.load(open(routed_cache_p, "rb"))
else:
    routed_cache = {}

# ─── Worker ──────────────────────────────────────────────────────
def worker(idx_row):
    i, row = idx_row
    key   = f"{i}::{row.get('from')}--{row.get('to')}"  # row-unique cache key
    score = row.get("score", float("nan"))
    sl_mi = row.geometry.length / 1609.34

    if key in routed_cache and not force_reroute:
        t,d,v,geom = routed_cache[key]
    else:
        res = route_one(row.geometry, key)
        routed_cache[key] = res
        t,d,v,geom = res or (None,None,None,None)
    return key, score, sl_mi, t, d, v, geom

# ─── Parallel routing ───────────────────────────────────────────
results, lines, unreachable = [],[],[]
with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
    for key,score,sl,t,d,v,geom in tqdm(pool.map(worker, list(corridors.iterrows())),
                                        total=len(corridors)):
        score_r = round(score,2) if pd.notnull(score) else None
        if t is not None:
            lines.append({"from_to": key, "score": score_r,
                          "geometry": geom, "straight_line": sl})
        else:
            unreachable.append({"from_to": key, "score": score_r})
        results.append({"from_to": key,"score":score_r,"straight_line_mi":sl,
                        "travel_time_hr":t,"network_dist_mi":d,"avg_speed_mph":v})

# ─── Save outputs ────────────────────────────────────────────────
pickle.dump(routed_cache, open(routed_cache_p, "wb"))
pd.DataFrame(results).to_csv(csv_out, index=False)
gpd.GeoDataFrame(lines, geometry="geometry", crs=proj_crs).to_file(geojson_out, driver="GeoJSON")
if unreachable: pd.DataFrame(unreachable).to_csv(unreach_out, index=False)

# ─── Folium preview map ──────────────────────────────────────────
print("Creating Folium preview map …")
gdf_routes_wgs = gpd.read_file(geojson_out).to_crs(wgs_crs)
df_stats       = pd.read_csv(csv_out)

def normalize(s): return s.strip().replace("–", "-").replace("—", "-")
gdf_routes_wgs["from_to"] = gdf_routes_wgs["from_to"].astype(str).apply(normalize)
df_stats["from_to"]       = df_stats["from_to"].astype(str).apply(normalize)
gdf_routes_wgs = gdf_routes_wgs.merge(df_stats, on="from_to", how="left")

if "score_y" in gdf_routes_wgs.columns:
    gdf_routes_wgs["score"] = gdf_routes_wgs.pop("score_y")
    gdf_routes_wgs.drop(columns=["score_x"], inplace=True)

cent = gdf_routes_wgs.geometry.union_all().centroid
m = folium.Map(location=[cent.y, cent.x], zoom_start=5)

dist_rank = gdf_routes_wgs["network_dist_mi"].rank(ascending=True)
cutoff    = dist_rank.quantile(0.3)
for i, r in gdf_routes_wgs.iterrows():
    folium.PolyLine(
        [(y,x) for x,y in r.geometry.coords],
        color  = "red" if dist_rank[i]<=cutoff else "blue",
        weight = 4,
        tooltip=(f"<b>{r['from_to']}</b><br>"
                 f"Score: {r['score']}<br>"
                 f"Straight-line: {r['straight_line_mi']:.1f} mi<br>"
                 f"Network dist: {r['network_dist_mi']:.1f} mi<br>"
                 f"Travel time: {r['travel_time_hr']:.2f} hr<br>"
                 f"Avg mph: {r['avg_speed_mph']:.1f}")
    ).add_to(m)

m.save(map_out)
print(f"✔  Saved → {map_out}")