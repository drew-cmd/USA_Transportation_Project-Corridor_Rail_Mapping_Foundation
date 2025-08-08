# USA Transportation Project Mapping Foundation

This project analyzes and visualizes the U.S. passenger and freight rail networks alongside metro areas and airports to identify and rank potential high-speed rail (HSR) corridors.
It integrates spatial datasets from federal sources, calculates corridor scores based on population interaction and distance, and uses network routing to estimate realistic travel times between major metro areas.
Outputs include ranked corridor tables, GeoJSON shapefiles, and interactive Folium maps for visualization.

---

## Project Structure
```
USA_TRANSPORTATION_PROJECT_MAPPING_FOUNDATION/
├── Data/
│ ├── NTAD_Amtrak_Routes_/Amtrak_Routes.shp
│ ├── NTAD_North_American_Rail_Network_Lines_/North_American_Rail_Network_Lines.shp
│ ├── corridors_top100.geojson
│ └── places_usa_2023.gpkg
│
├── Output/
│ ├── graph_cache.pkl
│ ├── routed_corridors_cache.pkl
│ ├── estimated_corridor_travel_times.csv
│ ├── corridor_routes.geojson
│ ├── unreachable_corridors.geojson             --if applicable
│ └── corridor_routes_map.html
│
├── corridor_rail_mapping.py
├── LICENSE
├── NOTICE
├── README.md
├── .gitattributes
├── .gitignore
```

---

## Features

- Loads and processes shapefiles for:
  - Amtrak routes (passenger network)
  - North American freight rail network (used as supplemental passenger paths for routing)
- Builds a routable rail network graph with:
  - Densified segments (≤ 2 km) for realistic curvature and routing
  - Edge weights based on travel time (distance / assumed speed)
- Routes the top 100 metro-to-metro corridors using A* pathfinding:
  - Prefers Amtrak-aligned paths but can fall back to freight lines
  - Stores results in cache files for fast re-runs
- Outputs:
  - Travel time, network distance, and average speed per corridor
  - A GeoJSON of routed paths
  - An interactive HTML map preview

---

## corridor_rail_mapping.py — Key Inputs & Outputs

Inputs:
- Data/NTAD_Amtrak_Routes_/Amtrak_Routes.shp – Passenger rail lines from DOT/NTAD
- Data/NTAD_North_American_Rail_Network_Lines_/North_American_Rail_Network_Lines.shp – Freight rail lines
- Data/corridors_top100.geojson – Candidate corridors to route

Outputs:
- Output/estimated_corridor_travel_times.csv 
    ( Travel time, distance, avg speed per corridor )
- Output/corridor_routes.geojson
    ( Routed paths for successful corridors )
- Output/unreachable_corridors.csv
    ( Corridors that could not be routed )
- Output/graph_cache.pkl
    ( Cached routing graph (NetworkX + spatial index) )
- Output/routed_corridors_cache.pkl
    ( Cached corridor results )
- Output/corridor_routes_map.html
    ( Interactive Folium map of routed corridors )

---

## Requirements

Install dependencies via pip:

```bash
pip install geopandas folium branca shapely pyproj pandas networkx tqdm scipy
```

Other system-level requirements:

GDAL/OGR (for reading shapefiles)
Git LFS (for handling large files tracked in this repo)

---

### How to Run

Prepare data
- Ensure the Data/ folder contains:
    - Amtrak shapefile
    - Freight rail shapefile
    - Top 100 corridors GeoJSON

Run the main script:

```
python "corridor_rail_mapping.py"

```

This will:

- Load and process the rail network
- Route each corridor using A*
- Save results to Output/ as CSV, GeoJSON, and HTML map



### Notes

Notes
- Caching:
    - graph_cache.pkl prevents rebuilding the rail graph each run
    - routed_corridors_cache.pkl prevents re-routing unchanged corridors
    - Set force_reroute=True in corridor_rail_mapping.py to ignore caches
- Viewing outputs:
    - Open .html files in your browser
    - Load .geojson in GIS software (e.g., QGIS)

- Large file handling:
```
git lfs install

```

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).

### Data Sources

The datasets used in this project originate from U.S. government sources (e.g., NTAD, TIGER/Line, ACS) and are in the public domain under 17 U.S.C. § 105.

For more information, see the [`NOTICE`](NOTICE) file.
