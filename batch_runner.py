import os
import shutil
import pandas as pd
import yaml
import sys
import glob
from pathlib import Path
from satellite_image_retrieval import retrieve_map_tiles

DATASET_ROOT = "/home/hamit/myProjects/SatelliteLocalization/_VisLoc_dataset"
EXPERIMENTS_ROOT = "/home/hamit/myProjects/SatelliteLocalization/experiments"
BASE_CONFIG_PATH = "/home/hamit/myProjects/SatelliteLocalization/config.yaml"
SUMMARY_CSV = "/home/hamit/myProjects/SatelliteLocalization/experiments_summary.csv"
MAX_QUERIES = 50
ZOOM_LEVELS = [17, 18]

def load_base_config():
    with open(BASE_CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

def save_config(config, path):
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def get_region_name(index_str):
    csv_path = os.path.join(DATASET_ROOT, "satellite_ coordinates_range.csv")
    if not os.path.exists(csv_path):
        return f"Region_{index_str}"
    df = pd.read_csv(csv_path)
    mapname = f"satellite{index_str}.tif"
    row = df[df['mapname'] == mapname]
    if not row.empty:
        return row.iloc[0]['region']
    return f"Region_{index_str}"

def get_latest_stats(output_dir):
    subdirs = sorted(glob.glob(os.path.join(output_dir, "*")), key=os.path.getmtime, reverse=True)
    for subdir in subdirs:
        stats_path = os.path.join(subdir, "localization_stats.txt")
        if os.path.exists(stats_path):
            stats = {}
            with open(stats_path, 'r') as f:
                content = f.read()
                for line in content.split('\n'):
                    if 'Success Rate:' in line:
                        stats['success_rate'] = line.split(':')[1].strip()
                    elif 'Average Error:' in line:
                        stats['avg_error'] = line.split(':')[1].strip()
            return stats
    return None

def process_region(index, summary_list):
    region_id = f"{index:02d}"
    region_name_raw = get_region_name(region_id)
    region_name = region_name_raw.replace(" ", "_").replace("-", "_")
    region_dir = os.path.join(DATASET_ROOT, region_id)
    csv_path = os.path.join(region_dir, f"{region_id}.csv")
    drone_img_dir = os.path.join(region_dir, "drone")
    
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    num_samples = min(len(df), MAX_QUERIES)
    subset = df.sample(n=num_samples, random_state=42).copy()
    
    for zoom in ZOOM_LEVELS:
        exp_name = f"{region_name}_{region_id}_zoom_{zoom}"
        exp_path = os.path.join(EXPERIMENTS_ROOT, exp_name)
        query_dir = os.path.join(exp_path, "data/query")
        map_dir = os.path.join(exp_path, "data/map")
        output_dir = os.path.join(exp_path, "data/output")
        
        if os.path.exists(output_dir):
            existing_stats = get_latest_stats(output_dir)
            if existing_stats:
                print(f"Skipping {exp_name}, already processed. Stats: {existing_stats}")
                summary_list.append({
                    "Region": region_name_raw,
                    "Zoom": zoom,
                    "Success Rate": existing_stats.get('success_rate'),
                    "Avg Error": existing_stats.get('avg_error')
                })
                continue

        os.makedirs(query_dir, exist_ok=True)
        os.makedirs(map_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        query_metadata = []
        for _, row in subset.iterrows():
            filename = str(row['filename'])
            src_img = os.path.join(drone_img_dir, filename)
            dst_img = os.path.join(query_dir, filename)
            if os.path.exists(src_img):
                if not os.path.exists(dst_img):
                    shutil.copy2(src_img, dst_img)
            
            query_metadata.append({
                "Filename": filename,
                "Latitude": row['lat'],
                "Longitude": row['lon'],
                "Altitude": row['height'],
                "Gimball_Roll": row['Kappa'],
                "Gimball_Pitch": -90.0 + row['Omega'],
                "Gimball_Yaw": 0.0,
                "Flight_Roll": 0.0,
                "Flight_Pitch": 0.0,
                "Flight_Yaw": row['Phi1']
            })
        
        pd.DataFrame(query_metadata).to_csv(os.path.join(query_dir, "photo_metadata.csv"), index=False)
        
        lats = subset['lat'].tolist()
        lons = subset['lon'].tolist()
        margin = 0.008 
        lat_max = max(lats) + margin
        lon_min = min(lons) - margin
        lat_min = min(lats) - margin
        lon_max = max(lons) + margin
        
        print(f"Downloading tiles for {exp_name}...")
        tiles = retrieve_map_tiles(lat_max, lon_min, lat_min, lon_max, zoom, map_dir)
        if tiles:
            pd.DataFrame(tiles).to_csv(os.path.join(map_dir, "map.csv"), index=False)
        else:
            print(f"No tiles downloaded for {exp_name}")
            continue
        
        config = load_base_config()
        config['data_paths'] = {
            'query_dir': os.path.abspath(str(query_dir)),
            'map_dir': os.path.abspath(str(map_dir)),
            'output_dir': os.path.abspath(str(output_dir)),
            'query_metadata': os.path.abspath(os.path.join(str(query_dir), "photo_metadata.csv")),
            'map_metadata': os.path.abspath(os.path.join(str(map_dir), "map.csv"))
        }
        config['preprocessing']['enabled'] = True
        config['preprocessing']['save_processed'] = True
        if 'warp' not in config['preprocessing']['steps']:
            config['preprocessing']['steps'].append('warp')
            
        config_path = os.path.join(exp_path, "config.yaml")
        save_config(config, config_path)
        
        print(f"Running localization for {exp_name}...")
        cmd = f"{sys.executable} /home/hamit/myProjects/SatelliteLocalization/localize.py --config {config_path}"
        os.system(cmd)
        
        stats = get_latest_stats(output_dir)
        if stats:
            summary_list.append({
                "Region": region_name_raw,
                "Zoom": zoom,
                "Success Rate": stats.get('success_rate'),
                "Avg Error": stats.get('avg_error')
            })

if __name__ == "__main__":
    summary_list = []
    for i in range(1, 12):
        print(f"\n{'='*20} Processing Region {i} {'='*20}")
        try:
            process_region(i, summary_list)
        except Exception as e:
            print(f"Error processing region {i}: {e}")
    
    if summary_list:
        summary_df = pd.DataFrame(summary_list)
        summary_df.to_csv(SUMMARY_CSV, index=False)
        print(f"\nAll regions processed. Summary saved to {SUMMARY_CSV}")
