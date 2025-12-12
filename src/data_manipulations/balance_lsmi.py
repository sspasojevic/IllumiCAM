
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import rawpy
from tqdm import tqdm
from sklearn.cluster import KMeans

# Add repo root to path to import config
# Assuming we are in IllumiCAM/src/data_manipulations
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # .../src/data_manipulations
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR)) # .../IllumiCAM
sys.path.insert(0, REPO_ROOT)

from config.config import LSMI_DATASET_ROOT, CLUSTER_CENTERS_PATH

# Constants
LSMI_ROOT = os.path.join(LSMI_DATASET_ROOT, "nikon")
META_FILE = os.path.join(LSMI_ROOT, "meta.json")
CLUSTER_CENTERS_FILE = CLUSTER_CENTERS_PATH



# Full Macbeth Color Checker Chart Coordinates (Source)
FULL_CELLCHART = np.float32([
    # Row 1
    [0.25, 0.25],   [2.75, 0.25],   [2.75, 2.75],   [0.25, 2.75],
    [3.00, 0.25],   [5.50, 0.25],   [5.50, 2.75],   [3.00, 2.75], 
    [5.75, 0.25],   [8.25, 0.25],   [8.25, 2.75],   [5.75, 2.75],
    [8.50, 0.25],   [11.00, 0.25],  [11.00, 2.75],  [8.50, 2.75],
    [11.25, 0.25],  [13.75, 0.25],  [13.75, 2.75],  [11.25, 2.75],
    [14.00, 0.25],  [16.50, 0.25],  [16.50, 2.75],  [14.00, 2.75],
    # Row 2  
    [0.25, 3.00],   [2.75, 3.00],   [2.75, 5.50],   [0.25, 5.50],
    [3.00, 3.00],   [5.50, 3.00],   [5.50, 5.50],   [3.00, 5.50],
    [5.75, 3.00],   [8.25, 3.00],   [8.25, 5.50],   [5.75, 5.50],
    [8.50, 3.00],   [11.00, 3.00],  [11.00, 5.50],  [8.50, 5.50],
    [11.25, 3.00],  [13.75, 3.00],  [13.75, 5.50],  [11.25, 5.50],
    [14.00, 3.00],  [16.50, 3.00],  [16.50, 5.50],  [14.00, 5.50],
    # Row 3
    [0.25, 5.75],   [2.75, 5.75],   [2.75, 8.25],   [0.25, 8.25],
    [3.00, 5.75],   [5.50, 5.75],   [5.50, 8.25],   [3.00, 8.25],
    [5.75, 5.75],   [8.25, 5.75],   [8.25, 8.25],   [5.75, 8.25],
    [8.50, 5.75],   [11.00, 5.75],  [11.00, 8.25],  [8.50, 8.25],
    [11.25, 5.75],  [13.75, 5.75],  [13.75, 8.25],  [11.25, 8.25],
    [14.00, 5.75],  [16.50, 5.75],  [16.50, 8.25],  [14.00, 8.25],
    # Row 4
    [0.25, 8.50],   [2.75, 8.50],   [2.75, 11.00],  [0.25, 11.00],
    [3.00, 8.50],   [5.50, 8.50],   [5.50, 11.00],  [3.00, 11.00],
    [5.75, 8.50],   [8.25, 8.50],   [8.25, 11.00],  [5.75, 11.00],
    [8.50, 8.50],   [11.00, 8.50],  [11.00, 11.00], [8.50, 11.00],
    [11.25, 8.50],  [13.75, 8.50],  [13.75, 11.00], [11.25, 11.00],
    [14.00, 8.50],  [16.50, 8.50],  [16.50, 11.00], [14.00, 11.00]
])
MCCBOX = np.float32([[0.00, 0.00], [16.75, 0.00], [16.75, 11.25], [0.00, 11.25]])

def manual_perspective_transform(points, h):
    points = np.array(points)
    if len(points.shape) != 2:
        points = points.reshape(-1, 2)
    points_homo = np.hstack([points, np.ones((points.shape[0], 1))])
    transformed = points_homo @ h.T
    transformed /= transformed[:, 2:3]
    return transformed[:, :2]

def get_patch_chroma(img, mcc_coord):
    h, w = img.shape[:2]
    if np.any(mcc_coord > np.array([w, h])):
        mcc_coord = mcc_coord * 0.5
        
    h_matrix = cv2.getPerspectiveTransform(MCCBOX, mcc_coord)
    if h_matrix is None: return None
    
    gray_patches_indices = [18, 19, 20, 21, 22, 23]
    patch_colors = []
    
    for idx in gray_patches_indices:
        corners_src = FULL_CELLCHART[idx*4 : (idx+1)*4]
        try:
            corners_dst = manual_perspective_transform(corners_src, h_matrix)
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(mask, corners_dst.astype(np.int32), 1)
            mean_val = cv2.mean(img, mask=mask)[:3]
            if np.sum(mean_val) > 0:
                patch_colors.append(mean_val)
        except Exception:
            continue
            
    if not patch_colors: return None
    avg_color = np.mean(patch_colors, axis=0)
    s = np.sum(avg_color)
    if s > 0:
        return avg_color / s
    return None

def extract_illuminants(meta_data, max_samples=None):
    illuminants = []
    places = list(meta_data.keys())
    
    if max_samples is not None and len(places) > max_samples:
        import random
        places = random.sample(places, max_samples)
        
    print(f"Extracting illuminants from {len(places)} samples...")
    
    for place in tqdm(places):
        if place not in meta_data: continue
            
        meta = meta_data[place]
        num_lights = meta.get("NumOfLights", 0)
        
        for i in range(num_lights):
            light_key = f"Light{i+1}"
            if light_key in meta:
                light_rgb = meta[light_key]
                if len(light_rgb) >= 3:
                    # Normalize to chromaticity (r = R / (R+G+B))
                    total = sum(light_rgb)
                    if total > 0:
                        r = light_rgb[0] / total
                        g = light_rgb[1] / total
                        b = light_rgb[2] / total
                        illuminants.append({
                            'r': r, 
                            'g': g, 
                            'b': b, 
                            'place': place, 
                            'light': i+1
                        })
            
    return pd.DataFrame(illuminants)



def main():
    # Load Data
    if os.path.exists(META_FILE):
        with open(META_FILE, "r") as f:
            meta_data = json.load(f)
        print(f"Loaded meta.json with {len(meta_data)} entries.")
    else:
        print("meta.json not found!")
        meta_data = {}

    # Run extraction
    df = extract_illuminants(meta_data, max_samples=None)

    # Ensure no duplicates
    df = df.drop_duplicates(subset=['place', 'light'])

    print(f"Extracted {len(df)} illuminants from {df['place'].nunique()} scenes.")

    # Filter for scenes with at least 2 illuminants
    place_counts = df['place'].value_counts()
    valid_places = place_counts[place_counts >= 2].index
    df = df[df['place'].isin(valid_places)].copy()
    print(f"Filtered to {len(df)} illuminants from {len(valid_places)} scenes (>= 2 lights).")


    # Assign Clusters
    if os.path.exists(CLUSTER_CENTERS_FILE):
        centers = np.load(CLUSTER_CENTERS_FILE, allow_pickle=True)
        if centers.shape == (): centers = centers.item()
        
        if isinstance(centers, dict):
            # Keys are 'Very_Warm', 'Warm', 'Neutral', 'Cool', 'Very_Cool'
            labels = list(centers.keys())
            center_points = np.array([centers[k] for k in labels])
        else:
            center_points = centers
            labels = [f"Cluster_{i}" for i in range(len(centers))]
            
        print(f"Loaded {len(center_points)} cluster centers.")
        
        # Ensure float64 for sklearn
        center_points = center_points.astype(np.float64)
        X = df[['r', 'g', 'b']].values.astype(np.float64)
        
        kmeans = KMeans(n_clusters=len(center_points), init=center_points, n_init=1)
        kmeans.fit(center_points)
        kmeans.cluster_centers_ = center_points
        
        df['cluster'] = kmeans.predict(X)
        
        # Map cluster index back to label
        cluster_names_map = {i: labels[i] for i in range(len(labels))}
        df['cluster_name'] = df['cluster'].map(cluster_names_map)
        
        print("Illuminant cluster counts (before distinct cluster filtering):")
        print(df['cluster_name'].value_counts())

        # Filter scenes: Must have illuminants from at least 2 DIFFERENT clusters
        valid_places_distinct = []
        for place in df['place'].unique():
            place_clusters = df[df['place'] == place]['cluster_name'].unique()
            if len(place_clusters) >= 2:
                valid_places_distinct.append(place)
                
        df = df[df['place'].isin(valid_places_distinct)].copy()
        print(f"\nFiltered to {len(df)} illuminants from {len(valid_places_distinct)} scenes (>= 2 distinct clusters).")
        
        print("Illuminant cluster counts (after distinct cluster filtering):")
        print(df['cluster_name'].value_counts())
        
        # Balance Scenes based on "Rarest" Cluster
        # Define rarity hierarchy (based on typical distribution or just preference)
        # We want to prioritize Very_Warm and Warm
        rarity_order = ["Very_Warm", "Warm", "Very_Cool", "Neutral", "Cool"]
        rarity_rank = {name: i for i, name in enumerate(rarity_order)}
        
        # Determine the "rarest" cluster for each scene
        scene_rarity = []
        for place in df['place'].unique():
            place_clusters = df[df['place'] == place]['cluster_name'].unique()
            # Find the cluster with the lowest rank (most rare)
            best_rank = float('inf')
            best_cluster = None
            for c in place_clusters:
                if c in rarity_rank:
                    rank = rarity_rank[c]
                    if rank < best_rank:
                        best_rank = rank
                        best_cluster = c
            
            if best_cluster:
                scene_rarity.append({'place': place, 'rarest_cluster': best_cluster})
                
        scene_df = pd.DataFrame(scene_rarity)
        print("\nScene counts by rarest cluster:")
        print(scene_df['rarest_cluster'].value_counts())
        
        # Balance scenes
        counts = scene_df['rarest_cluster'].value_counts()
        min_count = counts.min()
        print(f"Target scenes per category: {min_count}")
        
        balanced_places = []
        for cluster in counts.index:
            cluster_scenes = scene_df[scene_df['rarest_cluster'] == cluster]
            if len(cluster_scenes) > min_count:
                 sampled_scenes = cluster_scenes.sample(n=min_count, random_state=42)
            else:
                 sampled_scenes = cluster_scenes
            balanced_places.extend(sampled_scenes['place'].tolist())
            
        # Filter original dataframe to keep only selected places
        df_balanced = df[df['place'].isin(balanced_places)].copy()
        
        print(f"\nBalanced dataset size: {len(df_balanced)} illuminants from {len(balanced_places)} scenes.")
        print("Illuminant cluster counts (after balancing):")
        print(df_balanced['cluster_name'].value_counts())
        
        # Save balanced dataset
        df_balanced.to_csv("lsmi_balanced.csv", index=False)
        print("Saved balanced dataset to lsmi_balanced.csv")
        
    else:
        print("Cluster centers file not found!")

if __name__ == "__main__":
    main()
