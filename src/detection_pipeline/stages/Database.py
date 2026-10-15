#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 15:41:52 2026

@author: Harry Birch

Code to construct database of prominences and active regions from label parameter files.
Outputs a single csv for the databse for prominences, class 0, and active regions, class 1
"""

from pathlib import Path
import pandas as pd
import numpy as np
import csv


# Load and combine all csv together. Add parameters for limb position and convert t_obs to datetime
def load_all_properties(csv_dir: Path):
    csv_files = sorted(Path(csv_dir).glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")
    
    
    df = pd.concat(
     (pd.read_csv(csv_path) for csv_path in csv_files),
     ignore_index=True
     )

    df['t_obs'] = pd.to_datetime(df['t_obs'])
    df['limb_position'] = np.where(
        df['longitude'] < 0, "West", "East"
    )

    return df


def compute_iou(row1, row2):
    # Reconstruct corners from centre + half-dimensions
    def corners(row):
        x0 = row['x_centre_pix'] - row['box_w_pix'] / 2
        y0 = row['y_centre_pix'] - row['box_h_pix'] / 2
        x1 = row['x_centre_pix'] + row['box_w_pix'] / 2
        y1 = row['y_centre_pix'] + row['box_h_pix'] / 2
        return x0, y0, x1, y1
    
    ax0, ay0, ax1, ay1 = corners(row1)
    bx0, by0, bx1, by1 = corners(row2)

    # Intersection
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    intersection = max(0, ix1 - ix0) * max(0, iy1 - iy0)

    # Union
    area_a = row1['box_w_pix'] * row1['box_h_pix']
    area_b = row2['box_w_pix'] * row2['box_h_pix']
    union  = area_a + area_b - intersection

    return intersection / union if union > 0 else 0

# Function to return indexes of rows to remove containing overlapping box for each class
def remove_overlapping_boxes(dataframe, spatial_iou_threshold: float=0.25):
    to_remove = set()
    
    # Iterate over groups (images) and label classes# This ensures that only labels within the same classes are removed.
    for (t, label_class), group in dataframe.groupby(['t_obs', 'class']):
        
        #print(t, label_class)
        
        # outer loop - iterate over rows in group (labels)
        for i in range(len(group)):
            # inner loop
            for j in range(i+1, len(group)):
                # compute IoU for i and j
                iou = compute_iou(group.iloc[i], group.iloc[j])
                #print(i, j, iou)
                if iou > spatial_iou_threshold:
                    # Check which box to remove based on which has lower confidence
                    if group.iloc[i].confidence > group.iloc[j].confidence:
                        #print(f"removing box {j}")
                        to_remove.add(int(group.index[j])) # add original 'i' index to_remove
                    else:
                        #print(f"removing box {i}")
                        to_remove.add(int(group.index[i])) # add original 'j' index to_remove

    return to_remove                

# Function to return indexes of rows to remove containing overlapping box for each class
def track_events(dataframe,cadence=6, temporal_iou_threshold=0.5, event_persistence_cadence=0):
    df = dataframe.copy()
    df["event_id"] = -1  # New column for event_id set default to -1
    next_event_id = 0    # set default value for next_event_id
    matches = []         # Stores pairwise tracking links between frames for each class -> (match in frame t, with match in frame t+1)
    
    # tolernace for subsequent detections
    allowed_gap = (pd.Timedelta(hours=cadence)
                   + pd.Timedelta(event_persistence_cadence)
                   )
    # group dataframe by time and class
    for label_class, class_group in df.groupby('class'):
        grouped_df = class_group.groupby('t_obs')
        groups = list(grouped_df)
        class_matches = [] # Stores pairwise matches in a given class
        
        # case of no groups
        if len(groups) == 0:
            continue

        # Assign event_id to first frame of this class
        first_t, first_group = groups[0] 
        for idx in first_group.index:
            df.loc[idx, "event_id"] = next_event_id
            next_event_id += 1
        
        # Iterate over consecutive frame pairs for this class
        for group_idx, (t, group_t) in enumerate(groups[:-1]):
            t1, group_t1 = groups[group_idx + 1] # define next timestamp and group
            
            # Check frames are actually temporally adjacent (to better deal with less frequent AR class)
            # (skips matching if a class is absent for many frames between appearances)
            if (t1 - t) > allowed_gap:
                for idx in group_t1.index:
                    df.loc[idx, "event_id"] = next_event_id
                    next_event_id += 1
                continue
            
            # ensures code matches one object in frame t to one object in frame t+1
            used = set()  
            
            # Iterate over the rows in the next group
            for next_row_idx, next_row in group_t1.iterrows():
                # intialise best overlap score found for a candidate match and corresponding row.
                best_iou, best_current_row_idx = 0, None
                
                # Iterate over the rows in the current group            
                for current_row_idx, current_row in group_t.iterrows():
                    # Ignore any rows in the current group which have aready been used
                    if current_row_idx in used:
                        continue
                    # calculate IoU for the next and current rows
                    iou = compute_iou(next_row, current_row)
                    # Condition to keep track of the highest-IoU row
                    if iou > best_iou:
                        best_iou, best_current_row_idx = iou, current_row_idx
                
                # For matching boxes above Iou threshold, append matching row indices
                if best_iou > temporal_iou_threshold:
                    class_matches.append((best_current_row_idx, next_row_idx))
                    used.add(best_current_row_idx)
                    # inherit previous event id
                    event_id = df.loc[best_current_row_idx, "event_id"]
                    df.loc[next_row_idx, "event_id"] = event_id
                else:
                    # create new event_id for non-matching boxes
                    df.loc[next_row_idx, "event_id"] = next_event_id
                    next_event_id += 1
        matches.append(class_matches)    
        
    return df, matches


def lifetime(event_dataframe, cadence):
    df = event_dataframe.copy()
    df["lifetime"] = pd.Timedelta(0) # Add new datetime column for event lifetime
   
    for event_id, event_group in event_dataframe.groupby('event_id'):
       #print(f"{event_id} | size={len(event_group)}")
       df.loc[event_group.index, "lifetime"] = len(event_group)*pd.Timedelta(hours=cadence)  # New column for event_id lifetime

    return df


def create_database(
    csv_dir: Path,
    cadence: str | None = None,
    spatial_iou_threshold: float = 0.25,
    temporal_iou_threshold: float = 0.5,
    event_persistence_cadence: str = "0h",
):    
    global df
    
    df = load_all_properties(csv_dir)
    
    # Spatial filtering
    to_remove = remove_overlapping_boxes(dataframe=df)
    filtered_df = df.drop(df.index[list(to_remove)]) # Remove corresponding rows from df

    if cadence is None:
        # Calculate cadence based on t_obs differences. Assumes a constant cadence for observations
        cadence = round(
            (df['t_obs'].unique()[1] - df['t_obs'].unique()[0]) / pd.Timedelta(hours=1)
            )
    else:
        # Use DRMS cadence value
        cadence = pd.Timedelta(cadence) / pd.Timedelta(hours=1)
        
    # Track events and assign event_ids
    event_df, events = track_events(dataframe=filtered_df,
                                    cadence=cadence,
                                    temporal_iou_threshold=temporal_iou_threshold,
                                    event_persistence_cadence=event_persistence_cadence)  
    
    # Calculate event lifetime
    lifetime_df = lifetime(event_dataframe=event_df, cadence=cadence) 
    
    # save output as csv
    dst = csv_dir.parent.parent / "database.csv"
    lifetime_df.to_csv(dst, index=False)
    print(f"database.csv saved to {csv_dir.parent.parent}")
    
    return lifetime_df
            
