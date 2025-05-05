import numpy as np
import pandas as pd


def calculate_confidence_stats( detections ):

    if not detections:
        return {
            "mean": 0,
            "median": 0,
            "min": 0,
            "max": 0,
            "high_conf": 0,
            "medium_conf": 0,
            "low_conf": 0,
            "confidence_scores": []
        }

    confidence_scores = [ det['confidence' ] for det in detections ]

    return {
        "mean": np.mean( confidence_scores ),
        "median": np.median( confidence_scores ),
        "min": np.min( confidence_scores ),
        "max": np.max( confidence_scores ),
        "high_conf": sum( 1 for c in confidence_scores if c >= 0.7 ),
        "medium_conf": sum( 1 for c in confidence_scores if 0.5 <= c < 0.7 ),
        "low_conf": sum( 1 for c in confidence_scores if c < 0.5 ),
        "confidence_scores": confidence_scores
    }


def conf_score_for_hist(confidence_scores, bins=10):

    hist, edges = np.histogram( confidence_scores, bins=bins, range=(0, 1) )
    return pd.DataFrame({
        "bin_edges": [ f"{ edges[i]:.2f }-{ edges[ i + 1 ]:.2f }" for i in range( len(edges) - 1 ) ],
        "count": hist
    })


def boxes_area( detections ):

    if not detections:
        return {
            "mean": 0,
            "median": 0,
            "min": 0,
            "max": 0,
            "total": 0,
            "areas": []
        }

    areas = []
    for det in detections:
        x1, y1, x2, y2 = det[ 'bbox' ]
        area = (x2 - x1) * (y2 - y1)
        areas.append(area)

    return {
        "mean": np.mean( areas ),
        "median": np.median( areas ),
        "min": np.min( areas ),
        "max": np.max( areas ),
        "total": np.sum( areas ),
        "areas": areas
    }


def box_area_for_hist(areas, bins=10):

    hist, edges = np.histogram(areas, bins=bins)
    return pd.DataFrame({
        "bin_edges": [f"{edges[i]:.0f}–{edges[i + 1]:.0f}" for i in range(len(edges) - 1)],
        "count": hist
    })


def calculate_centers(detections):

    if not detections:
        return {
            "centers": [],
            "mean_x": 0,
            "mean_y": 0,
            "std_x": 0,
            "std_y": 0
        }

    centers = []
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        centers.append((center_x, center_y))

    centers_array = np.array(centers)
    return {
        "centers": centers,
        "mean_x": np.mean(centers_array[:, 0]),
        "mean_y": np.mean(centers_array[:, 1]),
        "std_x": np.std(centers_array[:, 0]),
        "std_y": np.std(centers_array[:, 1])
    }


def centers_for_graph( centers ):
    return pd.DataFrame( centers, columns=[ "x", "y" ] )


def density_grid(centers, image_size=(640, 640), grid_size=8):

    width, height = image_size
    grid = np.zeros((grid_size, grid_size))

    cell_w = width // grid_size
    cell_h = height // grid_size

    for x, y in centers:
        col = min(x // cell_w, grid_size - 1)
        row = min(y // cell_h, grid_size - 1)
        grid[row, col] += 1

    return grid


def count_horses_over_time(detections_by_frame):

    if not detections_by_frame:
        return {
            "counts": [],
            "mean_count": 0,
            "max_count": 0,
            "min_count": 0
        }

    counts = [len(frame_dets) for frame_dets in detections_by_frame]

    return {
        "counts": counts,
        "mean_count": np.mean(counts),
        "max_count": np.max(counts),
        "min_count": np.min(counts)
    }


def zone_active(detections_by_frame, width, height):
    zones = {
        "Top Left": 0,
        "Top Right": 0,
        "Bottom Left": 0,
        "Bottom Right": 0
    }

    mid_x = width / 2
    mid_y = height / 2

    for frame_detections in detections_by_frame:
        for detection in frame_detections:
            # Получаем центр bounding box'а
            x1, y1, x2, y2 = detection['bbox']
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            if center_x < mid_x and center_y < mid_y:
                zones["Top Left"] += 1
            elif center_x >= mid_x and center_y < mid_y:
                zones["Top Right"] += 1
            elif center_x < mid_x and center_y >= mid_y:
                zones["Bottom Left"] += 1
            else:
                zones["Bottom Right"] += 1

    return zones