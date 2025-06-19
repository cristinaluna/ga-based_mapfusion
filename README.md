# GeneticMapFusion

A Python class for aligning and fusing two binary occupancy maps using a genetic algorithm. It includes additional methods for comparison using cross-correlation and ORB+RANSAC-based alignment.

## Features

- **Genetic algorithm-based optimisation** for translation and rotation.
- **Fitness function** using Normalised Cross-Correlation (NCC) for robust evaluation of overlapping areas.
- **Edge-weighted IoU** available for additional metric analysis.
- **Fused map generation** after alignment.
- **Alternative alignment methods**:
  - Cross-correlation
  - ORB keypoints + RANSAC

## Requirements

- Python 3.7+
- NumPy
- SciPy
- scikit-image
- OpenCV
- Matplotlib
- PIL (Pillow)
- IPython (for Jupyter notebook display)

Install dependencies:

```bash
pip install numpy scipy scikit-image opencv-python matplotlib pillow ipython
```

## Usage

### 1. Initialisation

```python
from your_module import GeneticMapFusion

ref_map = ...       # Binary 2D numpy array
target_map = ...    # Binary 2D numpy array

fusion = GeneticMapFusion(ref_map, target_map)
```

### 2. Run Evolutionary Alignment

```python
best_params = fusion.evolve(
    tx_range=(-20, 20),
    ty_range=(-20, 20),
    theta_range=(-15, 15),
    fitness_threshold=0.80,
    min_generations=30,
    max_generations=1000
)
```

### 3. Fuse Maps with Optimal Parameters

```python
fused_map, ref_canvas, target_canvas = fusion.fuse_maps_aligned(best_params)
```

### 4. Alternative Alignment Methods

- **Cross-Correlation:**

```python
params_cc = fusion.align_with_cross_correlation()
```

- **ORB + RANSAC:**

```python
params_ransac = fusion.align_with_ransac()
```

## Output

- `best_params`: `[tx, ty, theta]` transformation to align `target_map` to `ref_map`.
- `fused_map`: Combined occupancy map with partial transparency (0.5) in overlapping areas.
- Optional visualisation using `matplotlib.pyplot.imshow`.

## Notes

- Input maps must be binary numpy arrays.
- Performance depends on image quality, overlap, and initial parameter ranges.
- The evolutionary method is slower but more robust for maps with partial overlaps or low keypoint matches.


# ROS 2 Node

We provide the ROS 2 node to test directly the algorithm reading occupancy grid maps from two different topics (robot 1 and robot 2). These topics are named /map1 and /map2, be sure to change the name if topics are named differently. Fused map is published to /fused_map

## Node structure

map_fusion_ga/
├── map_fusion_ga/
│   ├── __init__.py
│   ├── fusion_node.py
│   └── genetic_fusion.py   
├── launch/
│   └── fusion.launch.py
├── setup.py
├── package.xml

## Build and Run

    colcon build
    source install/setup.bash
    ros2 launch map_fusion_ga fusion.launch.py


# License

GNU GPLv3
