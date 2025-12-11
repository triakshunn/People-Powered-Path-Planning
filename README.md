# Leader-Follower Navigation with Social Force Model Integration

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation of a **Leader-Follower navigation algorithm** integrated with the **Social Force Model (SFM)** for socially-aware robot navigation in crowded environments. We compare our approach against an MPC-based baseline across multiple challenging scenarios.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation Scenarios](#evaluation-scenarios)
- [Evaluation Metrics](#evaluation-metrics)
- [Reproducing Results](#reproducing-results)
- [Citation](#citation)
- [License](#license)

---

## Overview

Navigating robots through dense human crowds requires balancing efficiency with social compliance. This project introduces a **Leader-Follower** approach where the robot identifies and follows pedestrians heading toward similar destinations, leveraging human navigation expertise to achieve socially compliant paths.

**Key Contributions:**
- A Leader-Follower planner that dynamically selects pedestrian leaders based on goal alignment and trajectory prediction
- Integration with Social Force Model for reactive collision avoidance
- Comparative evaluation against Model Predictive Control (MPC) baseline
- Novel evaluation metrics including **Path Smoothness** and **Invasion of Personal Space**

---

## Project Structure

```
ROB599LeaderFollowerAlgorithm/
├── simulator.py              # Base simulation environment with evaluation metrics
├── mpc_controller.py         # Social Force compliant MPC controller
├── leader_follower.py        # Leader-Follower planner implementation
├── LeaderFollowerSFM.py      # Leader-Follower + SFM integration module
├── environment.yml           # Conda environment specification
├── evals/                    # Evaluation notebooks for all scenarios
│   ├── BlackFridayLF.ipynb   # Black Friday scenario - Leader-Follower
│   ├── BlackFridayMPC.ipynb  # Black Friday scenario - MPC
│   ├── ConventionLF.ipynb    # Convention scenario - Leader-Follower
│   ├── ConventionMPC.ipynb   # Convention scenario - MPC
│   ├── JunctionLF.ipynb      # Junction scenario - Leader-Follower
│   ├── JunctionMPC.ipynb     # Junction scenario - MPC
│   ├── MuseumLF.ipynb        # Museum scenario - Leader-Follower
│   ├── MuseumMPC.ipynb       # Museum scenario - MPC
│   ├── PromenadeLF.ipynb     # Promenade scenario - Leader-Follower
│   ├── PromenadeMPC.ipynb    # Promenade scenario - MPC
│   ├── RoundaboutLF.ipynb    # Roundabout scenario - Leader-Follower
│   ├── RoundaboutMPC.ipynb   # Roundabout scenario - MPC
│   └── analyze_results.py    # Results analysis script
├── blender/                  # 3D visualization files
│   ├── BlackFriday.blend     # Black Friday scenario visualization
│   ├── Convention.blend      # Convention scenario visualization
│   ├── Junction.blend        # Junction scenario visualization
│   ├── Museum.blend          # Museum scenario visualization
│   └── Roundabout.blend      # Roundabout scenario visualization
├── results/                  # Evaluation logs (JSON format)
├── mpc_params_search/        # MPC parameter grid search notebooks
└── media/                    # Generated visualizations
```

### Module Descriptions

| Module | Description |
|--------|-------------|
| `simulator.py` | Core crowd simulation environment with agents, obstacles, Social Force Model, and evaluation metrics (path smoothness, personal space invasion) |
| `mpc_controller.py` | Model Predictive Control local planner with social force-aware cost functions for goal reaching, obstacle avoidance, and personal space preservation |
| `leader_follower.py` | Leader-Follower planner that identifies optimal pedestrian leaders based on goal alignment, velocity matching, and position scoring |
| `LeaderFollowerSFM.py` | Integration layer combining Leader-Follower subgoal planning with SFM for reactive navigation |

### Blender Visualizations

The `blender/` directory contains 3D visualization files for each evaluation scenario. These `.blend` files can be opened with [Blender](https://www.blender.org/) (version 3.0+) to visualize the simulation environments and generate high-quality renders of the navigation scenarios.

| File | Scenario |
|------|----------|
| `BlackFriday.blend` | High-density shopping environment |
| `Convention.blend` | Convention hall with corridors |
| `Junction.blend` | Intersection crossing |
| `Museum.blend` | Museum gallery space |
| `Roundabout.blend` | Circular pedestrian area |

---

## Installation

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution)
- Python 3.10+

### Setup Environment

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sebhelge/ROB599LeaderFollowerAlgorithm.git
   cd ROB599LeaderFollowerAlgorithm
   ```

2. **Create the conda environment:**
   ```bash
   conda env create -f environment.yml
   ```

3. **Activate the environment:**
   ```bash
   conda activate comphri_hw5
   ```

4. **Verify installation:**
   ```bash
   python -c "import numpy; import matplotlib; print('Installation successful!')"
   ```

---

## Usage

### Running the Simulator

```python
from simulator import CrowdSimulator, Agent, SocialForceModel

# Initialize simulation
sim = CrowdSimulator(time_step=0.25, max_steps=200)

# Add robot and humans
sim.add_robot(x=0, y=0, gx=10, gy=10)
sim.add_human(x=5, y=5, gx=0, gy=0)

# Set navigation policy
from LeaderFollowerSFM import Leader_Follower_SFM
policy = Leader_Follower_SFM(sim=sim)
sim.set_robot_policy(policy)

# Run simulation
while not sim.done:
    sim.step()

# Calculate metrics
metrics = sim.calculate_metrics()
```

### Using the MPC Controller

```python
from mpc_controller import MPCLocalPlanner

planner = MPCLocalPlanner(
    simulation=sim,
    horizon=10,
    dt=0.25,
    max_speed=1.4,
    wg=1.0,   # Goal weight
    ws=1.0,   # Static obstacle weight
    wd=1.0,   # Dynamic obstacle weight
    wps=0.5   # Personal space weight
)
```

---

## Evaluation Scenarios

We evaluate our approach across six challenging crowd scenarios:

| Scenario | Description |
|----------|-------------|
| **Black Friday** | High-density chaotic shopping environment with erratic pedestrian movements |
| **Convention** | Dense bidirectional pedestrian flow in narrow corridors |
| **Junction** | Multi-directional crossing at an intersection |
| **Museum** | Sparse crowds with unpredictable pedestrian behavior |
| **Promenade** | Linear walkway with mixed pedestrian flow patterns |
| **Roundabout** | Circular pedestrian flow with continuous merging and diverging |

---

## Evaluation Metrics

Our evaluation framework includes the following metrics:

| Metric | Description |
|--------|-------------|
| **Time to Goal** | Number of simulation steps to reach the goal |
| **Completion Rate** | Percentage of runs where the robot successfully reaches the goal |
| **Path Efficiency** | Ratio of actual path length to straight-line distance |
| **Average Speed** | Mean velocity throughout the trajectory |
| **Path Smoothness** | Measures trajectory roughness based on angular deviations |
| **Personal Space Invasion** | Count of instances where robot enters human personal space (3m bubble) |
| **Minimum Human Distance** | Closest approach to any pedestrian during navigation |
| **Collision Rate** | Percentage of runs resulting in collision |

---

## Reproducing Results

### Step 1: Activate Environment
```bash
conda activate comphri_hw5
```

### Step 2: Navigate to Evaluation Directory
```bash
cd evals/
```

### Step 3: Run Scenario Notebooks

Open and execute the Jupyter notebooks for each scenario:

| Scenario | Leader-Follower | MPC Baseline |
|----------|-----------------|--------------|
| Black Friday | `BlackFridayLF.ipynb` | `BlackFridayMPC.ipynb` |
| Convention | `ConventionLF.ipynb` | `ConventionMPC.ipynb` |
| Junction | `JunctionLF.ipynb` | `JunctionMPC.ipynb` |
| Museum | `MuseumLF.ipynb` | `MuseumMPC.ipynb` |
| Promenade | `PromenadeLF.ipynb` | `PromenadeMPC.ipynb` |
| Roundabout | `RoundaboutLF.ipynb` | `RoundaboutMPC.ipynb` |

```bash
jupyter notebook ConventionLF.ipynb
```

### Step 4: Analyze Results

```bash
python analyze_results.py ../results/convention_LeaderFollower_n100_20251207_210535.json
```

### Pre-computed Results

All evaluation results are stored in the `results/` directory as JSON files, containing metrics from 100 randomized runs per scenario.

---

## Citation

This project implements the Leader-Follower approach from the following paper:

```bibtex
@misc{liao2025followingneedrobotcrowd,
      title={Following Is All You Need: Robot Crowd Navigation Using People As Planners}, 
      author={Yuwen Liao and Xinhang Xu and Ruofei Bai and Yizhuo Yang and Muqing Cao and Shenghai Yuan and Lihua Xie},
      year={2025},
      eprint={2504.10828},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2504.10828}, 
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- ROB 599: Human-Robot Interaction, University of Michigan