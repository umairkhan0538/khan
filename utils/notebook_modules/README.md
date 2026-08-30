# notebook_modules

This directory contains the code that was previously written directly inside the
two notebooks in `../`. Each notebook now imports one of these modules, creates
a single shared `state` object, and calls the matching cell function for every
original code cell.

## Files

| File | Corresponding notebook | Cell functions |
| --- | --- | --- |
| `single_run_analysis.py` | `../01_single_run_analysis.ipynb` | `cell_00` … `cell_14` |
| `building_clustering.py` | `../1_similarity_reward_zero transfer.ipynb` | `cell_02`, `cell_04`, … `cell_41` |

## Cell function map

`single_run_analysis.py`

| Function | Purpose |
| --- | --- |
| `cell_00` | Imports, warnings, matplotlib-inline setup |
| `cell_01` | Paths and loading of the evaluated timeseries |
| `cell_02` | Requested episode-level KPIs and styled display |
| `cell_03` | Publication plotting style and `save_figure` helper |
| `cell_04` | Timeseries schema and data-integrity validation |
| `cell_05` | Plot 1: cumulative physical KPIs with zoomed insets |
| `cell_06` | Plot 3: weekly operational profile with peak-tariff shading |
| `cell_07` | Plot 6: battery SOC occupancy distribution |
| `cell_08` | Plot 7: battery charging/discharging behaviour |
| `cell_09` | Plot 10: PV energy allocation |
| `cell_10` | Plot 12: weighted reward/penalty contributions |
| `cell_11` | Plot 14: aggregate daily operating profiles |
| `cell_12` | Plot 15: TD3 action and realized battery heatmaps |
| `cell_13` | Plot 20: weekly cost/carbon trade-off |

`building_clustering.py`

| Function | Purpose |
| --- | --- |
| `cell_02` | Imports, session options, global config |
| `cell_04` | Dataset/schema paths, development interval, output config |
| `cell_06` | Discover files and inspect columns |
| `cell_08` | Load and validate development-period data + PV conversion |
| `cell_10` | Check physical scales and signs |
| `cell_12` | Optional battery metadata |
| `cell_15` | Physical descriptor extraction |
| `cell_17` | Descriptor distributions and correlation audit |
| `cell_19` | Select physically complementary clustering features |
| `cell_21` | Robust scaling |
| `cell_23` | Screen number of K-means clusters |
| `cell_25` | Final K and cluster summaries |
| `cell_27` | Initialization stability |
| `cell_29` | Day-bootstrap stability |
| `cell_31` | Hierarchical-clustering sensitivity |
| `cell_33` | True within-cluster medoids and portfolio medoid |
| `cell_35` | PCA visualization |
| `cell_37` | Cluster-wise cyclic load/PV profiles |
| `cell_39` | Shape-normalized load analysis |
| `cell_41` | Final development panel and exports |

## How the rewrite works

- Every notebook variable that previously lived in the IPython kernel
  namespace now lives on one shared object, created by `create_state()`.
- Every original code cell is a module function, for example
  `cell_06(state)`.
- Functions defined in a cell (e.g. `save_figure`, `validate_timeseries`,
  `extract_descriptors`) are attached to the shared `state` so later cells can
  still call them.
- Cell functions mutate `state` in place. Run them in notebook order, exactly
  like the original cells; later cells see the variables produced by earlier
  cells.
- The notebooks also expose `run_all(state=None)`, which executes every cell
  function on one shared state in order.

## Usage in a notebook

The rewritten notebooks already contain the setup cell. It looks like this
(module names differ per notebook):

```python
from pathlib import Path
import sys

_NOTEBOOK_DIR = Path.cwd().resolve()
if (_NOTEBOOK_DIR / 'notebook_modules').is_dir():
    _PACKAGE_ROOT = _NOTEBOOK_DIR
elif (_NOTEBOOK_DIR.parent / 'utils' / 'notebook_modules').is_dir():
    _PACKAGE_ROOT = _NOTEBOOK_DIR.parent / 'utils'
elif (_NOTEBOOK_DIR / 'utils' / 'notebook_modules').is_dir():
    _PACKAGE_ROOT = _NOTEBOOK_DIR / 'utils'
else:
    _PACKAGE_ROOT = _NOTEBOOK_DIR

if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from notebook_modules import single_run_analysis as m

state = m.create_state()
m.cell_00(state)
```

Every later code cell is simply:

```python
m.cell_06(state)
```

## Environment notes

- The modules import the same libraries the notebooks used: `numpy`, `pandas`,
  `matplotlib`, `seaborn`, `scipy`, `scikit-learn`, and `IPython`.
- `pandas` `.style` output needs `jinja2`, exactly as it did in the original
  notebook.
- The clustering notebook still expects the CityLearn dataset at the Windows
  path configured in `building_clustering.py::cell_04`; change that path if you
  work on a different machine.
