# -*- coding: utf-8 -*-
"""Auto-generated notebook code module.

The ``cell_XX`` functions below each contain the logic of one notebook code
cell. Notebook variables that used to live in the IPython kernel global
namespace are stored on a shared ``state`` object (a simple namespace). Each
notebook imports this module, creates one ``state`` object, and calls the cell
functions in order.
"""
from pathlib import Path
import json
import warnings
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import seaborn as sns

from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import RobustScaler

try:
    from IPython.display import display
except Exception:  # pragma: no cover
    display = None

try:
    _IPYTHON = get_ipython()
    if _IPYTHON is not None:
        _IPYTHON.run_line_magic('matplotlib', 'inline')
except Exception:
    pass

warnings.filterwarnings('default')
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 180)
sns.set_context('notebook')


def create_state():
    """Create a fresh shared notebook state object."""
    return SimpleNamespace()


def cell_02(state):
    """Cell 02: imports, session options, and global configuration."""
    from pathlib import Path
    import json
    import warnings

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import zscore
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.decomposition import PCA
    from sklearn.metrics import (
        adjusted_rand_score,
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )
    from sklearn.preprocessing import RobustScaler
    from matplotlib import ticker

    warnings.filterwarnings('default')
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.width', 180)
    sns.set_context('notebook')

    state.RANDOM_STATE = 0
    state.EPSILON = 1e-8

def cell_04(state):
    """Cell 04: dataset/schema paths, development interval, and output config."""
    from pathlib import Path


    # --------------------------------------------------------------------------
    # Dataset and schema paths
    # --------------------------------------------------------------------------
    state.DATA_PATH = Path(
        r'D:\Research\Citylearn_related\Citylearn_data\data\citylearn_challenge_2022_phase_all'
    )

    state.SCHEMA_PATH = state.DATA_PATH / 'schema.json'

    state.BUILDING_IDS = list(range(1, 18))


    # --------------------------------------------------------------------------
    # Development interval
    # --------------------------------------------------------------------------
    # Inclusive indices. The final test interval must remain excluded from
    # descriptor extraction, clustering, and representative-building selection.
    state.DEVELOPMENT_START = 0
    state.DEVELOPMENT_END = 7671


    # --------------------------------------------------------------------------
    # Building CSV columns
    # --------------------------------------------------------------------------
    state.LOAD_COLUMN = 'Equipment Electric Power [kWh]'

    # This is a normalized PV-generation profile in W/kW. It must be multiplied
    # by each building's PV nominal power from schema.json and divided by 1000.
    state.PV_COLUMN = 'Solar Generation [W/kW]'

    # One-hour simulation time step.
    state.TIME_STEP_HOURS = 1.0


    # --------------------------------------------------------------------------
    # Optional battery metadata
    # --------------------------------------------------------------------------
    # Provide a CSV path if battery capacity or nominal power differs by building.
    # Expected columns:
    # building_id, battery_capacity_kwh, battery_nominal_power_kw
    state.BATTERY_METADATA_PATH = None


    # --------------------------------------------------------------------------
    # K-means and cluster-validation settings
    # --------------------------------------------------------------------------
    # Candidate numbers of clusters to evaluate.
    state.K_VALUES = list(range(2, 6))

    # Number of independent centroid initializations within each K-means fit.
    state.KMEANS_N_INIT = 50

    # Maximum number of iterations for each K-means initialization.
    state.KMEANS_MAX_ITER = 500

    # Number of complete K-means repetitions used to assess sensitivity to
    # random initialization.
    state.N_INITIALIZATION_REPEATS = 50

    # Number of complete-day bootstrap samples used to assess temporal stability.
    state.N_DAY_BOOTSTRAPS = 100

    # Absolute Spearman-correlation threshold for flagging potentially
    # redundant physical descriptors.
    state.REDUNDANCY_THRESHOLD = 0.95


    # --------------------------------------------------------------------------
    # Output directory
    # --------------------------------------------------------------------------
    state.OUTPUT_DIR = Path('building_clustering_outputs')
    state.OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------------------------
    # Basic configuration checks
    # --------------------------------------------------------------------------
    if not state.DATA_PATH.is_dir():
        raise FileNotFoundError(
            'Dataset directory does not exist: {}'.format(state.DATA_PATH)
        )

    if not state.SCHEMA_PATH.is_file():
        raise FileNotFoundError(
            'CityLearn schema does not exist: {}'.format(state.SCHEMA_PATH)
        )

    if state.DEVELOPMENT_START < 0:
        raise ValueError(
            'DEVELOPMENT_START must be non-negative.'
        )

    if state.DEVELOPMENT_END < state.DEVELOPMENT_START:
        raise ValueError(
            'DEVELOPMENT_END must be greater than or equal to '
            'DEVELOPMENT_START.'
        )

    if state.TIME_STEP_HOURS <= 0.0:
        raise ValueError(
            'TIME_STEP_HOURS must be greater than zero.'
        )


    print('Dataset directory :', state.DATA_PATH)
    print('Schema file       :', state.SCHEMA_PATH)
    print(
        'Development range : {} to {} inclusive'.format(
            state.DEVELOPMENT_START,
            state.DEVELOPMENT_END
        )
    )
    print(
        'Development hours :',
        state.DEVELOPMENT_END - state.DEVELOPMENT_START + 1
    )
    print('Output directory  :', state.OUTPUT_DIR.resolve())

def cell_06(state):
    """Cell 06: discover files and inspect columns."""
    # --------------------------------------------------------------------------
    # Validate dataset files and inspect the building CSV structure
    # --------------------------------------------------------------------------

    if not state.DATA_PATH.is_dir():
        raise FileNotFoundError(
            'Dataset directory does not exist: {}'.format(
                state.DATA_PATH
            )
        )

    if not state.SCHEMA_PATH.is_file():
        raise FileNotFoundError(
            'CityLearn schema file does not exist: {}'.format(
                state.SCHEMA_PATH
            )
        )


    # Build and validate paths for all building CSV files.
    state.file_paths = {}
    state.missing_files = []

    for state.building_id in state.BUILDING_IDS:
        state.file_path = (
            state.DATA_PATH
            / 'Building_{}.csv'.format(state.building_id)
        )

        state.file_paths[state.building_id] = state.file_path

        if not state.file_path.is_file():
            state.missing_files.append(
                str(state.file_path)
            )

    if state.missing_files:
        raise FileNotFoundError(
            'The following building files are missing:\n{}'.format(
                '\n'.join(state.missing_files)
            )
        )


    # Read a short sample from the first building.
    state.sample_building_id = state.BUILDING_IDS[0]
    state.sample_file = state.file_paths[state.sample_building_id]

    state.sample = pd.read_csv(
        state.sample_file,
        nrows=5
    )


    # Validate the configured load and PV columns.
    state.required_columns = [
        state.LOAD_COLUMN,
        state.PV_COLUMN
    ]

    state.missing_columns = [
        state.column
        for state.column in state.required_columns
        if state.column not in state.sample.columns
    ]

    if state.missing_columns:
        raise KeyError(
            'The sample building file is missing the configured '
            'columns: {}.\nAvailable columns: {}'.format(
                state.missing_columns,
                state.sample.columns.tolist()
            )
        )


    # Display the inspected file and available columns.
    print(
        'Sample building file:',
        state.sample_file
    )

    print('\nAvailable columns:')

    for state.column in state.sample.columns:
        print(
            '  - {!r}'.format(state.column)
        )

    print('\nConfigured physical columns:')
    print(
        '  Load column:',
        state.LOAD_COLUMN
    )
    print(
        '  PV profile column:',
        state.PV_COLUMN
    )

    print('\nSample data:')
    display(state.sample)


    # Display basic values from the selected columns.
    print('\nLoad-column sample values:')
    print(
        state.sample[state.LOAD_COLUMN].to_list()
    )

    print('\nNormalized PV-profile sample values:')
    print(
        state.sample[state.PV_COLUMN].to_list()
    )


    print(
        '\nValidation successful: all {} building files, '
        'the schema file, and the configured load/PV columns '
        'are available.'.format(
            len(state.BUILDING_IDS)
        )
    )

    print(
        '\nImportant: the PV column is expressed in W/kW. '
        'The next preprocessing cell must multiply this profile '
        'by each building PV nominal power and divide by 1000 '
        'to obtain hourly building-level PV energy in kWh.'
    )

def cell_08(state):
    """Cell 08: load and validate development-period data (with PV conversion)."""
    # --------------------------------------------------------------------------
    # Load schema and extract building PV nominal powers
    # --------------------------------------------------------------------------

    if state.LOAD_COLUMN is None or state.PV_COLUMN is None:
        raise ValueError(
            'Set both LOAD_COLUMN and PV_COLUMN explicitly.'
        )

    if not state.SCHEMA_PATH.is_file():
        raise FileNotFoundError(
            'CityLearn schema was not found: {}'.format(
                state.SCHEMA_PATH
            )
        )

    with open(
        state.SCHEMA_PATH,
        'r',
        encoding='utf-8'
    ) as state.file:
        state.schema = json.load(state.file)


    state.pv_nominal_power = {}

    for state.building_id in state.BUILDING_IDS:
        state.building_name = 'Building_{}'.format(
            state.building_id
        )

        try:
            state.nominal_power_kw = state.schema[
                'buildings'
            ][
                state.building_name
            ][
                'pv'
            ][
                'attributes'
            ][
                'nominal_power'
            ]

        except KeyError as error:
            raise KeyError(
                'Could not find PV nominal power for {}. '
                'Expected schema path: '
                'buildings -> {} -> pv -> attributes '
                '-> nominal_power.'.format(
                    state.building_name,
                    state.building_name
                )
            ) from error

        state.nominal_power_kw = float(
            state.nominal_power_kw
        )

        if (
            not np.isfinite(state.nominal_power_kw)
            or state.nominal_power_kw < 0.0
        ):
            raise ValueError(
                'Invalid PV nominal power for {}: {}'.format(
                    state.building_name,
                    state.nominal_power_kw
                )
            )

        state.pv_nominal_power[state.building_id] = (
            state.nominal_power_kw
        )


    state.pv_power_table = pd.DataFrame({
        'building_id': state.BUILDING_IDS,
        'pv_nominal_power_kw': [
            state.pv_nominal_power[state.building_id]
            for state.building_id in state.BUILDING_IDS
        ]
    }).set_index('building_id')

    print('PV nominal powers extracted from schema:')
    display(state.pv_power_table)


    # --------------------------------------------------------------------------
    # Load and convert building load and PV data
    # --------------------------------------------------------------------------

    state.raw_data = {}
    state.full_lengths = {}

    for state.building_id in state.BUILDING_IDS:
        state.frame = pd.read_csv(
            state.file_paths[state.building_id]
        )

        state.required_columns = [
            state.LOAD_COLUMN,
            state.PV_COLUMN
        ]

        state.missing_columns = [
            state.column
            for state.column in state.required_columns
            if state.column not in state.frame.columns
        ]

        if state.missing_columns:
            raise KeyError(
                'Building {} is missing columns: {}'.format(
                    state.building_id,
                    state.missing_columns
                )
            )

        state.load = pd.to_numeric(
            state.frame[state.LOAD_COLUMN],
            errors='coerce'
        ).to_numpy(dtype=float)

        state.solar_generation_w_per_kw = pd.to_numeric(
            state.frame[state.PV_COLUMN],
            errors='coerce'
        ).to_numpy(dtype=float)

        state.nominal_power_kw = state.pv_nominal_power[
            state.building_id
        ]

        # Convert normalized PV profile from W/kW to actual hourly
        # building-level PV energy in kWh.
        state.pv = (
            state.solar_generation_w_per_kw
            * state.nominal_power_kw
            * state.TIME_STEP_HOURS
            / 1000.0
        )


        # ----------------------------------------------------------------------
        # Data-quality checks
        # ----------------------------------------------------------------------

        if not np.all(np.isfinite(state.load)):
            raise ValueError(
                'Nonfinite load values found in Building {}.'.format(
                    state.building_id
                )
            )

        if not np.all(
            np.isfinite(
                state.solar_generation_w_per_kw
            )
        ):
            raise ValueError(
                'Nonfinite normalized PV values found in '
                'Building {}.'.format(
                    state.building_id
                )
            )

        if not np.all(np.isfinite(state.pv)):
            raise ValueError(
                'Nonfinite converted PV values found in '
                'Building {}.'.format(
                    state.building_id
                )
            )

        if state.load.min() < -state.EPSILON:
            raise ValueError(
                'Negative load found in Building {}. '
                'Verify LOAD_COLUMN.'.format(
                    state.building_id
                )
            )

        if state.solar_generation_w_per_kw.min() < -state.EPSILON:
            raise ValueError(
                'Negative normalized PV profile found in '
                'Building {}. Verify the source data.'.format(
                    state.building_id
                )
            )

        if state.pv.min() < -state.EPSILON:
            raise ValueError(
                'Negative converted PV generation found in '
                'Building {}.'.format(
                    state.building_id
                )
            )

        # Sanity check for hourly PV generation.
        if (
            state.nominal_power_kw > 0.0
            and state.pv.max()
            > 1.5 * state.nominal_power_kw * state.TIME_STEP_HOURS
        ):
            raise ValueError(
                'Converted PV generation appears inconsistent '
                'for Building {}. '
                'Peak PV={:.4f} kWh per time step; '
                'nominal power={:.4f} kW.'.format(
                    state.building_id,
                    state.pv.max(),
                    state.nominal_power_kw
                )
            )

        if len(state.load) != len(state.pv):
            raise ValueError(
                'Load and PV lengths differ for Building {}: '
                'load={}, PV={}.'.format(
                    state.building_id,
                    len(state.load),
                    len(state.pv)
                )
            )


        # ----------------------------------------------------------------------
        # Select the development interval only
        # ----------------------------------------------------------------------

        state.full_lengths[state.building_id] = len(state.load)

        if (
            state.DEVELOPMENT_START < 0
            or state.DEVELOPMENT_END >= len(state.load)
        ):
            raise IndexError(
                'Development interval [{}, {}] is invalid '
                'for Building {} with {} rows.'.format(
                    state.DEVELOPMENT_START,
                    state.DEVELOPMENT_END,
                    state.building_id,
                    len(state.load)
                )
            )

        state.selection = slice(
            state.DEVELOPMENT_START,
            state.DEVELOPMENT_END + 1
        )

        state.raw_data[state.building_id] = {
            'load': state.load[state.selection].copy(),
            'pv': state.pv[state.selection].copy(),
            'solar_generation_w_per_kw':
                state.solar_generation_w_per_kw[
                    state.selection
                ].copy(),
            'pv_nominal_power_kw':
                state.nominal_power_kw
        }


    # --------------------------------------------------------------------------
    # Cross-building consistency checks
    # --------------------------------------------------------------------------

    if len(set(state.full_lengths.values())) != 1:
        raise ValueError(
            'Building time-series lengths differ: {}'.format(
                state.full_lengths
            )
        )

    state.N_DEVELOPMENT_HOURS = (
        state.DEVELOPMENT_END
        - state.DEVELOPMENT_START
        + 1
    )

    for state.building_id in state.BUILDING_IDS:
        if (
            len(state.raw_data[state.building_id]['load'])
            != state.N_DEVELOPMENT_HOURS
        ):
            raise ValueError(
                'Unexpected development-period length for '
                'Building {}.'.format(
                    state.building_id
                )
            )


    print(
        'Buildings loaded:',
        len(state.raw_data)
    )

    print(
        'Development hours:',
        state.N_DEVELOPMENT_HOURS
    )

    print(
        'Development days:',
        state.N_DEVELOPMENT_HOURS
        / (24.0 / state.TIME_STEP_HOURS)
    )

    print(
        'PV conversion applied: '
        'W/kW × nominal power (kW) × time-step duration (h) / 1000'
    )

def cell_10(state):
    """Cell 10: check physical scales and signs."""
    # --------------------------------------------------------------------------
    # Verify physical scales after converting PV from W/kW to hourly kWh
    # --------------------------------------------------------------------------

    state.scale_records = []

    for state.building_id in state.BUILDING_IDS:
        state.building_data = state.raw_data[state.building_id]

        state.load = np.asarray(
            state.building_data['load'],
            dtype=float
        )

        state.pv = np.asarray(
            state.building_data['pv'],
            dtype=float
        )

        state.nominal_power_kw = float(
            state.building_data['pv_nominal_power_kw']
        )

        if state.load.shape != state.pv.shape:
            raise ValueError(
                'Load and PV arrays have different shapes for '
                'Building {}: load={}, PV={}.'.format(
                    state.building_id,
                    state.load.shape,
                    state.pv.shape
                )
            )

        if state.load.size == 0:
            raise ValueError(
                'Building {} contains no development-period data.'.format(
                    state.building_id
                )
            )

        if not np.all(np.isfinite(state.load)):
            raise ValueError(
                'Building {} contains nonfinite load values.'.format(
                    state.building_id
                )
            )

        if not np.all(np.isfinite(state.pv)):
            raise ValueError(
                'Building {} contains nonfinite converted PV values.'.format(
                    state.building_id
                )
            )

        if state.load.min() < -state.EPSILON:
            raise ValueError(
                'Building {} contains negative physical load.'.format(
                    state.building_id
                )
            )

        if state.pv.min() < -state.EPSILON:
            raise ValueError(
                'Building {} contains negative physical PV generation.'.format(
                    state.building_id
                )
            )

        state.net_demand_without_storage = state.load - state.pv
        state.baseline_grid_import = np.maximum(
            state.net_demand_without_storage,
            0.0
        )
        state.baseline_grid_export = np.maximum(
            -state.net_demand_without_storage,
            0.0
        )

        state.total_load_kwh = float(state.load.sum())
        state.total_pv_kwh = float(state.pv.sum())

        state.scale_records.append({
            'building_id': state.building_id,
            'pv_nominal_power_kw': state.nominal_power_kw,

            'load_mean_kwh':
                float(state.load.mean()),
            'load_peak_kwh':
                float(state.load.max()),
            'load_total_kwh':
                state.total_load_kwh,

            'pv_mean_kwh':
                float(state.pv.mean()),
            'pv_peak_kwh':
                float(state.pv.max()),
            'pv_total_kwh':
                state.total_pv_kwh,

            'pv_to_load_ratio':
                state.total_pv_kwh
                / (state.total_load_kwh + state.EPSILON),

            'baseline_grid_import_kwh':
                float(state.baseline_grid_import.sum()),

            'baseline_grid_export_kwh':
                float(state.baseline_grid_export.sum()),

            'pv_surplus_frequency':
                float(
                    (
                        state.net_demand_without_storage < 0.0
                    ).mean()
                )
        })


    state.scale_checks = (
        pd.DataFrame(state.scale_records)
        .set_index('building_id')
    )

    if state.scale_checks.isna().any().any():
        raise ValueError(
            'The physical-scale table contains missing values.'
        )

    if not np.all(
        np.isfinite(
            state.scale_checks.to_numpy(dtype=float)
        )
    ):
        raise ValueError(
            'The physical-scale table contains nonfinite values.'
        )


    print(
        'Physical scale checks after converting '
        'Solar Generation [W/kW] to hourly PV energy '
    )

    display(
        state.scale_checks.round(4)
    )

def cell_12(state):
    """Cell 12: optional battery metadata."""
    # --------------------------------------------------------------------------
    # Extract battery metadata from the CityLearn experiment schema
    # --------------------------------------------------------------------------

    state.battery_records = []

    for state.building_id in state.BUILDING_IDS:
        state.building_name = 'Building_{}'.format(state.building_id)

        if state.building_name not in state.schema['buildings']:
            raise KeyError(
                'Building configuration not found in schema: {}'.format(
                    state.building_name
                )
            )

        state.building_configuration = state.schema['buildings'][state.building_name]

        # CityLearn schemas commonly use "electrical_storage".
        # Retain "battery" as a fallback for custom schemas.
        if 'electrical_storage' in state.building_configuration:
            state.storage_configuration = state.building_configuration[
                'electrical_storage'
            ]
        elif 'battery' in state.building_configuration:
            state.storage_configuration = state.building_configuration[
                'battery'
            ]
        else:
            raise KeyError(
                'No electrical-storage configuration was found for {}. '
                'Available keys: {}'.format(
                    state.building_name,
                    list(state.building_configuration.keys())
                )
            )

        state.storage_attributes = state.storage_configuration.get(
            'attributes',
            state.storage_configuration
        )

        if 'capacity' not in state.storage_attributes:
            raise KeyError(
                'Battery capacity was not found for {}. '
                'Storage configuration: {}'.format(
                    state.building_name,
                    state.storage_configuration
                )
            )

        if 'nominal_power' not in state.storage_attributes:
            raise KeyError(
                'Battery nominal power was not found for {}. '
                'Storage configuration: {}'.format(
                    state.building_name,
                    state.storage_configuration
                )
            )

        state.capacity_kwh = state.storage_attributes['capacity']
        state.nominal_power_kw = state.storage_attributes['nominal_power']

        # Stop if the schema requests autosizing rather than providing
        # resolved numerical values.
        if not isinstance(
            state.capacity_kwh,
            (int, float, np.integer, np.floating)
        ):
            raise TypeError(
                'Battery capacity for {} is not a resolved numerical value: '
                '{}. Use the generated experiment schema or extract the '
                'realized battery values from an initialized environment.'.format(
                    state.building_name,
                    state.capacity_kwh
                )
            )

        if not isinstance(
            state.nominal_power_kw,
            (int, float, np.integer, np.floating)
        ):
            raise TypeError(
                'Battery nominal power for {} is not a resolved numerical '
                'value: {}. Use the generated experiment schema or extract '
                'the realized battery values from an initialized '
                'environment.'.format(
                    state.building_name,
                    state.nominal_power_kw
                )
            )

        state.capacity_kwh = float(state.capacity_kwh)
        state.nominal_power_kw = float(state.nominal_power_kw)

        if (
            not np.isfinite(state.capacity_kwh)
            or state.capacity_kwh <= 0.0
        ):
            raise ValueError(
                'Invalid battery capacity for {}: {}'.format(
                    state.building_name,
                    state.capacity_kwh
                )
            )

        if (
            not np.isfinite(state.nominal_power_kw)
            or state.nominal_power_kw <= 0.0
        ):
            raise ValueError(
                'Invalid battery nominal power for {}: {}'.format(
                    state.building_name,
                    state.nominal_power_kw
                )
            )

        state.battery_records.append({
            'building_id': state.building_id,
            'battery_capacity_kwh': state.capacity_kwh,
            'battery_nominal_power_kw': state.nominal_power_kw
        })


    state.battery_metadata = (
        pd.DataFrame(state.battery_records)
        .set_index('building_id')
    )


    # --------------------------------------------------------------------------
    # Check whether battery specifications vary across buildings
    # --------------------------------------------------------------------------

    state.capacity_count = state.battery_metadata[
        'battery_capacity_kwh'
    ].nunique()

    state.power_count = state.battery_metadata[
        'battery_nominal_power_kw'
    ].nunique()

    display(
        state.battery_metadata.round(4)
    )

    print(
        'Unique battery capacities:',
        state.capacity_count
    )

    print(
        'Unique battery nominal powers:',
        state.power_count
    )


    if state.capacity_count == 1 and state.power_count == 1:
        print(
            '\nAll buildings use identical battery capacity and nominal '
            'power. Battery-relative descriptors will contain useful '
            'information only through their ratios to building demand.'
        )
    else:
        print(
            '\nBattery specifications differ across buildings. '
            'Battery-relative descriptors should be retained in the '
            'physical clustering analysis.'
        )

def cell_15(state):
    """Cell 15: physical descriptor extraction."""
    # --------------------------------------------------------------------------
    # Extract physically meaningful building descriptors
    # --------------------------------------------------------------------------

    def extract_descriptors(
        building_id,
        load,
        pv,
        battery_table=None
    ):
        """Return controller-independent physical descriptors for one building.

        Parameters
        ----------
        building_id : int
            CityLearn building identifier.

        load : array-like
            Hourly non-shiftable electrical load in kWh.

        pv : array-like
            Hourly building-level PV generation in kWh. The raw W/kW
            profile must already have been converted using PV nominal power.

        battery_table : pandas.DataFrame or None
            Optional table indexed by building_id with battery capacity
            and nominal power.

        Returns
        -------
        dict
            Physical load, PV, net-demand, grid-exchange, and optional
            battery-relative descriptors.
        """

        load = np.asarray(
            load,
            dtype=float
        ).reshape(-1)

        pv = np.asarray(
            pv,
            dtype=float
        ).reshape(-1)


        # ----------------------------------------------------------------------
        # Input validation
        # ----------------------------------------------------------------------

        if load.size == 0:
            raise ValueError(
                'Building {} has an empty load array.'.format(
                    building_id
                )
            )

        if pv.size == 0:
            raise ValueError(
                'Building {} has an empty PV array.'.format(
                    building_id
                )
            )

        if load.shape != pv.shape:
            raise ValueError(
                'Load and PV arrays have different shapes for '
                'Building {}: load={}, PV={}.'.format(
                    building_id,
                    load.shape,
                    pv.shape
                )
            )

        if not np.all(np.isfinite(load)):
            raise ValueError(
                'Building {} contains nonfinite load values.'.format(
                    building_id
                )
            )

        if not np.all(np.isfinite(pv)):
            raise ValueError(
                'Building {} contains nonfinite PV values.'.format(
                    building_id
                )
            )

        if load.min() < -state.EPSILON:
            raise ValueError(
                'Building {} contains negative physical load.'.format(
                    building_id
                )
            )

        if pv.min() < -state.EPSILON:
            raise ValueError(
                'Building {} contains negative physical PV generation.'.format(
                    building_id
                )
            )


        # ----------------------------------------------------------------------
        # Pre-storage grid exchange
        # ----------------------------------------------------------------------

        pre_storage_net_demand = load - pv

        baseline_grid_import = np.maximum(
            pre_storage_net_demand,
            0.0
        )

        baseline_grid_export = np.maximum(
            -pre_storage_net_demand,
            0.0
        )

        direct_pv_self_consumption = np.minimum(
            load,
            pv
        )

        pv_surplus_mask = (
            pre_storage_net_demand < 0.0
        )


        # ----------------------------------------------------------------------
        # Aggregate quantities
        # ----------------------------------------------------------------------

        total_load_kwh = float(
            load.sum()
        )

        mean_load_kwh = float(
            load.mean()
        )

        peak_load_kwh = float(
            load.max()
        )

        load_std_kwh = float(
            load.std(ddof=0)
        )

        total_pv_kwh = float(
            pv.sum()
        )

        total_direct_pv_use_kwh = float(
            direct_pv_self_consumption.sum()
        )

        total_baseline_import_kwh = float(
            baseline_grid_import.sum()
        )

        total_baseline_export_kwh = float(
            baseline_grid_export.sum()
        )

        n_days = (
            load.size
            * state.TIME_STEP_HOURS
            / 24.0
        )

        mean_daily_load_kwh = (
            total_load_kwh
            / (n_days + state.EPSILON)
        )


        # ----------------------------------------------------------------------
        # Physical consistency checks
        # ----------------------------------------------------------------------

        reconstructed_load = (
            total_baseline_import_kwh
            + total_direct_pv_use_kwh
        )

        if not np.isclose(
            reconstructed_load,
            total_load_kwh,
            rtol=1e-6,
            atol=1e-6
        ):
            raise ValueError(
                'Load-energy identity failed for Building {}. '
                'Load={:.6f}, reconstructed={:.6f}.'.format(
                    building_id,
                    total_load_kwh,
                    reconstructed_load
                )
            )

        reconstructed_pv = (
            total_direct_pv_use_kwh
            + total_baseline_export_kwh
        )

        if not np.isclose(
            reconstructed_pv,
            total_pv_kwh,
            rtol=1e-6,
            atol=1e-6
        ):
            raise ValueError(
                'PV-energy identity failed for Building {}. '
                'PV={:.6f}, reconstructed={:.6f}.'.format(
                    building_id,
                    total_pv_kwh,
                    reconstructed_pv
                )
            )


        # ----------------------------------------------------------------------
        # Load, PV, and net-demand descriptors
        # ----------------------------------------------------------------------

        result = {
            'building_id':
                int(building_id),

            # Load scale and variability
            'total_load_kwh':
                total_load_kwh,

            'mean_load_kwh':
                mean_load_kwh,

            'peak_load_kwh':
                peak_load_kwh,

            'load_std_kwh':
                load_std_kwh,

            'load_cv':
                load_std_kwh
                / (mean_load_kwh + state.EPSILON),

            'load_factor':
                mean_load_kwh
                / (peak_load_kwh + state.EPSILON),

            # PV generation and penetration
            'total_pv_kwh':
                total_pv_kwh,

            'mean_pv_kwh':
                float(pv.mean()),

            'peak_pv_kwh':
                float(pv.max()),

            'pv_to_load_ratio':
                total_pv_kwh
                / (total_load_kwh + state.EPSILON),

            # Direct PV utilization without storage
            'direct_pv_self_consumption_kwh':
                total_direct_pv_use_kwh,

            'direct_pv_self_consumption_rate':
                total_direct_pv_use_kwh
                / (total_pv_kwh + state.EPSILON),

            # PV surplus and baseline export
            'pv_surplus_frequency':
                float(pv_surplus_mask.mean()),

            'total_pv_surplus_kwh':
                total_baseline_export_kwh,

            'pv_surplus_fraction':
                total_baseline_export_kwh
                / (total_pv_kwh + state.EPSILON),

            # Pre-storage net demand
            'mean_net_demand_kwh':
                float(
                    pre_storage_net_demand.mean()
                ),

            'net_demand_std_kwh':
                float(
                    pre_storage_net_demand.std(
                        ddof=0
                    )
                ),

            'net_demand_cv':
                float(
                    pre_storage_net_demand.std(
                        ddof=0
                    )
                    / (
                        np.mean(
                            np.abs(
                                pre_storage_net_demand
                            )
                        )
                        + state.EPSILON
                    )
                ),

            'peak_positive_net_demand_kwh':
                float(
                    baseline_grid_import.max()
                ),

            # No-storage grid exchange
            'baseline_import_kwh':
                total_baseline_import_kwh,

            'baseline_export_kwh':
                total_baseline_export_kwh,

            'baseline_export_frequency':
                float(
                    (
                        baseline_grid_export
                        > state.EPSILON
                    ).mean()
                )
        }


        # ----------------------------------------------------------------------
        # Optional battery-relative descriptors
        # ----------------------------------------------------------------------

        if battery_table is not None:
            if building_id not in battery_table.index:
                raise KeyError(
                    'Battery metadata are missing for Building {}.'.format(
                        building_id
                    )
                )

            required_battery_columns = [
                'battery_capacity_kwh',
                'battery_nominal_power_kw'
            ]

            missing_battery_columns = [
                state.column
                for state.column in required_battery_columns
                if state.column not in battery_table.columns
            ]

            if missing_battery_columns:
                raise KeyError(
                    'Battery metadata are missing columns: {}'.format(
                        missing_battery_columns
                    )
                )

            battery_capacity_kwh = float(
                battery_table.loc[
                    building_id,
                    'battery_capacity_kwh'
                ]
            )

            battery_nominal_power_kw = float(
                battery_table.loc[
                    building_id,
                    'battery_nominal_power_kw'
                ]
            )

            if (
                not np.isfinite(
                    battery_capacity_kwh
                )
                or battery_capacity_kwh <= 0.0
            ):
                raise ValueError(
                    'Invalid battery capacity for Building {}: {}'.format(
                        building_id,
                        battery_capacity_kwh
                    )
                )

            if (
                not np.isfinite(
                    battery_nominal_power_kw
                )
                or battery_nominal_power_kw <= 0.0
            ):
                raise ValueError(
                    'Invalid battery nominal power for Building {}: {}'.format(
                        building_id,
                        battery_nominal_power_kw
                    )
                )

            result.update({
                'battery_capacity_kwh':
                    battery_capacity_kwh,

                'battery_nominal_power_kw':
                    battery_nominal_power_kw,

                'battery_capacity_to_daily_load':
                    battery_capacity_kwh
                    / (
                        mean_daily_load_kwh
                        + state.EPSILON
                    ),

                'battery_capacity_to_pv_surplus':
                    battery_capacity_kwh
                    / (
                        total_baseline_export_kwh
                        / (n_days + state.EPSILON)
                        + state.EPSILON
                    ),

                'battery_power_to_peak_load':
                    battery_nominal_power_kw
                    / (
                        peak_load_kwh
                        / state.TIME_STEP_HOURS
                        + state.EPSILON
                    ),

                'battery_power_to_peak_net_demand':
                    battery_nominal_power_kw
                    / (
                        baseline_grid_import.max()
                        / state.TIME_STEP_HOURS
                        + state.EPSILON
                    )
            })


        # ----------------------------------------------------------------------
        # Final descriptor validation
        # ----------------------------------------------------------------------

        numerical_values = np.asarray(
            list(result.values()),
            dtype=float
        )

        if not np.all(
            np.isfinite(
                numerical_values
            )
        ):
            raise ValueError(
                'Descriptor extraction generated nonfinite values '
                'for Building {}.'.format(
                    building_id
                )
            )

        return result
    state.extract_descriptors = extract_descriptors


    # --------------------------------------------------------------------------
    # Extract one descriptor row per building
    # --------------------------------------------------------------------------

    state.descriptor_records = []

    for state.building_id in state.BUILDING_IDS:
        state.descriptor_records.append(
            state.extract_descriptors(
                building_id=state.building_id,
                load=state.raw_data[state.building_id]['load'],
                pv=state.raw_data[state.building_id]['pv'],
                battery_table=state.battery_metadata
            )
        )


    state.descriptors = (
        pd.DataFrame(
            state.descriptor_records
        )
        .set_index('building_id')
        .sort_index()
    )


    # --------------------------------------------------------------------------
    # Validate the complete descriptor table
    # --------------------------------------------------------------------------

    if state.descriptors.shape[0] != len(state.BUILDING_IDS):
        raise ValueError(
            'Expected {} building rows but found {}.'.format(
                len(state.BUILDING_IDS),
                state.descriptors.shape[0]
            )
        )

    if state.descriptors.index.duplicated().any():
        raise ValueError(
            'The descriptor table contains duplicated building IDs.'
        )

    if state.descriptors.isna().any().any():
        state.missing_locations = (
            state.descriptors.isna()
            .stack()
        )

        state.missing_locations = state.missing_locations[
            state.missing_locations
        ]

        raise ValueError(
            'The descriptor table contains missing values at: {}'.format(
                state.missing_locations.index.tolist()
            )
        )

    if not np.all(
        np.isfinite(
            state.descriptors.to_numpy(
                dtype=float
            )
        )
    ):
        raise ValueError(
            'The descriptor table contains nonfinite values.'
        )


    # --------------------------------------------------------------------------
    # Identify constant descriptors
    # --------------------------------------------------------------------------

    state.constant_features = [
        state.column
        for state.column in state.descriptors.columns
        if state.descriptors[state.column].nunique(
            dropna=False
        ) <= 1
    ]

    print(
        'Constant descriptors:',
        state.constant_features
    )

    if state.constant_features:
        print(
            'Constant descriptors contain no information for clustering '
            'and should not be included in SELECTED_FEATURES.'
        )


    print(
        '\nPhysical descriptor table:'
    )

    display(
        state.descriptors.round(4)
    )


    # --------------------------------------------------------------------------
    # Save the reproducible descriptor table
    # --------------------------------------------------------------------------

    state.descriptor_output_path = (
        state.OUTPUT_DIR
        / 'building_physical_descriptors.csv'
    )

    state.descriptors.to_csv(
        state.descriptor_output_path
    )

    print(
        '\nDescriptor table saved to:',
        state.descriptor_output_path.resolve()
    )

def cell_17(state):
    """Cell 17: descriptor distributions and correlation audit."""
    # --------------------------------------------------------------------------
    # Visual audit of physical descriptor distributions across buildings
    # --------------------------------------------------------------------------

    # Descriptor names and publication-friendly axis labels.
    state.descriptor_labels = {
        # Load descriptors
        'mean_load_kwh':
            'Mean Load (kWh/step)',

        'peak_load_kwh':
            'Peak Load (kWh/step)',

        'load_cv':
            'Load Coefficient of Variation',

        'load_factor':
            'Load Factor',

        # PV descriptors
        'pv_to_load_ratio':
            'PV-to-Load Energy Ratio',

        'pv_surplus_frequency':
            'PV-Surplus Frequency',

        'pv_surplus_fraction':
            'PV-Surplus Fraction',

        # Pre-storage grid-exchange descriptors
        'baseline_import_kwh':
            'Baseline Grid Import (kWh)',

        'baseline_export_kwh':
            'Baseline Grid Export (kWh)',

        'net_demand_std_kwh':
            'Net-Demand Standard Deviation (kWh/step)',

        # Optional battery-relative descriptors
        'battery_capacity_to_daily_load':
            'Battery Capacity / Mean Daily Load',

        'battery_capacity_to_pv_surplus':
            'Battery Capacity / Mean Daily PV Surplus',

        'battery_power_to_peak_load':
            'Battery Power / Peak Load',

        'battery_power_to_peak_net_demand':
            'Battery Power / Peak Net Demand'
    }


    # Desired descriptor order for visual inspection.
    state.requested_audit_columns = [
        'mean_load_kwh',
        'peak_load_kwh',
        'load_cv',
        'load_factor',
        'pv_to_load_ratio',
        'pv_surplus_frequency',
        'pv_surplus_fraction',
        'baseline_import_kwh',
        'baseline_export_kwh',
        'net_demand_std_kwh',
        'battery_capacity_to_daily_load',
        'battery_capacity_to_pv_surplus',
        'battery_power_to_peak_load',
        'battery_power_to_peak_net_demand'
    ]


    # Retain only descriptors available in the current table.
    state.audit_columns = [
        state.column
        for state.column in state.requested_audit_columns
        if state.column in state.descriptors.columns
    ]


    if not state.audit_columns:
        raise ValueError(
            'None of the requested audit descriptors were found '
            'in the descriptor table.'
        )


    # Validate the selected descriptor values.
    state.audit_data = state.descriptors[
        state.audit_columns
    ].copy()


    if state.audit_data.isna().any().any():
        raise ValueError(
            'The selected audit descriptors contain missing values.'
        )


    if not np.all(
        np.isfinite(
            state.audit_data.to_numpy(dtype=float)
        )
    ):
        raise ValueError(
            'The selected audit descriptors contain '
            'NaN or infinite values.'
        )


    # --------------------------------------------------------------------------
    # Create subplot layout
    # --------------------------------------------------------------------------

    state.ncols = 2

    state.nrows = int(
        np.ceil(
            len(state.audit_columns)
            / float(state.ncols)
        )
    )


    state.fig, state.axes = plt.subplots(
        nrows=state.nrows,
        ncols=state.ncols,
        figsize=(13, 3.2 * state.nrows)
    )


    # Ensure consistent one-dimensional iteration over axes.
    state.axes = np.asarray(
        state.axes
    ).reshape(-1)


    state.building_labels = [
        'B{}'.format(state.building_id)
        for state.building_id in state.audit_data.index
    ]


    # Use colors to distinguish descriptor categories.
    state.category_colors = {
        'load': '#457b9d',
        'pv': '#2a9d8f',
        'grid': '#e76f51',
        'battery': '#8e6c8a'
    }


    def get_descriptor_color(column):
        """Return a consistent color according to descriptor category."""

        if column.startswith('battery_'):
            return state.category_colors['battery']

        if (
            column.startswith('pv_')
            or 'pv_surplus' in column
        ):
            return state.category_colors['pv']

        if (
            column.startswith('baseline_')
            or 'net_demand' in column
        ):
            return state.category_colors['grid']

        return state.category_colors['load']
    state.get_descriptor_color = get_descriptor_color


    # --------------------------------------------------------------------------
    # Plot every available descriptor
    # --------------------------------------------------------------------------

    for state.ax, state.column in zip(
        state.axes,
        state.audit_columns
    ):
        state.values = state.audit_data[
            state.column
        ].to_numpy(dtype=float)

        state.color = state.get_descriptor_color(
            state.column
        )

        state.ax.bar(
            state.building_labels,
            state.values,
            color=state.color,
            edgecolor='black',
            linewidth=0.4,
            alpha=0.85
        )

        state.readable_label = state.descriptor_labels.get(
            state.column,
            state.column.replace('_', ' ').title()
        )

        state.ax.set_title(
            state.readable_label,
            fontsize=11,
            fontweight='bold'
        )

        state.ax.set_xlabel(
            'Building'
        )

        state.ax.set_ylabel(
            'Value'
        )

        state.ax.tick_params(
            axis='x',
            labelrotation=0,
            labelsize=8
        )

        state.ax.grid(
            axis='y',
            linestyle='--',
            alpha=0.30
        )

        state.ax.set_axisbelow(
            True
        )


    # Hide unused subplot positions.
    for state.ax in state.axes[
        len(state.audit_columns):
    ]:
        state.ax.axis(
            'off'
        )


    state.fig.suptitle(
        'Physical Descriptor Distributions Across '
        'the CityLearn Building Portfolio',
        fontsize=14,
        fontweight='bold',
        y=1.005
    )


    plt.tight_layout()
    plt.show()


    print(
        'Displayed {} physical descriptors for {} buildings.'.format(
            len(state.audit_columns),
            len(state.audit_data)
        )
    )


    state.missing_optional_columns = [
        state.column
        for state.column in state.requested_audit_columns
        if state.column not in state.descriptors.columns
    ]


    if state.missing_optional_columns:
        print(
            '\nDescriptors not displayed because they are unavailable:'
        )

        for state.column in state.missing_optional_columns:
            print(
                '  -',
                state.column
            )

def cell_19(state):
    """Cell 19: select physically complementary clustering features."""
    # --------------------------------------------------------------------------
    # Select physically complementary descriptors for clustering
    # --------------------------------------------------------------------------

    # Core descriptors are selected to represent different dimensions of the
    # PV-battery control problem while limiting redundant feature weighting.
    state.SELECTED_FEATURES = [
        # Building-demand scale
        'peak_load_kwh',

        # Relative load variability
        'load_cv',

        # PV penetration relative to demand
        'pv_to_load_ratio',

        # Frequency of time steps with PV surplus
        'pv_surplus_frequency',

        # Fraction of PV generation exported without storage
        'pv_surplus_fraction',

        # Absolute variability of pre-storage net demand
        'net_demand_std_kwh',

        # Maximum import requirement without storage
        'peak_positive_net_demand_kwh',
    ]


    # Add battery-relative descriptors when battery metadata are available.
    state.optional_battery_features = [
        'battery_capacity_to_daily_load',
        'battery_power_to_peak_net_demand',
    ]

    for state.feature in state.optional_battery_features:
        if state.feature in state.descriptors.columns:
            state.SELECTED_FEATURES.append(state.feature)


    # --------------------------------------------------------------------------
    # Validate selected features
    # --------------------------------------------------------------------------

    state.missing_features = [
        state.feature
        for state.feature in state.SELECTED_FEATURES
        if state.feature not in state.descriptors.columns
    ]

    if state.missing_features:
        raise KeyError(
            'The following selected clustering features are missing: {}'.format(
                state.missing_features
            )
        )


    state.constant_selected_features = [
        state.feature
        for state.feature in state.SELECTED_FEATURES
        if state.descriptors[state.feature].nunique(
            dropna=False
        ) <= 1
    ]

    if state.constant_selected_features:
        raise ValueError(
            'The following selected features are constant across all buildings '
            'and provide no clustering information: {}'.format(
                state.constant_selected_features
            )
        )


    state.selected_data = state.descriptors[
        state.SELECTED_FEATURES
    ].copy()

    if state.selected_data.isna().any().any():
        raise ValueError(
            'The selected clustering features contain missing values.'
        )

    if not np.all(
        np.isfinite(
            state.selected_data.to_numpy(dtype=float)
        )
    ):
        raise ValueError(
            'The selected clustering features contain NaN or '
            'infinite values.'
        )


    # --------------------------------------------------------------------------
    # Audit correlation among the selected descriptors
    # --------------------------------------------------------------------------

    state.selected_feature_correlation = state.selected_data.corr(
        method='spearman'
    )

    state.high_correlation_pairs = []

    for state.left_index, state.left_feature in enumerate(
        state.SELECTED_FEATURES
    ):
        for state.right_feature in state.SELECTED_FEATURES[
            state.left_index + 1:
        ]:
            state.correlation = state.selected_feature_correlation.loc[
                state.left_feature,
                state.right_feature
            ]

            if abs(state.correlation) >= state.REDUNDANCY_THRESHOLD:
                state.high_correlation_pairs.append({
                    'feature_1': state.left_feature,
                    'feature_2': state.right_feature,
                    'spearman_correlation': state.correlation
                })


    print('Final clustering features:')

    for state.feature_number, state.feature in enumerate(
        state.SELECTED_FEATURES,
        start=1
    ):
        print(
            '  {}. {}'.format(
                state.feature_number,
                state.feature
            )
        )


    print(
        '\nNumber of selected features:',
        len(state.SELECTED_FEATURES)
    )


    if state.high_correlation_pairs:
        print(
            '\nWarning: the following selected feature pairs exceed '
            'the redundancy threshold of {:.2f}:'.format(
                state.REDUNDANCY_THRESHOLD
            )
        )

        state.high_correlation_table = pd.DataFrame(
            state.high_correlation_pairs
        )

        display(
            state.high_correlation_table.sort_values(
                'spearman_correlation',
                key=np.abs,
                ascending=False
            ).round(4)
        )

        print(
            'Review these pairs before final clustering. Retain both only '
            'when they represent distinct physical aspects of the '
            'battery-control problem.'
        )

    else:
        print(
            '\nNo selected feature pair exceeds the redundancy threshold '
            'of {:.2f}.'.format(
                state.REDUNDANCY_THRESHOLD
            )
        )

def cell_21(state):
    """Cell 21: robust scaling of selected descriptors."""
    # --------------------------------------------------------------------------
    # Robustly scale the selected physical descriptors
    # --------------------------------------------------------------------------

    # Preserve the building order explicitly.
    state.building_index = state.descriptors.index.copy()

    # Extract the selected descriptor matrix.
    state.X_physical = state.descriptors[
        state.SELECTED_FEATURES
    ].to_numpy(dtype=float)


    # --------------------------------------------------------------------------
    # Validate the unscaled matrix
    # --------------------------------------------------------------------------

    if state.X_physical.shape[0] != len(state.BUILDING_IDS):
        raise ValueError(
            'Expected {} buildings, but the physical descriptor matrix '
            'contains {} rows.'.format(
                len(state.BUILDING_IDS),
                state.X_physical.shape[0]
            )
        )

    if state.X_physical.shape[1] != len(state.SELECTED_FEATURES):
        raise ValueError(
            'Expected {} selected features, but the physical descriptor '
            'matrix contains {} columns.'.format(
                len(state.SELECTED_FEATURES),
                state.X_physical.shape[1]
            )
        )

    if not np.all(np.isfinite(state.X_physical)):
        raise ValueError(
            'The unscaled physical descriptor matrix contains '
            'NaN or infinite values.'
        )


    # --------------------------------------------------------------------------
    # Apply robust scaling
    # --------------------------------------------------------------------------

    # RobustScaler centers each descriptor using its median and scales it
    # using the interquartile range. This reduces the influence of extreme
    # buildings while keeping all selected descriptors on comparable scales.
    state.scaler = RobustScaler(
        with_centering=True,
        with_scaling=True,
        quantile_range=(25.0, 75.0)
    )

    state.X_scaled = state.scaler.fit_transform(
        state.X_physical
    )


    # --------------------------------------------------------------------------
    # Validate the scaled matrix
    # --------------------------------------------------------------------------

    if state.X_scaled.shape != state.X_physical.shape:
        raise ValueError(
            'The scaled descriptor matrix has an unexpected shape: '
            '{} instead of {}.'.format(
                state.X_scaled.shape,
                state.X_physical.shape
            )
        )

    if not np.all(np.isfinite(state.X_scaled)):
        raise ValueError(
            'The scaled descriptor matrix contains '
            'NaN or infinite values.'
        )


    # --------------------------------------------------------------------------
    # Create interpretable scaled-descriptor tables
    # --------------------------------------------------------------------------

    state.scaled_descriptors = pd.DataFrame(
        state.X_scaled,
        index=state.building_index,
        columns=state.SELECTED_FEATURES
    )

    state.scaled_descriptors.index.name = 'building_id'


    # Save the fitted scaling parameters for reproducibility.
    state.scaling_parameters = pd.DataFrame({
        'feature': state.SELECTED_FEATURES,
        'median_center': state.scaler.center_,
        'interquartile_range': state.scaler.scale_
    }).set_index('feature')


    print(
        'Scaled descriptor matrix shape:',
        state.scaled_descriptors.shape
    )

    print(
        '\nRobustly scaled physical descriptors:'
    )

    display(
        state.scaled_descriptors.round(3)
    )


    print(
        '\nRobust-scaling parameters:'
    )

    display(
        state.scaling_parameters.round(4)
    )


    # --------------------------------------------------------------------------
    # Save scaled data and scaling parameters
    # --------------------------------------------------------------------------

    state.scaled_descriptors.to_csv(
        state.OUTPUT_DIR
        / 'building_scaled_descriptors.csv'
    )

    state.scaling_parameters.to_csv(
        state.OUTPUT_DIR
        / 'robust_scaling_parameters.csv'
    )

    print(
        '\nScaled descriptors and scaling parameters saved to:',
        state.OUTPUT_DIR.resolve()
    )

def cell_23(state):
    """Cell 23: screen the number of K-means clusters."""
    # --------------------------------------------------------------------------
    # Evaluate candidate numbers of K-means clusters
    # --------------------------------------------------------------------------

    # Validate the scaled descriptor matrix.
    if state.X_scaled.ndim != 2:
        raise ValueError(
            'X_scaled must be a two-dimensional matrix. '
            'Received shape: {}'.format(state.X_scaled.shape)
        )

    state.n_buildings, state.n_features = state.X_scaled.shape

    if state.n_buildings != len(state.BUILDING_IDS):
        raise ValueError(
            'X_scaled contains {} building rows, but {} buildings '
            'were expected.'.format(
                state.n_buildings,
                len(state.BUILDING_IDS)
            )
        )

    if state.n_features != len(state.SELECTED_FEATURES):
        raise ValueError(
            'X_scaled contains {} features, but {} selected features '
            'were expected.'.format(
                state.n_features,
                len(state.SELECTED_FEATURES)
            )
        )

    if not np.all(np.isfinite(state.X_scaled)):
        raise ValueError(
            'X_scaled contains NaN or infinite values.'
        )


    # --------------------------------------------------------------------------
    # Validate candidate cluster counts
    # --------------------------------------------------------------------------

    state.valid_k_values = sorted({
        int(k)
        for k in state.K_VALUES
        if 2 <= int(k) < state.n_buildings
    })

    if not state.valid_k_values:
        raise ValueError(
            'K_VALUES must contain at least one integer satisfying '
            '2 <= K < number of buildings.'
        )

    state.excluded_k_values = [
        k
        for k in state.K_VALUES
        if int(k) not in state.valid_k_values
    ]

    if state.excluded_k_values:
        print(
            'Excluded invalid K values:',
            state.excluded_k_values
        )


    # --------------------------------------------------------------------------
    # Fit K-means and calculate internal validation indices
    # --------------------------------------------------------------------------

    state.quality_records = []
    state.cluster_models = {}
    state.cluster_labels_by_k = {}

    for state.n_clusters in state.valid_k_values:

        state.model = KMeans(
        n_clusters=state.n_clusters,
        random_state=state.RANDOM_STATE,
        n_init=state.KMEANS_N_INIT,
        max_iter=state.KMEANS_MAX_ITER,
        algorithm='full')

        state.labels_k = state.model.fit_predict(
            state.X_scaled
        )

        state.observed_clusters = np.unique(
            state.labels_k
        )

        if len(state.observed_clusters) != state.n_clusters:
            raise RuntimeError(
                'K-means requested {} clusters but produced {} '
                'nonempty clusters.'.format(
                    state.n_clusters,
                    len(state.observed_clusters)
                )
            )

        state.cluster_sizes = np.bincount(
            state.labels_k,
            minlength=state.n_clusters
        )

        state.silhouette = silhouette_score(
            state.X_scaled,
            state.labels_k,
            metric='euclidean'
        )

        state.calinski_harabasz = calinski_harabasz_score(
            state.X_scaled,
            state.labels_k
        )

        state.davies_bouldin = davies_bouldin_score(
            state.X_scaled,
            state.labels_k
        )

        state.quality_records.append({
            'n_clusters':
                state.n_clusters,

            'silhouette':
                float(state.silhouette),

            'calinski_harabasz':
                float(state.calinski_harabasz),

            'davies_bouldin':
                float(state.davies_bouldin),

            'inertia':
                float(state.model.inertia_),

            'minimum_cluster_size':
                int(state.cluster_sizes.min()),

            'maximum_cluster_size':
                int(state.cluster_sizes.max()),

            'cluster_size_imbalance':
                float(
                    state.cluster_sizes.max()
                    / state.cluster_sizes.min()
                ),

            'iterations_to_convergence':
                int(state.model.n_iter_),

            'reached_iteration_limit':
                bool(
                    state.model.n_iter_
                    >= state.KMEANS_MAX_ITER
                )
        })

        state.cluster_models[state.n_clusters] = state.model

        state.cluster_labels_by_k[state.n_clusters] = (
            state.labels_k.copy()
        )


    # --------------------------------------------------------------------------
    # Create and validate the quality table
    # --------------------------------------------------------------------------

    state.cluster_quality = (
        pd.DataFrame(state.quality_records)
        .sort_values('n_clusters')
        .reset_index(drop=True)
    )

    state.quality_metric_columns = [
        'silhouette',
        'calinski_harabasz',
        'davies_bouldin',
        'inertia'
    ]

    if not np.all(
        np.isfinite(
            state.cluster_quality[
                state.quality_metric_columns
            ].to_numpy(dtype=float)
        )
    ):
        raise ValueError(
            'The cluster-quality table contains '
            'NaN or infinite metric values.'
        )


    print(
        'K-means cluster-number evaluation:'
    )

    display(
        state.cluster_quality.round(4)
    )


    # --------------------------------------------------------------------------
    # Identify metric-specific candidates
    # --------------------------------------------------------------------------

    state.best_silhouette_k = int(
        state.cluster_quality.loc[
            state.cluster_quality[
                'silhouette'
            ].idxmax(),
            'n_clusters'
        ]
    )

    state.best_calinski_k = int(
        state.cluster_quality.loc[
            state.cluster_quality[
                'calinski_harabasz'
            ].idxmax(),
            'n_clusters'
        ]
    )

    state.best_davies_k = int(
        state.cluster_quality.loc[
            state.cluster_quality[
                'davies_bouldin'
            ].idxmin(),
            'n_clusters'
        ]
    )


    print(
        'Best K by silhouette score:',
        state.best_silhouette_k
    )

    print(
        'Best K by Calinski-Harabasz score:',
        state.best_calinski_k
    )

    print(
        'Best K by Davies-Bouldin score:',
        state.best_davies_k
    )

    print(
        '\nThese metric-specific values are diagnostics only. '
        'The final K must also consider cluster size, temporal '
        'stability, algorithm agreement, and physical interpretation.'
    )


    # --------------------------------------------------------------------------
    # Plot cluster-number diagnostics
    # --------------------------------------------------------------------------

    state.fig, state.axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(11, 8)
    )

    state.plot_definitions = [
        (
            'silhouette',
            'Silhouette Score',
            'Higher is better'
        ),
        (
            'calinski_harabasz',
            'Calinski-Harabasz Score',
            'Higher is better'
        ),
        (
            'davies_bouldin',
            'Davies-Bouldin Score',
            'Lower is better'
        ),
        (
            'inertia',
            'K-Means Inertia',
            'Inspect for an elbow'
        )
    ]


    for state.ax, (
        state.metric,
        state.y_label,
        state.interpretation
    ) in zip(
        state.axes.reshape(-1),
        state.plot_definitions
    ):
        state.ax.plot(
            state.cluster_quality['n_clusters'],
            state.cluster_quality[state.metric],
            marker='o',
            linewidth=1.8,
            markersize=6
        )

        state.ax.set_xlabel(
            'Number of Clusters, K'
        )

        state.ax.set_ylabel(
            state.y_label
        )

        state.ax.set_title(
            state.interpretation
        )

        state.ax.xaxis.set_major_locator(
            ticker.MaxNLocator(
                nbins=len(state.valid_k_values),
                integer=True
            )
        )

        state.ax.grid(
            linestyle='--',
            alpha=0.30
        )

        state.ax.set_axisbelow(
            True
        )


    state.fig.suptitle(
        'Internal Validation of Candidate K-Means Partitions',
        fontsize=14,
        fontweight='bold',
        y=1.01
    )

    plt.tight_layout()
    plt.show()


    # --------------------------------------------------------------------------
    # Plot cluster-size balance for each candidate K
    # --------------------------------------------------------------------------

    state.fig, state.axes = plt.subplots(
        nrows=1,
        ncols=len(state.valid_k_values),
        figsize=(4.0 * len(state.valid_k_values), 3.5),
        squeeze=False
    )

    for state.plot_index, state.n_clusters in enumerate(
        state.valid_k_values
    ):
        state.ax = state.axes[0, state.plot_index]

        state.labels_k = state.cluster_labels_by_k[
            state.n_clusters
        ]

        state.cluster_sizes = np.bincount(
            state.labels_k,
            minlength=state.n_clusters
        )

        state.ax.bar(
            np.arange(state.n_clusters),
            state.cluster_sizes,
            color='#457b9d',
            edgecolor='black',
            linewidth=0.5
        )

        state.ax.set_title(
            'K = {}'.format(state.n_clusters)
        )

        state.ax.set_xlabel(
            'Cluster'
        )

        state.ax.set_ylabel(
            'Number of Buildings'
        )

        state.ax.set_xticks(
            np.arange(state.n_clusters)
        )

        state.ax.yaxis.set_major_locator(
            ticker.MaxNLocator(
                nbins=5,
                integer=True
            )
        )

        state.ax.grid(
            axis='y',
            linestyle='--',
            alpha=0.30
        )

        state.ax.set_axisbelow(
            True
        )


    state.fig.suptitle(
        'Cluster-Size Balance for Candidate Partitions',
        fontsize=13,
        fontweight='bold',
        y=1.03
    )

    plt.tight_layout()
    plt.show()


    # --------------------------------------------------------------------------
    # Save cluster-number diagnostics
    # --------------------------------------------------------------------------

    state.quality_output_path = (
        state.OUTPUT_DIR
        / 'cluster_number_quality.csv'
    )

    state.cluster_quality.to_csv(
        state.quality_output_path,
        index=False
    )

    print(
        '\nCluster-quality results saved to:',
        state.quality_output_path.resolve()
    )

def cell_25(state):
    """Cell 25: select final K and inspect physical cluster summaries."""
    # --------------------------------------------------------------------------
    # Select the final K-means partition and summarize the clusters
    # --------------------------------------------------------------------------

    state.FINAL_K = 3


    # --------------------------------------------------------------------------
    # Validate the selected cluster count
    # --------------------------------------------------------------------------

    if state.FINAL_K not in state.cluster_models:
        raise ValueError(
            'FINAL_K={} was not screened. Available K values: {}'.format(
                state.FINAL_K,
                sorted(state.cluster_models.keys())
            )
        )

    if state.FINAL_K not in state.cluster_labels_by_k:
        raise KeyError(
            'Cluster labels are unavailable for FINAL_K={}.'.format(
                state.FINAL_K
            )
        )


    # --------------------------------------------------------------------------
    # Retrieve the fitted K-means model and labels
    # --------------------------------------------------------------------------

    state.kmeans_model = state.cluster_models[state.FINAL_K]

    state.cluster_labels = np.asarray(
        state.cluster_labels_by_k[state.FINAL_K],
        dtype=int
    ).copy()


    if len(state.cluster_labels) != len(state.BUILDING_IDS):
        raise ValueError(
            'Expected {} cluster labels, but found {}.'.format(
                len(state.BUILDING_IDS),
                len(state.cluster_labels)
            )
        )


    state.unique_clusters = np.unique(
        state.cluster_labels
    )


    if len(state.unique_clusters) != state.FINAL_K:
        raise ValueError(
            'Expected {} nonempty clusters, but found {}: {}'.format(
                state.FINAL_K,
                len(state.unique_clusters),
                state.unique_clusters.tolist()
            )
        )


    if state.kmeans_model.n_iter_ >= state.KMEANS_MAX_ITER:
        print(
            'WARNING: K-means reached the maximum iteration limit.'
        )
    else:
        print(
            'K-means converged in {} iterations.'.format(
                state.kmeans_model.n_iter_
            )
        )


    # --------------------------------------------------------------------------
    # Create the building-to-cluster assignment table
    # --------------------------------------------------------------------------

    state.cluster_assignments = pd.DataFrame({
        'building_id': state.BUILDING_IDS,
        'cluster': state.cluster_labels
    }).set_index(
        'building_id'
    ).sort_index()


    if state.cluster_assignments.index.duplicated().any():
        raise ValueError(
            'The cluster-assignment table contains duplicated building IDs.'
        )


    if not state.cluster_assignments.index.equals(
        state.descriptors.index
    ):
        raise ValueError(
            'Building order in cluster assignments does not match '
            'the descriptor-table order.'
        )


    # Join cluster labels with descriptors in physical units.
    state.clustered_descriptors = state.descriptors.join(
        state.cluster_assignments,
        how='inner'
    )


    # Join cluster labels with robustly scaled descriptors.
    state.clustered_scaled_descriptors = state.scaled_descriptors.join(
        state.cluster_assignments,
        how='inner'
    )


    if state.clustered_descriptors.shape[0] != len(state.BUILDING_IDS):
        raise ValueError(
            'Unexpected number of rows after joining descriptors '
            'and cluster assignments.'
        )


    # --------------------------------------------------------------------------
    # Report cluster sizes and members
    # --------------------------------------------------------------------------

    state.cluster_sizes = (
        state.cluster_assignments['cluster']
        .value_counts()
        .sort_index()
    )


    print('\nSelected number of clusters:', state.FINAL_K)

    print(
        'K-means inertia: {:.6f}'.format(
            state.kmeans_model.inertia_
        )
    )

    print('\nCluster sizes:')

    display(
        state.cluster_sizes.rename(
            'number_of_buildings'
        ).to_frame()
    )


    state.cluster_member_records = []

    print('Cluster members:')

    for state.cluster_id in sorted(state.unique_clusters):
        state.members = (
            state.cluster_assignments[
                state.cluster_assignments['cluster'] == state.cluster_id
            ]
            .index
            .astype(int)
            .tolist()
        )

        print(
            '  Cluster {}: {}'.format(
                state.cluster_id,
                state.members
            )
        )

        for state.building_id in state.members:
            state.cluster_member_records.append({
                'cluster': int(state.cluster_id),
                'building_id': int(state.building_id)
            })


    state.cluster_members_table = pd.DataFrame(
        state.cluster_member_records
    )


    # --------------------------------------------------------------------------
    # Summarize descriptors in their original physical units
    # --------------------------------------------------------------------------

    state.cluster_physical_medians = (
        state.clustered_descriptors
        .groupby('cluster')[state.SELECTED_FEATURES]
        .median()
    )

    state.cluster_physical_means = (
        state.clustered_descriptors
        .groupby('cluster')[state.SELECTED_FEATURES]
        .mean()
    )

    state.cluster_physical_std = (
        state.clustered_descriptors
        .groupby('cluster')[state.SELECTED_FEATURES]
        .std(ddof=0)
    )


    print('\nMedian physical descriptors by cluster:')

    display(
        state.cluster_physical_medians.round(4)
    )


    print('Mean physical descriptors by cluster:')

    display(
        state.cluster_physical_means.round(4)
    )


    print('Within-cluster descriptor standard deviations:')

    display(
        state.cluster_physical_std.round(4)
    )


    # --------------------------------------------------------------------------
    # Summarize descriptors in the robustly scaled clustering space
    # --------------------------------------------------------------------------

    state.cluster_scaled_medians = (
        state.clustered_scaled_descriptors
        .groupby('cluster')[state.SELECTED_FEATURES]
        .median()
    )

    state.cluster_scaled_means = (
        state.clustered_scaled_descriptors
        .groupby('cluster')[state.SELECTED_FEATURES]
        .mean()
    )


    print('Median robustly scaled descriptors by cluster:')

    display(
        state.cluster_scaled_medians.round(3)
    )


    print('Mean robustly scaled descriptors by cluster:')

    display(
        state.cluster_scaled_means.round(3)
    )


    # --------------------------------------------------------------------------
    # Create a compact cluster-membership summary
    # --------------------------------------------------------------------------

    state.cluster_profile_records = []

    for state.cluster_id in sorted(state.unique_clusters):
        state.members = (
            state.cluster_members_table[
                state.cluster_members_table['cluster'] == state.cluster_id
            ]['building_id']
            .astype(int)
            .tolist()
        )

        state.cluster_profile_records.append({
            'cluster': int(state.cluster_id),
            'number_of_buildings': len(state.members),
            'members': ', '.join(
                'B{}'.format(state.building_id)
                for state.building_id in state.members
            )
        })


    state.cluster_profile_table = (
        pd.DataFrame(state.cluster_profile_records)
        .set_index('cluster')
    )


    print('Compact cluster profile:')

    display(
        state.cluster_profile_table
    )


    # --------------------------------------------------------------------------
    # Safely save assignments and cluster summaries
    # --------------------------------------------------------------------------

    state.OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    state.output_tables = {
        'building_cluster_assignments.csv':
            state.cluster_assignments.reset_index(),

        'cluster_members.csv':
            state.cluster_members_table,

        'cluster_physical_medians.csv':
            state.cluster_physical_medians.reset_index(),

        'cluster_physical_means.csv':
            state.cluster_physical_means.reset_index(),

        'cluster_physical_standard_deviations.csv':
            state.cluster_physical_std.reset_index(),

        'cluster_scaled_medians.csv':
            state.cluster_scaled_medians.reset_index(),

        'cluster_scaled_means.csv':
            state.cluster_scaled_means.reset_index(),

        'cluster_profile.csv':
            state.cluster_profile_table.reset_index()
    }


    for state.filename, state.output_table in state.output_tables.items():
        state.output_path = state.OUTPUT_DIR / state.filename

        try:
            state.output_table.to_csv(
                state.output_path,
                index=False
            )

        except PermissionError:
            state.alternative_path = (
                state.OUTPUT_DIR
                / state.filename.replace(
                    '.csv',
                    '_new.csv'
                )
            )

            state.output_table.to_csv(
                state.alternative_path,
                index=False
            )

            print(
                'WARNING: {} could not be overwritten, probably '
                'because it is open in Excel. Saved as {}.'.format(
                    state.filename,
                    state.alternative_path.name
                )
            )


    print(
        '\nCluster assignments and summaries saved to:'
    )

    print(
        state.OUTPUT_DIR.resolve()
    )

def cell_27(state):
    """Cell 27: initialization stability."""
    state.repeat_labels = []
    state.repeat_inertias = []
    for state.seed in range(state.N_INITIALIZATION_REPEATS):
        state.model = KMeans(n_clusters=state.FINAL_K, random_state=state.seed,
                       n_init=state.KMEANS_N_INIT, max_iter=state.KMEANS_MAX_ITER)
        state.repeat_labels.append(state.model.fit_predict(state.X_scaled))
        state.repeat_inertias.append(state.model.inertia_)

    state.ari_values = []
    for state.i in range(len(state.repeat_labels)):
        for state.j in range(state.i+1, len(state.repeat_labels)):
            state.ari_values.append(adjusted_rand_score(state.repeat_labels[state.i], state.repeat_labels[state.j]))

    state.initialization_stability = {
        'mean_pairwise_ari': float(np.mean(state.ari_values)),
        'minimum_pairwise_ari': float(np.min(state.ari_values)),
        'maximum_pairwise_ari': float(np.max(state.ari_values)),
        'inertia_mean': float(np.mean(state.repeat_inertias)),
        'inertia_std': float(np.std(state.repeat_inertias, ddof=1)),
    }
    print(json.dumps(state.initialization_stability, indent=2))

def cell_29(state):
    """Cell 29: day-bootstrap stability."""
    # --------------------------------------------------------------------------
    # Evaluate K-means stability across random initializations
    # --------------------------------------------------------------------------

    if state.FINAL_K not in state.cluster_models:
        raise ValueError(
            'FINAL_K={} was not previously evaluated.'.format(
                state.FINAL_K
            )
        )
    if len(state.cluster_labels) != state.X_scaled.shape[0]:
        raise ValueError(
            'Reference cluster labels and X_scaled contain '
            'different numbers of buildings.'
        )

    if state.N_INITIALIZATION_REPEATS < 2:
        raise ValueError(
            'N_INITIALIZATION_REPEATS must be at least 2.'
        )

    if not np.all(np.isfinite(state.X_scaled)):
        raise ValueError(
            'X_scaled contains NaN or infinite values.'
        )


    # --------------------------------------------------------------------------
    # Repeat the complete K-means fitting procedure
    # --------------------------------------------------------------------------

    state.repeat_labels = []
    state.repeat_inertias = []
    state.repeat_iterations = []
    state.repeat_reference_ari = []
    state.repeat_records = []

    for state.repeat_seed in range(
        state.N_INITIALIZATION_REPEATS
    ):
        state.repeated_model = KMeans(
            n_clusters=state.FINAL_K,
            random_state=state.repeat_seed,
            n_init=state.KMEANS_N_INIT,
            max_iter=state.KMEANS_MAX_ITER,

            # Compatible with scikit-learn 1.0.2.
            algorithm='full'
        )

        state.repeated_labels = state.repeated_model.fit_predict(
            state.X_scaled
        )

        state.unique_repeated_clusters = np.unique(
            state.repeated_labels
        )

        if len(state.unique_repeated_clusters) != state.FINAL_K:
            raise RuntimeError(
                'Repeated K-means fit with random_state={} '
                'produced {} nonempty clusters instead of {}.'.format(
                    state.repeat_seed,
                    len(state.unique_repeated_clusters),
                    state.FINAL_K
                )
            )

        state.reference_ari = adjusted_rand_score(
            state.cluster_labels,
            state.repeated_labels
        )

        state.repeat_labels.append(
            state.repeated_labels.copy()
        )

        state.repeat_inertias.append(
            float(state.repeated_model.inertia_)
        )

        state.repeat_iterations.append(
            int(state.repeated_model.n_iter_)
        )

        state.repeat_reference_ari.append(
            float(state.reference_ari)
        )

        state.repeat_records.append({
            'random_state':
                state.repeat_seed,

            'adjusted_rand_index_vs_reference':
                float(state.reference_ari),

            'inertia':
                float(state.repeated_model.inertia_),

            'iterations_to_convergence':
                int(state.repeated_model.n_iter_),

            'reached_iteration_limit':
                bool(
                    state.repeated_model.n_iter_
                    >= state.KMEANS_MAX_ITER
                )
        })


    state.repeat_results = pd.DataFrame(
        state.repeat_records
    )


    # --------------------------------------------------------------------------
    # Calculate pairwise ARI across all repeated partitions
    # --------------------------------------------------------------------------

    state.pairwise_ari_records = []

    for state.left_index in range(
        len(state.repeat_labels)
    ):
        for state.right_index in range(
            state.left_index + 1,
            len(state.repeat_labels)
        ):
            state.ari_value = adjusted_rand_score(
                state.repeat_labels[state.left_index],
                state.repeat_labels[state.right_index]
            )

            state.pairwise_ari_records.append({
                'left_random_state':
                    state.left_index,

                'right_random_state':
                    state.right_index,

                'adjusted_rand_index':
                    float(state.ari_value)
            })


    state.pairwise_ari_table = pd.DataFrame(
        state.pairwise_ari_records
    )

    state.pairwise_ari_values = state.pairwise_ari_table[
        'adjusted_rand_index'
    ].to_numpy(dtype=float)


    if not np.all(
        np.isfinite(state.pairwise_ari_values)
    ):
        raise ValueError(
            'Pairwise ARI calculations contain '
            'nonfinite values.'
        )


    # --------------------------------------------------------------------------
    # Summarize initialization stability
    # --------------------------------------------------------------------------

    state.initialization_stability = {
        'final_k':
            int(state.FINAL_K),

        'number_of_repeats':
            int(state.N_INITIALIZATION_REPEATS),

        'number_of_pairwise_comparisons':
            int(len(state.pairwise_ari_values)),

        'mean_pairwise_ari':
            float(
                np.mean(state.pairwise_ari_values)
            ),

        'median_pairwise_ari':
            float(
                np.median(state.pairwise_ari_values)
            ),

        'minimum_pairwise_ari':
            float(
                np.min(state.pairwise_ari_values)
            ),

        'maximum_pairwise_ari':
            float(
                np.max(state.pairwise_ari_values)
            ),

        'pairwise_ari_std':
            float(
                np.std(
                    state.pairwise_ari_values,
                    ddof=1
                )
            ),

        'mean_ari_vs_reference':
            float(
                np.mean(
                    state.repeat_reference_ari
                )
            ),

        'minimum_ari_vs_reference':
            float(
                np.min(
                    state.repeat_reference_ari
                )
            ),

        'inertia_mean':
            float(
                np.mean(
                    state.repeat_inertias
                )
            ),

        'inertia_std':
            float(
                np.std(
                    state.repeat_inertias,
                    ddof=1
                )
            ),

        'inertia_minimum':
            float(
                np.min(
                    state.repeat_inertias
                )
            ),

        'inertia_maximum':
            float(
                np.max(
                    state.repeat_inertias
                )
            ),

        'mean_iterations_to_convergence':
            float(
                np.mean(
                    state.repeat_iterations
                )
            ),

        'maximum_iterations_to_convergence':
            int(
                np.max(
                    state.repeat_iterations
                )
            ),

        'number_reaching_iteration_limit':
            int(
                state.repeat_results[
                    'reached_iteration_limit'
                ].sum()
            )
    }


    print(
        'K-means initialization-stability summary:'
    )

    print(
        json.dumps(
            state.initialization_stability,
            indent=2
        )
    )


    print(
        '\nResults from individual random-state repetitions:'
    )

    display(
        state.repeat_results.round(6)
    )


    # --------------------------------------------------------------------------
    # Plot the stability distributions
    # --------------------------------------------------------------------------

    state.fig, state.axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12, 4)
    )


    sns.histplot(
        state.pairwise_ari_values,
        bins=15,
        kde=False,
        ax=state.axes[0],
        color='#457b9d'
    )

    state.axes[0].set_xlabel(
        'Pairwise Adjusted Rand Index'
    )

    state.axes[0].set_ylabel(
        'Number of Partition Pairs'
    )

    state.axes[0].set_title(
        'Agreement Among Repeated K-Means Fits'
    )

    state.axes[0].grid(
        axis='y',
        linestyle='--',
        alpha=0.30
    )


    state.axes[1].scatter(
        state.repeat_results['random_state'],
        state.repeat_results[
            'adjusted_rand_index_vs_reference'
        ],
        color='#e76f51',
        edgecolor='black',
        linewidth=0.4
    )

    state.axes[1].axhline(
        y=1.0,
        color='black',
        linestyle='--',
        linewidth=1.0,
        label='Perfect agreement'
    )

    state.axes[1].set_xlabel(
        'Random State'
    )

    state.axes[1].set_ylabel(
        'ARI Versus Reference Partition'
    )

    state.axes[1].set_title(
        'Agreement with the Selected Partition'
    )

    state.axes[1].set_ylim(
        min(
            -0.05,
            state.repeat_results[
                'adjusted_rand_index_vs_reference'
            ].min() - 0.05
        ),
        1.05
    )

    state.axes[1].grid(
        linestyle='--',
        alpha=0.30
    )

    state.axes[1].legend(
        frameon=False
    )


    state.fig.suptitle(
        'K-Means Sensitivity to Random Initialization',
        fontsize=14,
        fontweight='bold',
        y=1.02
    )

    plt.tight_layout()
    plt.show()


    # --------------------------------------------------------------------------
    # Save initialization-stability results safely
    # --------------------------------------------------------------------------

    state.OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    state.stability_outputs = {
        'kmeans_initialization_repeats.csv':
            state.repeat_results,

        'kmeans_pairwise_ari.csv':
            state.pairwise_ari_table
    }


    for state.filename, state.output_table in state.stability_outputs.items():
        state.output_path = state.OUTPUT_DIR / state.filename

        try:
            state.output_table.to_csv(
                state.output_path,
                index=False
            )

        except PermissionError:
            state.alternative_path = (
                state.OUTPUT_DIR
                / state.filename.replace(
                    '.csv',
                    '_new.csv'
                )
            )

            state.output_table.to_csv(
                state.alternative_path,
                index=False
            )

            print(
                'WARNING: {} could not be overwritten, probably '
                'because it is open in Excel. Saved as {}.'.format(
                    state.filename,
                    state.alternative_path.name
                )
            )


    state.summary_output_path = (
        state.OUTPUT_DIR
        / 'kmeans_initialization_stability.json'
    )

    try:
        with open(
            state.summary_output_path,
            'w',
            encoding='utf-8'
        ) as state.file:
            json.dump(
                state.initialization_stability,
                state.file,
                indent=2
            )

    except PermissionError:
        state.alternative_summary_path = (
            state.OUTPUT_DIR
            / 'kmeans_initialization_stability_new.json'
        )

        with open(
            state.alternative_summary_path,
            'w',
            encoding='utf-8'
        ) as state.file:
            json.dump(
                state.initialization_stability,
                state.file,
                indent=2
            )

        print(
            'WARNING: The original stability-summary file could not '
            'be overwritten. Saved as: {}'.format(
                state.alternative_summary_path.name
            )
        )


    print(
        '\nInitialization-stability outputs saved to:'
    )

    print(
        state.OUTPUT_DIR.resolve()
    )

def cell_31(state):
    """Cell 31: hierarchical-clustering sensitivity."""
    # --------------------------------------------------------------------------
    # Compare K-means with hierarchical agglomerative clustering
    # --------------------------------------------------------------------------

    # Validate the reference K-means partition.
    if state.X_scaled.ndim != 2:
        raise ValueError(
            'X_scaled must be a two-dimensional matrix. '
            'Received shape: {}.'.format(
                state.X_scaled.shape
            )
        )

    if state.X_scaled.shape[0] != len(state.BUILDING_IDS):
        raise ValueError(
            'X_scaled contains {} rows, but {} buildings were expected.'.format(
                state.X_scaled.shape[0],
                len(state.BUILDING_IDS)
            )
        )

    if len(state.cluster_labels) != state.X_scaled.shape[0]:
        raise ValueError(
            'The K-means labels and X_scaled contain different '
            'numbers of buildings.'
        )

    if len(np.unique(state.cluster_labels)) != state.FINAL_K:
        raise ValueError(
            'The K-means reference partition contains {} clusters, '
            'but FINAL_K is {}.'.format(
                len(np.unique(state.cluster_labels)),
                state.FINAL_K
            )
        )

    if not np.all(np.isfinite(state.X_scaled)):
        raise ValueError(
            'X_scaled contains NaN or infinite values.'
        )


    # --------------------------------------------------------------------------
    # Fit hierarchical agglomerative clustering
    # --------------------------------------------------------------------------

    # Ward linkage minimizes within-cluster variance and requires
    # Euclidean distance.
    state.hierarchical_model = AgglomerativeClustering(
        n_clusters=state.FINAL_K,
        affinity='euclidean',
        linkage='ward'
    )

    state.hierarchical_labels = state.hierarchical_model.fit_predict(
        state.X_scaled
    )

    state.hierarchical_labels = np.asarray(
        state.hierarchical_labels,
        dtype=int
    )


    if len(state.hierarchical_labels) != len(state.BUILDING_IDS):
        raise ValueError(
            'Hierarchical clustering returned an unexpected '
            'number of labels.'
        )

    state.unique_hierarchical_clusters = np.unique(
        state.hierarchical_labels
    )

    if len(state.unique_hierarchical_clusters) != state.FINAL_K:
        raise ValueError(
            'Hierarchical clustering produced {} nonempty clusters '
            'instead of {}.'.format(
                len(state.unique_hierarchical_clusters),
                state.FINAL_K
            )
        )


    # --------------------------------------------------------------------------
    # Calculate label-invariant agreement
    # --------------------------------------------------------------------------

    state.algorithm_agreement_ari = adjusted_rand_score(
        state.cluster_labels,
        state.hierarchical_labels
    )


    # --------------------------------------------------------------------------
    # Create a building-level comparison table
    # --------------------------------------------------------------------------

    state.comparison = pd.DataFrame({
        'building_id':
            state.BUILDING_IDS,

        'kmeans_cluster':
            state.cluster_labels,

        'hierarchical_cluster':
            state.hierarchical_labels
    }).set_index(
        'building_id'
    )


    # Numerical labels are arbitrary across algorithms. Therefore,
    # direct equality of cluster numbers is not a valid agreement test.
    # ARI is the primary label-invariant agreement measure.


    # --------------------------------------------------------------------------
    # Report cluster sizes and members
    # --------------------------------------------------------------------------

    state.kmeans_sizes = (
        state.comparison['kmeans_cluster']
        .value_counts()
        .sort_index()
        .rename('kmeans_size')
    )

    state.hierarchical_sizes = (
        state.comparison['hierarchical_cluster']
        .value_counts()
        .sort_index()
        .rename('hierarchical_size')
    )


    print(
        'K-means versus hierarchical clustering ARI: {:.4f}'.format(
            state.algorithm_agreement_ari
        )
    )


    if np.isclose(
        state.algorithm_agreement_ari,
        1.0
    ):
        print(
            'Interpretation: the two methods produced identical '
            'building groupings, apart from arbitrary cluster-label names.'
        )

    elif state.algorithm_agreement_ari >= 0.80:
        print(
            'Interpretation: the two clustering methods show strong '
            'agreement, with limited differences in building assignment.'
        )

    elif state.algorithm_agreement_ari >= 0.50:
        print(
            'Interpretation: the methods show moderate agreement. '
            'The buildings with differing assignments should be examined.'
        )

    else:
        print(
            'Interpretation: agreement is weak. The selected partition '
            'is sensitive to the clustering algorithm and should be '
            'investigated before selecting TD3 development buildings.'
        )


    print('\nCluster sizes produced by K-means:')

    display(
        state.kmeans_sizes.to_frame()
    )


    print('Cluster sizes produced by hierarchical clustering:')

    display(
        state.hierarchical_sizes.to_frame()
    )


    print('Building-level cluster comparison:')

    display(
        state.comparison
    )


    print('K-means cluster members:')

    for state.cluster_id in sorted(
        state.comparison['kmeans_cluster'].unique()
    ):
        state.members = (
            state.comparison[
                state.comparison['kmeans_cluster']
                == state.cluster_id
            ]
            .index
            .astype(int)
            .tolist()
        )

        print(
            '  Cluster {}: {}'.format(
                state.cluster_id,
                state.members
            )
        )


    print('\nHierarchical cluster members:')

    for state.cluster_id in sorted(
        state.comparison[
            'hierarchical_cluster'
        ].unique()
    ):
        state.members = (
            state.comparison[
                state.comparison['hierarchical_cluster']
                == state.cluster_id
            ]
            .index
            .astype(int)
            .tolist()
        )

        print(
            '  Cluster {}: {}'.format(
                state.cluster_id,
                state.members
            )
        )


    # --------------------------------------------------------------------------
    # Save the algorithm-comparison results safely
    # --------------------------------------------------------------------------

    state.OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    state.comparison_output_path = (
        state.OUTPUT_DIR
        / 'kmeans_hierarchical_comparison.csv'
    )

    try:
        state.comparison.reset_index().to_csv(
            state.comparison_output_path,
            index=False
        )

    except PermissionError:
        state.alternative_output_path = (
            state.OUTPUT_DIR
            / 'kmeans_hierarchical_comparison_new.csv'
        )

        state.comparison.reset_index().to_csv(
            state.alternative_output_path,
            index=False
        )

        print(
            '\nWARNING: The original comparison file could not be '
            'overwritten, probably because it is open in Excel.'
        )

        print(
            'Results were saved instead to:',
            state.alternative_output_path.resolve()
        )

    else:
        print(
            '\nAlgorithm-comparison results saved to:',
            state.comparison_output_path.resolve()
        )

def cell_33(state):
    """Cell 33: true within-cluster medoids and overall portfolio medoid."""
    # --------------------------------------------------------------------------
    # Calculate cluster medoids and core-portfolio representative buildings
    # --------------------------------------------------------------------------

    # Buildings identified through descriptor inspection as outlier cases.
    # These buildings remain in the final transfer study but are separated
    # from the primary TD3 hyperparameter-development panel.
    state.OUTLIER_BUILDINGS = [12, 15]


    # --------------------------------------------------------------------------
    # Validate clustering inputs
    # --------------------------------------------------------------------------

    if state.X_scaled.ndim != 2:
        raise ValueError(
            'X_scaled must be a two-dimensional matrix. '
            'Received shape: {}.'.format(
                state.X_scaled.shape
            )
        )

    if state.X_scaled.shape[0] != len(state.BUILDING_IDS):
        raise ValueError(
            'X_scaled contains {} rows, but {} buildings were expected.'.format(
                state.X_scaled.shape[0],
                len(state.BUILDING_IDS)
            )
        )

    if len(state.cluster_labels) != len(state.BUILDING_IDS):
        raise ValueError(
            'Expected {} cluster labels, but found {}.'.format(
                len(state.BUILDING_IDS),
                len(state.cluster_labels)
            )
        )

    if not np.all(np.isfinite(state.X_scaled)):
        raise ValueError(
            'X_scaled contains NaN or infinite values.'
        )

    state.missing_outlier_buildings = sorted(
        set(state.OUTLIER_BUILDINGS)
        - set(state.BUILDING_IDS)
    )

    if state.missing_outlier_buildings:
        raise ValueError(
            'The following declared outlier buildings are not present: {}'.format(
                state.missing_outlier_buildings
            )
        )


    # --------------------------------------------------------------------------
    # Calculate the full pairwise Euclidean-distance matrix
    # --------------------------------------------------------------------------

    state.distance_matrix = squareform(
        pdist(
            state.X_scaled,
            metric='euclidean'
        )
    )

    if state.distance_matrix.shape != (
        len(state.BUILDING_IDS),
        len(state.BUILDING_IDS)
    ):
        raise ValueError(
            'The pairwise distance matrix has an unexpected shape: {}.'.format(
                state.distance_matrix.shape
            )
        )

    if not np.allclose(
        state.distance_matrix,
        state.distance_matrix.T,
        rtol=1e-10,
        atol=1e-10
    ):
        raise ValueError(
            'The pairwise distance matrix is not symmetric.'
        )


    # Create a labelled distance table for interpretation and export.
    state.distance_table = pd.DataFrame(
        state.distance_matrix,
        index=state.BUILDING_IDS,
        columns=state.BUILDING_IDS
    )

    state.distance_table.index.name = 'building_id'


    # --------------------------------------------------------------------------
    # Identify the cluster containing the declared outlier buildings
    # --------------------------------------------------------------------------

    state.building_to_cluster = {
        int(state.building_id): int(cluster_label)
        for state.building_id, cluster_label in zip(
            state.BUILDING_IDS,
            state.cluster_labels
        )
    }

    state.outlier_cluster_labels = {
        state.building_to_cluster[state.building_id]
        for state.building_id in state.OUTLIER_BUILDINGS
    }

    if len(state.outlier_cluster_labels) != 1:
        raise ValueError(
            'The declared outlier buildings do not belong to one common '
            'cluster. Their cluster labels are: {}.'.format(
                sorted(state.outlier_cluster_labels)
            )
        )

    state.OUTLIER_CLUSTER = int(
        next(iter(state.outlier_cluster_labels))
    )

    print(
        'Outlier cluster:',
        state.OUTLIER_CLUSTER
    )

    print(
        'Declared outlier buildings:',
        state.OUTLIER_BUILDINGS
    )


    # --------------------------------------------------------------------------
    # Calculate true within-cluster medoids
    # --------------------------------------------------------------------------

    state.cluster_medoid_records = []
    state.cluster_medoids = {}

    for state.cluster_id in sorted(
        np.unique(state.cluster_labels)
    ):
        state.member_indices = np.where(
            state.cluster_labels == state.cluster_id
        )[0]

        state.member_buildings = [
            int(state.BUILDING_IDS[index])
            for index in state.member_indices
        ]

        state.within_cluster_distances = state.distance_matrix[
            np.ix_(
                state.member_indices,
                state.member_indices
            )
        ]

        state.total_within_cluster_distances = (
            state.within_cluster_distances.sum(axis=1)
        )

        state.minimum_total_distance = float(
            state.total_within_cluster_distances.min()
        )

        # More than one medoid candidate may occur, particularly in a
        # two-building cluster, where both members have equal total distance.
        state.tied_local_positions = np.where(
            np.isclose(
                state.total_within_cluster_distances,
                state.minimum_total_distance,
                rtol=1e-10,
                atol=1e-10
            )
        )[0]

        state.tied_medoid_buildings = [
            state.member_buildings[state.position]
            for state.position in state.tied_local_positions
        ]

        # Use the lowest building ID only as a deterministic storage rule.
        # All tied candidates are retained and reported.
        state.selected_medoid_building = int(
            min(state.tied_medoid_buildings)
        )

        state.cluster_medoids[int(state.cluster_id)] = (
            state.selected_medoid_building
        )

        state.cluster_role = (
            'outlier_stress_test'
            if int(state.cluster_id) == state.OUTLIER_CLUSTER
            else 'core_development'
        )

        state.cluster_medoid_records.append({
            'cluster':
                int(state.cluster_id),

            'cluster_role':
                state.cluster_role,

            'cluster_size':
                int(len(state.member_buildings)),

            'cluster_members':
                ', '.join(
                    'B{}'.format(state.building_id)
                    for state.building_id in state.member_buildings
                ),

            'selected_medoid_building':
                state.selected_medoid_building,

            'all_tied_medoid_candidates':
                ', '.join(
                    'B{}'.format(state.building_id)
                    for state.building_id in state.tied_medoid_buildings
                ),

            'number_of_tied_medoid_candidates':
                int(len(state.tied_medoid_buildings)),

            'minimum_total_within_cluster_distance':
                state.minimum_total_distance,

            'mean_pairwise_within_cluster_distance':
                float(
                    state.within_cluster_distances[
                        np.triu_indices(
                            len(state.member_indices),
                            k=1
                        )
                    ].mean()
                )
                if len(state.member_indices) > 1
                else 0.0
        })


    state.cluster_medoids_table = pd.DataFrame(
        state.cluster_medoid_records
    ).sort_values(
        'cluster'
    ).reset_index(
        drop=True
    )


    print(
        '\nCluster medoids and cluster roles:'
    )

    display(
        state.cluster_medoids_table
    )


    # --------------------------------------------------------------------------
    # Select the primary core TD3 development panel
    # --------------------------------------------------------------------------

    state.core_development_panel = (
        state.cluster_medoids_table[
            state.cluster_medoids_table[
                'cluster_role'
            ] == 'core_development'
        ]
        .copy()
        .reset_index(drop=True)
    )


    if state.core_development_panel.empty:
        raise ValueError(
            'No core-cluster medoids were identified.'
        )


    print(
        'Primary TD3 development buildings:'
    )

    for state._, state.row in state.core_development_panel.iterrows():
        print(
            '  Cluster {}: Building {}'.format(
                int(state.row['cluster']),
                int(state.row['selected_medoid_building'])
            )
        )


    # --------------------------------------------------------------------------
    # Calculate the overall medoid of all 17 buildings
    # --------------------------------------------------------------------------

    state.total_distance_to_full_portfolio = (
        state.distance_matrix.sum(axis=1)
    )

    state.overall_medoid_index = int(
        np.argmin(
            state.total_distance_to_full_portfolio
        )
    )

    state.overall_medoid_building = int(
        state.BUILDING_IDS[
            state.overall_medoid_index
        ]
    )


    state.overall_medoid_ranking = pd.DataFrame({
        'building_id':
            state.BUILDING_IDS,

        'cluster':
            state.cluster_labels,

        'is_declared_outlier': [
            state.building_id in state.OUTLIER_BUILDINGS
            for state.building_id in state.BUILDING_IDS
        ],

        'total_distance_to_full_portfolio':
            state.total_distance_to_full_portfolio
    }).sort_values(
        'total_distance_to_full_portfolio'
    ).reset_index(
        drop=True
    )

    state.overall_medoid_ranking[
        'full_portfolio_medoid_rank'
    ] = np.arange(
        1,
        len(state.overall_medoid_ranking) + 1
    )


    print(
        '\nOverall medoid across all 17 buildings: Building {}'.format(
            state.overall_medoid_building
        )
    )


    # --------------------------------------------------------------------------
    # Calculate the medoid of the 15-building core portfolio
    # --------------------------------------------------------------------------

    state.core_buildings = [
        state.building_id
        for state.building_id in state.BUILDING_IDS
        if state.building_id not in state.OUTLIER_BUILDINGS
    ]

    state.core_indices = np.asarray([
        state.BUILDING_IDS.index(
            state.building_id
        )
        for state.building_id in state.core_buildings
    ], dtype=int)

    state.core_distance_matrix = state.distance_matrix[
        np.ix_(
            state.core_indices,
            state.core_indices
        )
    ]

    state.total_distance_to_core_portfolio = (
        state.core_distance_matrix.sum(axis=1)
    )

    state.core_medoid_local_index = int(
        np.argmin(
            state.total_distance_to_core_portfolio
        )
    )

    state.overall_core_medoid_building = int(
        state.core_buildings[
            state.core_medoid_local_index
        ]
    )


    state.core_medoid_ranking = pd.DataFrame({
        'building_id':
            state.core_buildings,

        'cluster': [
            state.building_to_cluster[state.building_id]
            for state.building_id in state.core_buildings
        ],

        'total_distance_to_core_portfolio':
            state.total_distance_to_core_portfolio
    }).sort_values(
        'total_distance_to_core_portfolio'
    ).reset_index(
        drop=True
    )

    state.core_medoid_ranking[
        'core_portfolio_medoid_rank'
    ] = np.arange(
        1,
        len(state.core_medoid_ranking) + 1
    )


    print(
        'Overall medoid of the 15-building core portfolio: '
        'Building {}'.format(
            state.overall_core_medoid_building
        )
    )


    print(
        '\nFull-portfolio medoid ranking:'
    )

    display(
        state.overall_medoid_ranking.round(4)
    )


    print(
        'Core-portfolio medoid ranking:'
    )

    display(
        state.core_medoid_ranking.round(4)
    )


    # --------------------------------------------------------------------------
    # Define outlier stress-test candidates
    # --------------------------------------------------------------------------

    state.outlier_stress_test_candidates = pd.DataFrame({
        'building_id':
            state.OUTLIER_BUILDINGS,

        'cluster': [
            state.building_to_cluster[state.building_id]
            for state.building_id in state.OUTLIER_BUILDINGS
        ],

        'distance_to_full_portfolio': [
            state.total_distance_to_full_portfolio[
                state.BUILDING_IDS.index(state.building_id)
            ]
            for state.building_id in state.OUTLIER_BUILDINGS
        ],

        'distance_to_core_medoid': [
            state.distance_matrix[
                state.BUILDING_IDS.index(state.building_id),
                state.BUILDING_IDS.index(
                    state.overall_core_medoid_building
                )
            ]
            for state.building_id in state.OUTLIER_BUILDINGS
        ]
    }).sort_values(
        'distance_to_full_portfolio'
    ).reset_index(
        drop=True
    )


    print(
        'Outlier stress-test candidates:'
    )

    display(
        state.outlier_stress_test_candidates.round(4)
    )


    print(
        '\nRecommended methodology:'
    )

    print(
        '  Primary TD3 tuning: use the medoid from each core cluster.'
    )

    print(
        '  Single-building fallback: use Building {}.'.format(
            state.overall_core_medoid_building
        )
    )

    print(
        '  Outlier robustness test: evaluate Building 12 separately.'
    )

    print(
        '  Final transfer matrix: retain all 17 buildings, including '
        'Buildings 12 and 15.'
    )


    # --------------------------------------------------------------------------
    # Safely export medoid and distance results
    # --------------------------------------------------------------------------

    state.OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    state.output_tables = {
        'pairwise_physical_distances.csv':
            state.distance_table.reset_index(),

        'cluster_medoids_and_roles.csv':
            state.cluster_medoids_table,

        'core_td3_development_panel.csv':
            state.core_development_panel,

        'full_portfolio_medoid_ranking.csv':
            state.overall_medoid_ranking,

        'core_portfolio_medoid_ranking.csv':
            state.core_medoid_ranking,

        'outlier_stress_test_candidates.csv':
            state.outlier_stress_test_candidates
    }


    for state.filename, state.output_table in state.output_tables.items():
        state.output_path = state.OUTPUT_DIR / state.filename

        try:
            state.output_table.to_csv(
                state.output_path,
                index=False
            )

        except PermissionError:
            state.alternative_path = (
                state.OUTPUT_DIR
                / state.filename.replace(
                    '.csv',
                    '_new.csv'
                )
            )

            state.output_table.to_csv(
                state.alternative_path,
                index=False
            )

            print(
                'WARNING: {} could not be overwritten, probably because '
                'it is open in Excel. Saved as {}.'.format(
                    state.filename,
                    state.alternative_path.name
                )
            )


    print(
        '\nMedoid and distance outputs saved to:'
    )

    print(
        state.OUTPUT_DIR.resolve()
    )

def cell_35(state):
    """Cell 35: PCA visualization only."""
    # --------------------------------------------------------------------------
    # Visualize physical building clusters using two-dimensional PCA
    # --------------------------------------------------------------------------

    # PCA is used only for visualization.
    # Clustering and medoid selection remain based on the complete
    # robustly scaled physical-descriptor matrix, X_scaled.

    if state.X_scaled.ndim != 2:
        raise ValueError(
            'X_scaled must be a two-dimensional matrix. '
            'Received shape: {}.'.format(
                state.X_scaled.shape
            )
        )

    if state.X_scaled.shape[0] != len(state.BUILDING_IDS):
        raise ValueError(
            'X_scaled contains {} rows, but {} buildings were expected.'.format(
                state.X_scaled.shape[0],
                len(state.BUILDING_IDS)
            )
        )

    if len(state.cluster_labels) != len(state.BUILDING_IDS):
        raise ValueError(
            'Expected {} cluster labels, but found {}.'.format(
                len(state.BUILDING_IDS),
                len(state.cluster_labels)
            )
        )

    if not np.all(np.isfinite(state.X_scaled)):
        raise ValueError(
            'X_scaled contains NaN or infinite values.'
        )

    if len(np.unique(state.cluster_labels)) != state.FINAL_K:
        raise ValueError(
            'The cluster-label array contains {} clusters, '
            'but FINAL_K is {}.'.format(
                len(np.unique(state.cluster_labels)),
                state.FINAL_K
            )
        )

    if 'OUTLIER_BUILDINGS' not in globals():
        state.OUTLIER_BUILDINGS = [12, 15]


    # --------------------------------------------------------------------------
    # Fit PCA for visualization
    # --------------------------------------------------------------------------

    state.pca_model = PCA(
        n_components=2
    )

    state.X_pca = state.pca_model.fit_transform(
        state.X_scaled
    )

    state.explained_variance = (
        state.pca_model.explained_variance_ratio_
    )

    state.total_explained_variance = float(
        state.explained_variance.sum()
    )


    if not np.all(np.isfinite(state.X_pca)):
        raise ValueError(
            'PCA produced NaN or infinite coordinates.'
        )


    # --------------------------------------------------------------------------
    # Build the PCA plotting table
    # --------------------------------------------------------------------------

    state.plot_data = pd.DataFrame({
        'building_id':
            state.BUILDING_IDS,

        'PC1':
            state.X_pca[:, 0],

        'PC2':
            state.X_pca[:, 1],

        'cluster':
            state.cluster_labels.astype(int)
    })


    state.plot_data['cluster_label'] = state.plot_data[
        'cluster'
    ].apply(
        lambda value: 'Cluster {}'.format(
            int(value)
        )
    )


    state.plot_data['is_outlier'] = state.plot_data[
        'building_id'
    ].isin(
        state.OUTLIER_BUILDINGS
    )


    # Identify the core-cluster medoids from the updated medoid table.
    state.core_medoid_buildings = (
        state.cluster_medoids_table[
            state.cluster_medoids_table[
                'cluster_role'
            ] == 'core_development'
        ][
            'selected_medoid_building'
        ]
        .astype(int)
        .tolist()
    )


    state.plot_data['is_core_medoid'] = state.plot_data[
        'building_id'
    ].isin(
        state.core_medoid_buildings
    )


    state.plot_data['is_overall_core_medoid'] = (
        state.plot_data['building_id']
        == state.overall_core_medoid_building
    )


    # --------------------------------------------------------------------------
    # Create PCA scatter plot
    # --------------------------------------------------------------------------

    state.fig, state.ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(11, 8)
    )


    sns.scatterplot(
        data=state.plot_data,
        x='PC1',
        y='PC2',
        hue='cluster_label',
        palette='Set2',
        s=120,
        edgecolor='black',
        linewidth=0.8,
        ax=state.ax,
        zorder=2
    )


    # --------------------------------------------------------------------------
    # Annotate every building
    # --------------------------------------------------------------------------

    for state._, state.row in state.plot_data.iterrows():
        state.building_id = int(
            state.row['building_id']
        )

        if state.row['is_outlier']:
            state.label_color = '#b22222'
            state.font_weight = 'bold'
        elif state.row['is_core_medoid']:
            state.label_color = 'black'
            state.font_weight = 'bold'
        else:
            state.label_color = '#333333'
            state.font_weight = 'normal'

        state.ax.annotate(
            'B{}'.format(state.building_id),
            xy=(
                state.row['PC1'],
                state.row['PC2']
            ),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            fontweight=state.font_weight,
            color=state.label_color,
            zorder=5
        )


    # --------------------------------------------------------------------------
    # Highlight core-cluster medoids
    # --------------------------------------------------------------------------

    for state.building_id in state.core_medoid_buildings:
        state.position = state.BUILDING_IDS.index(
            state.building_id
        )

        state.ax.scatter(
            state.X_pca[state.position, 0],
            state.X_pca[state.position, 1],
            marker='X',
            s=280,
            color='black',
            edgecolor='white',
            linewidth=1.2,
            zorder=6,
            label='Core-cluster medoid'
        )


    # --------------------------------------------------------------------------
    # Highlight the overall core-portfolio medoid
    # --------------------------------------------------------------------------

    state.overall_core_position = state.BUILDING_IDS.index(
        state.overall_core_medoid_building
    )

    state.ax.scatter(
        state.X_pca[state.overall_core_position, 0],
        state.X_pca[state.overall_core_position, 1],
        marker='*',
        s=420,
        color='#ffd166',
        edgecolor='black',
        linewidth=1.2,
        zorder=7,
        label='Overall core medoid'
    )


    # --------------------------------------------------------------------------
    # Highlight outlier buildings
    # --------------------------------------------------------------------------

    for state.outlier_index, state.building_id in enumerate(
        state.OUTLIER_BUILDINGS
    ):
        state.position = state.BUILDING_IDS.index(
            state.building_id
        )

        state.ax.scatter(
            state.X_pca[state.position, 0],
            state.X_pca[state.position, 1],
            marker='D',
            s=190,
            facecolor='none',
            edgecolor='#b22222',
            linewidth=2.0,
            zorder=6,
            label=(
                'Outlier building'
                if state.outlier_index == 0
                else None
            )
        )


    # --------------------------------------------------------------------------
    # Axis labels and interpretation warning
    # --------------------------------------------------------------------------

    state.ax.set_xlabel(
        'Principal Component 1 ({:.1f}% variance)'.format(
            100.0
            * state.explained_variance[0]
        )
    )

    state.ax.set_ylabel(
        'Principal Component 2 ({:.1f}% variance)'.format(
            100.0
            * state.explained_variance[1]
        )
    )

    state.ax.set_title(
        'PCA Visualization of Physical Building Clusters,\n'
        'Core Medoids, and Outlier Buildings',
        fontsize=14,
        fontweight='bold'
    )

    state.ax.grid(
        linestyle='--',
        alpha=0.30
    )

    state.ax.set_axisbelow(
        True
    )


    # --------------------------------------------------------------------------
    # Remove duplicate legend entries
    # --------------------------------------------------------------------------

    state.legend_handles, state.legend_labels = (
        state.ax.get_legend_handles_labels()
    )

    state.unique_legend_entries = {}

    for state.handle, state.label in zip(
        state.legend_handles,
        state.legend_labels
    ):
        if state.label and state.label not in state.unique_legend_entries:
            state.unique_legend_entries[state.label] = state.handle


    state.ax.legend(
        state.unique_legend_entries.values(),
        state.unique_legend_entries.keys(),
        title='Building Group',
        loc='best',
        frameon=True
    )


    # Add a clarification inside the figure.
    state.ax.text(
        0.01,
        0.01,
        (
            'PCA is used only for visualization. '
            'Clustering and medoid selection use the full '
            'scaled descriptor space.'
        ),
        transform=state.ax.transAxes,
        fontsize=8,
        color='#444444',
        verticalalignment='bottom'
    )


    plt.tight_layout()
    plt.show()


    # --------------------------------------------------------------------------
    # Report PCA information
    # --------------------------------------------------------------------------

    print(
        'Variance explained by PC1: {:.2f}%'.format(
            100.0
            * state.explained_variance[0]
        )
    )

    print(
        'Variance explained by PC2: {:.2f}%'.format(
            100.0
            * state.explained_variance[1]
        )
    )

    print(
        'Total variance explained by the two-dimensional projection: '
        '{:.2f}%'.format(
            100.0
            * state.total_explained_variance
        )
    )

    print(
        '\nCore-cluster medoid buildings:',
        state.core_medoid_buildings
    )

    print(
        'Overall core-portfolio medoid: Building {}'.format(
            state.overall_core_medoid_building
        )
    )

    print(
        'Outlier buildings:',
        state.OUTLIER_BUILDINGS
    )


    if state.total_explained_variance < 0.70:
        print(
            '\nCAUTION: The first two principal components explain less '
            'than 70% of the total descriptor variance. Distances and '
            'apparent separation in this two-dimensional figure should '
            'not be interpreted as the complete physical relationship.'
        )


    # --------------------------------------------------------------------------
    # Save PCA coordinates and figure safely
    # --------------------------------------------------------------------------

    state.OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    state.pca_output_table = state.plot_data.copy()

    state.pca_output_table[
        'is_core_medoid'
    ] = state.pca_output_table[
        'is_core_medoid'
    ].astype(bool)

    state.pca_output_table[
        'is_overall_core_medoid'
    ] = state.pca_output_table[
        'is_overall_core_medoid'
    ].astype(bool)


    state.pca_table_path = (
        state.OUTPUT_DIR
        / 'building_pca_coordinates.csv'
    )

    try:
        state.pca_output_table.to_csv(
            state.pca_table_path,
            index=False
        )

    except PermissionError:
        state.alternative_pca_table_path = (
            state.OUTPUT_DIR
            / 'building_pca_coordinates_new.csv'
        )

        state.pca_output_table.to_csv(
            state.alternative_pca_table_path,
            index=False
        )

        print(
            '\nWARNING: The original PCA-coordinate file could not be '
            'overwritten, probably because it is open in Excel.'
        )

        print(
            'PCA coordinates were saved instead to:',
            state.alternative_pca_table_path.resolve()
        )


    state.figure_path = (
        state.OUTPUT_DIR
        / 'physical_building_clusters_pca.png'
    )

    try:
        state.fig.savefig(
            state.figure_path,
            dpi=300,
            bbox_inches='tight'
        )

    except PermissionError:
        state.alternative_figure_path = (
            state.OUTPUT_DIR
            / 'physical_building_clusters_pca_new.png'
        )

        state.fig.savefig(
            state.alternative_figure_path,
            dpi=300,
            bbox_inches='tight'
        )

        print(
            '\nWARNING: The original PCA figure could not be overwritten.'
        )

        print(
            'The figure was saved instead to:',
            state.alternative_figure_path.resolve()
        )

    else:
        print(
            '\nPCA figure saved to:',
            state.figure_path.resolve()
        )

def cell_37(state):
    """Cell 37: cluster-wise cyclic load and PV profiles."""
    # --------------------------------------------------------------------------
    # Plot cluster-wise cyclic daily load and PV profiles
    # --------------------------------------------------------------------------

    def calculate_cyclic_profile(
        member_ids,
        variable,
        time_step_hours=1.0
    ):
        """Calculate cluster-level cyclic daily profiles.

        Each building is first reduced to its mean daily profile. The cluster
        median and interquartile range are then calculated across buildings.

        Parameters
        ----------
        member_ids : list of int
            Building identifiers belonging to one cluster.

        variable : str
            Key in raw_data, normally 'load' or 'pv'.

        time_step_hours : float
            Duration of one simulation time step in hours.

        Returns
        -------
        dict
            Cluster median, 25th percentile, 75th percentile, and individual
            building cyclic profiles.
        """

        if not member_ids:
            raise ValueError(
                'member_ids must contain at least one building.'
            )

        if variable not in ['load', 'pv']:
            raise ValueError(
                "variable must be either 'load' or 'pv'. "
                "Received: {!r}.".format(variable)
            )

        if (
            not np.isfinite(time_step_hours)
            or time_step_hours <= 0.0
        ):
            raise ValueError(
                'time_step_hours must be finite and greater than zero.'
            )

        steps_per_day = int(
            round(
                24.0 / time_step_hours
            )
        )

        if not np.isclose(
            steps_per_day * time_step_hours,
            24.0,
            rtol=1e-8,
            atol=1e-8
        ):
            raise ValueError(
                'TIME_STEP_HOURS={} does not produce an integer number '
                'of time steps per day.'.format(
                    time_step_hours
                )
            )

        building_profiles = []
        valid_building_ids = []



        for building_id in member_ids:
            if building_id not in state.raw_data:
                raise KeyError(
                    'raw_data does not contain Building {}.'.format(
                        building_id
                    )
                )

            if variable not in state.raw_data[building_id]:
                raise KeyError(
                    "raw_data[{}] does not contain variable {!r}.".format(
                        building_id,
                        variable
                    )
                )

            values = np.asarray(
                state.raw_data[building_id][variable],
                dtype=float
            ).reshape(-1)




            if values.size == 0:
                raise ValueError(
                    'Building {} has no data for variable {!r}.'.format(
                        building_id,
                        variable
                    )
                )

            if not np.all(np.isfinite(values)):
                raise ValueError(
                    'Building {} contains nonfinite values for '
                    'variable {!r}.'.format(
                        building_id,
                        variable
                    )
                )

            if values.min() < -state.EPSILON:
                raise ValueError(
                    'Building {} contains negative physical values for '
                    'variable {!r}.'.format(
                        building_id,
                        variable
                    )
                )

            n_complete_days = (
                values.size
                // steps_per_day
            )

            if n_complete_days < 1:
                raise ValueError(
                    'Building {} does not contain a complete day '
                    'for variable {!r}.'.format(
                        building_id,
                        variable
                    )
                )

            # Omit only an incomplete final day.
            n_complete_steps = (
                n_complete_days
                * steps_per_day
            )

            trimmed_values = values[
                :n_complete_steps
            ]

            daily_profiles = trimmed_values.reshape(
                n_complete_days,
                steps_per_day
            )

            # Each building contributes one mean cyclic profile, preventing
            # buildings with more observations from receiving greater weight.
            mean_building_profile = daily_profiles.mean(
                axis=0
            )

            building_profiles.append(
                mean_building_profile
            )

            valid_building_ids.append(
                int(building_id)
            )

        building_profiles = np.vstack(
            building_profiles
        )

        return {
            'building_ids':
                valid_building_ids,

            'individual_profiles':
                building_profiles,

            'median':
                np.median(
                    building_profiles,
                    axis=0
                ),

            'q25':
                np.quantile(
                    building_profiles,
                    0.25,
                    axis=0
                ),

            'q75':
                np.quantile(
                    building_profiles,
                    0.75,
                    axis=0
                ),

            'minimum':
                np.min(
                    building_profiles,
                    axis=0
                ),

            'maximum':
                np.max(
                    building_profiles,
                    axis=0
                ),

            'steps_per_day':
                steps_per_day,

            'number_of_buildings':
                len(valid_building_ids)
        }
    state.calculate_cyclic_profile = calculate_cyclic_profile


    # --------------------------------------------------------------------------
    # Validate clustering information
    # --------------------------------------------------------------------------

    state.cluster_labels = np.asarray(
        state.cluster_labels,
        dtype=int
    )

    if len(state.cluster_labels) != len(state.BUILDING_IDS):
        raise ValueError(
            'Expected {} cluster labels, but found {}.'.format(
                len(state.BUILDING_IDS),
                len(state.cluster_labels)
            )
        )

    state.cluster_ids = sorted(
        np.unique(state.cluster_labels)
    )

    if len(state.cluster_ids) != state.FINAL_K:
        raise ValueError(
            'Expected {} clusters, but found {} unique labels: {}.'.format(
                state.FINAL_K,
                len(state.cluster_ids),
                state.cluster_ids
            )
        )

    if 'OUTLIER_BUILDINGS' not in globals():
        state.OUTLIER_BUILDINGS = [12, 15]

    if 'OUTLIER_CLUSTER' not in globals():
        state.outlier_cluster_values = {
            int(
                state.cluster_labels[
                    state.BUILDING_IDS.index(state.building_id)
                ]
            )
            for state.building_id in state.OUTLIER_BUILDINGS
        }

        if len(state.outlier_cluster_values) != 1:
            raise ValueError(
                'The declared outlier buildings do not belong to '
                'one common cluster.'
            )

        state.OUTLIER_CLUSTER = int(
            next(
                iter(state.outlier_cluster_values)
            )
        )


    # --------------------------------------------------------------------------
    # Define daily time axis
    # --------------------------------------------------------------------------

    state.steps_per_day = int(
        round(
            24.0 / state.TIME_STEP_HOURS
        )
    )

    state.time_of_day = (
        np.arange(state.steps_per_day)
        * state.TIME_STEP_HOURS
    )


    # --------------------------------------------------------------------------
    # Create plotting layout
    # --------------------------------------------------------------------------

    state.fig, state.axes = plt.subplots(
        nrows=state.FINAL_K,
        ncols=2,
        figsize=(
            14,
            3.8 * state.FINAL_K
        ),
        squeeze=False
    )


    state.plotting_variables = [
        {
            'variable': 'load',
            'title': 'Non-Shiftable Electrical Load',
            'color': '#e76f51'
        },
        {
            'variable': 'pv',
            'title': 'PV Generation',
            'color': '#2a9d8f'
        }
    ]


    # --------------------------------------------------------------------------
    # Plot each cluster
    # --------------------------------------------------------------------------

    state.profile_summary_records = []

    for state.row_index, state.cluster_id in enumerate(
        state.cluster_ids
    ):
        state.members = [
            int(state.building_id)
            for state.building_id, state.label in zip(
                state.BUILDING_IDS,
                state.cluster_labels
            )
            if int(state.label) == int(state.cluster_id)
        ]

        if not state.members:
            raise ValueError(
                'Cluster {} contains no buildings.'.format(
                    state.cluster_id
                )
            )

        state.cluster_role = (
            'Outlier stress-test cluster'
            if int(state.cluster_id) == int(state.OUTLIER_CLUSTER)
            else 'Core development cluster'
        )

        for state.column_index, state.plot_settings in enumerate(
            state.plotting_variables
        ):
            state.variable = state.plot_settings[
                'variable'
            ]

            state.profile_summary = state.calculate_cyclic_profile(
                member_ids=state.members,
                variable=state.variable,
                time_step_hours=state.TIME_STEP_HOURS
            )

            state.ax = state.axes[
                state.row_index,
                state.column_index
            ]

            # Plot individual building profiles using faint lines.
            for state.profile_index, state.building_profile in enumerate(
                state.profile_summary[
                    'individual_profiles'
                ]
            ):
                state.building_id = state.profile_summary[
                    'building_ids'
                ][state.profile_index]

                if state.building_id in state.OUTLIER_BUILDINGS:
                    state.individual_color = '#b22222'
                    state.individual_alpha = 0.75
                    state.individual_width = 1.2
                else:
                    state.individual_color = state.plot_settings[
                        'color'
                    ]
                    state.individual_alpha = 0.25
                    state.individual_width = 0.9

                state.ax.plot(
                    state.time_of_day,
                    state.building_profile,
                    color=state.individual_color,
                    linewidth=state.individual_width,
                    alpha=state.individual_alpha,
                    label=(
                        'B{}'.format(state.building_id)
                        if len(state.members) <= 4
                        else None
                    ),
                    zorder=1
                )

            # Plot the interquartile range across building profiles.
            state.ax.fill_between(
                state.time_of_day,
                state.profile_summary['q25'],
                state.profile_summary['q75'],
                color=state.plot_settings['color'],
                alpha=0.20,
                label='Interquartile range',
                zorder=2
            )

            # Plot the cluster median.
            state.ax.plot(
                state.time_of_day,
                state.profile_summary['median'],
                color='black',
                linewidth=2.2,
                label='Cluster median',
                zorder=3
            )

            # Mark the cluster medoid if available.
            state.medoid_rows = state.cluster_medoids_table[
                state.cluster_medoids_table[
                    'cluster'
                ] == state.cluster_id
            ]

            if not state.medoid_rows.empty:
                state.medoid_building = int(
                    state.medoid_rows.iloc[0][
                        'selected_medoid_building'
                    ]
                )

                if state.medoid_building in state.members:
                    state.medoid_position = state.members.index(
                        state.medoid_building
                    )

                    state.medoid_profile = state.profile_summary[
                        'individual_profiles'
                    ][
                        state.medoid_position
                    ]

                    state.ax.plot(
                        state.time_of_day,
                        state.medoid_profile,
                        color='#ffd166',
                        linewidth=2.0,
                        linestyle='--',
                        label='Medoid B{}'.format(
                            state.medoid_building
                        ),
                        zorder=4
                    )

            state.ax.set_title(
                'Cluster {}: {} [{}]\nMembers: {}'.format(
                    state.cluster_id,
                    state.plot_settings['title'],
                    state.cluster_role,
                    ', '.join(
                        'B{}'.format(state.building_id)
                        for state.building_id in state.members
                    )
                ),
                fontsize=11,
                fontweight='bold'
            )

            state.ax.set_xlabel(
                'Hour of Day'
            )

            state.ax.set_ylabel(
                'Energy per Time Step (kWh)'
            )

            state.ax.set_xlim(
                0.0,
                24.0 - state.TIME_STEP_HOURS
            )

            state.ax.xaxis.set_major_locator(
                ticker.MaxNLocator(
                    nbins=8,
                    integer=True
                )
            )

            state.ax.grid(
                linestyle='--',
                alpha=0.30
            )

            state.ax.set_axisbelow(
                True
            )

            # Remove duplicate legend entries.
            state.handles, state.labels = (
                state.ax.get_legend_handles_labels()
            )

            state.unique_entries = {}

            for state.handle, state.label in zip(
                state.handles,
                state.labels
            ):
                if (
                    state.label
                    and state.label not in state.unique_entries
                ):
                    state.unique_entries[state.label] = state.handle

            state.ax.legend(
                state.unique_entries.values(),
                state.unique_entries.keys(),
                loc='best',
                frameon=False,
                fontsize=8
            )

            for state.time_index, state.time_value in enumerate(
                state.time_of_day
            ):
                state.profile_summary_records.append({
                    'cluster':
                        int(state.cluster_id),

                    'cluster_role':
                        state.cluster_role,

                    'variable':
                        state.variable,

                    'hour_of_day':
                        float(state.time_value),

                    'cluster_median':
                        float(
                            state.profile_summary[
                                'median'
                            ][state.time_index]
                        ),

                    'cluster_q25':
                        float(
                            state.profile_summary[
                                'q25'
                            ][state.time_index]
                        ),

                    'cluster_q75':
                        float(
                            state.profile_summary[
                                'q75'
                            ][state.time_index]
                        ),

                    'number_of_buildings':
                        int(
                            state.profile_summary[
                                'number_of_buildings'
                            ]
                        )
                })


    state.fig.suptitle(
        'Cluster-Wise Cyclic Load and PV Profiles',
        fontsize=15,
        fontweight='bold',
        y=1.01
    )

    plt.tight_layout()
    plt.show()


    # --------------------------------------------------------------------------
    # Save profile summaries and figure
    # --------------------------------------------------------------------------

    state.cluster_profile_summary = pd.DataFrame(
        state.profile_summary_records
    )

    state.OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    state.profile_output_path = (
        state.OUTPUT_DIR
        / 'cluster_cyclic_profile_summary.csv'
    )

    try:
        state.cluster_profile_summary.to_csv(
            state.profile_output_path,
            index=False
        )

    except PermissionError:
        state.alternative_profile_path = (
            state.OUTPUT_DIR
            / 'cluster_cyclic_profile_summary_new.csv'
        )

        state.cluster_profile_summary.to_csv(
            state.alternative_profile_path,
            index=False
        )

        print(
            'WARNING: The original profile-summary file could not be '
            'overwritten, probably because it is open in Excel.'
        )

        print(
            'Profile summary saved instead to:',
            state.alternative_profile_path.resolve()
        )


    state.figure_path = (
        state.OUTPUT_DIR
        / 'cluster_cyclic_load_pv_profiles.png'
    )

    try:
        state.fig.savefig(
            state.figure_path,
            dpi=300,
            bbox_inches='tight'
        )

    except PermissionError:
        state.alternative_figure_path = (
            state.OUTPUT_DIR
            / 'cluster_cyclic_load_pv_profiles_new.png'
        )

        state.fig.savefig(
            state.alternative_figure_path,
            dpi=300,
            bbox_inches='tight'
        )

        print(
            'WARNING: The original cyclic-profile figure could not '
            'be overwritten.'
        )

        print(
            'Figure saved instead to:',
            state.alternative_figure_path.resolve()
        )

    else:
        print(
            'Cyclic-profile figure saved to:',
            state.figure_path.resolve()
        )

def cell_39(state):
    """Cell 39: supplementary shape-normalized load analysis."""
    # --------------------------------------------------------------------------
    # Supplementary analysis of shape-normalized load profiles
    # --------------------------------------------------------------------------

    # This analysis compares temporal load shapes after removing differences
    # in mean load and magnitude. It supports interpretation but does not replace
    # the physical-descriptor clustering used for TD3 building selection.

    if 'OUTLIER_BUILDINGS' not in globals():
        state.OUTLIER_BUILDINGS = [12, 15]

    if len(state.BUILDING_IDS) < 2:
        raise ValueError(
            'At least two buildings are required for load-shape analysis.'
        )

    if len(state.cluster_labels) != len(state.BUILDING_IDS):
        raise ValueError(
            'Expected {} physical-cluster labels, but found {}.'.format(
                len(state.BUILDING_IDS),
                len(state.cluster_labels)
            )
        )

    state.shape_normalized_load = {}
    state.load_lengths = {}

    for state.building_id in state.BUILDING_IDS:
        if state.building_id not in state.raw_data:
            raise KeyError(
                'raw_data does not contain Building {}.'.format(state.building_id)
            )

        if 'load' not in state.raw_data[state.building_id]:
            raise KeyError(
                "raw_data[{}] does not contain 'load'.".format(state.building_id)
            )

        state.load = np.asarray(
            state.raw_data[state.building_id]['load'],
            dtype=float
        ).reshape(-1)

        if state.load.size == 0:
            raise ValueError(
                'Building {} contains an empty load series.'.format(state.building_id)
            )

        if not np.all(np.isfinite(state.load)):
            raise ValueError(
                'Building {} contains nonfinite load values.'.format(state.building_id)
            )

        state.load_standard_deviation = float(state.load.std(ddof=0))

        if state.load_standard_deviation <= state.EPSILON:
            raise ValueError(
                'Building {} has a constant or nearly constant load profile.'.format(
                    state.building_id
                )
            )

        state.normalized_load = zscore(
            state.load,
            ddof=0,
            nan_policy='raise'
        )

        state.normalized_load = np.asarray(
            state.normalized_load,
            dtype=float
        ).reshape(-1)

        if not np.all(np.isfinite(state.normalized_load)):
            raise ValueError(
                'Shape normalization generated nonfinite values for Building {}.'.format(
                    state.building_id
                )
            )

        state.shape_normalized_load[state.building_id] = state.normalized_load
        state.load_lengths[state.building_id] = int(state.normalized_load.size)

    if len(set(state.load_lengths.values())) != 1:
        raise ValueError(
            'Shape-normalized load lengths differ across buildings: {}'.format(
                state.load_lengths
            )
        )

    state.X_shape = np.vstack([
        state.shape_normalized_load[state.building_id]
        for state.building_id in state.BUILDING_IDS
    ])

    if state.X_shape.shape[0] != len(state.BUILDING_IDS):
        raise ValueError(
            'The shape matrix contains an unexpected number of buildings.'
        )

    if not np.all(np.isfinite(state.X_shape)):
        raise ValueError(
            'The shape-normalized load matrix contains NaN or infinite values.'
        )

    state.shape_correlation = np.corrcoef(state.X_shape)
    state.expected_matrix_shape = (len(state.BUILDING_IDS), len(state.BUILDING_IDS))

    if state.shape_correlation.shape != state.expected_matrix_shape:
        raise ValueError(
            'The load-shape correlation matrix has an unexpected shape: {}.'.format(
                state.shape_correlation.shape
            )
        )

    if not np.all(np.isfinite(state.shape_correlation)):
        raise ValueError(
            'The load-shape correlation matrix contains NaN or infinite values.'
        )

    if not np.allclose(
        np.diag(state.shape_correlation),
        1.0,
        rtol=1e-8,
        atol=1e-8
    ):
        raise ValueError(
            'The diagonal of the correlation matrix is not equal to one.'
        )

    state.building_labels = [
        'B{}'.format(state.building_id)
        for state.building_id in state.BUILDING_IDS
    ]

    state.fig, state.ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(11, 9)
    )

    sns.heatmap(
        state.shape_correlation,
        xticklabels=state.building_labels,
        yticklabels=state.building_labels,
        cmap='coolwarm',
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
        annot=True,
        fmt='.2f',
        square=True,
        linewidths=0.25,
        cbar_kws={'label': 'Pearson Correlation'},
        ax=state.ax
    )

    state.ax.set_title(
        'Shape-Normalized Development-Period Load Similarity',
        fontsize=14,
        fontweight='bold'
    )
    state.ax.set_xlabel('Building')
    state.ax.set_ylabel('Building')
    plt.tight_layout()
    plt.show()

    state.shape_distance = squareform(
        pdist(state.X_shape, metric='euclidean')
    )

    if state.shape_distance.shape != state.expected_matrix_shape:
        raise ValueError(
            'The load-shape distance matrix has an unexpected shape: {}.'.format(
                state.shape_distance.shape
            )
        )

    if not np.all(np.isfinite(state.shape_distance)):
        raise ValueError(
            'The load-shape distance matrix contains NaN or infinite values.'
        )

    if not np.allclose(
        state.shape_distance,
        state.shape_distance.T,
        rtol=1e-10,
        atol=1e-10
    ):
        raise ValueError(
            'The load-shape distance matrix is not symmetric.'
        )

    state.neighbor_records = []

    for state.building_position, state.building_id in enumerate(state.BUILDING_IDS):
        state.sorted_neighbor_positions = np.argsort(
            state.shape_distance[state.building_position]
        )
        state.neighbor_rank = 0

        for state.neighbor_position in state.sorted_neighbor_positions:
            if state.neighbor_position == state.building_position:
                continue

            state.neighbor_rank += 1
            state.neighbor_building = int(state.BUILDING_IDS[state.neighbor_position])

            state.neighbor_records.append({
                'building_id': int(state.building_id),
                'neighbor_building': state.neighbor_building,
                'shape_distance': float(
                    state.shape_distance[state.building_position, state.neighbor_position]
                ),
                'shape_correlation': float(
                    state.shape_correlation[state.building_position, state.neighbor_position]
                ),
                'neighbor_rank': int(state.neighbor_rank),
                'same_physical_cluster': bool(
                    state.cluster_labels[state.building_position]
                    == state.cluster_labels[state.neighbor_position]
                ),
                'source_is_outlier': bool(
                    state.building_id in state.OUTLIER_BUILDINGS
                ),
                'neighbor_is_outlier': bool(
                    state.neighbor_building in state.OUTLIER_BUILDINGS
                )
            })

    state.nearest_neighbors = pd.DataFrame(state.neighbor_records)

    if state.nearest_neighbors.empty:
        raise RuntimeError('The nearest-neighbor table is empty.')

    state.top_neighbor_count = 3

    state.top_nearest_neighbors = state.nearest_neighbors[
        state.nearest_neighbors['neighbor_rank'] <= state.top_neighbor_count
    ].sort_values([
        'building_id',
        'neighbor_rank'
    ]).reset_index(drop=True)

    print('Three nearest load-shape neighbors for each building:')
    display(
        state.top_nearest_neighbors.round({
            'shape_distance': 4,
            'shape_correlation': 4
        })
    )

    state.nearest_neighbor_agreement = state.top_nearest_neighbors[
        'same_physical_cluster'
    ].mean()

    print(
        'Fraction of the top {} load-shape neighbors that belong '
        'to the same physical cluster: {:.2%}'.format(
            state.top_neighbor_count,
            state.nearest_neighbor_agreement
        )
    )

    state.outlier_neighbor_summary = state.top_nearest_neighbors[
        state.top_nearest_neighbors['source_is_outlier']
    ].copy()

    if not state.outlier_neighbor_summary.empty:
        print('\nNearest load-shape neighbors of the outlier buildings:')
        display(
            state.outlier_neighbor_summary.round({
                'shape_distance': 4,
                'shape_correlation': 4
            })
        )

    state.shape_correlation_table = pd.DataFrame(
        state.shape_correlation,
        index=state.BUILDING_IDS,
        columns=state.BUILDING_IDS
    )
    state.shape_correlation_table.index.name = 'building_id'

    state.shape_distance_table = pd.DataFrame(
        state.shape_distance,
        index=state.BUILDING_IDS,
        columns=state.BUILDING_IDS
    )
    state.shape_distance_table.index.name = 'building_id'

    state.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    state.shape_output_tables = {
        'load_shape_correlation_matrix.csv': state.shape_correlation_table.reset_index(),
        'load_shape_distance_matrix.csv': state.shape_distance_table.reset_index(),
        'load_shape_nearest_neighbors.csv': state.nearest_neighbors,
        'load_shape_top_three_neighbors.csv': state.top_nearest_neighbors
    }

    for state.filename, state.output_table in state.shape_output_tables.items():
        state.output_path = state.OUTPUT_DIR / state.filename

        try:
            state.output_table.to_csv(state.output_path, index=False)
        except PermissionError:
            state.alternative_output_path = (
                state.OUTPUT_DIR / state.filename.replace('.csv', '_new.csv')
            )
            state.output_table.to_csv(state.alternative_output_path, index=False)
            print(
                'WARNING: {} could not be overwritten. Saved as {}.'.format(
                    state.filename,
                    state.alternative_output_path.name
                )
            )

    state.figure_output_path = state.OUTPUT_DIR / 'load_shape_correlation_heatmap.png'

    try:
        state.fig.savefig(
            state.figure_output_path,
            dpi=300,
            bbox_inches='tight'
        )
    except PermissionError:
        state.alternative_figure_path = (
            state.OUTPUT_DIR / 'load_shape_correlation_heatmap_new.png'
        )
        state.fig.savefig(
            state.alternative_figure_path,
            dpi=300,
            bbox_inches='tight'
        )
        print(
            'WARNING: Heatmap saved instead to:',
            state.alternative_figure_path.resolve()
        )
    else:
        print('\nLoad-shape heatmap saved to:', state.figure_output_path.resolve())

def cell_41(state):
    """Cell 41: final development panel and exports."""
    # --------------------------------------------------------------------------
    # Finalize and export clustering results
    # --------------------------------------------------------------------------

    from datetime import datetime


    # --------------------------------------------------------------------------
    # Validate the essential analysis objects
    # --------------------------------------------------------------------------

    state.required_objects = [
        'descriptors',
        'cluster_assignments',
        'cluster_quality',
        'cluster_medoids_table',
        'cluster_labels',
        'SELECTED_FEATURES',
        'FINAL_K',
        'overall_medoid_building',
        'overall_core_medoid_building',
        'initialization_stability',
        'algorithm_agreement_ari',
    ]

    state.missing_objects = [
        object_name
        for object_name in state.required_objects
        if object_name not in globals()
    ]

    if state.missing_objects:
        raise NameError(
            'The following required analysis objects are unavailable: '
            '{}'.format(state.missing_objects)
        )


    # --------------------------------------------------------------------------
    # Define and validate outlier information
    # --------------------------------------------------------------------------

    if 'OUTLIER_BUILDINGS' not in globals():
        state.OUTLIER_BUILDINGS = [12, 15]

    if 'OUTLIER_CLUSTER' not in globals():
        state.outlier_cluster_labels = set()

        for state.building_id in state.OUTLIER_BUILDINGS:
            state.building_position = state.BUILDING_IDS.index(state.building_id)

            state.outlier_cluster_labels.add(
                int(state.cluster_labels[state.building_position])
            )

        if len(state.outlier_cluster_labels) != 1:
            raise ValueError(
                'Buildings 12 and 15 do not belong to one common cluster. '
                'Observed cluster labels: {}'.format(
                    sorted(state.outlier_cluster_labels)
                )
            )

        state.OUTLIER_CLUSTER = int(
            list(state.outlier_cluster_labels)[0]
        )


    # --------------------------------------------------------------------------
    # Treat day-bootstrap analysis as optional during notebook development
    # --------------------------------------------------------------------------

    if 'bootstrap_stability' in globals():
        state.bootstrap_completed = True

        if len(bootstrap_stability) == 0:
            raise ValueError(
                'bootstrap_stability exists but contains no values.'
            )

        state.bootstrap_values = np.asarray(
            bootstrap_stability,
            dtype=float
        )

        if not np.all(np.isfinite(state.bootstrap_values)):
            raise ValueError(
                'bootstrap_stability contains NaN or infinite values.'
            )

        state.bootstrap_count = int(
            len(state.bootstrap_values)
        )

        state.bootstrap_ari_mean = float(
            np.mean(state.bootstrap_values)
        )

        state.bootstrap_ari_median = float(
            np.median(state.bootstrap_values)
        )

        state.bootstrap_ari_std = float(
            np.std(
                state.bootstrap_values,
                ddof=1
            )
        ) if state.bootstrap_count > 1 else 0.0

        state.bootstrap_ari_minimum = float(
            np.min(state.bootstrap_values)
        )

        state.bootstrap_ari_maximum = float(
            np.max(state.bootstrap_values)
        )

    else:
        state.bootstrap_completed = False
        state.bootstrap_count = 0
        state.bootstrap_ari_mean = None
        state.bootstrap_ari_median = None
        state.bootstrap_ari_std = None
        state.bootstrap_ari_minimum = None
        state.bootstrap_ari_maximum = None

        print(
            'WARNING: Day-bootstrap stability analysis has not been run.'
        )

        print(
            'Bootstrap metadata will be saved as null. Run the '
            'day-bootstrap cell before freezing the final development panel.'
        )


    # --------------------------------------------------------------------------
    # Validate the medoid table
    # --------------------------------------------------------------------------

    state.required_medoid_columns = [
        'cluster',
        'cluster_role',
        'selected_medoid_building',
    ]

    state.missing_medoid_columns = [
        state.column
        for state.column in state.required_medoid_columns
        if state.column not in state.cluster_medoids_table.columns
    ]

    if state.missing_medoid_columns:
        raise KeyError(
            'cluster_medoids_table is missing columns: {}'.format(
                state.missing_medoid_columns
            )
        )


    # --------------------------------------------------------------------------
    # Create the primary multi-building TD3 development panel
    # --------------------------------------------------------------------------

    state.development_panel = state.cluster_medoids_table.loc[
        state.cluster_medoids_table['cluster_role']
        == 'core_development'
    ].copy()

    state.development_panel = state.development_panel.reset_index(
        drop=True
    )

    if state.development_panel.empty:
        raise ValueError(
            'No core-cluster medoids were identified.'
        )

    state.development_panel = state.development_panel.merge(
        state.descriptors.reset_index(),
        left_on='selected_medoid_building',
        right_on='building_id',
        how='left',
        validate='one_to_one'
    )

    if state.development_panel['building_id'].isna().any():
        raise ValueError(
            'One or more core medoids could not be matched '
            'to the physical descriptor table.'
        )

    state.development_panel['is_overall_core_medoid'] = (
        state.development_panel['selected_medoid_building'].astype(int)
        == int(state.overall_core_medoid_building)
    )

    state.development_panel['recommended_use'] = (
        'primary_td3_hyperparameter_development'
    )


    print('Primary TD3 development panel:')

    display(state.development_panel)


    print(
        'Single-building fallback: Building {}'.format(
            state.overall_core_medoid_building
        )
    )

    print(
        'Outlier stress-test building: Building 12'
    )

    print(
        'Additional outlier retained in transfer matrix: Building 15'
    )


    # --------------------------------------------------------------------------
    # Create the complete building assignment table
    # --------------------------------------------------------------------------

    state.final_assignments = state.descriptors.join(
        state.cluster_assignments,
        how='inner'
    ).reset_index()

    if state.final_assignments.shape[0] != len(state.BUILDING_IDS):
        raise ValueError(
            'Expected {} building-assignment rows but found {}.'.format(
                len(state.BUILDING_IDS),
                state.final_assignments.shape[0]
            )
        )

    if state.final_assignments['building_id'].duplicated().any():
        raise ValueError(
            'The final assignment table contains duplicated building IDs.'
        )


    state.medoid_map = {}

    state.cluster_role_map = {}

    for state._, state.row in state.cluster_medoids_table.iterrows():
        state.cluster_id = int(state.row['cluster'])

        state.medoid_map[state.cluster_id] = int(
            state.row['selected_medoid_building']
        )

        state.cluster_role_map[state.cluster_id] = str(
            state.row['cluster_role']
        )


    state.final_assignments['cluster_role'] = (
        state.final_assignments['cluster']
        .map(state.cluster_role_map)
    )

    if state.final_assignments['cluster_role'].isna().any():
        raise ValueError(
            'At least one cluster could not be assigned a cluster role.'
        )


    state.final_assignments['is_cluster_medoid'] = [
        int(state.building_id) == state.medoid_map[int(state.cluster_id)]
        for state.building_id, state.cluster_id in zip(
            state.final_assignments['building_id'],
            state.final_assignments['cluster']
        )
    ]


    state.final_assignments['is_declared_outlier'] = (
        state.final_assignments['building_id']
        .isin(state.OUTLIER_BUILDINGS)
    )


    state.final_assignments[
        'is_overall_full_portfolio_medoid'
    ] = (
        state.final_assignments['building_id']
        == int(state.overall_medoid_building)
    )


    state.final_assignments[
        'is_overall_core_portfolio_medoid'
    ] = (
        state.final_assignments['building_id']
        == int(state.overall_core_medoid_building)
    )


    state.core_medoid_buildings = (
        state.development_panel['selected_medoid_building']
        .astype(int)
        .tolist()
    )


    state.final_assignments[
        'is_primary_td3_development_building'
    ] = (
        state.final_assignments['building_id']
        .isin(state.core_medoid_buildings)
    )


    state.final_assignments[
        'is_outlier_stress_test_candidate'
    ] = (
        state.final_assignments['building_id']
        .isin(state.OUTLIER_BUILDINGS)
    )


    print('\nFinal building assignments:')

    display(state.final_assignments)


    # --------------------------------------------------------------------------
    # Create a compact methodological selection table
    # --------------------------------------------------------------------------

    state.selection_records = []


    for state._, state.row in state.development_panel.iterrows():
        state.selection_records.append({
            'building_id':
                int(state.row['selected_medoid_building']),

            'cluster':
                int(state.row['cluster']),

            'selection_role':
                'core_cluster_medoid',

            'recommended_use':
                'primary_td3_hyperparameter_development'
        })


    state.selection_records.append({
        'building_id':
            int(state.overall_core_medoid_building),

        'cluster':
            int(
                state.final_assignments.loc[
                    state.final_assignments['building_id']
                    == state.overall_core_medoid_building,
                    'cluster'
                ].iloc[0]
            ),

        'selection_role':
            'overall_core_portfolio_medoid',

        'recommended_use':
            'single_building_fallback'
    })


    state.selection_records.append({
        'building_id': 12,

        'cluster':
            int(
                state.final_assignments.loc[
                    state.final_assignments['building_id'] == 12,
                    'cluster'
                ].iloc[0]
            ),

        'selection_role':
            'outlier_building',

        'recommended_use':
            'outlier_stress_test'
    })


    state.selection_records.append({
        'building_id': 15,

        'cluster':
            int(
                state.final_assignments.loc[
                    state.final_assignments['building_id'] == 15,
                    'cluster'
                ].iloc[0]
            ),

        'selection_role':
            'outlier_building',

        'recommended_use':
            'final_transfer_matrix'
    })


    state.methodological_selection = pd.DataFrame(
        state.selection_records
    )

    state.methodological_selection = (
        state.methodological_selection
        .drop_duplicates(
            subset=[
                'building_id',
                'selection_role'
            ]
        )
        .sort_values([
            'selection_role',
            'building_id'
        ])
        .reset_index(drop=True)
    )


    print('\nMethodological building-selection summary:')

    display(state.methodological_selection)


    # --------------------------------------------------------------------------
    # Safe export functions
    # --------------------------------------------------------------------------

    state.OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    state.timestamp = datetime.now().strftime(
        '%Y%m%d_%H%M%S'
    )


    def save_csv_safely(
        dataframe,
        filename,
        include_index=False
    ):
        """Save a CSV and use a timestamp if the original file is locked."""

        output_path = state.OUTPUT_DIR / filename

        try:
            dataframe.to_csv(
                output_path,
                index=include_index
            )

            print('Saved:', output_path.resolve())

            return output_path

        except PermissionError:
            alternative_filename = filename.replace(
                '.csv',
                '_{}.csv'.format(state.timestamp)
            )

            alternative_path = (
                state.OUTPUT_DIR
                / alternative_filename
            )

            dataframe.to_csv(
                alternative_path,
                index=include_index
            )

            print(
                'WARNING: {} could not be overwritten, probably '
                'because it is open in Excel.'.format(filename)
            )

            print('Saved instead:', alternative_path.resolve())

            return alternative_path
    state.save_csv_safely = save_csv_safely


    def save_json_safely(
        data,
        filename
    ):
        """Save JSON and use a timestamp if the original file is locked."""

        output_path = state.OUTPUT_DIR / filename

        try:
            with open(
                output_path,
                'w',
                encoding='utf-8'
            ) as stream:
                json.dump(
                    data,
                    stream,
                    indent=2
                )

            print('Saved:', output_path.resolve())

            return output_path

        except PermissionError:
            alternative_filename = filename.replace(
                '.json',
                '_{}.json'.format(state.timestamp)
            )

            alternative_path = (
                state.OUTPUT_DIR
                / alternative_filename
            )

            with open(
                alternative_path,
                'w',
                encoding='utf-8'
            ) as stream:
                json.dump(
                    data,
                    stream,
                    indent=2
                )

            print(
                'WARNING: {} could not be overwritten.'.format(
                    filename
                )
            )

            print('Saved instead:', alternative_path.resolve())

            return alternative_path
    state.save_json_safely = save_json_safely


    # --------------------------------------------------------------------------
    # Export primary result tables
    # --------------------------------------------------------------------------

    state.save_csv_safely(
        state.final_assignments,
        'building_cluster_assignments.csv'
    )

    state.save_csv_safely(
        state.cluster_quality,
        'cluster_number_quality.csv'
    )

    state.save_csv_safely(
        state.cluster_medoids_table,
        'cluster_medoids_and_roles.csv'
    )

    state.save_csv_safely(
        state.development_panel,
        'core_td3_development_panel.csv'
    )

    state.save_csv_safely(
        state.methodological_selection,
        'methodological_building_selection.csv'
    )


    # Export optional tables only when available.
    if 'nearest_neighbors' in globals():
        state.save_csv_safely(
            state.nearest_neighbors,
            'load_shape_nearest_neighbors.csv'
        )

    if 'top_nearest_neighbors' in globals():
        state.save_csv_safely(
            state.top_nearest_neighbors,
            'load_shape_top_three_neighbors.csv'
        )

    if 'overall_medoid_ranking' in globals():
        state.save_csv_safely(
            state.overall_medoid_ranking,
            'full_portfolio_medoid_ranking.csv'
        )

    if 'core_medoid_ranking' in globals():
        state.save_csv_safely(
            state.core_medoid_ranking,
            'core_portfolio_medoid_ranking.csv'
        )

    if 'outlier_stress_test_candidates' in globals():
        state.save_csv_safely(
            state.outlier_stress_test_candidates,
            'outlier_stress_test_candidates.csv'
        )


    # --------------------------------------------------------------------------
    # Construct complete reproducibility metadata
    # --------------------------------------------------------------------------

    state.metadata = {
        'dataset': {
            'data_path':
                str(state.DATA_PATH),

            'schema_path':
                str(state.SCHEMA_PATH),

            'building_ids': [
                int(state.building_id)
                for state.building_id in state.BUILDING_IDS
            ],

            'development_start':
                int(state.DEVELOPMENT_START),

            'development_end':
                int(state.DEVELOPMENT_END),

            'development_time_steps':
                int(
                    state.DEVELOPMENT_END
                    - state.DEVELOPMENT_START
                    + 1
                ),

            'time_step_hours':
                float(state.TIME_STEP_HOURS),

            'load_column':
                str(state.LOAD_COLUMN),

            'pv_column':
                str(state.PV_COLUMN),

            'pv_conversion':
                (
                    'W/kW profile multiplied by PV nominal power in kW, '
                    'multiplied by time-step duration in hours, '
                    'and divided by 1000.'
                )
        },

        'clustering': {
            'method':
                'KMeans',

            'selected_features': [
                str(state.feature)
                for state.feature in state.SELECTED_FEATURES
            ],

            'robust_scaling':
                True,

            'robust_scaling_quantile_range': [
                25.0,
                75.0
            ],

            'evaluated_k_values': [
                int(value)
                for value in state.K_VALUES
            ],

            'final_k':
                int(state.FINAL_K),

            'random_state':
                int(state.RANDOM_STATE),

            'kmeans_n_init':
                int(state.KMEANS_N_INIT),

            'kmeans_max_iter':
                int(state.KMEANS_MAX_ITER),

            'redundancy_threshold':
                float(state.REDUNDANCY_THRESHOLD)
        },

        'building_selection': {
            'outlier_buildings': [
                int(state.building_id)
                for state.building_id in state.OUTLIER_BUILDINGS
            ],

            'outlier_cluster':
                int(state.OUTLIER_CLUSTER),

            'cluster_medoids': {
                str(state.cluster_id): int(state.building_id)
                for state.cluster_id, state.building_id
                in state.medoid_map.items()
            },

            'core_cluster_medoids':
                state.core_medoid_buildings,

            'overall_full_portfolio_medoid':
                int(state.overall_medoid_building),

            'overall_core_portfolio_medoid':
                int(state.overall_core_medoid_building),

            'recommended_outlier_stress_test':
                12,

            'methodological_note':
                (
                    'Core-cluster medoids are used for primary TD3 '
                    'hyperparameter development. Building 12 is used '
                    'as an outlier stress test. Buildings 12 and 15 '
                    'remain included in final source-policy training '
                    'and directed zero-shot transfer evaluation.'
                )
        },

        'stability': {
            'initialization_stability':
                state.initialization_stability,

            'day_bootstrap_completed':
                bool(state.bootstrap_completed),

            'number_of_day_bootstraps':
                int(state.bootstrap_count),

            'day_bootstrap_ari_mean':
                state.bootstrap_ari_mean,

            'day_bootstrap_ari_median':
                state.bootstrap_ari_median,

            'day_bootstrap_ari_std':
                state.bootstrap_ari_std,

            'day_bootstrap_ari_minimum':
                state.bootstrap_ari_minimum,

            'day_bootstrap_ari_maximum':
                state.bootstrap_ari_maximum,

            'kmeans_hierarchical_agreement_ari':
                float(state.algorithm_agreement_ari)
        },

        'td3_development': {
            'multi_building_panel':
                state.core_medoid_buildings,

            'single_building_fallback':
                int(state.overall_core_medoid_building),

            'outlier_stress_test':
                12,

            'all_buildings_retained_for_final_transfer':
                True
        }
    }


    # --------------------------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------------------------

    state.save_json_safely(
        state.metadata,
        'clustering_metadata.json'
    )


    # --------------------------------------------------------------------------
    # Final summary
    # --------------------------------------------------------------------------

    print('\nSaved clustering outputs:')

    for state.item in sorted(
        state.OUTPUT_DIR.iterdir()
    ):
        print('  -', state.item.name)


    print('\nFinal methodological recommendation:')

    print(
        '  Primary TD3 development panel:',
        state.core_medoid_buildings
    )

    print(
        '  Single-building fallback: Building {}'.format(
            state.overall_core_medoid_building
        )
    )

    print(
        '  Outlier stress test: Building 12'
    )

    print(
        '  Final zero-shot transfer analysis: all 17 buildings'
    )

    if not state.bootstrap_completed:
        print(
            '\nIMPORTANT: Run the day-bootstrap stability cell before '
            'freezing the final TD3 development panel.'
        )



# ---------------------------------------------------------------------------
# Ordered list of cell functions for run_all().
# ---------------------------------------------------------------------------
_CELL_FUNCTIONS = [

    cell_02,
    cell_04,
    cell_06,
    cell_08,
    cell_10,
    cell_12,
    cell_15,
    cell_17,
    cell_19,
    cell_21,
    cell_23,
    cell_25,
    cell_27,
    cell_29,
    cell_31,
    cell_33,
    cell_35,
    cell_37,
    cell_39,
    cell_41,

]

def run_all(state=None):
    """Run every cell function in notebook order on one shared state."""
    if state is None:
        state = create_state()
    for fn in _CELL_FUNCTIONS:
        fn(state)
    return state

__all__ = ['create_state', 'run_all']
