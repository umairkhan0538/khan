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


def cell_00(state):
    """Cell 00: module imports, warnings, and matplotlib inline setup."""

    from pathlib import Path
    import warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    import seaborn as sns
    from scipy import stats
    warnings.resetwarnings()
    warnings.filterwarnings('default')

def cell_01(state):
    """Cell 01: path configuration and loading of the evaluated timeseries."""

    state.SRC_DIR = Path.cwd().resolve()
    state.EXPERIMENT_DIR = state.SRC_DIR.parent
    state.RESULTS_DIR = state.EXPERIMENT_DIR / 'results'
    state.BUILDING_RESULTS_DIR = state.RESULTS_DIR / '2_Building_2'
    state.AGENTS_DIR = state.RESULTS_DIR / 'agents'
    state.FIGURES_DIR = state.BUILDING_RESULTS_DIR / 'analysis_figures'
    state.TABLES_DIR = state.BUILDING_RESULTS_DIR / 'analysis_tables'
    state.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    state.TABLES_DIR.mkdir(parents=True, exist_ok=True)


    # =============================================================================
    # Load the evaluated timeseries
    # =============================================================================

    state.TIMESERIES_PATH = state.BUILDING_RESULTS_DIR / 'timeseries.csv'

    if not state.TIMESERIES_PATH.is_file():
        raise FileNotFoundError(f'Timeseries file was not found:\n{state.TIMESERIES_PATH}')

    state.df = pd.read_csv(state.TIMESERIES_PATH)

    if state.df.empty:
        raise ValueError(f'Timeseries file contains no records:\n{state.TIMESERIES_PATH}')

def cell_02(state):
    """Cell 02: requested episode-level KPI calculation and display."""
    # =============================================================================
    # Requested episode-level KPI calculation and display
    # =============================================================================

    # This cell calculates the requested KPIs directly from the loaded
    # timeseries DataFrame named "df".
    #
    # No output file is created.
    #
    # Definitions:
    #
    # Controlled grid import:
    #     sum(grid_import)
    #
    # Controlled grid export:
    #     sum(grid_export)
    #
    # PV-only baseline grid import:
    #     sum(max(net_baseline, 0))
    #
    # PV-only baseline grid export:
    #     sum(max(-net_baseline, 0))
    #
    # Avoidable export:
    #     max(grid_export - projected_unavoidable_export, 0)
    #
    # PV self-consumption rate:
    #     100 * total_pv_self_consumed / total_pv_generation


    # =============================================================================
    # Configuration
    # =============================================================================

    state.KPI_ATOL = 1e-8
    state.KPI_RTOL = 1e-6
    state.DENOMINATOR_TOLERANCE = 1e-10


    # =============================================================================
    # Validate required columns
    # =============================================================================

    state.required_kpi_columns = [
        'reward',
        'load',
        'pv_generation',
        'net_baseline',
        'grid_import',
        'grid_export',
        'unavoidable_export',
        'avoidable_export',
        'raw_cost',
        'raw_emission',
        'baseline_grid_import',
        'positive_baseline_cost',
        'positive_baseline_emission',
        'pv_self_consumed',
    ]

    state.missing_kpi_columns = [
        state.column
        for state.column in state.required_kpi_columns
        if state.column not in state.df.columns
    ]

    if state.missing_kpi_columns:
        raise KeyError(
            'The requested KPIs cannot be calculated because the '
            'following columns are missing:\n'
            f'{state.missing_kpi_columns}'
        )

    if state.df.empty:
        raise ValueError(
            'The timeseries DataFrame contains no transitions.'
        )


    # =============================================================================
    # Validate finite numerical values
    # =============================================================================

    state.non_finite_columns = []

    for state.column in state.required_kpi_columns:

        state.values = state.df[state.column].to_numpy(
            dtype=float
        )

        if not np.isfinite(
            state.values
        ).all():
            state.non_finite_columns.append(
                state.column
            )

    if state.non_finite_columns:
        raise ValueError(
            'NaN or infinite values were detected in the following '
            'KPI columns:\n'
            f'{state.non_finite_columns}'
        )


    # =============================================================================
    # Validate baseline energy identity
    # =============================================================================

    state.expected_net_baseline = (
        state.df['load']
        - state.df['pv_generation']
    )

    state.baseline_identity_passed = np.allclose(
        state.df['net_baseline'],
        state.expected_net_baseline,
        atol=state.KPI_ATOL,
        rtol=state.KPI_RTOL,
    )

    if not state.baseline_identity_passed:

        state.maximum_error = float(
            np.max(
                np.abs(
                    state.df['net_baseline']
                    - state.expected_net_baseline
                )
            )
        )

        raise ValueError(
            'The baseline net-demand identity failed:\n'
            'net_baseline != load - pv_generation\n'
            f'Maximum absolute error: {state.maximum_error:.10f} kWh'
        )


    # =============================================================================
    # Reconstruct PV-only baseline grid import and export
    # =============================================================================

    state.baseline_grid_import_series = np.maximum(
        state.df['net_baseline'].to_numpy(
            dtype=float
        ),
        0.0,
    )

    state.baseline_grid_export_series = np.maximum(
        -state.df['net_baseline'].to_numpy(
            dtype=float
        ),
        0.0,
    )


    # Validate the saved baseline grid-import series.
    state.baseline_import_identity_passed = np.allclose(
        state.df['baseline_grid_import'].to_numpy(
            dtype=float
        ),
        state.baseline_grid_import_series,
        atol=state.KPI_ATOL,
        rtol=state.KPI_RTOL,
    )

    if not state.baseline_import_identity_passed:

        state.maximum_error = float(
            np.max(
                np.abs(
                    state.df['baseline_grid_import'].to_numpy(
                        dtype=float
                    )
                    - state.baseline_grid_import_series
                )
            )
        )

        raise ValueError(
            'The baseline grid-import identity failed.\n'
            f'Maximum absolute error: {state.maximum_error:.10f} kWh'
        )


    # =============================================================================
    # Validate avoidable-export definition
    # =============================================================================

    state.expected_avoidable_export = np.maximum(
        (
            state.df['grid_export'].to_numpy(
                dtype=float
            )
            - state.df['unavoidable_export'].to_numpy(
                dtype=float
            )
        ),
        0.0,
    )

    state.avoidable_export_identity_passed = np.allclose(
        state.df['avoidable_export'].to_numpy(
            dtype=float
        ),
        state.expected_avoidable_export,
        atol=state.KPI_ATOL,
        rtol=state.KPI_RTOL,
    )

    if not state.avoidable_export_identity_passed:

        state.maximum_error = float(
            np.max(
                np.abs(
                    state.df['avoidable_export'].to_numpy(
                        dtype=float
                    )
                    - state.expected_avoidable_export
                )
            )
        )

        raise ValueError(
            'The avoidable-export identity failed:\n'
            'avoidable_export != max(grid_export - unavoidable_export, 0)\n'
            f'Maximum absolute error: {state.maximum_error:.10f} kWh'
        )


    # =============================================================================
    # Safe ratio function
    # =============================================================================

    def safe_kpi_ratio(
        numerator,
        denominator,
        ratio_name,
        tolerance=state.DENOMINATOR_TOLERANCE,
    ):
        """
        Calculate a KPI ratio after validating the denominator.
        """

        numerator = float(
            numerator
        )

        denominator = float(
            denominator
        )

        if not np.isfinite(numerator):
            raise ValueError(
                f'The numerator for "{ratio_name}" is not finite.'
            )

        if not np.isfinite(denominator):
            raise ValueError(
                f'The denominator for "{ratio_name}" is not finite.'
            )

        if abs(denominator) <= tolerance:
            raise ZeroDivisionError(
                f'The denominator for "{ratio_name}" is zero or '
                f'near zero: {denominator:.12g}'
            )

        return numerator / denominator
    state.safe_kpi_ratio = safe_kpi_ratio


    # =============================================================================
    # 1. Accumulated reward
    # =============================================================================

    state.accumulated_reward = float(
        state.df['reward'].sum()
    )


    # =============================================================================
    # 2. Grid-import KPIs
    # =============================================================================

    state.total_grid_import_kwh = float(
        state.df['grid_import'].sum()
    )

    state.baseline_grid_import_kwh = float(
        state.baseline_grid_import_series.sum()
    )

    state.grid_import_ratio = state.safe_kpi_ratio(
        numerator=state.total_grid_import_kwh,
        denominator=state.baseline_grid_import_kwh,
        ratio_name='Grid-import ratio',
    )


    # =============================================================================
    # 3. Grid-export KPIs
    # =============================================================================

    state.total_grid_export_kwh = float(
        state.df['grid_export'].sum()
    )

    state.baseline_grid_export_kwh = float(
        state.baseline_grid_export_series.sum()
    )


    # =============================================================================
    # 4. Export composition and PV utilization
    # =============================================================================

    # This reproduces the evaluator's total of the saved projected
    # unavoidable-export quantity.
    state.total_unavoidable_export_kwh = float(
        state.df['unavoidable_export'].sum()
    )

    state.total_avoidable_export_kwh = float(
        state.df['avoidable_export'].sum()
    )

    state.total_pv_generation_kwh = float(
        state.df['pv_generation'].sum()
    )

    state.total_pv_self_consumed_kwh = float(
        state.df['pv_self_consumed'].sum()
    )

    state.pv_self_consumption_rate_percent = (
        100.0
        * state.safe_kpi_ratio(
            numerator=state.total_pv_self_consumed_kwh,
            denominator=state.total_pv_generation_kwh,
            ratio_name='PV self-consumption rate',
        )
    )


    # =============================================================================
    # 5. Electricity-cost KPIs
    # =============================================================================

    state.total_electricity_cost = float(
        state.df['raw_cost'].sum()
    )

    state.baseline_electricity_cost = float(
        state.df['positive_baseline_cost'].sum()
    )

    state.electricity_cost_ratio = state.safe_kpi_ratio(
        numerator=state.total_electricity_cost,
        denominator=state.baseline_electricity_cost,
        ratio_name='Electricity-cost ratio',
    )


    # =============================================================================
    # 6. Carbon-emission KPIs
    # =============================================================================

    state.total_carbon_emission_kg = float(
        state.df['raw_emission'].sum()
    )

    state.baseline_carbon_emission_kg = float(
        state.df['positive_baseline_emission'].sum()
    )

    state.carbon_emission_ratio = state.safe_kpi_ratio(
        numerator=state.total_carbon_emission_kg,
        denominator=state.baseline_carbon_emission_kg,
        ratio_name='Carbon-emission ratio',
    )


    # =============================================================================
    # Optional percentage improvements for direct interpretation
    # =============================================================================

    state.grid_import_reduction_percent = (
        100.0
        * (
            1.0
            - state.grid_import_ratio
        )
    )

    state.electricity_cost_reduction_percent = (
        100.0
        * (
            1.0
            - state.electricity_cost_ratio
        )
    )

    state.carbon_emission_reduction_percent = (
        100.0
        * (
            1.0
            - state.carbon_emission_ratio
        )
    )


    # =============================================================================
    # Create KPI DataFrame in the requested order
    # =============================================================================

    state.requested_kpi_results = pd.DataFrame({
        'Section': [
            'Reward',

            'Grid import',
            'Grid import',
            'Grid import',

            'Grid export',
            'Grid export',

            'Export and PV',
            'Export and PV',
            'Export and PV',

            'Electricity cost',
            'Electricity cost',
            'Electricity cost',

            'Carbon emissions',
            'Carbon emissions',
            'Carbon emissions',
        ],

        'KPI': [
            'Accumulated reward',

            'Total grid import',
            'Baseline grid import',
            'Grid-import ratio',

            'Total grid export',
            'Baseline grid export',

            'Projected unavoidable export',
            'Avoidable export',
            'PV self-consumption rate',

            'Total electricity cost',
            'Baseline electricity cost',
            'Electricity-cost ratio',

            'Total carbon emissions',
            'Baseline carbon emissions',
            'Carbon-emission ratio',
        ],

        'Value': [
            state.accumulated_reward,

            state.total_grid_import_kwh,
            state.baseline_grid_import_kwh,
            state.grid_import_ratio,

            state.total_grid_export_kwh,
            state.baseline_grid_export_kwh,

            state.total_unavoidable_export_kwh,
            state.total_avoidable_export_kwh,
            state.pv_self_consumption_rate_percent,

            state.total_electricity_cost,
            state.baseline_electricity_cost,
            state.electricity_cost_ratio,

            state.total_carbon_emission_kg,
            state.baseline_carbon_emission_kg,
            state.carbon_emission_ratio,
        ],

        'Unit': [
            'dimensionless',

            'kWh',
            'kWh',
            'TD3 / baseline',

            'kWh',
            'kWh',

            'kWh',
            'kWh',
            '%',

            'monetary units',
            'monetary units',
            'TD3 / baseline',

            'kgCO2e',
            'kgCO2e',
            'TD3 / baseline',
        ],

        'Interpretation': [
            'Scalar evaluation return for the current reward configuration',

            'Grid energy imported by the controlled system',
            'Grid energy imported by the PV-only baseline',
            (
                f'{state.grid_import_reduction_percent:.2f}% lower than baseline'
                if state.grid_import_ratio < 1.0
                else
                f'{abs(state.grid_import_reduction_percent):.2f}% higher than baseline'
            ),

            'Grid energy exported by the controlled system',
            'Grid energy exported by the PV-only baseline',

            'Saved projected unavoidable-export quantity',
            'Controlled export exceeding projected unavoidable export',
            'Share of generated PV allocated to load and battery charging',

            'Controlled import-only electricity cost',
            'PV-only baseline electricity cost',
            (
                f'{state.electricity_cost_reduction_percent:.2f}% lower than baseline'
                if state.electricity_cost_ratio < 1.0
                else
                f'{abs(state.electricity_cost_reduction_percent):.2f}% higher than baseline'
            ),

            'Controlled grid-related operational emissions',
            'PV-only baseline grid-related operational emissions',
            (
                f'{state.carbon_emission_reduction_percent:.2f}% lower than baseline'
                if state.carbon_emission_ratio < 1.0
                else
                f'{abs(state.carbon_emission_reduction_percent):.2f}% higher than baseline'
            ),
        ],
    })


    # =============================================================================
    # Attractive notebook display
    # =============================================================================

    # Soft pastel backgrounds with dark text for improved readability.
    state.section_colors = {
        'Reward': '#F8F0FA',             # Very light purple
        'Grid import': '#EDF6FC',         # Very light blue
        'Grid export': '#FFF6EB',         # Very light orange
        'Export and PV': '#EEF8F0',       # Very light green
        'Electricity cost': '#FFFBEA',    # Very light yellow
        'Carbon emissions': '#ECF9FA',    # Very light cyan
    }


    def apply_section_color(row):
        """
        Apply a soft section-specific background with dark readable text.
        """

        background_color = state.section_colors.get(
            row['Section'],
            '#FFFFFF',
        )

        return [
            (
                f'background-color: {background_color}; '
                'color: #1F2933;'
            )
            for _ in row
        ]
    state.apply_section_color = apply_section_color


    state.requested_kpi_styled = (
        state.requested_kpi_results.style

        # Display values with exactly three decimal places.
        .format({
            'Value': '{:,.3f}',
        })

        # Apply soft category-specific background shading.
        .apply(
            state.apply_section_color,
            axis=1,
        )

        # Keep the original KPI-name formatting.
        .set_properties(
            subset=[
                'KPI',
            ],
            **{
                'font-weight': 'bold',
                'text-align': 'left',
                'color': '#1F2933',
            }
        )

        # Keep the original numerical formatting and alignment.
        .set_properties(
            subset=[
                'Value',
            ],
            **{
                'font-family': 'monospace',
                'font-weight': 'bold',
                'text-align': 'right',
                'color': '#0B2239',
            }
        )

        # Keep section and unit columns centered.
        .set_properties(
            subset=[
                'Section',
                'Unit',
            ],
            **{
                'text-align': 'center',
                'color': '#263238',
            }
        )

        # Keep interpretation text left aligned.
        .set_properties(
            subset=[
                'Interpretation',
            ],
            **{
                'text-align': 'left',
                'color': '#263238',
            }
        )

        .set_table_styles([
            {
                'selector': 'caption',
                'props': [
                    ('font-size', '15px'),
                    ('font-weight', 'bold'),
                    ('color', '#222222'),
                    ('background-color', '#FFFFFF'),
                    ('text-align', 'left'),
                    ('padding-bottom', '10px'),
                ],
            },
            {
                'selector': 'th',
                'props': [
                    ('background-color', '#2E5D7B'),
                    ('color', 'white'),
                    ('font-weight', 'bold'),
                    ('text-align', 'center'),
                    ('padding', '8px'),
                    ('border', '1px solid #C7CDD1'),
                ],
            },
            {
                'selector': 'td',
                'props': [
                    ('padding', '7px'),
                    ('border', '1px solid #C7CDD1'),
                    ('vertical-align', 'middle'),
                ],
            },
            {
                'selector': 'table',
                'props': [
                    ('border-collapse', 'collapse'),
                    ('width', '100%'),
                    ('background-color', '#FFFFFF'),
                ],
            },
        ])

        .set_caption(
            'Requested Episode-Level KPIs: TD3-Controlled System '
            'versus PV-Only Baseline'
        )

        .hide_index()
    )


    display(
        state.requested_kpi_styled
    )


def cell_03(state):
    """Cell 03: publication plotting style and shared figure-saving helper."""
    # =============================================================================
    # Publication plotting style
    # =============================================================================

    # General-purpose colors
    state.BLUE = '#2E86AB'
    state.ORANGE = '#E07B39'
    state.GREEN = '#27AE60'
    state.RED = '#C0392B'
    state.PURPLE = '#8E44AD'
    state.TEAL = '#16A085'
    state.GRAY = '#7F8C8D'
    state.LIGHT_GRAY = '#ECF0F1'

    # Meaning-specific colors
    state.COLOR_TD3 = '#1F77B4'
    state.COLOR_BASELINE = '#D62728'
    state.COLOR_PV = '#2CA02C'
    state.COLOR_ACCENT = '#FF7F0E'

    # Additional physical-variable colors
    state.COLOR_LOAD = '#4D4D4D'
    state.COLOR_BATTERY = '#FF7F0E'
    state.COLOR_GRID_IMPORT = '#2E86AB'
    state.COLOR_GRID_EXPORT = '#E07B39'
    state.COLOR_PRICE = '#8E44AD'
    state.COLOR_CARBON = '#16A085'


    # =============================================================================
    # Global Matplotlib settings
    # =============================================================================

    plt.rcParams.update({
        # Font
        'font.family': 'DejaVu Sans',
        'font.size': 10,

        # Axes
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'axes.labelsize': 10,
        'axes.facecolor': 'white',

        # Grid
        'grid.color': '#E8E8E8',
        'grid.linewidth': 0.6,

        # Tick labels
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,

        # Legend
        'legend.fontsize': 9,
        'legend.framealpha': 0.85,
        'legend.edgecolor': '#CCCCCC',

        # Figure display
        'figure.dpi': 150,
        'figure.facecolor': 'white',

        # Saved figures
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
    })


    # =============================================================================
    # Figure-saving function
    # =============================================================================

    def save_figure(
        fig,
        name,
        save_png=True,
        save_pdf=True,
        show=True,
        close=True,
    ):
        """
        Save a Matplotlib figure in the analysis_figures directory.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Matplotlib figure to save.

        name : str or Path
            Filename or filename stem, such as 'soc_profile' or
            'soc_profile.png'.

        save_png : bool, default=True
            Save a 300-dpi PNG copy.

        save_pdf : bool, default=True
            Save a vector PDF copy.

        show : bool, default=True
            Display the figure inside the notebook.

        close : bool, default=True
            Close the figure after saving and displaying.
        """

        if fig is None:
            raise ValueError('A valid Matplotlib figure must be supplied.')

        state.FIGURES_DIR.mkdir(parents=True,exist_ok=True,)

        figure_stem = Path(name).stem

        if not figure_stem:
            raise ValueError('The figure name cannot be empty.')
        saved_paths = []
        if save_png:
            png_path = state.FIGURES_DIR / f'{figure_stem}.png'

            fig.savefig(
                png_path,
                dpi=300,
                bbox_inches='tight',
                facecolor='white',)

            saved_paths.append(png_path)

        if save_pdf:
            pdf_path = state.FIGURES_DIR / f'{figure_stem}.pdf'
            fig.savefig(pdf_path,bbox_inches='tight',facecolor='white',)
            saved_paths.append(pdf_path)

        if not saved_paths:
            raise ValueError('At least one output format must be enabled.')

        for saved_path in saved_paths:
            print(f'Saved: {saved_path}')

        if show:
            plt.show()

        if close:
            plt.close(fig)

        return saved_paths
    state.save_figure = save_figure

def cell_04(state):
    """Cell 04: timeseries schema/data-integrity validation."""
    # =============================================================================
    # Timeseries schema and data-integrity validation
    # =============================================================================

    state.REQUIRED_COLUMNS = [
        'transition',
        'action_time_step',
        'reward_time_step',
        'environment_time_step_after_action',
        'action',
        'reward',
        'pre_action_soc',
        'post_action_soc',
        'load',
        'pv_generation',
        'base_net_demand',
        'battery_consumption',
        'net_consumption',
        'grid_import',
        'grid_export',
        'unavoidable_export',
        'avoidable_export',
        'electricity_price',
        'carbon_intensity',
        'raw_grid_penalty',
        'raw_cost',
        'raw_emission',
        'normalized_grid',
        'normalized_cost',
        'normalized_emission',
        'weighted_grid',
        'weighted_cost',
        'weighted_emission',
        'terminal_soc_penalty',
        'net_baseline',
        'baseline_grid_import',
        'positive_baseline_cost',
        'positive_baseline_emission',
    ]


    def validate_timeseries(
        timeseries,
        expected_transitions=None,
        atol=1e-8,
        rtol=1e-6,
    ):
        """
        Validate one complete corrected evaluation episode.

        The function reports structural, timing, physical,
        reward, and SOC-continuity checks without modifying
        the raw timeseries.
        """

        missing_columns = [
            state.column
            for state.column in state.REQUIRED_COLUMNS
            if state.column not in timeseries.columns
        ]

        if missing_columns:
            raise KeyError(
                'Required timeseries columns are missing:\n'
                f'{missing_columns}'
            )

        if timeseries.empty:
            raise ValueError(
                'The timeseries contains no transitions.'
            )

        numeric_data = timeseries.select_dtypes(
            include=[np.number],
        )

        nan_count = int(timeseries.isna().sum().sum())
        infinite_count = int(
            np.isinf(numeric_data.to_numpy()).sum()
        )

        checks = {
            'No missing values':
                nan_count == 0,

            'No infinite values':
                infinite_count == 0,

            'Unique transition identifiers':
                not timeseries['transition'].duplicated().any(),

            'Continuous transition identifiers':
                timeseries['transition'].equals(
                    pd.Series(
                        np.arange(len(timeseries)),
                        index=timeseries.index,
                        name='transition',
                    )
                ),

            'Transition equals action time step':
                (
                    timeseries['transition']
                    == timeseries['action_time_step']
                ).all(),

            'Corrected transition timing':
                (
                    timeseries['reward_time_step']
                    == timeseries['action_time_step'] + 1
                ).all(),

            'Environment post-action time alignment':
                (
                    timeseries[
                        'environment_time_step_after_action'
                    ]
                    == timeseries['reward_time_step']
                ).all(),

            'Base-net-demand identity':
                np.allclose(
                    timeseries['base_net_demand'],
                    (
                        timeseries['load']
                        - timeseries['pv_generation']
                    ),
                    atol=atol,
                    rtol=rtol,
                ),

            'Net-consumption identity':
                np.allclose(
                    timeseries['net_consumption'],
                    (
                        timeseries['base_net_demand']
                        + timeseries['battery_consumption']
                    ),
                    atol=atol,
                    rtol=rtol,
                ),

            'Grid-import identity':
                np.allclose(
                    timeseries['grid_import'],
                    np.maximum(
                        timeseries['net_consumption'],
                        0.0,
                    ),
                    atol=atol,
                    rtol=rtol,
                ),

            'Grid-export identity':
                np.allclose(
                    timeseries['grid_export'],
                    np.maximum(
                        -timeseries['net_consumption'],
                        0.0,
                    ),
                    atol=atol,
                    rtol=rtol,
                ),

            'Unavoidable-export identity':
                np.allclose(
                    timeseries['unavoidable_export'],
                    np.maximum(
                        -timeseries['projected_grid_target'],
                        0.0,
                    ),
                    atol=atol,
                    rtol=rtol,
                ),

            'Avoidable-export identity':
                np.allclose(
                    timeseries['avoidable_export'],
                    np.maximum(
                        (
                            timeseries['grid_export']
                            - timeseries['unavoidable_export']
                        ),
                        0.0,
                    ),
                    atol=atol,
                    rtol=rtol,
                ),

            'Raw-cost identity':
                np.allclose(
                    timeseries['raw_cost'],
                    (
                        timeseries['grid_import']
                        * timeseries['electricity_price']
                    ),
                    atol=atol,
                    rtol=rtol,
                ),

            'Raw-emission identity':
                np.allclose(
                    timeseries['raw_emission'],
                    (
                        timeseries['grid_import']
                        * timeseries['carbon_intensity']
                    ),
                    atol=atol,
                    rtol=rtol,
                ),

            'Reward identity':
                np.allclose(
                    timeseries['reward'],
                    -(
                        timeseries['weighted_grid']
                        + timeseries['weighted_cost']
                        + timeseries['weighted_emission']
                    )
                    + timeseries['terminal_soc_penalty'],
                    atol=atol,
                    rtol=rtol,
                ),
        }

        soc_continuity = np.allclose(
            timeseries['pre_action_soc'].iloc[1:].to_numpy(),
            timeseries['post_action_soc'].iloc[:-1].to_numpy(),
            atol=atol,
            rtol=rtol,
        )

        checks['Inter-transition SOC continuity'] = soc_continuity

        if expected_transitions is not None:
            checks['Expected transition count'] = (
                len(timeseries) == expected_transitions
            )

        validation_report = pd.DataFrame({
            'check': list(checks.keys()),
            'passed': list(checks.values()),
        })

        validation_report['status'] = np.where(
            validation_report['passed'],
            'PASS',
            'FAIL',
        )

        return validation_report
    state.validate_timeseries = validate_timeseries
    # ══════════════════════════════════════════════════════════════════════════════
    # ══════════════════════════════════════════════════════════════════════════════
    state.validation_report = state.validate_timeseries(
        state.df,
        expected_transitions=8759,
    )

    display(state.validation_report)
    # ══════════════════════════════════════════════════════════════════════════════
    # ══════════════════════════════════════════════════════════════════════════════
    state.validation_report.to_csv(
        state.TABLES_DIR / 'timeseries_validation_report.csv',
        index=False,
    )

def cell_05(state):
    """Cell 05: Plot 1 - cumulative physical KPIs with zoomed insets."""
    # =============================================================================
    # Plot 1: Cumulative physical KPIs with final-period zoomed insets
    # =============================================================================

    # This figure describes one complete evaluation episode.
    # It illustrates temporal accumulation but does not represent
    # multi-seed uncertainty or statistical significance.

    from mpl_toolkits.axes_grid1.inset_locator import (
        inset_axes,
        mark_inset,
    )


    state.TIME_COLUMN = 'reward_time_step'

    # Fraction of the complete evaluation horizon shown in each inset.
    # A value of 0.10 means that the final 10% of the episode is enlarged.
    state.ZOOM_FRACTION = 0.10


    # =============================================================================
    # Plot configuration
    # =============================================================================

    state.kpi_pairs = [
        {
            'controller': 'grid_import',
            'baseline': 'baseline_grid_import',
            'ylabel': 'Cumulative grid import (kWh)',
            'title': 'Grid Import',
            'color': state.COLOR_GRID_IMPORT,
            'unit': 'kWh',
        },
        {
            'controller': 'raw_cost',
            'baseline': 'positive_baseline_cost',
            'ylabel': (
                'Cumulative electricity cost\n'
                '(monetary units)'
            ),
            'title': 'Electricity Cost',
            'color': state.ORANGE,
            'unit': 'monetary units',
        },
        {
            'controller': 'raw_emission',
            'baseline': 'positive_baseline_emission',
            'ylabel': (
                'Cumulative carbon emissions '
                '(kgCO$_2$e)'
            ),
            'title': 'Carbon Emissions',
            'color': state.COLOR_CARBON,
            'unit': 'kgCO$_2$e',
        },
    ]


    # KPI summary boxes are positioned below the upper-left legends.
    state.summary_box_positions = {
        'Grid Import': (0.035, 0.735),
        'Electricity Cost': (0.035, 0.735),
        'Carbon Emissions': (0.035, 0.735),
    }


    # Inset positions are expressed in parent-axis fractional coordinates:
    #
    # (left position, bottom position, width, height)
    #
    # The bottom position is raised to prevent the inset tick labels from
    # overlapping with the main x-axis tick labels and x-axis title.
    state.inset_positions = {
        'Grid Import': (0.55, 0.18, 0.42, 0.34),
        'Electricity Cost': (0.55, 0.18, 0.42, 0.34),
        'Carbon Emissions': (0.55, 0.18, 0.42, 0.34),
    }


    # =============================================================================
    # Validate plot configuration
    # =============================================================================

    if not 0.0 < state.ZOOM_FRACTION < 1.0:
        raise ValueError(
            'ZOOM_FRACTION must be strictly between 0 and 1.'
        )


    state.required_plot_columns = {
        state.TIME_COLUMN,
    }

    for state.item in state.kpi_pairs:
        state.required_plot_columns.add(
            state.item['controller']
        )
        state.required_plot_columns.add(
            state.item['baseline']
        )


    state.missing_plot_columns = sorted(
        state.required_plot_columns - set(state.df.columns)
    )

    if state.missing_plot_columns:
        raise KeyError(
            'The cumulative-KPI plot cannot be generated because '
            'the following columns are missing:\n'
            f'{state.missing_plot_columns}'
        )


    # Confirm that all plotted columns contain finite numerical values.
    state.non_finite_columns = []

    for state.column in sorted(state.required_plot_columns):

        state.column_values = state.df[state.column].to_numpy(
            dtype=float
        )

        if not np.isfinite(state.column_values).all():
            state.non_finite_columns.append(
                state.column
            )


    if state.non_finite_columns:
        raise ValueError(
            'NaN or infinite values were detected in the '
            'following plot columns:\n'
            f'{state.non_finite_columns}'
        )


    # =============================================================================
    # Validate the time axis
    # =============================================================================

    state.time_values = state.df[
        state.TIME_COLUMN
    ].astype(float)


    if not state.time_values.is_monotonic_increasing:
        raise ValueError(
            f'{state.TIME_COLUMN} is not monotonically increasing.'
        )


    if state.time_values.duplicated().any():
        raise ValueError(
            f'{state.TIME_COLUMN} contains duplicated values.'
        )


    # =============================================================================
    # Determine the zoomed evaluation interval
    # =============================================================================

    state.number_of_transitions = len(state.df)

    state.zoom_start_index = max(
        0,
        int(
            np.floor(
                state.number_of_transitions
                * (1.0 - state.ZOOM_FRACTION)
            )
        ),
    )

    state.zoom_start_time = float(
        state.time_values.iloc[
            state.zoom_start_index
        ]
    )

    state.zoom_end_time = float(
        state.time_values.iloc[-1]
    )


    # =============================================================================
    # Create the main figure
    # =============================================================================

    state.fig, state.axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(15.8, 5.4),
        sharex=True,
    )


    # Store the numerical values used in the figure.
    state.plot_summary_records = []


    # =============================================================================
    # Generate each KPI panel
    # =============================================================================

    for state.ax, state.item in zip(
        state.axes,
        state.kpi_pairs,
    ):

        # -------------------------------------------------------------------------
        # Calculate cumulative trajectories
        # -------------------------------------------------------------------------

        state.controller_cumulative = (
            state.df[state.item['controller']]
            .cumsum()
            .astype(float)
        )

        state.baseline_cumulative = (
            state.df[state.item['baseline']]
            .cumsum()
            .astype(float)
        )


        # -------------------------------------------------------------------------
        # Calculate final episode-level values
        # -------------------------------------------------------------------------

        state.controller_total = float(
            state.controller_cumulative.iloc[-1]
        )

        state.baseline_total = float(
            state.baseline_cumulative.iloc[-1]
        )

        state.final_difference = (
            state.baseline_total
            - state.controller_total
        )

        state.final_time = float(
            state.time_values.iloc[-1]
        )


        # -------------------------------------------------------------------------
        # Calculate final ratio and percentage change
        # -------------------------------------------------------------------------

        if np.isclose(
            state.baseline_total,
            0.0,
            atol=1e-12,
        ):
            state.final_ratio = np.nan
            state.percentage_reduction = np.nan

            state.change_label = 'Percentage change undefined'

            state.annotation_text = (
                f'TD3: {state.controller_total:,.1f}\n'
                f'Baseline: {state.baseline_total:,.1f}\n'
                'Ratio: undefined'
            )

        else:
            state.final_ratio = (
                state.controller_total
                / state.baseline_total
            )

            state.percentage_reduction = (
                100.0
                * state.final_difference
                / state.baseline_total
            )

            if state.percentage_reduction >= 0.0:
                state.change_label = (
                    f'{state.percentage_reduction:.1f}% reduction'
                )
            else:
                state.change_label = (
                    f'{abs(state.percentage_reduction):.1f}% increase'
                )

            state.annotation_text = (
                f'TD3: {state.controller_total:,.1f}\n'
                f'Baseline: {state.baseline_total:,.1f}\n'
                f'Ratio: {state.final_ratio:.3f}\n'
                f'{state.change_label}'
            )


        # -------------------------------------------------------------------------
        # Store numerical results for table export
        # -------------------------------------------------------------------------

        state.plot_summary_records.append({
            'metric': state.item['title'],
            'controller_column': state.item['controller'],
            'baseline_column': state.item['baseline'],
            'td3_total': state.controller_total,
            'baseline_total': state.baseline_total,
            'baseline_minus_td3': state.final_difference,
            'td3_to_baseline_ratio': state.final_ratio,
            'percentage_reduction': state.percentage_reduction,
            'unit': state.item['unit'],
            'zoom_start_time_step': state.zoom_start_time,
            'zoom_end_time_step': state.zoom_end_time,
        })


        # -------------------------------------------------------------------------
        # Plot the baseline cumulative trajectory
        # -------------------------------------------------------------------------

        state.ax.plot(
            state.time_values,
            state.baseline_cumulative,
            color=state.COLOR_BASELINE,
            linewidth=1.6,
            linestyle='--',
            label='Baseline',
            zorder=2,
        )


        # -------------------------------------------------------------------------
        # Plot the TD3 cumulative trajectory
        # -------------------------------------------------------------------------

        state.ax.plot(
            state.time_values,
            state.controller_cumulative,
            color=state.item['color'],
            linewidth=1.9,
            linestyle='-',
            label='TD3 controller',
            zorder=3,
        )


        # -------------------------------------------------------------------------
        # Shade regions where TD3 remains cumulatively below baseline
        # -------------------------------------------------------------------------

        state.improvement_mask = (
            state.controller_cumulative
            < state.baseline_cumulative
        )

        state.ax.fill_between(
            state.time_values,
            state.controller_cumulative,
            state.baseline_cumulative,
            where=state.improvement_mask,
            interpolate=True,
            color=state.item['color'],
            alpha=0.14,
            label='Lower than baseline',
            zorder=1,
        )


        # -------------------------------------------------------------------------
        # Mark the final baseline and TD3 values on the main plot
        # -------------------------------------------------------------------------

        state.ax.scatter(
            state.final_time,
            state.baseline_total,
            color=state.COLOR_BASELINE,
            marker='D',
            s=42,
            edgecolor='white',
            linewidth=0.8,
            zorder=6,
            clip_on=False,
        )

        state.ax.scatter(
            state.final_time,
            state.controller_total,
            color=state.item['color'],
            marker='o',
            s=46,
            edgecolor='white',
            linewidth=0.8,
            zorder=7,
            clip_on=False,
        )


        # -------------------------------------------------------------------------
        # Place the legend at the top-left
        # -------------------------------------------------------------------------

        state.ax.legend(
            loc='upper left',
            bbox_to_anchor=(
                0.025,
                0.985,
            ),
            borderaxespad=0.0,
            frameon=True,
            framealpha=0.95,
            facecolor='white',
            edgecolor='#CCCCCC',
            fontsize=8,
            labelspacing=0.30,
            handlelength=2.2,
            handletextpad=0.55,
        )


        # -------------------------------------------------------------------------
        # Add the numerical KPI box below the legend
        # -------------------------------------------------------------------------

        state.ax.annotate(
            state.annotation_text,

            # Connect the summary box with the final TD3 point.
            xy=(
                state.final_time,
                state.controller_total,
            ),

            # Position the box directly below the legend.
            xytext=state.summary_box_positions[
                state.item['title']
            ],

            xycoords='data',
            textcoords='axes fraction',

            horizontalalignment='left',
            verticalalignment='top',

            fontsize=8.2,
            fontweight='bold',
            linespacing=1.15,
            color='#222222',

            bbox={
                'boxstyle': 'round,pad=0.40',
                'facecolor': 'white',
                'edgecolor': state.item['color'],
                'linewidth': 1.25,
                'alpha': 0.96,
            },

            arrowprops={
                'arrowstyle': '-',
                'color': state.item['color'],
                'linewidth': 0.85,
                'alpha': 0.75,
                'connectionstyle': 'arc3,rad=0.05',
            },

            zorder=8,
        )


        # =========================================================================
        # Create a raised inset showing the final 10% of the episode
        # =========================================================================

        state.inset_left, state.inset_bottom, state.inset_width, state.inset_height = (
            state.inset_positions[
                state.item['title']
            ]
        )

        state.inset_ax = inset_axes(
            state.ax,
            width='100%',
            height='100%',
            loc='lower left',
            bbox_to_anchor=(
                state.inset_left,
                state.inset_bottom,
                state.inset_width,
                state.inset_height,
            ),
            bbox_transform=state.ax.transAxes,
            borderpad=0.0,
        )


        # -------------------------------------------------------------------------
        # Plot the cumulative trajectories inside the inset
        # -------------------------------------------------------------------------

        state.inset_ax.plot(
            state.time_values,
            state.baseline_cumulative,
            color=state.COLOR_BASELINE,
            linewidth=1.25,
            linestyle='--',
            zorder=2,
        )

        state.inset_ax.plot(
            state.time_values,
            state.controller_cumulative,
            color=state.item['color'],
            linewidth=1.45,
            linestyle='-',
            zorder=3,
        )

        state.inset_ax.fill_between(
            state.time_values,
            state.controller_cumulative,
            state.baseline_cumulative,
            where=state.improvement_mask,
            interpolate=True,
            color=state.item['color'],
            alpha=0.18,
            zorder=1,
        )


        # -------------------------------------------------------------------------
        # Extract the data included in the zoomed interval
        # -------------------------------------------------------------------------

        state.zoom_controller_values = (
            state.controller_cumulative.iloc[
                state.zoom_start_index:
            ]
        )

        state.zoom_baseline_values = (
            state.baseline_cumulative.iloc[
                state.zoom_start_index:
            ]
        )

        state.zoom_all_values = np.concatenate([
            state.zoom_controller_values.to_numpy(
                dtype=float
            ),
            state.zoom_baseline_values.to_numpy(
                dtype=float
            ),
        ])


        state.zoom_y_min = float(
            np.min(
                state.zoom_all_values
            )
        )

        state.zoom_y_max = float(
            np.max(
                state.zoom_all_values
            )
        )

        state.zoom_y_range = (
            state.zoom_y_max
            - state.zoom_y_min
        )


        # -------------------------------------------------------------------------
        # Add vertical spacing around the zoomed values
        # -------------------------------------------------------------------------

        if np.isclose(
            state.zoom_y_range,
            0.0,
            atol=1e-12,
        ):
            state.zoom_y_padding = max(
                abs(state.zoom_y_max) * 0.03,
                1.0,
            )
        else:
            state.zoom_y_padding = (
                0.08
                * state.zoom_y_range
            )


        state.inset_ax.set_xlim(
            state.zoom_start_time,
            state.zoom_end_time,
        )

        state.inset_ax.set_ylim(
            state.zoom_y_min - state.zoom_y_padding,
            state.zoom_y_max + state.zoom_y_padding,
        )


        # -------------------------------------------------------------------------
        # Mark final values inside the inset
        # -------------------------------------------------------------------------

        state.inset_ax.scatter(
            state.final_time,
            state.baseline_total,
            color=state.COLOR_BASELINE,
            marker='D',
            s=28,
            edgecolor='white',
            linewidth=0.6,
            zorder=5,
            clip_on=False,
        )

        state.inset_ax.scatter(
            state.final_time,
            state.controller_total,
            color=state.item['color'],
            marker='o',
            s=32,
            edgecolor='white',
            linewidth=0.6,
            zorder=6,
            clip_on=False,
        )


        # -------------------------------------------------------------------------
        # Add final percentage change inside the inset
        # -------------------------------------------------------------------------

        if np.isfinite(
            state.percentage_reduction
        ):
            if state.percentage_reduction >= 0.0:
                state.inset_change_text = (
                    f'{state.percentage_reduction:.1f}%\n'
                    'lower'
                )

                state.inset_change_color = (
                    state.item['color']
                )

            else:
                state.inset_change_text = (
                    f'{abs(state.percentage_reduction):.1f}%\n'
                    'higher'
                )

                state.inset_change_color = state.RED

        else:
            state.inset_change_text = (
                'Change\nundefined'
            )

            state.inset_change_color = state.GRAY


        state.inset_ax.text(
            0.96,
            0.06,
            state.inset_change_text,

            transform=state.inset_ax.transAxes,

            horizontalalignment='right',
            verticalalignment='bottom',

            fontsize=7.4,
            fontweight='bold',

            color=state.inset_change_color,

            bbox={
                'boxstyle': 'round,pad=0.24',
                'facecolor': 'white',
                'edgecolor': state.inset_change_color,
                'linewidth': 0.8,
                'alpha': 0.93,
            },

            zorder=8,
        )


        # -------------------------------------------------------------------------
        # Place inset title inside the inset
        # -------------------------------------------------------------------------

        state.inset_ax.text(
            0.50,
            0.96,
            'Final 10% of episode',

            transform=state.inset_ax.transAxes,

            horizontalalignment='center',
            verticalalignment='top',

            fontsize=7.2,
            fontweight='bold',
            color='#222222',

            bbox={
                'boxstyle': 'round,pad=0.18',
                'facecolor': 'white',
                'edgecolor': 'none',
                'alpha': 0.82,
            },

            zorder=10,
        )


        # -------------------------------------------------------------------------
        # Format inset ticks and grid
        # -------------------------------------------------------------------------

        state.inset_ax.tick_params(
            axis='both',
            which='major',
            labelsize=6.2,
            length=2.3,
            pad=1.5,
        )

        state.inset_ax.xaxis.set_major_locator(
            ticker.MaxNLocator(
                nbins=4,
                integer=True,
            )
        )

        state.inset_ax.yaxis.set_major_locator(
            ticker.MaxNLocator(
                nbins=4,
            )
        )

        state.inset_ax.xaxis.set_major_formatter(
            ticker.StrMethodFormatter(
                '{x:,.0f}'
            )
        )

        state.inset_ax.yaxis.set_major_formatter(
            ticker.StrMethodFormatter(
                '{x:,.0f}'
            )
        )

        state.inset_ax.grid(
            True,
            color='#E8E8E8',
            linewidth=0.45,
            alpha=0.8,
        )

        # The inset does not need separate axis titles.
        state.inset_ax.set_xlabel('')
        state.inset_ax.set_ylabel('')


        # Keep all four boundaries visible around the inset.
        for state.spine in state.inset_ax.spines.values():
            state.spine.set_visible(True)
            state.spine.set_linewidth(0.8)
            state.spine.set_color('#555555')


        # -------------------------------------------------------------------------
        # Draw dashed zoom rectangle and connector lines
        # -------------------------------------------------------------------------

        mark_inset(
            state.ax,
            state.inset_ax,
            loc1=2,
            loc2=4,
            facecolor='none',
            edgecolor='#666666',
            linewidth=0.85,
            linestyle='--',
            alpha=0.85,
            zorder=5,
        )


        # -------------------------------------------------------------------------
        # Format the main axes
        # -------------------------------------------------------------------------

        state.ax.set_xlabel(
            'Reward time step (hour)'
        )

        state.ax.set_ylabel(
            state.item['ylabel']
        )

        state.ax.set_title(
            state.item['title'],
            pad=9,
        )

        state.ax.xaxis.set_major_locator(
            ticker.MaxNLocator(
                nbins=5,
                integer=True,
            )
        )

        state.ax.xaxis.set_major_formatter(
            ticker.StrMethodFormatter(
                '{x:,.0f}'
            )
        )

        state.ax.yaxis.set_major_formatter(
            ticker.StrMethodFormatter(
                '{x:,.0f}'
            )
        )

        # Add enough horizontal and vertical margin for final markers.
        state.ax.margins(
            x=0.04,
            y=0.08,
        )


    # =============================================================================
    # Final figure formatting
    # =============================================================================

    state.fig.suptitle(
        'Cumulative Performance: TD3 Controller versus Baseline',
        fontsize=13,
        fontweight='bold',
        y=0.995,
    )

    state.fig.tight_layout(
        rect=[
            0.0,
            0.035,
            1.0,
            0.955,
        ],
        w_pad=2.0,
    )


    # =============================================================================
    # Save the figure
    # =============================================================================

    state.save_figure(
        state.fig,
        'plot01_cumulative_kpis_with_zoomed_insets',
    )


    # =============================================================================
    # Save and display the numerical summary
    # =============================================================================

    state.plot01_summary = pd.DataFrame(
        state.plot_summary_records
    )

    state.plot01_summary_path = (
        state.TABLES_DIR
        / 'plot01_cumulative_kpis_with_zoomed_insets_summary.csv'
    )

    state.plot01_summary.to_csv(
        state.plot01_summary_path,
        index=False,
    )

    display(
        state.plot01_summary
    )

    print(
        'Plot 1 numerical summary saved to:\n'
        f'{state.plot01_summary_path}'
    )

def cell_06(state):
    """Cell 06: Plot 3 - weekly operational profile with peak-tariff shading."""
    # =============================================================================
    # Plot 3: Selected weekly operational profile with peak-tariff shading
    # =============================================================================

    # This figure illustrates one sequential 168-hour evaluation period.
    # It is a single-run operational diagnostic and does not represent
    # multi-seed uncertainty or statistical significance.


    # =============================================================================
    # User configuration
    # =============================================================================

    state.SELECTED_WEEK = 20

    state.HOURS_PER_DAY = 24
    state.DAYS_PER_WEEK = 7
    state.HOURS_PER_WEEK = state.HOURS_PER_DAY * state.DAYS_PER_WEEK

    state.TIME_COLUMN = 'reward_time_step'

    # Verified tariff values observed in the current timeseries.
    state.OFF_PEAK_PRICES = {
        0.21,
        0.22,
    }

    state.PEAK_PRICES = {
        0.40,
        0.50,
        0.54,
    }

    state.PRICE_TOLERANCE = 1e-8

    # Plot colors
    state.PEAK_SHADE_COLOR = '#F6C98D'
    state.PEAK_SHADE_ALPHA = 0.22

    state.DAY_SEPARATOR_COLOR = '#808080'
    state.DAY_SEPARATOR_ALPHA = 0.75


    # =============================================================================
    # Validate selected-week configuration
    # =============================================================================

    if not isinstance(state.SELECTED_WEEK, int):
        raise TypeError(
            'SELECTED_WEEK must be an integer.'
        )

    if state.SELECTED_WEEK < 1:
        raise ValueError(
            'SELECTED_WEEK must be greater than or equal to 1.'
        )


    # =============================================================================
    # Validate required columns
    # =============================================================================

    state.weekly_required_columns = [
        'transition',
        state.TIME_COLUMN,
        'pre_action_soc_percent',
        'post_action_soc_percent',
        'load',
        'pv_generation',
        'net_baseline',
        'net_consumption',
        'battery_consumption',
        'electricity_price',
    ]

    state.weekly_missing_columns = [
        state.column
        for state.column in state.weekly_required_columns
        if state.column not in state.df.columns
    ]

    if state.weekly_missing_columns:
        raise KeyError(
            'The weekly operational-profile figure cannot be generated '
            'because the following columns are missing:\n'
            f'{state.weekly_missing_columns}'
        )


    # =============================================================================
    # Check for NaN and infinite values
    # =============================================================================

    state.weekly_non_finite_columns = []

    for state.column in state.weekly_required_columns:

        state.column_values = state.df[state.column].to_numpy(
            dtype=float
        )

        if not np.isfinite(state.column_values).all():
            state.weekly_non_finite_columns.append(
                state.column
            )

    if state.weekly_non_finite_columns:
        raise ValueError(
            'NaN or infinite values were detected in the following '
            'weekly-profile columns:\n'
            f'{state.weekly_non_finite_columns}'
        )


    # =============================================================================
    # Validate observed tariff values
    # =============================================================================

    state.verified_prices = (
        state.OFF_PEAK_PRICES
        | state.PEAK_PRICES
    )

    state.observed_prices = sorted(
        state.df['electricity_price']
        .round(8)
        .unique()
        .tolist()
    )

    state.unexpected_prices = [
        price
        for price in state.observed_prices
        if not any(
            np.isclose(
                price,
                verified_price,
                atol=state.PRICE_TOLERANCE,
                rtol=0.0,
            )
            for verified_price in state.verified_prices
        )
    ]

    if state.unexpected_prices:
        raise ValueError(
            'The timeseries contains electricity-price values that are '
            'not included in the verified tariff mapping:\n'
            f'{state.unexpected_prices}\n'
            'Verify the source pricing data before updating the mapping.'
        )

    print(
        'Verified electricity-price values:',
        state.observed_prices,
    )


    # =============================================================================
    # Construct sequential evaluation-week identifiers
    # =============================================================================

    state.weekly_df = state.df.copy()

    state.weekly_df['evaluation_week'] = (
        state.weekly_df['transition']
        // state.HOURS_PER_WEEK
    ) + 1

    state.available_weeks = sorted(
        state.weekly_df['evaluation_week']
        .unique()
        .tolist()
    )

    if state.SELECTED_WEEK not in state.available_weeks:
        raise ValueError(
            f'Sequential evaluation Week {state.SELECTED_WEEK} is unavailable. '
            f'Available weeks extend from '
            f'{min(state.available_weeks)} to {max(state.available_weeks)}.'
        )


    # =============================================================================
    # Extract the selected week
    # =============================================================================

    state.week = (
        state.weekly_df.loc[
            state.weekly_df['evaluation_week']
            == state.SELECTED_WEEK
        ]
        .copy()
        .reset_index(drop=True)
    )

    if len(state.week) != state.HOURS_PER_WEEK:
        raise ValueError(
            f'Sequential evaluation Week {state.SELECTED_WEEK} contains '
            f'{len(state.week)} transitions instead of the expected '
            f'{state.HOURS_PER_WEEK}.'
        )


    # Local hour index from 0 to 167.
    state.week['hour_within_week'] = np.arange(
        len(state.week),
        dtype=int,
    )

    state.hours = state.week[
        'hour_within_week'
    ]


    # =============================================================================
    # Classify tariff periods
    # =============================================================================

    def belongs_to_price_set(
        price,
        verified_price_set,
        tolerance=state.PRICE_TOLERANCE,
    ):
        """
        Return True when a price matches a value in the verified set.
        """

        return any(
            np.isclose(
                price,
                verified_price,
                atol=tolerance,
                rtol=0.0,
            )
            for verified_price in verified_price_set
        )
    state.belongs_to_price_set = belongs_to_price_set


    state.week['is_peak_tariff'] = (
        state.week['electricity_price']
        .apply(
            lambda price: state.belongs_to_price_set(
                price,
                state.PEAK_PRICES,
            )
        )
    )

    state.week['is_off_peak_tariff'] = (
        state.week['electricity_price']
        .apply(
            lambda price: state.belongs_to_price_set(
                price,
                state.OFF_PEAK_PRICES,
            )
        )
    )


    state.classified_tariff_mask = (
        state.week['is_peak_tariff']
        | state.week['is_off_peak_tariff']
    )

    if not state.classified_tariff_mask.all():

        state.unclassified_week_prices = sorted(
            state.week.loc[
                ~state.classified_tariff_mask,
                'electricity_price',
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            'Some selected-week tariff values could not be classified:\n'
            f'{state.unclassified_week_prices}'
        )


    # =============================================================================
    # Find contiguous peak-tariff intervals
    # =============================================================================

    def find_contiguous_intervals(
        boolean_values,
    ):
        """
        Convert a Boolean hourly mask into contiguous plot intervals.

        Each hourly observation is treated as an interval centred on
        that hour. For example, hour 16 covers 15.5 to 16.5.
        """

        boolean_values = np.asarray(
            boolean_values,
            dtype=bool,
        )

        intervals = []
        interval_start = None

        for index, is_active in enumerate(
            boolean_values
        ):

            if is_active and interval_start is None:
                interval_start = index

            if interval_start is not None:

                is_last_index = (
                    index == len(boolean_values) - 1
                )

                if not is_active:

                    interval_end = index - 1

                    intervals.append((
                        interval_start - 0.5,
                        interval_end + 0.5,
                    ))

                    interval_start = None

                elif is_last_index:

                    interval_end = index

                    intervals.append((
                        interval_start - 0.5,
                        interval_end + 0.5,
                    ))

                    interval_start = None

        return intervals
    state.find_contiguous_intervals = find_contiguous_intervals


    state.peak_intervals = state.find_contiguous_intervals(
        state.week['is_peak_tariff']
    )

    if not state.peak_intervals:
        raise ValueError(
            f'No verified peak-tariff intervals were found in '
            f'Sequential Week {state.SELECTED_WEEK}.'
        )


    print(
        f'Peak-tariff intervals in Week {state.SELECTED_WEEK}:'
    )

    for state.interval_start, state.interval_end in state.peak_intervals:

        print(
            f'  Hour {state.interval_start + 0.5:.0f} '
            f'to {state.interval_end + 0.5:.0f}'
        )


    # =============================================================================
    # Shared background function
    # =============================================================================

    def add_weekly_background(
        axis,
        peak_periods,
    ):
        """
        Add peak-tariff shading and daily separators to a subplot.
        """

        # Shade peak-tariff periods.
        for interval_start, interval_end in peak_periods:

            axis.axvspan(
                interval_start,
                interval_end,
                color=state.PEAK_SHADE_COLOR,
                alpha=state.PEAK_SHADE_ALPHA,
                linewidth=0.0,
                zorder=0,
            )

        # Add daily separators after Days 1 to 6.
        for day_boundary in range(
            state.HOURS_PER_DAY,
            state.HOURS_PER_WEEK,
            state.HOURS_PER_DAY,
        ):

            axis.axvline(
                x=day_boundary - 0.5,
                color=state.DAY_SEPARATOR_COLOR,
                linewidth=0.75,
                linestyle=':',
                alpha=state.DAY_SEPARATOR_ALPHA,
                zorder=1,
            )

        axis.set_xlim(
            -0.5,
            state.HOURS_PER_WEEK - 0.5,
        )
    state.add_weekly_background = add_weekly_background


    # =============================================================================
    # Create operational masks
    # =============================================================================

    state.charging_mask = (
        state.week['battery_consumption']
        > 0.0
    )

    state.discharging_mask = (
        state.week['battery_consumption']
        < 0.0
    )

    state.grid_import_mask = (
        state.week['net_consumption']
        >= 0.0
    )

    state.grid_export_mask = (
        state.week['net_consumption']
        < 0.0
    )


    # =============================================================================
    # Calculate a weekly numerical summary
    # =============================================================================

    state.baseline_grid_import = float(
        np.maximum(
            state.week['net_baseline'],
            0.0,
        ).sum()
    )

    state.controlled_grid_import = float(
        np.maximum(
            state.week['net_consumption'],
            0.0,
        ).sum()
    )

    state.baseline_grid_export = float(
        np.maximum(
            -state.week['net_baseline'],
            0.0,
        ).sum()
    )

    state.controlled_grid_export = float(
        np.maximum(
            -state.week['net_consumption'],
            0.0,
        ).sum()
    )

    state.charging_energy = float(
        state.week.loc[
            state.charging_mask,
            'battery_consumption',
        ].sum()
    )

    state.discharging_energy_magnitude = float(
        -state.week.loc[
            state.discharging_mask,
            'battery_consumption',
        ].sum()
    )


    if np.isclose(
        state.baseline_grid_import,
        0.0,
        atol=1e-12,
    ):
        state.weekly_grid_import_ratio = np.nan
    else:
        state.weekly_grid_import_ratio = (
            state.controlled_grid_import
            / state.baseline_grid_import
        )


    state.weekly_summary = pd.DataFrame([{
        'selected_week': state.SELECTED_WEEK,
        'start_transition': int(
            state.week['transition'].iloc[0]
        ),
        'end_transition': int(
            state.week['transition'].iloc[-1]
        ),
        'start_reward_time_step': int(
            state.week[state.TIME_COLUMN].iloc[0]
        ),
        'end_reward_time_step': int(
            state.week[state.TIME_COLUMN].iloc[-1]
        ),
        'number_of_transitions': len(state.week),
        'total_load_kwh': float(
            state.week['load'].sum()
        ),
        'total_pv_generation_kwh': float(
            state.week['pv_generation'].sum()
        ),
        'baseline_grid_import_kwh': state.baseline_grid_import,
        'controlled_grid_import_kwh': state.controlled_grid_import,
        'grid_import_ratio': state.weekly_grid_import_ratio,
        'baseline_grid_export_kwh': state.baseline_grid_export,
        'controlled_grid_export_kwh': state.controlled_grid_export,
        'charging_energy_kwh': state.charging_energy,
        'discharging_energy_magnitude_kwh':
            state.discharging_energy_magnitude,
        'initial_soc_percent': float(
            state.week['pre_action_soc_percent'].iloc[0]
        ),
        'final_soc_percent': float(
            state.week['post_action_soc_percent'].iloc[-1]
        ),
        'minimum_soc_percent': float(
            state.week['post_action_soc_percent'].min()
        ),
        'maximum_soc_percent': float(
            state.week['post_action_soc_percent'].max()
        ),
        'peak_tariff_hours': int(
            state.week['is_peak_tariff'].sum()
        ),
        'off_peak_tariff_hours': int(
            state.week['is_off_peak_tariff'].sum()
        ),
    }])


    # =============================================================================
    # Create four vertically aligned subplots
    # =============================================================================

    # IMPORTANT:
    # plt.subplots uses "ncols", with an s.
    # Legend functions use "ncol", without an s, in this Matplotlib version.

    state.fig, state.axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(13.5, 11.2),
        sharex=True,
    )


    # Add shared peak shading and daily separators.
    for state.axis in state.axes:

        state.add_weekly_background(
            state.axis,
            state.peak_intervals,
        )

        state.axis.set_axisbelow(
            True
        )


    # =============================================================================
    # Panel (a): Building load and PV generation
    # =============================================================================

    state.axes[0].fill_between(
        state.hours,
        0.0,
        state.week['pv_generation'],
        color=state.COLOR_PV,
        alpha=0.24,
        linewidth=0.0,
        zorder=2,
    )

    state.axes[0].plot(
        state.hours,
        state.week['pv_generation'],
        color=state.COLOR_PV,
        linewidth=1.45,
        linestyle='-.',
        label='PV generation',
        zorder=4,
    )

    state.axes[0].plot(
        state.hours,
        state.week['load'],
        color=state.ORANGE,
        linewidth=1.55,
        linestyle='-',
        label='Building load',
        zorder=5,
    )

    state.axes[0].set_ylabel(
        'Energy per time step\n(kWh)'
    )

    state.axes[0].set_title(
        '(a) Building Load and PV Generation',
        loc='left',
        pad=7,
    )

    state.axes[0].legend(
        loc='upper right',
        ncol=2,
        frameon=True,
        framealpha=0.93,
        facecolor='white',
        edgecolor='#CCCCCC',
    )


    # =============================================================================
    # Panel (b): Baseline and TD3-controlled net grid exchange
    # =============================================================================

    state.axes[1].plot(
        state.hours,
        state.week['net_baseline'],
        color=state.COLOR_BASELINE,
        linewidth=1.35,
        linestyle='--',
        label='Baseline without storage',
        zorder=4,
    )

    state.axes[1].plot(
        state.hours,
        state.week['net_consumption'],
        color=state.COLOR_TD3,
        linewidth=1.55,
        linestyle='-',
        label='TD3-controlled',
        zorder=5,
    )

    state.axes[1].axhline(
        y=0.0,
        color='#333333',
        linewidth=0.85,
        linestyle='-',
        zorder=3,
    )

    state.axes[1].fill_between(
        state.hours,
        0.0,
        state.week['net_consumption'],
        where=state.grid_import_mask,
        interpolate=True,
        color=state.COLOR_GRID_IMPORT,
        alpha=0.07,
        zorder=2,
    )

    state.axes[1].fill_between(
        state.hours,
        0.0,
        state.week['net_consumption'],
        where=state.grid_export_mask,
        interpolate=True,
        color=state.COLOR_GRID_EXPORT,
        alpha=0.08,
        zorder=2,
    )

    state.axes[1].set_ylabel(
        'Net grid exchange\n(kWh)'
    )

    state.axes[1].set_title(
        '(b) Baseline and TD3-Controlled Net Grid Exchange',
        loc='left',
        pad=7,
    )

    state.axes[1].legend(
        loc='upper right',
        ncol=2,
        frameon=True,
        framealpha=0.93,
        facecolor='white',
        edgecolor='#CCCCCC',
    )


    # =============================================================================
    # Panel (c): Realized battery charging and discharging
    # =============================================================================

    state.axes[2].bar(
        state.hours[state.charging_mask],
        state.week.loc[
            state.charging_mask,
            'battery_consumption',
        ],
        width=0.82,
        color=state.COLOR_BATTERY,
        alpha=0.86,
        label='Charging',
        zorder=4,
    )

    state.axes[2].bar(
        state.hours[state.discharging_mask],
        state.week.loc[
            state.discharging_mask,
            'battery_consumption',
        ],
        width=0.82,
        color=state.PURPLE,
        alpha=0.86,
        label='Discharging',
        zorder=4,
    )

    state.axes[2].axhline(
        y=0.0,
        color='#333333',
        linewidth=0.85,
        linestyle='-',
        zorder=3,
    )

    state.axes[2].set_ylabel(
        'Battery exchange\n(kWh)'
    )

    state.axes[2].set_title(
        '(c) Realized Battery Charging and Discharging',
        loc='left',
        pad=7,
    )

    state.axes[2].legend(
        loc='upper right',
        ncol=2,
        frameon=True,
        framealpha=0.93,
        facecolor='white',
        edgecolor='#CCCCCC',
    )


    # =============================================================================
    # Panel (d): Post-action battery SOC
    # =============================================================================

    state.axes[3].fill_between(
        state.hours,
        0.0,
        state.week['post_action_soc_percent'],
        color=state.TEAL,
        alpha=0.10,
        linewidth=0.0,
        zorder=2,
    )

    state.axes[3].plot(
        state.hours,
        state.week['post_action_soc_percent'],
        color=state.TEAL,
        linewidth=1.85,
        linestyle='-',
        label='Post-action SOC',
        zorder=5,
    )

    state.axes[3].axhline(
        y=0.0,
        color='#555555',
        linewidth=0.75,
        linestyle=':',
        zorder=3,
    )

    state.axes[3].axhline(
        y=100.0,
        color='#555555',
        linewidth=0.75,
        linestyle=':',
        zorder=3,
    )

    state.axes[3].set_ylabel(
        'Battery SOC (%)'
    )

    state.axes[3].set_ylim(
        -3.0,
        103.0,
    )

    state.axes[3].set_title(
        '(d) Post-Action Battery State of Charge',
        loc='left',
        pad=7,
    )

    state.axes[3].legend(
        loc='upper right',
        frameon=True,
        framealpha=0.93,
        facecolor='white',
        edgecolor='#CCCCCC',
    )


    # =============================================================================
    # Shared x-axis formatting
    # =============================================================================

    state.day_midpoint_positions = (
        np.arange(
            state.DAYS_PER_WEEK
        )
        * state.HOURS_PER_DAY
        + (state.HOURS_PER_DAY - 1) / 2
    )

    state.axes[3].set_xticks(
        state.day_midpoint_positions
    )

    state.axes[3].set_xticklabels([
        f'Day {day_number}'
        for day_number in range(
            1,
            state.DAYS_PER_WEEK + 1,
        )
    ])

    state.axes[3].set_xlabel(
        'Sequential day within selected evaluation week'
    )


    # Format y-axis values consistently.
    for state.axis in state.axes:

        state.axis.yaxis.set_major_formatter(
            ticker.StrMethodFormatter(
                '{x:,.1f}'
            )
        )


    # =============================================================================
    # Create shared tariff legend
    # =============================================================================

    state.peak_tariff_patch = mpatches.Patch(
        facecolor=state.PEAK_SHADE_COLOR,
        edgecolor='#D9A45D',
        linewidth=0.8,
        alpha=state.PEAK_SHADE_ALPHA,
        label='Peak tariff period',
    )

    state.off_peak_patch = mpatches.Patch(
        facecolor='white',
        edgecolor='#B0B0B0',
        linewidth=0.8,
        label='Unshaded: off-peak tariff period',
    )


    # IMPORTANT:
    # fig.legend uses "ncol", without an s, in this Matplotlib version.

    state.fig.legend(
        handles=[
            state.peak_tariff_patch,
            state.off_peak_patch,
        ],
        loc='upper center',
        bbox_to_anchor=(
            0.5,
            0.952,
        ),
        ncol=2,
        frameon=True,
        framealpha=0.96,
        facecolor='white',
        edgecolor='#CCCCCC',
        fontsize=9,
        handlelength=2.2,
        columnspacing=1.8,
    )


    # =============================================================================
    # Figure title and sign-convention note
    # =============================================================================

    state.fig.suptitle(
        (
            'Selected Weekly Operational Profile: '
            f'Sequential Evaluation Week {state.SELECTED_WEEK}'
        ),
        fontsize=13,
        fontweight='bold',
        y=0.995,
    )

    state.fig.text(
        0.5,
        0.008,
        (
            'Positive battery exchange denotes charging and negative '
            'battery exchange denotes discharging. Positive net grid '
            'exchange denotes import and negative net grid exchange '
            'denotes export.'
        ),
        horizontalalignment='center',
        verticalalignment='bottom',
        fontsize=8.4,
        color='#444444',
    )


    # Leave space for the title, shared tariff legend, and bottom note.
    state.fig.tight_layout(
        rect=[
            0.0,
            0.037,
            1.0,
            0.925,
        ],
        h_pad=1.10,
    )


    # =============================================================================
    # Save the figure
    # =============================================================================

    state.figure_name = (
        f'plot03_weekly_operational_profile_'
        f'week_{state.SELECTED_WEEK:02d}_with_peak_tariff'
    )

    state.save_figure(
        state.fig,
        state.figure_name,
    )


    # =============================================================================
    # Save and display the weekly numerical summary
    # =============================================================================

    state.weekly_summary_path = (
        state.TABLES_DIR
        / (
            f'plot03_weekly_operational_profile_'
            f'week_{state.SELECTED_WEEK:02d}_summary.csv'
        )
    )

    state.weekly_summary.to_csv(
        state.weekly_summary_path,
        index=False,
    )

    display(
        state.weekly_summary
    )

    print(
        'Weekly operational-profile summary saved to:\n'
        f'{state.weekly_summary_path}'
    )

def cell_07(state):
    """Cell 07: Plot 6 - battery SOC occupancy distribution."""
    # =============================================================================
    # Plot 6: Battery SOC occupancy distribution
    # =============================================================================

    # This figure describes the distribution of post-action battery SOC
    # over one complete evaluation episode.
    #
    # The left panel includes all transitions.
    # The right panel is conditional on SOC exceeding the explicitly
    # defined near-empty tolerance.
    #
    # This is a single-run diagnostic and does not represent
    # multi-seed uncertainty.

    state.SOC_COLUMN = 'post_action_soc_percent'

    # Values at or below this tolerance are classified as near-empty.
    state.SOC_EMPTY_TOLERANCE_PERCENT = 0.5

    # Values at or above this threshold are classified as near-full.
    state.SOC_FULL_TOLERANCE_PERCENT = 99.5

    # Forty equal-width bins over the physical 0% to 100% SOC range.
    state.SOC_BIN_EDGES = np.linspace(
        0.0,
        100.0,
        41,
    )


    # =============================================================================
    # Validate the SOC data
    # =============================================================================

    if state.SOC_COLUMN not in state.df.columns:
        raise KeyError(
            f'The SOC-distribution plot requires the column '
            f'"{state.SOC_COLUMN}", but the column is unavailable.'
        )


    state.soc_all = state.df[
        state.SOC_COLUMN
    ].astype(float)


    if state.soc_all.empty:
        raise ValueError(
            'The SOC series contains no observations.'
        )


    if not np.isfinite(
        state.soc_all.to_numpy()
    ).all():
        raise ValueError(
            f'The column "{state.SOC_COLUMN}" contains NaN or infinite values.'
        )


    # Allow a small tolerance for floating-point representation.
    if (
        state.soc_all.min() < -1e-8
        or state.soc_all.max() > 100.0 + 1e-8
    ):
        raise ValueError(
            'Post-action SOC contains values outside the expected '
            '0% to 100% physical range.\n'
            f'Observed range: {state.soc_all.min():.6f}% to '
            f'{state.soc_all.max():.6f}%.'
        )


    if not (
        0.0
        <= state.SOC_EMPTY_TOLERANCE_PERCENT
        < state.SOC_FULL_TOLERANCE_PERCENT
        <= 100.0
    ):
        raise ValueError(
            'The SOC empty and full thresholds are invalid.'
        )


    # =============================================================================
    # Construct SOC subsets
    # =============================================================================

    state.near_empty_mask = (
        state.soc_all
        <= state.SOC_EMPTY_TOLERANCE_PERCENT
    )

    state.near_full_mask = (
        state.soc_all
        >= state.SOC_FULL_TOLERANCE_PERCENT
    )

    state.soc_active = state.soc_all.loc[
        ~state.near_empty_mask
    ]


    if state.soc_active.empty:
        raise ValueError(
            'No SOC observations exceed the near-empty tolerance of '
            f'{state.SOC_EMPTY_TOLERANCE_PERCENT:.1f}%.'
        )


    # =============================================================================
    # Calculate descriptive SOC statistics
    # =============================================================================

    state.soc_mean = float(
        state.soc_all.mean()
    )

    state.soc_median = float(
        state.soc_all.median()
    )

    state.soc_active_mean = float(
        state.soc_active.mean()
    )

    state.soc_active_median = float(
        state.soc_active.median()
    )

    state.near_empty_percentage = (
        100.0
        * float(
            state.near_empty_mask.mean()
        )
    )

    state.near_full_percentage = (
        100.0
        * float(
            state.near_full_mask.mean()
        )
    )

    state.active_soc_percentage = (
        100.0
        * len(state.soc_active)
        / len(state.soc_all)
    )


    # =============================================================================
    # Create histogram weights as percentage of observations
    # =============================================================================

    state.all_soc_weights = np.full(
        shape=len(state.soc_all),
        fill_value=100.0 / len(state.soc_all),
    )

    state.active_soc_weights = np.full(
        shape=len(state.soc_active),
        fill_value=100.0 / len(state.soc_active),
    )


    # =============================================================================
    # Create the figure
    # =============================================================================

    state.fig, state.axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(11.5, 4.8),
        sharex=True,
    )


    # =============================================================================
    # Panel (a): Complete SOC distribution
    # =============================================================================

    state.axes[0].hist(
        state.soc_all,
        bins=state.SOC_BIN_EDGES,
        weights=state.all_soc_weights,
        color=state.TEAL,
        edgecolor='white',
        linewidth=0.55,
        alpha=0.88,
    )

    state.axes[0].axvline(
        state.soc_mean,
        color=state.RED,
        linewidth=1.5,
        linestyle='--',
        label=f'Mean: {state.soc_mean:.1f}%',
    )

    state.axes[0].axvline(
        state.soc_median,
        color='#333333',
        linewidth=1.3,
        linestyle=':',
        label=f'Median: {state.soc_median:.1f}%',
    )

    state.axes[0].axvspan(
        0.0,
        state.SOC_EMPTY_TOLERANCE_PERCENT,
        color=state.GRAY,
        alpha=0.15,
        linewidth=0.0,
    )

    state.axes[0].axvspan(
        state.SOC_FULL_TOLERANCE_PERCENT,
        100.0,
        color=state.ORANGE,
        alpha=0.15,
        linewidth=0.0,
    )

    state.axes[0].set_xlabel(
        'Post-action battery SOC (%)'
    )

    state.axes[0].set_ylabel(
        'Share of all transitions (%)'
    )

    state.axes[0].set_title(
        '(a) Complete SOC Occupancy',
        loc='left',
    )

    state.axes[0].legend(
        loc='upper center',
        frameon=True,
        framealpha=0.93,
        facecolor='white',
        edgecolor='#CCCCCC',
    )


    # Add occupancy statistics.
    state.axes[0].text(
        0.97,
        0.96,
        (
            f'Near empty: {state.near_empty_percentage:.1f}%\n'
            f'Near full: {state.near_full_percentage:.1f}%'
        ),
        transform=state.axes[0].transAxes,
        horizontalalignment='right',
        verticalalignment='top',
        fontsize=8.5,
        fontweight='bold',
        bbox={
            'boxstyle': 'round,pad=0.35',
            'facecolor': 'white',
            'edgecolor': state.TEAL,
            'linewidth': 1.0,
            'alpha': 0.95,
        },
    )


    # =============================================================================
    # Panel (b): SOC distribution conditional on active stored energy
    # =============================================================================

    state.axes[1].hist(
        state.soc_active,
        bins=state.SOC_BIN_EDGES,
        weights=state.active_soc_weights,
        color=state.PURPLE,
        edgecolor='white',
        linewidth=0.55,
        alpha=0.88,
    )

    state.axes[1].axvline(
        state.soc_active_mean,
        color=state.RED,
        linewidth=1.5,
        linestyle='--',
        label=f'Mean: {state.soc_active_mean:.1f}%',
    )

    state.axes[1].axvline(
        state.soc_active_median,
        color='#333333',
        linewidth=1.3,
        linestyle=':',
        label=f'Median: {state.soc_active_median:.1f}%',
    )

    state.axes[1].axvspan(
        state.SOC_FULL_TOLERANCE_PERCENT,
        100.0,
        color=state.ORANGE,
        alpha=0.15,
        linewidth=0.0,
    )

    state.axes[1].set_xlabel(
        'Post-action battery SOC (%)'
    )

    state.axes[1].set_ylabel(
        'Share of active-SOC transitions (%)'
    )

    state.axes[1].set_title(
        (
            '(b) Conditional SOC Occupancy '
            f'(SOC > {state.SOC_EMPTY_TOLERANCE_PERCENT:.1f}%)'
        ),
        loc='left',
    )

    state.axes[1].legend(
        loc='upper center',
        frameon=True,
        framealpha=0.93,
        facecolor='white',
        edgecolor='#CCCCCC',
    )


    state.axes[1].text(
        0.97,
        0.96,
        (
            f'Active observations: {len(state.soc_active):,}\n'
            f'Share of episode: {state.active_soc_percentage:.1f}%'
        ),
        transform=state.axes[1].transAxes,
        horizontalalignment='right',
        verticalalignment='top',
        fontsize=8.5,
        fontweight='bold',
        bbox={
            'boxstyle': 'round,pad=0.35',
            'facecolor': 'white',
            'edgecolor': state.PURPLE,
            'linewidth': 1.0,
            'alpha': 0.95,
        },
    )


    # =============================================================================
    # Shared axis formatting
    # =============================================================================

    for state.axis in state.axes:

        state.axis.set_xlim(
            0.0,
            100.0,
        )

        state.axis.xaxis.set_major_locator(
            ticker.MaxNLocator(
                nbins=6,
            )
        )

        state.axis.yaxis.set_major_formatter(
            ticker.PercentFormatter(
                xmax=100.0,
                decimals=0,
            )
        )


    # =============================================================================
    # Figure title and explanatory note
    # =============================================================================

    state.fig.suptitle(
        'Battery State-of-Charge Occupancy Distribution',
        fontsize=13,
        fontweight='bold',
        y=0.99,
    )

    state.fig.text(
        0.5,
        0.01,
        (
            'Near-empty and near-full occupancy are defined as '
            f'SOC ≤ {state.SOC_EMPTY_TOLERANCE_PERCENT:.1f}% and '
            f'SOC ≥ {state.SOC_FULL_TOLERANCE_PERCENT:.1f}%, respectively. '
            'The right panel is conditional on positive stored energy.'
        ),
        horizontalalignment='center',
        fontsize=8.3,
        color='#444444',
    )

    state.fig.tight_layout(
        rect=[
            0.0,
            0.06,
            1.0,
            0.94,
        ],
        w_pad=2.0,
    )


    # =============================================================================
    # Save the figure
    # =============================================================================

    state.save_figure(
        state.fig,
        'plot06_soc_occupancy_distribution',
    )


    # =============================================================================
    # Save the numerical summary
    # =============================================================================

    state.plot06_summary = pd.DataFrame([{
        'soc_column': state.SOC_COLUMN,
        'number_of_transitions': len(state.soc_all),
        'mean_soc_percent': state.soc_mean,
        'median_soc_percent': state.soc_median,
        'minimum_soc_percent': float(
            state.soc_all.min()
        ),
        'maximum_soc_percent': float(
            state.soc_all.max()
        ),
        'near_empty_threshold_percent':
            state.SOC_EMPTY_TOLERANCE_PERCENT,
        'near_empty_transition_count': int(
            state.near_empty_mask.sum()
        ),
        'near_empty_occupancy_percent':
            state.near_empty_percentage,
        'near_full_threshold_percent':
            state.SOC_FULL_TOLERANCE_PERCENT,
        'near_full_transition_count': int(
            state.near_full_mask.sum()
        ),
        'near_full_occupancy_percent':
            state.near_full_percentage,
        'active_soc_transition_count':
            len(state.soc_active),
        'active_soc_occupancy_percent':
            state.active_soc_percentage,
        'active_soc_mean_percent':
            state.soc_active_mean,
        'active_soc_median_percent':
            state.soc_active_median,
    }])


    state.plot06_summary_path = (
        state.TABLES_DIR
        / 'plot06_soc_occupancy_distribution_summary.csv'
    )

    state.plot06_summary.to_csv(
        state.plot06_summary_path,
        index=False,
    )

    display(
        state.plot06_summary
    )

    print(
        'SOC-distribution summary saved to:\n'
        f'{state.plot06_summary_path}'
    )

def cell_08(state):
    """Cell 08: Plot 7 - battery charging/discharging behaviour."""
    # =============================================================================
    # Plot 7: Battery charging and discharging behaviour
    # =============================================================================

    # This figure characterizes realized battery operation over one complete
    # evaluation episode.
    #
    # Panels (a) and (b) show the magnitudes of active charging and
    # discharging events. Panel (c) shows the percentage of transitions
    # classified as charging, discharging, or idle.
    #
    # This is a single-run behavioural diagnostic and does not represent
    # multi-seed uncertainty or battery-degradation performance.


    # =============================================================================
    # Configuration
    # =============================================================================

    state.BATTERY_COLUMN = 'battery_consumption'

    # Realized battery exchanges with absolute magnitude less than or equal
    # to this tolerance are classified as operationally idle.
    state.BATTERY_IDLE_TOLERANCE_KWH = 1e-6

    # Number of common histogram intervals.
    state.NUMBER_OF_BINS = 30

    # State colors
    state.CHARGING_COLOR = state.GREEN
    state.DISCHARGING_COLOR = state.RED
    state.IDLE_COLOR = state.GRAY


    # =============================================================================
    # Validate the battery-consumption column
    # =============================================================================

    if state.BATTERY_COLUMN not in state.df.columns:
        raise KeyError(
            'The charging and discharging plot requires the column '
            f'"{state.BATTERY_COLUMN}", but the column is unavailable.'
        )


    state.battery_exchange = state.df[
        state.BATTERY_COLUMN
    ].astype(float)


    if state.battery_exchange.empty:
        raise ValueError(
            'The battery-consumption series contains no transitions.'
        )


    if not np.isfinite(
        state.battery_exchange.to_numpy()
    ).all():
        raise ValueError(
            f'The column "{state.BATTERY_COLUMN}" contains NaN or '
            'infinite values.'
        )


    if state.BATTERY_IDLE_TOLERANCE_KWH < 0.0:
        raise ValueError(
            'BATTERY_IDLE_TOLERANCE_KWH cannot be negative.'
        )


    # =============================================================================
    # Classify operational states using the numerical deadband
    # =============================================================================

    state.charging_mask = (
        state.battery_exchange
        > state.BATTERY_IDLE_TOLERANCE_KWH
    )

    state.discharging_mask = (
        state.battery_exchange
        < -state.BATTERY_IDLE_TOLERANCE_KWH
    )

    state.idle_mask = (
        state.battery_exchange.abs()
        <= state.BATTERY_IDLE_TOLERANCE_KWH
    )


    # Verify that each transition belongs to exactly one state.
    state.state_membership_count = (
        state.charging_mask.astype(int)
        + state.discharging_mask.astype(int)
        + state.idle_mask.astype(int)
    )

    if not (
        state.state_membership_count == 1
    ).all():
        raise ValueError(
            'Battery operating-state classification failed. '
            'At least one transition was assigned to zero or '
            'multiple operational states.'
        )


    # =============================================================================
    # Extract charging and discharging magnitudes
    # =============================================================================

    state.charging_events = state.battery_exchange.loc[
        state.charging_mask
    ].copy()

    # Convert negative discharge exchanges to positive magnitudes.
    state.discharging_magnitudes = (
        -state.battery_exchange.loc[
            state.discharging_mask
        ]
    ).copy()


    if state.charging_events.empty:
        raise ValueError(
            'No charging events were identified using the selected '
            f'idle tolerance of {state.BATTERY_IDLE_TOLERANCE_KWH:.1e} kWh.'
        )


    if state.discharging_magnitudes.empty:
        raise ValueError(
            'No discharging events were identified using the selected '
            f'idle tolerance of {state.BATTERY_IDLE_TOLERANCE_KWH:.1e} kWh.'
        )


    # =============================================================================
    # Calculate transition counts and occupancy percentages
    # =============================================================================

    state.number_of_transitions = len(
        state.battery_exchange
    )

    state.charging_count = int(
        state.charging_mask.sum()
    )

    state.discharging_count = int(
        state.discharging_mask.sum()
    )

    state.idle_count = int(
        state.idle_mask.sum()
    )


    state.charging_percentage = (
        100.0
        * state.charging_count
        / state.number_of_transitions
    )

    state.discharging_percentage = (
        100.0
        * state.discharging_count
        / state.number_of_transitions
    )

    state.idle_percentage = (
        100.0
        * state.idle_count
        / state.number_of_transitions
    )


    # Confirm that the percentages sum to 100%.
    state.occupancy_percentage_sum = (
        state.charging_percentage
        + state.discharging_percentage
        + state.idle_percentage
    )

    if not np.isclose(
        state.occupancy_percentage_sum,
        100.0,
        atol=1e-8,
        rtol=0.0,
    ):
        raise ValueError(
            'Battery operating-state percentages do not sum to 100%.'
        )


    # =============================================================================
    # Calculate energy and event-magnitude statistics
    # =============================================================================

    state.total_charging_energy = float(
        state.charging_events.sum()
    )

    state.total_discharging_magnitude = float(
        state.discharging_magnitudes.sum()
    )

    state.mean_charging_magnitude = float(
        state.charging_events.mean()
    )

    state.median_charging_magnitude = float(
        state.charging_events.median()
    )

    state.maximum_charging_magnitude = float(
        state.charging_events.max()
    )

    state.mean_discharging_magnitude = float(
        state.discharging_magnitudes.mean()
    )

    state.median_discharging_magnitude = float(
        state.discharging_magnitudes.median()
    )

    state.maximum_discharging_magnitude = float(
        state.discharging_magnitudes.max()
    )


    # =============================================================================
    # Calculate active charge-discharge direction reversals
    # =============================================================================

    # Map every transition to:
    #  1 = charging
    #  0 = idle
    # -1 = discharging

    state.operating_state = np.zeros(
        state.number_of_transitions,
        dtype=int,
    )

    state.operating_state[
        state.charging_mask.to_numpy()
    ] = 1

    state.operating_state[
        state.discharging_mask.to_numpy()
    ] = -1


    # Remove idle observations before calculating direction reversals.
    # Consequently, charging -> idle -> discharging is counted as one
    # active-state reversal rather than two separate sign changes.
    state.active_operating_state = state.operating_state[
        state.operating_state != 0
    ]

    if len(state.active_operating_state) >= 2:

        state.active_direction_reversals = int(
            np.sum(
                state.active_operating_state[1:]
                != state.active_operating_state[:-1]
            )
        )

    else:

        state.active_direction_reversals = 0


    # =============================================================================
    # Construct common histogram bins
    # =============================================================================

    state.maximum_event_magnitude = max(
        state.maximum_charging_magnitude,
        state.maximum_discharging_magnitude,
    )


    if np.isclose(
        state.maximum_event_magnitude,
        0.0,
        atol=1e-12,
    ):
        raise ValueError(
            'The maximum active battery-exchange magnitude is zero.'
        )


    state.common_bin_edges = np.linspace(
        0.0,
        state.maximum_event_magnitude,
        state.NUMBER_OF_BINS + 1,
    )


    # Normalize the histograms to percentage of events.
    state.charging_histogram_weights = np.full(
        shape=len(state.charging_events),
        fill_value=(
            100.0
            / len(state.charging_events)
        ),
    )

    state.discharging_histogram_weights = np.full(
        shape=len(state.discharging_magnitudes),
        fill_value=(
            100.0
            / len(state.discharging_magnitudes)
        ),
    )


    # =============================================================================
    # Create the figure
    # =============================================================================

    state.fig, state.axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(14.0, 4.8),
    )


    # =============================================================================
    # Panel (a): Charging-event magnitude distribution
    # =============================================================================

    state.axes[0].hist(
        state.charging_events,
        bins=state.common_bin_edges,
        weights=state.charging_histogram_weights,
        color=state.CHARGING_COLOR,
        edgecolor='white',
        linewidth=0.55,
        alpha=0.88,
    )

    state.axes[0].axvline(
        state.mean_charging_magnitude,
        color='#222222',
        linewidth=1.45,
        linestyle='--',
        label=(
            f'Mean: '
            f'{state.mean_charging_magnitude:.2f} kWh'
        ),
    )

    state.axes[0].axvline(
        state.median_charging_magnitude,
        color=state.PURPLE,
        linewidth=1.35,
        linestyle=':',
        label=(
            f'Median: '
            f'{state.median_charging_magnitude:.2f} kWh'
        ),
    )

    state.axes[0].set_xlabel(
        'Charging energy per time step (kWh)'
    )

    state.axes[0].set_ylabel(
        'Share of charging events (%)'
    )

    state.axes[0].set_title(
        (
            '(a) Charging-Event Magnitude\n'
            f'{state.charging_count:,} transitions'
        ),
        loc='left',
    )

    state.axes[0].legend(
        loc='upper right',
        frameon=True,
        framealpha=0.93,
        facecolor='white',
        edgecolor='#CCCCCC',
        fontsize=8,
    )


    state.axes[0].text(
        0.97,
        0.72,
        (
            f'Total: {state.total_charging_energy:,.1f} kWh\n'
            f'Maximum: {state.maximum_charging_magnitude:.2f} kWh'
        ),
        transform=state.axes[0].transAxes,
        horizontalalignment='right',
        verticalalignment='top',
        fontsize=8.3,
        fontweight='bold',
        bbox={
            'boxstyle': 'round,pad=0.35',
            'facecolor': 'white',
            'edgecolor': state.CHARGING_COLOR,
            'linewidth': 1.0,
            'alpha': 0.95,
        },
    )


    # =============================================================================
    # Panel (b): Discharging-event magnitude distribution
    # =============================================================================

    state.axes[1].hist(
        state.discharging_magnitudes,
        bins=state.common_bin_edges,
        weights=state.discharging_histogram_weights,
        color=state.DISCHARGING_COLOR,
        edgecolor='white',
        linewidth=0.55,
        alpha=0.88,
    )

    state.axes[1].axvline(
        state.mean_discharging_magnitude,
        color='#222222',
        linewidth=1.45,
        linestyle='--',
        label=(
            f'Mean: '
            f'{state.mean_discharging_magnitude:.2f} kWh'
        ),
    )

    state.axes[1].axvline(
        state.median_discharging_magnitude,
        color=state.PURPLE,
        linewidth=1.35,
        linestyle=':',
        label=(
            f'Median: '
            f'{state.median_discharging_magnitude:.2f} kWh'
        ),
    )

    state.axes[1].set_xlabel(
        'Discharging energy magnitude per time step (kWh)'
    )

    state.axes[1].set_ylabel(
        'Share of discharging events (%)'
    )

    state.axes[1].set_title(
        (
            '(b) Discharging-Event Magnitude\n'
            f'{state.discharging_count:,} transitions'
        ),
        loc='left',
    )

    state.axes[1].legend(
        loc='upper right',
        frameon=True,
        framealpha=0.93,
        facecolor='white',
        edgecolor='#CCCCCC',
        fontsize=8,
    )


    state.axes[1].text(
        0.97,
        0.72,
        (
            f'Total: {state.total_discharging_magnitude:,.1f} kWh\n'
            f'Maximum: {state.maximum_discharging_magnitude:.2f} kWh'
        ),
        transform=state.axes[1].transAxes,
        horizontalalignment='right',
        verticalalignment='top',
        fontsize=8.3,
        fontweight='bold',
        bbox={
            'boxstyle': 'round,pad=0.35',
            'facecolor': 'white',
            'edgecolor': state.DISCHARGING_COLOR,
            'linewidth': 1.0,
            'alpha': 0.95,
        },
    )


    # =============================================================================
    # Panel (c): Operational-state occupancy pie chart
    # =============================================================================

    state.operating_percentages = [
        state.charging_percentage,
        state.discharging_percentage,
        state.idle_percentage,
    ]

    state.operating_labels = [
        'Charging',
        'Discharging',
        'Idle',
    ]

    state.operating_colors = [
        state.CHARGING_COLOR,
        state.DISCHARGING_COLOR,
        state.IDLE_COLOR,
    ]


    state.wedges, state.pie_texts, state.percentage_texts = state.axes[2].pie(
        state.operating_percentages,
        labels=state.operating_labels,
        colors=state.operating_colors,
        autopct='%1.1f%%',
        startangle=90,
        counterclock=False,
        pctdistance=0.72,
        labeldistance=1.06,
        wedgeprops={
            'edgecolor': 'white',
            'linewidth': 1.3,
        },
        textprops={
            'fontsize': 9,
        },
    )


    # Improve the readability of percentage labels.
    for state.percentage_text in state.percentage_texts:

        state.percentage_text.set_fontweight(
            'bold'
        )

        state.percentage_text.set_color(
            'white'
        )


    # Use dark text if the idle segment is too light for white text.
    if len(state.percentage_texts) >= 3:

        state.percentage_texts[2].set_color(
            '#222222'
        )


    state.axes[2].set_title(
        '(c) Operational-State Occupancy',
        loc='left',
    )


    # Add numerical counts below the pie chart.
    state.axes[2].text(
        0.5,
        -0.08,
        (
            f'Charging: {state.charging_count:,} | '
            f'Discharging: {state.discharging_count:,} | '
            f'Idle: {state.idle_count:,}'
        ),
        transform=state.axes[2].transAxes,
        horizontalalignment='center',
        verticalalignment='top',
        fontsize=8.2,
        color='#333333',
    )


    # Keep the pie chart circular.
    state.axes[2].axis(
        'equal'
    )


    # =============================================================================
    # Shared histogram formatting
    # =============================================================================

    for state.axis in state.axes[:2]:

        state.axis.set_xlim(
            0.0,
            state.maximum_event_magnitude
            * 1.02,
        )

        state.axis.xaxis.set_major_locator(
            ticker.MaxNLocator(
                nbins=5,
            )
        )

        state.axis.yaxis.set_major_locator(
            ticker.MaxNLocator(
                nbins=5,
            )
        )

        state.axis.yaxis.set_major_formatter(
            ticker.PercentFormatter(
                xmax=100.0,
                decimals=0,
            )
        )


    # =============================================================================
    # Figure title and methodological note
    # =============================================================================

    state.fig.suptitle(
        'Battery Operating-State Occupancy and Exchange Magnitudes',
        fontsize=13,
        fontweight='bold',
        y=0.99,
    )


    state.fig.text(
        0.5,
        0.01,
        (
            'Positive realized battery exchange is classified as charging; '
            'negative exchange is classified as discharging. Exchanges with '
            f'absolute magnitude ≤ {state.BATTERY_IDLE_TOLERANCE_KWH:.1e} kWh '
            'are classified as idle.'
        ),
        horizontalalignment='center',
        verticalalignment='bottom',
        fontsize=8.2,
        color='#444444',
    )


    state.fig.tight_layout(
        rect=[
            0.0,
            0.07,
            1.0,
            0.94,
        ],
        w_pad=1.8,
    )


    # =============================================================================
    # Save the figure
    # =============================================================================

    state.save_figure(
        state.fig,
        'plot07_battery_charge_discharge_behaviour',
    )


    # =============================================================================
    # Create and save the numerical summary
    # =============================================================================

    state.plot07_summary = pd.DataFrame([{
        'battery_column': state.BATTERY_COLUMN,
        'idle_tolerance_kwh':
            state.BATTERY_IDLE_TOLERANCE_KWH,
        'number_of_transitions':
            state.number_of_transitions,

        'charging_transition_count':
            state.charging_count,
        'charging_transition_percent':
            state.charging_percentage,

        'discharging_transition_count':
            state.discharging_count,
        'discharging_transition_percent':
            state.discharging_percentage,

        'idle_transition_count':
            state.idle_count,
        'idle_transition_percent':
            state.idle_percentage,

        'total_charging_energy_kwh':
            state.total_charging_energy,
        'total_discharge_magnitude_kwh':
            state.total_discharging_magnitude,

        'mean_charging_event_kwh':
            state.mean_charging_magnitude,
        'median_charging_event_kwh':
            state.median_charging_magnitude,
        'maximum_charging_event_kwh':
            state.maximum_charging_magnitude,

        'mean_discharge_event_kwh':
            state.mean_discharging_magnitude,
        'median_discharge_event_kwh':
            state.median_discharging_magnitude,
        'maximum_discharge_event_kwh':
            state.maximum_discharging_magnitude,

        'active_direction_reversals':
            state.active_direction_reversals,
    }])


    state.plot07_summary_path = (
        state.TABLES_DIR
        / 'plot07_battery_charge_discharge_behaviour_summary.csv'
    )

    state.plot07_summary.to_csv(
        state.plot07_summary_path,
        index=False,
    )

    display(
        state.plot07_summary
    )

    print(
        'Battery-operation summary saved to:\n'
        f'{state.plot07_summary_path}'
    )

def cell_09(state):
    """Cell 09: Plot 10 - PV energy allocation."""
    # =============================================================================
    # Plot 10: PV energy allocation and weekly utilization
    # =============================================================================

    # This figure decomposes PV generation into:
    #
    # 1. PV supplied directly to the building load.
    # 2. PV allocated to battery charging during simultaneous PV surplus.
    # 3. Residual PV not self-consumed.
    #
    # The residual PV component is an allocation residual. It must not be
    # interpreted automatically as avoidable export, curtailment, or total
    # grid export.
    #
    # Panel (a) reports the allocation over the complete evaluation horizon.
    # Panel (b) reports the same allocation for complete sequential weeks.
    #
    # This is a single-run diagnostic and does not represent multi-seed
    # uncertainty or statistical significance.


    # =============================================================================
    # Configuration
    # =============================================================================

    state.HOURS_PER_DAY = 24
    state.DAYS_PER_WEEK = 7
    state.HOURS_PER_WEEK = (
        state.HOURS_PER_DAY
        * state.DAYS_PER_WEEK
    )

    state.PV_IDENTITY_ATOL = 1e-8
    state.PV_IDENTITY_RTOL = 1e-6

    # Colors used consistently across both panels.
    state.PV_TO_LOAD_COLOR = state.GREEN
    state.PV_TO_BATTERY_COLOR = state.TEAL
    state.PV_RESIDUAL_COLOR = state.ORANGE


    # =============================================================================
    # Validate required columns
    # =============================================================================

    state.pv_required_columns = [
        'transition',
        'pv_generation',
        'pv_to_load',
        'pv_to_battery',
        'pv_self_consumed',
    ]

    state.pv_missing_columns = [
        state.column
        for state.column in state.pv_required_columns
        if state.column not in state.df.columns
    ]

    if state.pv_missing_columns:
        raise KeyError(
            'The PV-allocation figure cannot be generated because '
            'the following columns are missing:\n'
            f'{state.pv_missing_columns}'
        )


    # =============================================================================
    # Validate numerical values
    # =============================================================================

    state.pv_non_finite_columns = []

    for state.column in state.pv_required_columns:

        state.column_values = state.df[state.column].to_numpy(
            dtype=float
        )

        if not np.isfinite(
            state.column_values
        ).all():
            state.pv_non_finite_columns.append(
                state.column
            )

    if state.pv_non_finite_columns:
        raise ValueError(
            'NaN or infinite values were detected in the following '
            'PV-allocation columns:\n'
            f'{state.pv_non_finite_columns}'
        )


    # =============================================================================
    # Copy only required analysis data
    # =============================================================================

    state.pv_df = state.df[
        state.pv_required_columns
    ].copy()


    # =============================================================================
    # Check non-negativity
    # =============================================================================

    state.pv_nonnegative_columns = [
        'pv_generation',
        'pv_to_load',
        'pv_to_battery',
        'pv_self_consumed',
    ]

    state.negative_value_report = {}

    for state.column in state.pv_nonnegative_columns:

        state.substantial_negative_mask = (
            state.pv_df[state.column]
            < -state.PV_IDENTITY_ATOL
        )

        state.negative_count = int(
            state.substantial_negative_mask.sum()
        )

        if state.negative_count > 0:
            state.negative_value_report[state.column] = (
                state.negative_count
            )

    if state.negative_value_report:
        raise ValueError(
            'Substantial negative values were detected in variables '
            'that should be nonnegative:\n'
            f'{state.negative_value_report}'
        )


    # Set only negligible floating-point negatives to zero.
    for state.column in state.pv_nonnegative_columns:

        state.near_zero_negative_mask = (
            (state.pv_df[state.column] < 0.0)
            & (
                state.pv_df[state.column]
                >= -state.PV_IDENTITY_ATOL
            )
        )

        state.pv_df.loc[
            state.near_zero_negative_mask,
            state.column,
        ] = 0.0


    # =============================================================================
    # Validate the PV self-consumption identity
    # =============================================================================

    state.expected_pv_self_consumed = (
        state.pv_df['pv_to_load']
        + state.pv_df['pv_to_battery']
    )

    state.pv_self_consumption_identity_mask = np.isclose(
        state.pv_df['pv_self_consumed'],
        state.expected_pv_self_consumed,
        atol=state.PV_IDENTITY_ATOL,
        rtol=state.PV_IDENTITY_RTOL,
    )

    if not state.pv_self_consumption_identity_mask.all():

        state.invalid_identity_count = int(
            (
                ~state.pv_self_consumption_identity_mask
            ).sum()
        )

        state.maximum_identity_error = float(
            np.max(
                np.abs(
                    state.pv_df['pv_self_consumed']
                    - state.expected_pv_self_consumed
                )
            )
        )

        raise ValueError(
            'The PV self-consumption identity failed.\n'
            f'Invalid transitions: {state.invalid_identity_count}\n'
            f'Maximum absolute error: '
            f'{state.maximum_identity_error:.10f} kWh'
        )


    # =============================================================================
    # Check that self-consumed PV does not exceed generated PV
    # =============================================================================

    state.pv_exceedance = (
        state.pv_df['pv_self_consumed']
        - state.pv_df['pv_generation']
    )

    state.substantial_exceedance_mask = (
        state.pv_exceedance
        > state.PV_IDENTITY_ATOL
    )

    if state.substantial_exceedance_mask.any():

        state.exceedance_count = int(
            state.substantial_exceedance_mask.sum()
        )

        state.maximum_exceedance = float(
            state.pv_exceedance.max()
        )

        raise ValueError(
            'PV self-consumption exceeds PV generation at one or '
            'more transitions.\n'
            f'Invalid transitions: {state.exceedance_count}\n'
            f'Maximum exceedance: '
            f'{state.maximum_exceedance:.10f} kWh'
        )


    # =============================================================================
    # Calculate residual PV before applying numerical clipping
    # =============================================================================

    state.pv_df['pv_residual_raw'] = (
        state.pv_df['pv_generation']
        - state.pv_df['pv_self_consumed']
    )

    state.minimum_raw_residual = float(
        state.pv_df['pv_residual_raw'].min()
    )

    if state.minimum_raw_residual < -state.PV_IDENTITY_ATOL:
        raise ValueError(
            'The raw residual PV contains a substantial negative '
            'value, indicating an allocation inconsistency.\n'
            f'Minimum raw residual: '
            f'{state.minimum_raw_residual:.10f} kWh'
        )


    # Clip only negligible negative values caused by floating-point arithmetic.
    state.pv_df['pv_residual'] = (
        state.pv_df['pv_residual_raw']
        .clip(
            lower=0.0
        )
    )


    # =============================================================================
    # Validate complete transition-level PV allocation
    # =============================================================================

    state.reconstructed_pv_generation = (
        state.pv_df['pv_to_load']
        + state.pv_df['pv_to_battery']
        + state.pv_df['pv_residual']
    )

    state.pv_allocation_identity_mask = np.isclose(
        state.pv_df['pv_generation'],
        state.reconstructed_pv_generation,
        atol=state.PV_IDENTITY_ATOL,
        rtol=state.PV_IDENTITY_RTOL,
    )

    if not state.pv_allocation_identity_mask.all():

        state.invalid_allocation_count = int(
            (
                ~state.pv_allocation_identity_mask
            ).sum()
        )

        state.maximum_allocation_error = float(
            np.max(
                np.abs(
                    state.pv_df['pv_generation']
                    - state.reconstructed_pv_generation
                )
            )
        )

        raise ValueError(
            'The complete PV-allocation identity failed.\n'
            f'Invalid transitions: {state.invalid_allocation_count}\n'
            f'Maximum absolute error: '
            f'{state.maximum_allocation_error:.10f} kWh'
        )


    # =============================================================================
    # Calculate evaluation-horizon totals
    # =============================================================================

    state.total_pv_generation = float(
        state.pv_df['pv_generation'].sum()
    )

    state.total_pv_to_load = float(
        state.pv_df['pv_to_load'].sum()
    )

    state.total_pv_to_battery = float(
        state.pv_df['pv_to_battery'].sum()
    )

    state.total_pv_self_consumed = float(
        state.pv_df['pv_self_consumed'].sum()
    )

    state.total_pv_residual = float(
        state.pv_df['pv_residual'].sum()
    )


    if np.isclose(
        state.total_pv_generation,
        0.0,
        atol=state.PV_IDENTITY_ATOL,
    ):
        raise ValueError(
            'Total PV generation is zero. '
            'PV allocation percentages cannot be calculated.'
        )


    # =============================================================================
    # Calculate allocation percentages
    # =============================================================================

    state.pv_to_load_percentage = (
        100.0
        * state.total_pv_to_load
        / state.total_pv_generation
    )

    state.pv_to_battery_percentage = (
        100.0
        * state.total_pv_to_battery
        / state.total_pv_generation
    )

    state.pv_residual_percentage = (
        100.0
        * state.total_pv_residual
        / state.total_pv_generation
    )

    state.pv_self_consumption_rate = (
        100.0
        * state.total_pv_self_consumed
        / state.total_pv_generation
    )

    state.allocation_percentage_sum = (
        state.pv_to_load_percentage
        + state.pv_to_battery_percentage
        + state.pv_residual_percentage
    )

    if not np.isclose(
        state.allocation_percentage_sum,
        100.0,
        atol=1e-6,
        rtol=0.0,
    ):
        raise ValueError(
            'PV allocation percentages do not sum to 100%.\n'
            f'Observed sum: {state.allocation_percentage_sum:.8f}%'
        )


    # =============================================================================
    # Construct sequential week identifiers
    # =============================================================================

    state.pv_df['evaluation_week'] = (
        state.pv_df['transition']
        // state.HOURS_PER_WEEK
    ) + 1

    state.weekly_transition_counts = (
        state.pv_df.groupby(
            'evaluation_week'
        )
        .size()
        .rename(
            'number_of_transitions'
        )
    )

    state.complete_week_numbers = (
        state.weekly_transition_counts.loc[
            state.weekly_transition_counts
            == state.HOURS_PER_WEEK
        ]
        .index
        .tolist()
    )

    state.partial_week_numbers = (
        state.weekly_transition_counts.loc[
            state.weekly_transition_counts
            != state.HOURS_PER_WEEK
        ]
        .index
        .tolist()
    )

    if not state.complete_week_numbers:
        raise ValueError(
            'No complete 168-transition evaluation weeks were found.'
        )

    if state.partial_week_numbers:
        print(
            'Partial sequential weeks excluded from Panel (b):',
            state.partial_week_numbers,
        )


    # =============================================================================
    # Aggregate complete sequential weeks
    # =============================================================================

    state.weekly_pv_allocation = (
        state.pv_df.loc[
            state.pv_df['evaluation_week']
            .isin(
                state.complete_week_numbers
            )
        ]
        .groupby(
            'evaluation_week',
            as_index=False,
        )[
            [
                'pv_generation',
                'pv_to_load',
                'pv_to_battery',
                'pv_self_consumed',
                'pv_residual',
            ]
        ]
        .sum()
    )


    state.weekly_pv_allocation[
        'pv_self_consumption_rate_percent'
    ] = (
        100.0
        * state.weekly_pv_allocation[
            'pv_self_consumed'
        ]
        / state.weekly_pv_allocation[
            'pv_generation'
        ]
    )


    # =============================================================================
    # Validate weekly PV allocation
    # =============================================================================

    state.weekly_reconstructed_pv = (
        state.weekly_pv_allocation[
            'pv_to_load'
        ]
        + state.weekly_pv_allocation[
            'pv_to_battery'
        ]
        + state.weekly_pv_allocation[
            'pv_residual'
        ]
    )

    state.weekly_identity_mask = np.isclose(
        state.weekly_pv_allocation[
            'pv_generation'
        ],
        state.weekly_reconstructed_pv,
        atol=state.PV_IDENTITY_ATOL,
        rtol=state.PV_IDENTITY_RTOL,
    )

    if not state.weekly_identity_mask.all():
        raise ValueError(
            'The weekly PV-allocation identity failed.'
        )


    # =============================================================================
    # Create the two-panel figure
    # =============================================================================

    state.fig, state.axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(14.5, 5.5),
        gridspec_kw={
            'width_ratios': [
                0.9,
                1.7,
            ],
        },
    )


    # =============================================================================
    # Panel (a): Evaluation-horizon PV allocation donut chart
    # =============================================================================

    state.allocation_values = [
        state.total_pv_to_load,
        state.total_pv_to_battery,
        state.total_pv_residual,
    ]

    state.allocation_labels = [
        'Direct to building load',
        'Allocated to battery',
        'Residual PV',
    ]

    state.allocation_colors = [
        state.PV_TO_LOAD_COLOR,
        state.PV_TO_BATTERY_COLOR,
        state.PV_RESIDUAL_COLOR,
    ]


    state.wedges, state.pie_labels, state.percentage_labels = state.axes[0].pie(
        state.allocation_values,
        labels=state.allocation_labels,
        colors=state.allocation_colors,
        autopct='%1.1f%%',
        startangle=90,
        counterclock=False,
        pctdistance=0.76,
        labeldistance=1.08,
        wedgeprops={
            'edgecolor': 'white',
            'linewidth': 1.3,
            'width': 0.42,
        },
        textprops={
            'fontsize': 8.5,
        },
    )


    # Format percentage text inside the donut.
    for state.percentage_text in state.percentage_labels:

        state.percentage_text.set_fontweight(
            'bold'
        )

        state.percentage_text.set_color(
            'white'
        )


    # Add total PV generation in the centre.
    state.axes[0].text(
        0.0,
        0.04,
        f'{state.total_pv_generation:,.1f}',
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=13,
        fontweight='bold',
        color='#222222',
    )

    state.axes[0].text(
        0.0,
        -0.14,
        'Total PV\n(kWh)',
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=8.5,
        color='#444444',
    )


    state.axes[0].set_title(
        '(a) PV Allocation Over the Evaluation Horizon',
        loc='left',
        pad=12,
    )

    state.axes[0].axis(
        'equal'
    )


    # Add exact absolute allocation values.
    state.axes[0].text(
        0.50,
        -0.12,
        (
            f'Direct to load: '
            f'{state.total_pv_to_load:,.1f} kWh\n'
            f'To battery: '
            f'{state.total_pv_to_battery:,.1f} kWh\n'
            f'Residual PV: '
            f'{state.total_pv_residual:,.1f} kWh\n'
            f'Self-consumption rate: '
            f'{state.pv_self_consumption_rate:.1f}%'
        ),
        transform=state.axes[0].transAxes,
        horizontalalignment='center',
        verticalalignment='top',
        fontsize=8.4,
        bbox={
            'boxstyle': 'round,pad=0.40',
            'facecolor': 'white',
            'edgecolor': '#CCCCCC',
            'linewidth': 1.0,
            'alpha': 0.96,
        },
    )


    # =============================================================================
    # Panel (b): Weekly stacked PV allocation
    # =============================================================================

    state.weekly_x = np.arange(
        len(
            state.weekly_pv_allocation
        )
    )

    state.weekly_to_load = state.weekly_pv_allocation[
        'pv_to_load'
    ].to_numpy()

    state.weekly_to_battery = state.weekly_pv_allocation[
        'pv_to_battery'
    ].to_numpy()

    state.weekly_residual = state.weekly_pv_allocation[
        'pv_residual'
    ].to_numpy()


    state.axes[1].bar(
        state.weekly_x,
        state.weekly_to_load,
        width=0.82,
        color=state.PV_TO_LOAD_COLOR,
        alpha=0.90,
        label='Direct to building load',
        zorder=3,
    )

    state.axes[1].bar(
        state.weekly_x,
        state.weekly_to_battery,
        width=0.82,
        bottom=state.weekly_to_load,
        color=state.PV_TO_BATTERY_COLOR,
        alpha=0.90,
        label='Allocated to battery',
        zorder=3,
    )

    state.axes[1].bar(
        state.weekly_x,
        state.weekly_residual,
        width=0.82,
        bottom=(
            state.weekly_to_load
            + state.weekly_to_battery
        ),
        color=state.PV_RESIDUAL_COLOR,
        alpha=0.85,
        label='Residual PV not self-consumed',
        zorder=3,
    )


    state.axes[1].set_xlabel(
        'Sequential complete evaluation week'
    )

    state.axes[1].set_ylabel(
        'Weekly PV energy allocation (kWh)'
    )

    state.axes[1].set_title(
        '(b) Sequential Weekly PV Allocation',
        loc='left',
        pad=12,
    )


    state.axes[1].legend(
        loc='upper center',
        ncol=1,
        frameon=True,
        framealpha=0.94,
        facecolor='white',
        edgecolor='#CCCCCC',
        fontsize=8,
    )


    # Show a manageable number of weekly labels.
    state.number_of_complete_weeks = len(
        state.weekly_pv_allocation
    )

    state.weekly_tick_step = max(
        1,
        int(
            np.ceil(
                state.number_of_complete_weeks
                / 13.0
            )
        ),
    )

    state.weekly_tick_positions = np.arange(
        0,
        state.number_of_complete_weeks,
        state.weekly_tick_step,
    )

    state.weekly_tick_labels = (
        state.weekly_pv_allocation[
            'evaluation_week'
        ]
        .iloc[
            state.weekly_tick_positions
        ]
        .astype(int)
        .astype(str)
        .tolist()
    )


    state.axes[1].set_xticks(
        state.weekly_tick_positions
    )

    state.axes[1].set_xticklabels(
        state.weekly_tick_labels
    )


    state.axes[1].xaxis.set_major_locator(
        ticker.MaxNLocator(
            nbins=13,
            integer=True,
        )
    )

    state.axes[1].yaxis.set_major_locator(
        ticker.MaxNLocator(
            nbins=6,
        )
    )

    state.axes[1].yaxis.set_major_formatter(
        ticker.StrMethodFormatter(
            '{x:,.0f}'
        )
    )


    # =============================================================================
    # Figure title and explanatory note
    # =============================================================================

    state.fig.suptitle(
        'PV Energy Allocation and Temporal Utilization',
        fontsize=13,
        fontweight='bold',
        y=0.99,
    )


    state.fig.text(
        0.5,
        0.01,
        (
            'PV allocated to battery charging is inferred from simultaneous '
            'positive battery charging and available PV surplus. Residual PV '
            'is not automatically equivalent to avoidable export, total grid '
            'export, or curtailed energy.'
        ),
        horizontalalignment='center',
        verticalalignment='bottom',
        fontsize=8.2,
        color='#444444',
    )


    state.fig.tight_layout(
        rect=[
            0.0,
            0.075,
            1.0,
            0.94,
        ],
        w_pad=2.3,
    )


    # =============================================================================
    # Save the figure
    # =============================================================================

    state.save_figure(
        state.fig,
        'plot10_pv_energy_allocation',
    )


    # =============================================================================
    # Create and save evaluation-horizon summary
    # =============================================================================

    state.plot10_summary = pd.DataFrame([{
        'number_of_transitions': len(
            state.pv_df
        ),
        'total_pv_generation_kwh':
            state.total_pv_generation,
        'pv_to_load_kwh':
            state.total_pv_to_load,
        'pv_to_load_percent':
            state.pv_to_load_percentage,
        'pv_to_battery_kwh':
            state.total_pv_to_battery,
        'pv_to_battery_percent':
            state.pv_to_battery_percentage,
        'pv_self_consumed_kwh':
            state.total_pv_self_consumed,
        'pv_self_consumption_rate_percent':
            state.pv_self_consumption_rate,
        'residual_pv_kwh':
            state.total_pv_residual,
        'residual_pv_percent':
            state.pv_residual_percentage,
        'minimum_raw_residual_kwh':
            state.minimum_raw_residual,
        'number_of_complete_weeks':
            len(
                state.complete_week_numbers
            ),
        'number_of_partial_weeks':
            len(
                state.partial_week_numbers
            ),
        'partial_week_numbers':
            ', '.join([
                str(week_number)
                for week_number in state.partial_week_numbers
            ]),
    }])


    state.plot10_summary_path = (
        state.TABLES_DIR
        / 'plot10_pv_energy_allocation_summary.csv'
    )

    state.plot10_weekly_path = (
        state.TABLES_DIR
        / 'plot10_weekly_pv_energy_allocation.csv'
    )


    state.plot10_summary.to_csv(
        state.plot10_summary_path,
        index=False,
    )

    state.weekly_pv_allocation.to_csv(
        state.plot10_weekly_path,
        index=False,
    )


    display(
        state.plot10_summary
    )

    display(
        state.weekly_pv_allocation.head()
    )


    print(
        'PV-allocation summary saved to:\n'
        f'{state.plot10_summary_path}'
    )

    print(
        'Weekly PV-allocation table saved to:\n'
        f'{state.plot10_weekly_path}'
    )

def cell_10(state):
    """Cell 10: Plot 12 - weighted reward/penalty contributions."""
    # =============================================================================
    # Plot 12: Realized weighted reward-penalty contributions
    # =============================================================================

    # This figure explains how the three weighted normalized objectives
    # contribute to the scalar reward for one evaluated configuration.
    #
    # Panel (a) shows complete-window trailing 168-hour mean penalties.
    # Panel (b) compares nominal reward weights with realized accumulated
    # penalty shares over the complete evaluation episode.
    #
    # Cross-configuration physical performance must still be evaluated using
    # unweighted avoidable export, electricity cost, and carbon emissions.
    #
    # The terminal SOC penalty is reported separately because it is an
    # episode-boundary fairness condition rather than a physical objective.

    state.TIME_COLUMN = 'reward_time_step'
    state.ROLLING_WINDOW_HOURS = 168

    state.COMPONENT_CONFIGURATION = [
        {
            'name': 'Avoidable export',
            'normalized_column': 'normalized_grid',
            'weighted_column': 'weighted_grid',
            'color': state.COLOR_GRID_EXPORT,
        },
        {
            'name': 'Electricity cost',
            'normalized_column': 'normalized_cost',
            'weighted_column': 'weighted_cost',
            'color': state.ORANGE,
        },
        {
            'name': 'Carbon emissions',
            'normalized_column': 'normalized_emission',
            'weighted_column': 'weighted_emission',
            'color': state.COLOR_CARBON,
        },
    ]

    state.TERMINAL_COLUMN = 'terminal_soc_penalty'

    state.IDENTITY_ATOL = 1e-8
    state.IDENTITY_RTOL = 1e-6


    # =============================================================================
    # Validate required columns
    # =============================================================================

    state.required_component_columns = [
        state.TIME_COLUMN,
        'reward',
        state.TERMINAL_COLUMN,
    ]

    for state.component in state.COMPONENT_CONFIGURATION:
        state.required_component_columns.append(
            state.component['normalized_column']
        )
        state.required_component_columns.append(
            state.component['weighted_column']
        )


    state.missing_component_columns = [
        state.column
        for state.column in state.required_component_columns
        if state.column not in state.df.columns
    ]

    if state.missing_component_columns:
        raise KeyError(
            'The reward-component figure cannot be generated because '
            'the following columns are missing:\n'
            f'{state.missing_component_columns}'
        )


    # =============================================================================
    # Validate numerical data
    # =============================================================================

    state.non_finite_component_columns = []

    for state.column in state.required_component_columns:

        state.column_values = state.df[state.column].to_numpy(
            dtype=float
        )

        if not np.isfinite(
            state.column_values
        ).all():
            state.non_finite_component_columns.append(
                state.column
            )

    if state.non_finite_component_columns:
        raise ValueError(
            'NaN or infinite values were detected in the following '
            'reward-component columns:\n'
            f'{state.non_finite_component_columns}'
        )


    if not state.df[state.TIME_COLUMN].is_monotonic_increasing:
        raise ValueError(
            f'The column "{state.TIME_COLUMN}" is not monotonically increasing.'
        )


    if state.df[state.TIME_COLUMN].duplicated().any():
        raise ValueError(
            f'The column "{state.TIME_COLUMN}" contains duplicated values.'
        )


    if state.ROLLING_WINDOW_HOURS <= 0:
        raise ValueError(
            'ROLLING_WINDOW_HOURS must be greater than zero.'
        )


    if len(state.df) < state.ROLLING_WINDOW_HOURS:
        raise ValueError(
            f'The timeseries contains {len(state.df)} transitions, which is '
            f'shorter than the requested rolling window of '
            f'{state.ROLLING_WINDOW_HOURS} transitions.'
        )


    # =============================================================================
    # Validate weighted-penalty non-negativity
    # =============================================================================

    state.negative_component_report = {}

    for state.component in state.COMPONENT_CONFIGURATION:

        state.weighted_column = state.component[
            'weighted_column'
        ]

        state.negative_mask = (
            state.df[state.weighted_column]
            < -state.IDENTITY_ATOL
        )

        state.negative_count = int(
            state.negative_mask.sum()
        )

        if state.negative_count > 0:
            state.negative_component_report[
                state.weighted_column
            ] = state.negative_count


    if state.negative_component_report:
        raise ValueError(
            'Negative weighted penalty values were detected. '
            'Review the reward accounting before plotting:\n'
            f'{state.negative_component_report}'
        )


    # =============================================================================
    # Infer and validate nominal reward weights
    # =============================================================================

    def infer_constant_weight(
        normalized_values,
        weighted_values,
        component_name,
        tolerance=1e-12,
    ):
        """
        Infer one nominal reward weight from weighted and normalized values.

        Rows with normalized values near zero are excluded because their
        weighted-to-normalized ratio is numerically undefined.
        """

        valid_mask = (
            normalized_values.abs()
            > tolerance
        )

        if not valid_mask.any():
            raise ValueError(
                f'The nominal weight for "{component_name}" cannot be '
                'inferred because all normalized values are zero.'
            )

        inferred_ratios = (
            weighted_values.loc[
                valid_mask
            ]
            / normalized_values.loc[
                valid_mask
            ]
        )

        inferred_weight = float(
            inferred_ratios.median()
        )

        constant_weight_mask = np.isclose(
            inferred_ratios,
            inferred_weight,
            atol=state.IDENTITY_ATOL,
            rtol=state.IDENTITY_RTOL,
        )

        if not constant_weight_mask.all():

            maximum_weight_deviation = float(
                np.max(
                    np.abs(
                        inferred_ratios
                        - inferred_weight
                    )
                )
            )

            raise ValueError(
                f'The inferred weight for "{component_name}" is not '
                'constant across the evaluation episode.\n'
                f'Maximum deviation: {maximum_weight_deviation:.10f}'
            )

        if (
            inferred_weight < -state.IDENTITY_ATOL
            or inferred_weight > 1.0 + state.IDENTITY_ATOL
        ):
            raise ValueError(
                f'The inferred weight for "{component_name}" is outside '
                f'the expected interval [0, 1]: {inferred_weight:.8f}'
            )

        return inferred_weight
    state.infer_constant_weight = infer_constant_weight


    state.nominal_weights = []

    for state.component in state.COMPONENT_CONFIGURATION:

        state.inferred_weight = state.infer_constant_weight(
            normalized_values=state.df[
                state.component['normalized_column']
            ].astype(float),
            weighted_values=state.df[
                state.component['weighted_column']
            ].astype(float),
            component_name=state.component['name'],
        )

        state.nominal_weights.append(
            state.inferred_weight
        )


    state.nominal_weights = np.asarray(
        state.nominal_weights,
        dtype=float,
    )

    state.nominal_weight_sum = float(
        state.nominal_weights.sum()
    )


    if not np.isclose(
        state.nominal_weight_sum,
        1.0,
        atol=1e-6,
        rtol=0.0,
    ):
        raise ValueError(
            'The inferred nominal reward weights do not sum to one.\n'
            f'Observed sum: {state.nominal_weight_sum:.10f}'
        )


    # =============================================================================
    # Validate weighted-component identities
    # =============================================================================

    for state.component, state.nominal_weight in zip(
        state.COMPONENT_CONFIGURATION,
        state.nominal_weights,
    ):

        state.expected_weighted_component = (
            state.nominal_weight
            * state.df[
                state.component['normalized_column']
            ]
        )

        state.component_identity_mask = np.isclose(
            state.df[
                state.component['weighted_column']
            ],
            state.expected_weighted_component,
            atol=state.IDENTITY_ATOL,
            rtol=state.IDENTITY_RTOL,
        )

        if not state.component_identity_mask.all():

            state.invalid_component_rows = int(
                (
                    ~state.component_identity_mask
                ).sum()
            )

            state.maximum_component_error = float(
                np.max(
                    np.abs(
                        state.df[
                            state.component['weighted_column']
                        ]
                        - state.expected_weighted_component
                    )
                )
            )

            raise ValueError(
                f'The weighted-component identity failed for '
                f'"{state.component["name"]}".\n'
                f'Invalid transitions: {state.invalid_component_rows}\n'
                f'Maximum absolute error: '
                f'{state.maximum_component_error:.10f}'
            )


    # =============================================================================
    # Validate the complete reward identity
    # =============================================================================

    state.weighted_penalty_sum = (
        state.df['weighted_grid']
        + state.df['weighted_cost']
        + state.df['weighted_emission']
    )

    state.expected_reward = (
        -state.weighted_penalty_sum
        + state.df[state.TERMINAL_COLUMN]
    )

    state.reward_identity_mask = np.isclose(
        state.df['reward'],
        state.expected_reward,
        atol=state.IDENTITY_ATOL,
        rtol=state.IDENTITY_RTOL,
    )

    if not state.reward_identity_mask.all():

        state.invalid_reward_rows = int(
            (
                ~state.reward_identity_mask
            ).sum()
        )

        state.maximum_reward_error = float(
            np.max(
                np.abs(
                    state.df['reward']
                    - state.expected_reward
                )
            )
        )

        raise ValueError(
            'The reward identity failed before plotting.\n'
            f'Invalid transitions: {state.invalid_reward_rows}\n'
            f'Maximum absolute error: {state.maximum_reward_error:.10f}'
        )


    # =============================================================================
    # Calculate complete-window rolling penalty contributions
    # =============================================================================

    state.time_values = state.df[
        state.TIME_COLUMN
    ].astype(float)


    state.rolling_component_data = {}

    for state.component in state.COMPONENT_CONFIGURATION:

        state.weighted_column = state.component[
            'weighted_column'
        ]

        state.rolling_component_data[
            state.weighted_column
        ] = (
            state.df[state.weighted_column]
            .rolling(
                window=state.ROLLING_WINDOW_HOURS,
                min_periods=state.ROLLING_WINDOW_HOURS,
            )
            .mean()
        )


    state.rolling_component_df = pd.DataFrame(
        state.rolling_component_data
    )

    state.valid_rolling_mask = (
        state.rolling_component_df
        .notna()
        .all(
            axis=1
        )
    )

    if not state.valid_rolling_mask.any():
        raise ValueError(
            'No complete rolling windows were generated.'
        )


    state.rolling_time_values = state.time_values.loc[
        state.valid_rolling_mask
    ]

    state.rolling_component_df = (
        state.rolling_component_df.loc[
            state.valid_rolling_mask
        ]
    )


    # =============================================================================
    # Calculate episode-total realized contributions
    # =============================================================================

    state.episode_component_totals = []

    for state.component in state.COMPONENT_CONFIGURATION:

        state.episode_component_totals.append(
            float(
                state.df[
                    state.component['weighted_column']
                ].sum()
            )
        )


    state.episode_component_totals = np.asarray(
        state.episode_component_totals,
        dtype=float,
    )

    state.total_weighted_penalty = float(
        state.episode_component_totals.sum()
    )


    if np.isclose(
        state.total_weighted_penalty,
        0.0,
        atol=1e-12,
    ):
        raise ValueError(
            'The accumulated weighted penalty is zero, so realized '
            'contribution shares cannot be calculated.'
        )


    state.realized_contribution_shares = (
        100.0
        * state.episode_component_totals
        / state.total_weighted_penalty
    )

    state.nominal_weight_percentages = (
        100.0
        * state.nominal_weights
    )


    if not np.isclose(
        state.realized_contribution_shares.sum(),
        100.0,
        atol=1e-6,
        rtol=0.0,
    ):
        raise ValueError(
            'The realized contribution shares do not sum to 100%.'
        )


    state.terminal_soc_penalty_total = float(
        state.df[state.TERMINAL_COLUMN].sum()
    )

    state.accumulated_reward = float(
        state.df['reward'].sum()
    )

    state.reconstructed_accumulated_reward = (
        -state.total_weighted_penalty
        + state.terminal_soc_penalty_total
    )


    if not np.isclose(
        state.accumulated_reward,
        state.reconstructed_accumulated_reward,
        atol=1e-6,
        rtol=state.IDENTITY_RTOL,
    ):
        raise ValueError(
            'The accumulated reward does not match the accumulated '
            'component reconstruction.'
        )


    # =============================================================================
    # Create two-panel manuscript figure
    # =============================================================================

    state.fig, state.axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(14.0, 5.2),
        gridspec_kw={
            'width_ratios': [
                1.8,
                1.0,
            ],
        },
    )


    # =============================================================================
    # Panel (a): Rolling weighted penalty contributions
    # =============================================================================

    for state.component in state.COMPONENT_CONFIGURATION:

        state.weighted_column = state.component[
            'weighted_column'
        ]

        state.axes[0].plot(
            state.rolling_time_values,
            state.rolling_component_df[
                state.weighted_column
            ],
            color=state.component['color'],
            linewidth=1.6,
            label=state.component['name'],
            zorder=3,
        )


    state.axes[0].set_xlabel(
        'Reward time step (hour)'
    )

    state.axes[0].set_ylabel(
        'Weighted normalized penalty\n'
        '(trailing 168-hour mean)'
    )

    state.axes[0].set_title(
        '(a) Temporal Weighted-Penalty Contributions',
        loc='left',
        pad=9,
    )


    state.axes[0].xaxis.set_major_locator(
        ticker.MaxNLocator(
            nbins=7,
            integer=True,
        )
    )

    state.axes[0].xaxis.set_major_formatter(
        ticker.StrMethodFormatter(
            '{x:,.0f}'
        )
    )

    state.axes[0].yaxis.set_major_locator(
        ticker.MaxNLocator(
            nbins=6,
        )
    )


    state.axes[0].legend(
        loc='upper right',
        ncol=1,
        frameon=True,
        framealpha=0.94,
        facecolor='white',
        edgecolor='#CCCCCC',
        fontsize=8.5,
    )


    state.axes[0].margins(
        x=0.015,
        y=0.08,
    )


    # Add nominal weights for context.
    state.nominal_weight_lines = [
        'Nominal weights'
    ]

    for state.component, state.nominal_weight in zip(
        state.COMPONENT_CONFIGURATION,
        state.nominal_weights,
    ):
        state.nominal_weight_lines.append(
            f'{state.component["name"]}: {state.nominal_weight:.2f}'
        )

    state.nominal_weight_text = '\n'.join(
        state.nominal_weight_lines
    )


    state.axes[0].text(
        0.02,
        0.96,
        state.nominal_weight_text,
        transform=state.axes[0].transAxes,
        horizontalalignment='left',
        verticalalignment='top',
        fontsize=8.4,
        fontweight='bold',
        linespacing=1.15,
        bbox={
            'boxstyle': 'round,pad=0.38',
            'facecolor': 'white',
            'edgecolor': '#777777',
            'linewidth': 1.0,
            'alpha': 0.95,
        },
        zorder=5,
    )


    # =============================================================================
    # Panel (b): Nominal weights versus realized contribution shares
    # =============================================================================

    state.component_names = [
        state.component['name']
        for state.component in state.COMPONENT_CONFIGURATION
    ]

    state.component_colors = [
        state.component['color']
        for state.component in state.COMPONENT_CONFIGURATION
    ]


    state.comparison_rows = [
        'Nominal weights',
        'Realized contributions',
    ]

    state.comparison_values = [
        state.nominal_weight_percentages,
        state.realized_contribution_shares,
    ]

    state.bar_y_positions = [
        1,
        0,
    ]


    for state.row_position, state.row_values in zip(
        state.bar_y_positions,
        state.comparison_values,
    ):

        state.cumulative_left = 0.0

        for state.component_index, (
            state.component_name,
            state.component_color,
        ) in enumerate(
            zip(
                state.component_names,
                state.component_colors,
            )
        ):

            state.component_value = float(
                state.row_values[
                    state.component_index
                ]
            )

            state.axes[1].barh(
                y=state.row_position,
                width=state.component_value,
                left=state.cumulative_left,
                height=0.55,
                color=state.component_color,
                edgecolor='white',
                linewidth=1.0,
                zorder=3,
            )

            # Show the percentage inside the segment when space permits.
            if state.component_value >= 6.0:

                state.axes[1].text(
                    state.cumulative_left
                    + state.component_value / 2.0,
                    state.row_position,
                    f'{state.component_value:.1f}%',
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=8.5,
                    fontweight='bold',
                    color='white',
                    zorder=5,
                )

            state.cumulative_left += state.component_value


    state.axes[1].set_xlim(
        0.0,
        100.0,
    )

    state.axes[1].set_ylim(
        -0.65,
        1.65,
    )

    state.axes[1].set_yticks(
        state.bar_y_positions
    )

    state.axes[1].set_yticklabels(
        state.comparison_rows
    )

    state.axes[1].set_xlabel(
        'Contribution share (%)'
    )

    state.axes[1].set_title(
        '(b) Nominal versus Realized Contributions',
        loc='left',
        pad=9,
    )


    state.axes[1].xaxis.set_major_locator(
        ticker.MaxNLocator(
            nbins=5,
        )
    )

    state.axes[1].xaxis.set_major_formatter(
        ticker.PercentFormatter(
            xmax=100.0,
            decimals=0,
        )
    )


    # Component legend for the horizontal bars.
    state.component_legend_handles = [
        mpatches.Patch(
            facecolor=state.component['color'],
            edgecolor='white',
            label=state.component['name'],
        )
        for state.component in state.COMPONENT_CONFIGURATION
    ]


    state.axes[1].legend(
        handles=state.component_legend_handles,
        loc='lower center',
        bbox_to_anchor=(
            0.5,
            -0.34,
        ),
        ncol=1,
        frameon=True,
        framealpha=0.94,
        facecolor='white',
        edgecolor='#CCCCCC',
        fontsize=8.2,
    )


    # Add accumulated penalty and terminal-condition information.
    state.episode_summary_text = (
        f'Total weighted penalty: '
        f'{state.total_weighted_penalty:,.2f}\n'
        f'Terminal SOC penalty: '
        f'{state.terminal_soc_penalty_total:,.2f}\n'
        f'Accumulated reward: '
        f'{state.accumulated_reward:,.2f}'
    )


    state.axes[1].text(
        0.98,
        0.04,
        state.episode_summary_text,
        transform=state.axes[1].transAxes,
        horizontalalignment='right',
        verticalalignment='bottom',
        fontsize=8.2,
        fontweight='bold',
        linespacing=1.15,
        bbox={
            'boxstyle': 'round,pad=0.38',
            'facecolor': 'white',
            'edgecolor': '#777777',
            'linewidth': 1.0,
            'alpha': 0.95,
        },
        zorder=6,
    )


    # =============================================================================
    # Figure-level title and interpretation note
    # =============================================================================

    state.fig.suptitle(
        'Realized Weighted Reward-Penalty Contributions',
        fontsize=13,
        fontweight='bold',
        y=0.99,
    )


    state.fig.text(
        0.5,
        0.01,
        (
            'Realized contribution shares depend jointly on nominal weights '
            'and normalized objective magnitudes. The terminal SOC penalty '
            'is reported separately and excluded from the three-objective '
            'contribution shares.'
        ),
        horizontalalignment='center',
        verticalalignment='bottom',
        fontsize=8.2,
        color='#444444',
    )


    state.fig.tight_layout(
        rect=[
            0.0,
            0.095,
            1.0,
            0.94,
        ],
        w_pad=2.4,
    )


    # =============================================================================
    # Save figure
    # =============================================================================

    state.save_figure(
        state.fig,
        'plot12_weighted_reward_penalty_contributions',
    )


    # =============================================================================
    # Create and save numerical summary table
    # =============================================================================

    state.plot12_summary_records = []

    for state.component_index, state.component in enumerate(
        state.COMPONENT_CONFIGURATION
    ):

        state.plot12_summary_records.append({
            'component': state.component['name'],
            'normalized_column':
                state.component['normalized_column'],
            'weighted_column':
                state.component['weighted_column'],
            'nominal_weight':
                state.nominal_weights[state.component_index],
            'nominal_weight_percent':
                state.nominal_weight_percentages[state.component_index],
            'accumulated_weighted_penalty':
                state.episode_component_totals[state.component_index],
            'realized_contribution_percent':
                state.realized_contribution_shares[state.component_index],
            'difference_realized_minus_nominal_percentage_points':
                (
                    state.realized_contribution_shares[state.component_index]
                    - state.nominal_weight_percentages[state.component_index]
                ),
        })


    state.plot12_summary = pd.DataFrame(
        state.plot12_summary_records
    )


    # Add episode-level quantities to each row for traceability.
    state.plot12_summary[
        'rolling_window_hours'
    ] = state.ROLLING_WINDOW_HOURS

    state.plot12_summary[
        'number_of_transitions'
    ] = len(state.df)

    state.plot12_summary[
        'total_weighted_penalty'
    ] = state.total_weighted_penalty

    state.plot12_summary[
        'terminal_soc_penalty'
    ] = state.terminal_soc_penalty_total

    state.plot12_summary[
        'accumulated_reward'
    ] = state.accumulated_reward


    state.plot12_summary_path = (
        state.TABLES_DIR
        / 'plot12_weighted_reward_penalty_contributions_summary.csv'
    )


    state.plot12_summary.to_csv(
        state.plot12_summary_path,
        index=False,
    )


    display(
        state.plot12_summary
    )


    print(
        'Reward-contribution summary saved to:\n'
        f'{state.plot12_summary_path}'
    )

def cell_11(state):
    """Cell 11: Plot 14 - aggregate daily operating profiles."""
    # =============================================================================
    # Plot 14: Aggregate daily operating profiles
    # =============================================================================

    # This figure summarizes recurring 24-hour operating patterns over the
    # complete evaluation horizon.
    #
    # Solid lines show hourly medians.
    # Dark ribbons show the 25th to 75th percentile range.
    # Light ribbons show the 10th to 90th percentile range.
    #
    # The ribbons describe variation across evaluation days. They do not
    # represent multi-seed uncertainty or confidence intervals.


    # =============================================================================
    # Configuration
    # =============================================================================

    state.TIME_COLUMN = 'reward_time_step'
    state.HOURS_PER_DAY = 24

    # Hour-alignment offset:
    #
    # hour_of_day = (reward_time_step + HOUR_OFFSET) % 24
    #
    # Keep HOUR_OFFSET = 0 only if reward_time_step modulo 24 is verified
    # to match the CityLearn hour-of-day convention.
    state.HOUR_OFFSET = 0

    # Verified peak-tariff prices observed in the evaluated timeseries.
    state.PEAK_PRICES = {
        0.40,
        0.50,
        0.54,
    }

    state.PRICE_TOLERANCE = 1e-8

    # An hour is treated as a recurring peak-tariff hour if at least
    # 50% of observations at that cyclic hour use a peak tariff.
    state.PEAK_HOUR_FRACTION_THRESHOLD = 0.50

    # Quantile limits
    state.LOWER_OUTER_QUANTILE = 0.10
    state.LOWER_INNER_QUANTILE = 0.25
    state.UPPER_INNER_QUANTILE = 0.75
    state.UPPER_OUTER_QUANTILE = 0.90

    # Colors
    state.PEAK_SHADE_COLOR = '#F6C98D'
    state.PEAK_SHADE_ALPHA = 0.20

    state.LOAD_COLOR = state.ORANGE
    state.PV_COLOR = state.COLOR_PV
    state.BASELINE_COLOR = state.COLOR_BASELINE
    state.TD3_COLOR = state.COLOR_TD3
    state.BATTERY_COLOR = state.PURPLE
    state.SOC_COLOR = state.TEAL


    # =============================================================================
    # Validate required columns
    # =============================================================================

    state.required_daily_columns = [
        state.TIME_COLUMN,
        'electricity_price',
        'load',
        'pv_generation',
        'net_baseline',
        'net_consumption',
        'battery_consumption',
        'post_action_soc_percent',
    ]

    state.missing_daily_columns = [
        state.column
        for state.column in state.required_daily_columns
        if state.column not in state.df.columns
    ]

    if state.missing_daily_columns:
        raise KeyError(
            'The aggregate daily-profile figure cannot be generated '
            'because the following columns are missing:\n'
            f'{state.missing_daily_columns}'
        )


    # =============================================================================
    # Validate numerical values
    # =============================================================================

    state.non_finite_daily_columns = []

    for state.column in state.required_daily_columns:

        state.column_values = state.df[state.column].to_numpy(
            dtype=float
        )

        if not np.isfinite(
            state.column_values
        ).all():
            state.non_finite_daily_columns.append(
                state.column
            )

    if state.non_finite_daily_columns:
        raise ValueError(
            'NaN or infinite values were detected in the following '
            'daily-profile columns:\n'
            f'{state.non_finite_daily_columns}'
        )


    if not state.df[state.TIME_COLUMN].is_monotonic_increasing:
        raise ValueError(
            f'The column "{state.TIME_COLUMN}" is not monotonically increasing.'
        )

    if state.df[state.TIME_COLUMN].duplicated().any():
        raise ValueError(
            f'The column "{state.TIME_COLUMN}" contains duplicated values.'
        )


    # =============================================================================
    # Create cyclic hour-of-day
    # =============================================================================

    state.profile_df = state.df[
        state.required_daily_columns
    ].copy()

    state.profile_df['hour_of_day'] = (
        (
            state.profile_df[state.TIME_COLUMN].astype(int)
            + state.HOUR_OFFSET
        )
    )


    state.observed_hours = sorted(
        state.profile_df['hour_of_day']
        .unique()
        .tolist()
    )

    state.expected_hours = list(
        range(state.HOURS_PER_DAY)
    )

    if state.observed_hours != state.expected_hours:
        raise ValueError(
            'The derived cyclic-hour variable does not contain all '
            'expected hours from 0 to 23.\n'
            f'Observed hours: {state.observed_hours}'
        )


    state.hourly_observation_counts = (
        state.profile_df.groupby(
            'hour_of_day'
        )
        .size()
        .reindex(
            state.expected_hours
        )
    )

    if state.hourly_observation_counts.isna().any():
        raise ValueError(
            'At least one hour of day has no observations.'
        )

    print(
        'Observations per cyclic hour:',
        int(state.hourly_observation_counts.min()),
        'to',
        int(state.hourly_observation_counts.max()),
    )

    print(
        'Hour alignment uses HOUR_OFFSET =',
        state.HOUR_OFFSET,
    )


    # =============================================================================
    # Identify recurring peak-tariff hours
    # =============================================================================

    def is_peak_price(
        price,
        peak_prices=state.PEAK_PRICES,
        tolerance=state.PRICE_TOLERANCE,
    ):
        """Return True if price matches one verified peak-tariff value."""

        return any(
            np.isclose(
                price,
                peak_price,
                atol=tolerance,
                rtol=0.0,
            )
            for peak_price in peak_prices
        )
    state.is_peak_price = is_peak_price


    state.profile_df['is_peak_tariff'] = (
        state.profile_df['electricity_price']
        .apply(
            state.is_peak_price
        )
    )


    state.hourly_peak_fraction = (
        state.profile_df.groupby(
            'hour_of_day'
        )[
            'is_peak_tariff'
        ]
        .mean()
        .reindex(
            state.expected_hours
        )
    )


    state.recurring_peak_hour_mask = (
        state.hourly_peak_fraction
        >= state.PEAK_HOUR_FRACTION_THRESHOLD
    )


    # =============================================================================
    # Convert peak-hour mask into contiguous intervals
    # =============================================================================

    def find_contiguous_hour_intervals(
        boolean_values,
    ):
        """
        Convert a Boolean hourly mask into contiguous plotting intervals.

        For example, active hours 16 through 20 are displayed from
        15.5 through 20.5.
        """

        boolean_values = np.asarray(
            boolean_values,
            dtype=bool,
        )

        intervals = []
        interval_start = None

        for index, is_active in enumerate(
            boolean_values
        ):

            if is_active and interval_start is None:
                interval_start = index

            if interval_start is not None:

                is_last_index = (
                    index
                    == len(boolean_values) - 1
                )

                if not is_active:

                    interval_end = index - 1

                    intervals.append((
                        interval_start - 0.5,
                        interval_end + 0.5,
                    ))

                    interval_start = None

                elif is_last_index:

                    interval_end = index

                    intervals.append((
                        interval_start - 0.5,
                        interval_end + 0.5,
                    ))

                    interval_start = None

        return intervals
    state.find_contiguous_hour_intervals = find_contiguous_hour_intervals


    state.peak_hour_intervals = (
        state.find_contiguous_hour_intervals(
            state.recurring_peak_hour_mask.to_numpy()
        )
    )


    print('Recurring peak-tariff cyclic-hour intervals:')

    for state.interval_start, state.interval_end in state.peak_hour_intervals:

        print(
            f'  Hour {state.interval_start + 0.5:.0f} '
            f'to hour {state.interval_end + 0.5:.0f}'
        )


    # =============================================================================
    # Calculate robust hourly profiles
    # =============================================================================

    def calculate_hourly_profile(
        data,
        value_column,
        hour_column='hour_of_day',
    ):
        """
        Calculate robust hourly statistics for one variable.
        """

        grouped_values = data.groupby(
            hour_column
        )[
            value_column
        ]

        profile = pd.DataFrame({
            'count': grouped_values.count(),
            'mean': grouped_values.mean(),
            'median': grouped_values.median(),
            'q10': grouped_values.quantile(
                state.LOWER_OUTER_QUANTILE
            ),
            'q25': grouped_values.quantile(
                state.LOWER_INNER_QUANTILE
            ),
            'q75': grouped_values.quantile(
                state.UPPER_INNER_QUANTILE
            ),
            'q90': grouped_values.quantile(
                state.UPPER_OUTER_QUANTILE
            ),
        })

        profile = profile.reindex(
            state.expected_hours
        )

        if profile.isna().any().any():
            raise ValueError(
                f'The hourly profile for "{value_column}" '
                'contains missing values.'
            )

        return profile
    state.calculate_hourly_profile = calculate_hourly_profile


    state.load_profile = state.calculate_hourly_profile(
        state.profile_df,
        'load',
    )

    state.pv_profile = state.calculate_hourly_profile(
        state.profile_df,
        'pv_generation',
    )

    state.baseline_profile = state.calculate_hourly_profile(
        state.profile_df,
        'net_baseline',
    )

    state.controlled_profile = state.calculate_hourly_profile(
        state.profile_df,
        'net_consumption',
    )

    state.battery_profile = state.calculate_hourly_profile(
        state.profile_df,
        'battery_consumption',
    )

    state.soc_profile = state.calculate_hourly_profile(
        state.profile_df,
        'post_action_soc_percent',
    )


    state.hour_values = np.arange(
        state.HOURS_PER_DAY,
        dtype=float,
    )


    # =============================================================================
    # Plotting helper functions
    # =============================================================================

    def add_peak_tariff_shading(
        axis,
    ):
        """Add recurring peak-tariff shading to one subplot."""

        for interval_start, interval_end in state.peak_hour_intervals:

            axis.axvspan(
                interval_start,
                interval_end,
                color=state.PEAK_SHADE_COLOR,
                alpha=state.PEAK_SHADE_ALPHA,
                linewidth=0.0,
                zorder=0,
            )
    state.add_peak_tariff_shading = add_peak_tariff_shading


    def plot_quantile_profile(
        axis,
        profile,
        color,
        label,
        linestyle='-',
        linewidth=2.0,
        inner_alpha=0.22,
        outer_alpha=0.08,
    ):
        """
        Plot median, interquartile range, and 10th to 90th percentile.
        """

        # 10th to 90th percentile range
        axis.fill_between(
            state.hour_values,
            profile['q10'].to_numpy(),
            profile['q90'].to_numpy(),
            color=color,
            alpha=outer_alpha,
            linewidth=0.0,
            zorder=1,
        )

        # 25th to 75th percentile range
        axis.fill_between(
            state.hour_values,
            profile['q25'].to_numpy(),
            profile['q75'].to_numpy(),
            color=color,
            alpha=inner_alpha,
            linewidth=0.0,
            zorder=2,
        )

        # Median profile
        axis.plot(
            state.hour_values,
            profile['median'].to_numpy(),
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            label=label,
            zorder=4,
        )
    state.plot_quantile_profile = plot_quantile_profile


    # =============================================================================
    # Create four-panel figure
    # =============================================================================

    state.fig, state.axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(14.0, 8.2),
        sharex=True,
    )


    state.hour_ticks = np.arange(
        0,
        24,
        3,
    )

    state.hour_tick_labels = [
        '00:00',
        '03:00',
        '06:00',
        '09:00',
        '12:00',
        '15:00',
        '18:00',
        '21:00',
    ]


    for state.axis in state.axes.flat:

        state.add_peak_tariff_shading(
            state.axis
        )

        state.axis.set_xlim(
            -0.5,
            23.5,
        )

        state.axis.set_xticks(
            state.hour_ticks
        )

        state.axis.set_xticklabels(
            state.hour_tick_labels
        )

        state.axis.xaxis.set_minor_locator(
            ticker.MultipleLocator(
                1
            )
        )

        state.axis.tick_params(
            axis='x',
            which='minor',
            length=2.0,
        )

        state.axis.set_axisbelow(
            True
        )


    # =============================================================================
    # Panel (a): Building load and PV generation
    # =============================================================================

    state.plot_quantile_profile(
        axis=state.axes[0, 0],
        profile=state.load_profile,
        color=state.LOAD_COLOR,
        label='Building load',
    )

    state.plot_quantile_profile(
        axis=state.axes[0, 0],
        profile=state.pv_profile,
        color=state.PV_COLOR,
        label='PV generation',
        linestyle='-.',
    )


    state.axes[0, 0].set_ylabel(
        'Energy per time step (kWh)'
    )

    state.axes[0, 0].set_title(
        '(a) Building Load and PV Generation',
        loc='left',
        pad=9,
    )

    state.axes[0, 0].legend(
        loc='upper right',
        ncol=2,
        frameon=True,
        framealpha=0.94,
        facecolor='white',
        edgecolor='#CCCCCC',
        fontsize=8.5,
    )


    # =============================================================================
    # Panel (b): Baseline and controlled net grid exchange
    # =============================================================================

    state.plot_quantile_profile(
        axis=state.axes[0, 1],
        profile=state.baseline_profile,
        color=state.BASELINE_COLOR,
        label='Baseline without storage',
        linestyle='--',
    )

    state.plot_quantile_profile(
        axis=state.axes[0, 1],
        profile=state.controlled_profile,
        color=state.TD3_COLOR,
        label='TD3-controlled',
    )


    state.axes[0, 1].axhline(
        y=0.0,
        color='#333333',
        linewidth=0.85,
        linestyle='-',
        zorder=3,
    )

    state.axes[0, 1].set_ylabel(
        'Net grid exchange (kWh)'
    )

    state.axes[0, 1].set_title(
        '(b) Baseline and Controlled Net Grid Exchange',
        loc='left',
        pad=9,
    )

    state.axes[0, 1].legend(
        loc='upper right',
        ncol=1,
        frameon=True,
        framealpha=0.94,
        facecolor='white',
        edgecolor='#CCCCCC',
        fontsize=8.5,
    )


    # =============================================================================
    # Panel (c): Realized battery charging and discharging
    # =============================================================================

    state.plot_quantile_profile(
        axis=state.axes[1, 0],
        profile=state.battery_profile,
        color=state.BATTERY_COLOR,
        label='Median battery exchange',
    )


    state.axes[1, 0].axhline(
        y=0.0,
        color='#333333',
        linewidth=0.85,
        linestyle='-',
        zorder=3,
    )


    state.axes[1, 0].text(
        0.02,
        0.96,
        'Positive: charging',
        transform=state.axes[1, 0].transAxes,
        horizontalalignment='left',
        verticalalignment='top',
        fontsize=8.2,
        fontweight='bold',
        color=state.BATTERY_COLOR,
    )

    state.axes[1, 0].text(
        0.02,
        0.04,
        'Negative: discharging',
        transform=state.axes[1, 0].transAxes,
        horizontalalignment='left',
        verticalalignment='bottom',
        fontsize=8.2,
        fontweight='bold',
        color=state.BATTERY_COLOR,
    )


    state.axes[1, 0].set_xlabel(
        'Cyclic hour of day'
    )

    state.axes[1, 0].set_ylabel(
        'Battery exchange (kWh)'
    )

    state.axes[1, 0].set_title(
        '(c) Realized Battery Charging and Discharging',
        loc='left',
        pad=9,
    )

    state.axes[1, 0].legend(
        loc='upper right',
        ncol=1,
        frameon=True,
        framealpha=0.94,
        facecolor='white',
        edgecolor='#CCCCCC',
        fontsize=8.3,
    )


    # =============================================================================
    # Panel (d): Post-action battery SOC
    # =============================================================================

    state.plot_quantile_profile(
        axis=state.axes[1, 1],
        profile=state.soc_profile,
        color=state.SOC_COLOR,
        label='Median post-action SOC',
    )


    state.axes[1, 1].axhline(
        y=0.0,
        color='#555555',
        linewidth=0.7,
        linestyle=':',
        zorder=3,
    )

    state.axes[1, 1].axhline(
        y=100.0,
        color='#555555',
        linewidth=0.7,
        linestyle=':',
        zorder=3,
    )

    state.axes[1, 1].set_ylim(
        -3.0,
        103.0,
    )

    state.axes[1, 1].set_xlabel(
        'Cyclic hour of day'
    )

    state.axes[1, 1].set_ylabel(
        'Post-action battery SOC (%)'
    )

    state.axes[1, 1].set_title(
        '(d) Battery State of Charge',
        loc='left',
        pad=9,
    )

    state.axes[1, 1].legend(
        loc='upper right',
        ncol=1,
        frameon=True,
        framealpha=0.94,
        facecolor='white',
        edgecolor='#CCCCCC',
        fontsize=8.3,
    )


    # =============================================================================
    # Shared explanatory legend
    # =============================================================================

    state.peak_tariff_patch = mpatches.Patch(
        facecolor=state.PEAK_SHADE_COLOR,
        edgecolor='#D9A45D',
        linewidth=0.8,
        alpha=state.PEAK_SHADE_ALPHA,
        label='Recurring peak-tariff hours',
    )

    state.inner_ribbon_patch = mpatches.Patch(
        facecolor=state.GRAY,
        edgecolor='none',
        alpha=0.28,
        label='25th–75th percentile',
    )

    state.outer_ribbon_patch = mpatches.Patch(
        facecolor=state.GRAY,
        edgecolor='none',
        alpha=0.10,
        label='10th–90th percentile',
    )


    state.fig.legend(
        handles=[
            state.peak_tariff_patch,
            state.inner_ribbon_patch,
            state.outer_ribbon_patch,
        ],
        loc='upper center',
        bbox_to_anchor=(
            0.5,
            0.947,
        ),
        ncol=3,
        frameon=True,
        framealpha=0.95,
        facecolor='white',
        edgecolor='#CCCCCC',
        fontsize=8.5,
        columnspacing=1.7,
        handlelength=2.0,
    )


    # =============================================================================
    # Figure title and explanatory note
    # =============================================================================

    state.fig.suptitle(
        'Aggregate Daily Operating Profiles',
        fontsize=13,
        fontweight='bold',
        y=0.992,
    )


    state.fig.text(
        0.5,
        0.01,
        (
            'Solid lines show hourly medians across the evaluation horizon. '
            'Ribbons represent variability across evaluation days and not '
            'multi-seed confidence intervals. '
            f'Cyclic-hour alignment uses HOUR_OFFSET = {state.HOUR_OFFSET}.'
        ))

def cell_12(state):
    """Cell 12: Plot 15 - TD3 action and realized battery heatmaps."""
    # =============================================================================
    # Plot 15: Requested TD3 action and realized battery-exchange heatmaps
    # =============================================================================

    # This figure compares the requested TD3 action with the realized battery
    # exchange over the complete evaluation horizon.
    #
    # Red indicates discharging or a discharge request.
    # Green indicates charging or a charge request.
    # White indicates zero or near-zero activity.
    #
    # The figure uses sequential evaluation day and cyclic transition hour
    # because an explicit timestamp is not available in timeseries.csv.
    #
    # This is a single-run policy-behaviour diagnostic and does not represent
    # multi-seed uncertainty.


    # =============================================================================
    # Configuration
    # =============================================================================

    state.ACTION_COLUMN = 'action'
    state.BATTERY_COLUMN = 'battery_consumption'
    state.TRANSITION_COLUMN = 'transition'

    state.HOURS_PER_DAY = 24

    # Cyclic-hour alignment:
    #
    # cyclic_hour = (transition + HOUR_OFFSET) % 24
    #
    # Keep HOUR_OFFSET = 0 only if transition 0 is verified as the first
    # intended hour of the daily cycle.
    state.HOUR_OFFSET = 0

    # Use this percentile to define the symmetric realized-battery color scale.
    # A percentile below 1.0 prevents rare extremes from compressing the
    # visual contrast of most battery exchanges.
    state.BATTERY_DISPLAY_PERCENTILE = 0.99

    # Numerical tolerance for identifying near-zero activity.
    state.ACTIVITY_TOLERANCE = 1e-8


    # =============================================================================
    # Validate configuration
    # =============================================================================

    if not isinstance(state.HOUR_OFFSET, int):
        raise TypeError(
            'HOUR_OFFSET must be an integer.'
        )

    if not 0.0 < state.BATTERY_DISPLAY_PERCENTILE <= 1.0:
        raise ValueError(
            'BATTERY_DISPLAY_PERCENTILE must be in the interval (0, 1].'
        )

    if state.ACTIVITY_TOLERANCE < 0.0:
        raise ValueError(
            'ACTIVITY_TOLERANCE cannot be negative.'
        )


    # =============================================================================
    # Validate required columns
    # =============================================================================

    state.required_heatmap_columns = [
        state.TRANSITION_COLUMN,
        state.ACTION_COLUMN,
        state.BATTERY_COLUMN,
    ]

    state.missing_heatmap_columns = [
        state.column
        for state.column in state.required_heatmap_columns
        if state.column not in state.df.columns
    ]

    if state.missing_heatmap_columns:
        raise KeyError(
            'The action and battery-exchange heatmaps cannot be generated '
            'because the following columns are missing:\n'
            f'{state.missing_heatmap_columns}'
        )


    # =============================================================================
    # Validate numerical data
    # =============================================================================

    for state.column in state.required_heatmap_columns:

        state.column_values = state.df[state.column].to_numpy(
            dtype=float
        )

        if not np.isfinite(
            state.column_values
        ).all():
            raise ValueError(
                f'The column "{state.column}" contains NaN or infinite values.'
            )


    if state.df[state.TRANSITION_COLUMN].duplicated().any():
        raise ValueError(
            f'The column "{state.TRANSITION_COLUMN}" contains duplicate values.'
        )


    if not state.df[state.TRANSITION_COLUMN].is_monotonic_increasing:
        raise ValueError(
            f'The column "{state.TRANSITION_COLUMN}" is not monotonically increasing.'
        )


    state.expected_transitions = np.arange(
        len(state.df)
    )

    if not np.array_equal(
        state.df[state.TRANSITION_COLUMN].to_numpy(
            dtype=int
        ),
        state.expected_transitions,
    ):
        raise ValueError(
            'The transition sequence is not continuous from 0 to '
            f'{len(state.df) - 1}.'
        )


    # =============================================================================
    # Construct sequential day and cyclic transition hour
    # =============================================================================

    state.heatmap_df = state.df[
        state.required_heatmap_columns
    ].copy()


    state.heatmap_df['sequential_day'] = (
        state.heatmap_df[state.TRANSITION_COLUMN]
        // state.HOURS_PER_DAY
    ) + 1


    state.heatmap_df['cyclic_hour'] = (
        (
            state.heatmap_df[state.TRANSITION_COLUMN]
            + state.HOUR_OFFSET
        )
    )


    state.observed_hours = sorted(
        state.heatmap_df['cyclic_hour']
        .unique()
        .tolist()
    )

    state.expected_hours = list(
        range(state.HOURS_PER_DAY)
    )

    if state.observed_hours != state.expected_hours:
        raise ValueError(
            'The derived cyclic-hour variable does not contain all '
            'expected hours from 0 through 23.\n'
            f'Observed hours: {state.observed_hours}'
        )


    state.number_of_days = int(
        state.heatmap_df['sequential_day'].max()
    )


    # =============================================================================
    # Create complete hour-by-day pivot tables
    # =============================================================================

    state.action_pivot = state.heatmap_df.pivot_table(
        index='cyclic_hour',
        columns='sequential_day',
        values=state.ACTION_COLUMN,
        aggfunc='mean',
    )


    state.battery_pivot = state.heatmap_df.pivot_table(
        index='cyclic_hour',
        columns='sequential_day',
        values=state.BATTERY_COLUMN,
        aggfunc='mean',
    )


    state.complete_day_index = np.arange(
        1,
        state.number_of_days + 1,
        dtype=int,
    )


    state.action_pivot = state.action_pivot.reindex(
        index=state.expected_hours,
        columns=state.complete_day_index,
    )


    state.battery_pivot = state.battery_pivot.reindex(
        index=state.expected_hours,
        columns=state.complete_day_index,
    )


    # The final sequential day may be partial. Missing cells remain NaN and
    # are shown using the missing-data color.
    state.missing_action_cells = int(
        state.action_pivot.isna().sum().sum()
    )

    state.missing_battery_cells = int(
        state.battery_pivot.isna().sum().sum()
    )


    print(
        'Sequential evaluation days:',
        state.number_of_days,
    )

    print(
        'Missing action heatmap cells:',
        state.missing_action_cells,
    )

    print(
        'Missing realized-battery heatmap cells:',
        state.missing_battery_cells,
    )

    print(
        'Cyclic-hour alignment uses HOUR_OFFSET =',
        state.HOUR_OFFSET,
    )


    # =============================================================================
    # Validate action range
    # =============================================================================

    state.minimum_action = float(
        state.heatmap_df[state.ACTION_COLUMN].min()
    )

    state.maximum_action = float(
        state.heatmap_df[state.ACTION_COLUMN].max()
    )


    if (
        state.minimum_action < -1.0 - 1e-6
        or state.maximum_action > 1.0 + 1e-6
    ):
        raise ValueError(
            'The TD3 action contains values outside the expected '
            'range [-1, 1].\n'
            f'Observed range: {state.minimum_action:.8f} to '
            f'{state.maximum_action:.8f}'
        )


    # =============================================================================
    # Determine symmetric battery-exchange display range
    # =============================================================================

    state.absolute_battery_exchange = (
        state.heatmap_df[state.BATTERY_COLUMN]
        .abs()
    )


    state.battery_display_limit = float(
        state.absolute_battery_exchange.quantile(
            state.BATTERY_DISPLAY_PERCENTILE
        )
    )


    if np.isclose(
        state.battery_display_limit,
        0.0,
        atol=state.ACTIVITY_TOLERANCE,
    ):
        raise ValueError(
            'The realized battery exchange contains no meaningful '
            'nonzero variation.'
        )


    state.battery_values_outside_display_range = int(
        (
            state.absolute_battery_exchange
            > state.battery_display_limit
        ).sum()
    )

    state.battery_percentage_outside_display_range = (
        100.0
        * state.battery_values_outside_display_range
        / len(state.heatmap_df)
    )


    # =============================================================================
    # Construct diverging color map
    # =============================================================================

    state.action_cmap = LinearSegmentedColormap.from_list(
        'discharge_idle_charge',
        [
            state.RED,
            '#FFFFFF',
            state.GREEN,
        ],
        N=256,
    )


    # Display missing cells in light gray.
    state.action_cmap.set_bad(
        color='#E6E6E6'
    )


    # =============================================================================
    # Calculate behavioural summary statistics
    # =============================================================================

    state.action_sign = np.zeros(
        len(state.heatmap_df),
        dtype=int,
    )

    state.action_sign[
        state.heatmap_df[state.ACTION_COLUMN]
        > state.ACTIVITY_TOLERANCE
    ] = 1

    state.action_sign[
        state.heatmap_df[state.ACTION_COLUMN]
        < -state.ACTIVITY_TOLERANCE
    ] = -1


    state.battery_sign = np.zeros(
        len(state.heatmap_df),
        dtype=int,
    )

    state.battery_sign[
        state.heatmap_df[state.BATTERY_COLUMN]
        > state.ACTIVITY_TOLERANCE
    ] = 1

    state.battery_sign[
        state.heatmap_df[state.BATTERY_COLUMN]
        < -state.ACTIVITY_TOLERANCE
    ] = -1


    # Requested action and realized battery exchange have matching active
    # directions when both are positive or both are negative.
    state.matching_active_direction = (
        (state.action_sign == state.battery_sign)
        & (state.action_sign != 0)
    )


    # Requested operation that results in near-zero realized exchange.
    state.requested_but_not_realized = (
        (state.action_sign != 0)
        & (state.battery_sign == 0)
    )


    state.opposite_realized_direction = (
        (state.action_sign != 0)
        & (state.battery_sign != 0)
        & (state.action_sign != state.battery_sign)
    )


    state.matching_active_percentage = (
        100.0
        * state.matching_active_direction.mean()
    )

    state.requested_not_realized_percentage = (
        100.0
        * state.requested_but_not_realized.mean()
    )

    state.opposite_direction_percentage = (
        100.0
        * state.opposite_realized_direction.mean()
    )


    # =============================================================================
    # Create two aligned heatmaps
    # =============================================================================

    state.fig, state.axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(14.5, 8.0),
        sharex=True,
    )


    # =============================================================================
    # Panel (a): Requested TD3 action
    # =============================================================================

    state.action_image = state.axes[0].imshow(
        state.action_pivot.to_numpy(),
        aspect='auto',
        interpolation='nearest',
        cmap=state.action_cmap,
        vmin=-1.0,
        vmax=1.0,
        origin='upper',
        extent=[
            0.5,
            state.number_of_days + 0.5,
            23.5,
            -0.5,
        ],
    )


    state.action_colorbar = state.fig.colorbar(
        state.action_image,
        ax=state.axes[0],
        shrink=0.88,
        pad=0.015,
    )

    state.action_colorbar.set_label(
        'Requested action\n'
        '(-1 discharge, +1 charge)'
    )

    state.action_colorbar.set_ticks([
        -1.0,
        -0.5,
        0.0,
        0.5,
        1.0,
    ])


    state.axes[0].set_ylabel(
        'Cyclic transition hour'
    )

    state.axes[0].set_title(
        '(a) Requested TD3 Action',
        loc='left',
        pad=9,
    )


    # =============================================================================
    # Panel (b): Realized battery exchange
    # =============================================================================

    state.battery_image = state.axes[1].imshow(
        state.battery_pivot.to_numpy(),
        aspect='auto',
        interpolation='nearest',
        cmap=state.action_cmap,
        vmin=-state.battery_display_limit,
        vmax=state.battery_display_limit,
        origin='upper',
        extent=[
            0.5,
            state.number_of_days + 0.5,
            23.5,
            -0.5,
        ],
    )


    state.battery_colorbar = state.fig.colorbar(
        state.battery_image,
        ax=state.axes[1],
        shrink=0.88,
        pad=0.015,
    )

    state.battery_colorbar.set_label(
        'Realized battery exchange (kWh)\n'
        'Negative discharge, positive charge'
    )


    state.battery_colorbar.locator = ticker.MaxNLocator(
        nbins=5
    )

    state.battery_colorbar.update_ticks()


    state.axes[1].set_xlabel(
        'Sequential evaluation day'
    )

    state.axes[1].set_ylabel(
        'Cyclic transition hour'
    )

    state.axes[1].set_title(
        '(b) Realized Battery Charging and Discharging',
        loc='left',
        pad=9,
    )


    # =============================================================================
    # Shared axis formatting
    # =============================================================================

    state.hour_tick_positions = np.arange(
        0,
        24,
        3,
    )

    state.hour_tick_labels = [
        '00:00',
        '03:00',
        '06:00',
        '09:00',
        '12:00',
        '15:00',
        '18:00',
        '21:00',
    ]


    state.day_tick_step = max(
        1,
        int(
            np.ceil(
                state.number_of_days
                / 12.0
            )
        ),
    )


    state.day_tick_positions = np.arange(
        1,
        state.number_of_days + 1,
        state.day_tick_step,
    )


    for state.axis in state.axes:

        state.axis.set_yticks(
            state.hour_tick_positions
        )

        state.axis.set_yticklabels(
            state.hour_tick_labels
        )

        state.axis.set_xticks(
            state.day_tick_positions
        )

        state.axis.xaxis.set_major_formatter(
            ticker.StrMethodFormatter(
                '{x:,.0f}'
            )
        )

        state.axis.grid(
            False
        )


    # =============================================================================
    # Add diagnostic summary box
    # =============================================================================

    state.diagnostic_text = (
        f'Active direction matched: '
        f'{state.matching_active_percentage:.1f}%\n'
        f'Requested but not realized: '
        f'{state.requested_not_realized_percentage:.1f}%\n'
        f'Opposite realized direction: '
        f'{state.opposite_direction_percentage:.2f}%'
    )


    state.axes[1].text(
        0.99,
        0.98,
        state.diagnostic_text,
        transform=state.axes[1].transAxes,
        horizontalalignment='right',
        verticalalignment='top',
        fontsize=8.4,
        fontweight='bold',
        linespacing=1.2,
        bbox={
            'boxstyle': 'round,pad=0.38',
            'facecolor': 'white',
            'edgecolor': '#777777',
            'linewidth': 1.0,
            'alpha': 0.94,
        },
        zorder=5,
    )


    # =============================================================================
    # Figure title and interpretation note
    # =============================================================================

    state.fig.suptitle(
        'Full-Horizon TD3 Action and Realized Battery-Exchange Patterns',
        fontsize=13,
        fontweight='bold',
        y=0.99,
    )


    state.fig.text(
        0.5,
        0.012,
        (
            'The requested action is a policy command, whereas realized '
            'battery exchange reflects SOC and operational feasibility. '
            f'The battery color scale is symmetric and limited to the '
            f'{state.BATTERY_DISPLAY_PERCENTILE * 100:.0f}th percentile of '
            'absolute realized exchange. '
            f'Cyclic-hour alignment uses HOUR_OFFSET = {state.HOUR_OFFSET}.'
        ),
        horizontalalignment='center',
        verticalalignment='bottom',
        fontsize=8.1,
        color='#444444',
    )


    state.fig.tight_layout(
        rect=[
            0.0,
            0.065,
            1.0,
            0.95,
        ],
        h_pad=1.9,
    )


    # =============================================================================
    # Save the figure
    # =============================================================================

    state.save_figure(
        state.fig,
        'plot15_td3_action_and_realized_battery_heatmaps',
    )


    # =============================================================================
    # Save numerical summary
    # =============================================================================

    state.plot15_summary = pd.DataFrame([{
        'number_of_transitions': len(
            state.heatmap_df
        ),
        'number_of_sequential_days':
            state.number_of_days,
        'hour_offset':
            state.HOUR_OFFSET,
        'minimum_action':
            state.minimum_action,
        'maximum_action':
            state.maximum_action,
        'battery_display_percentile':
            state.BATTERY_DISPLAY_PERCENTILE,
        'battery_display_limit_kwh':
            state.battery_display_limit,
        'battery_values_above_display_limit':
            state.battery_values_outside_display_range,
        'battery_percent_above_display_limit':
            state.battery_percentage_outside_display_range,
        'active_direction_match_percent':
            state.matching_active_percentage,
        'requested_but_not_realized_percent':
            state.requested_not_realized_percentage,
        'opposite_realized_direction_percent':
            state.opposite_direction_percentage,
        'missing_action_heatmap_cells':
            state.missing_action_cells,
        'missing_battery_heatmap_cells':
            state.missing_battery_cells,
    }])


    state.plot15_summary_path = (
        state.TABLES_DIR
        / 'plot15_td3_action_and_realized_battery_heatmaps_summary.csv'
    )


    state.plot15_summary.to_csv(
        state.plot15_summary_path,
        index=False,
    )


    display(
        state.plot15_summary
    )


    print(
        'Action and battery heatmap summary saved to:\n'
        f'{state.plot15_summary_path}'
    )

def cell_13(state):
    """Cell 13: Plot 20 - weekly cost/carbon trade-off."""
    # =============================================================================
    # Plot 20: Weekly electricity-cost and carbon-emission trade-off
    # =============================================================================

    # This figure examines whether the TD3 controller reduces electricity
    # cost and grid-related carbon emissions simultaneously over complete
    # sequential evaluation weeks.
    #
    # Panel (a) shows the movement from baseline to TD3 in absolute units.
    # Panel (b) shows the percentage changes relative to the baseline.
    #
    # This is a single-run temporal diagnostic. It is not a Pareto-front
    # analysis and does not represent multi-seed uncertainty.


    # =============================================================================
    # Configuration
    # =============================================================================

    state.HOURS_PER_DAY = 24
    state.DAYS_PER_WEEK = 7
    state.HOURS_PER_WEEK = (
        state.HOURS_PER_DAY
        * state.DAYS_PER_WEEK
    )

    state.TRANSITION_COLUMN = 'transition'

    state.CONTROLLER_COST_COLUMN = 'raw_cost'
    state.CONTROLLER_EMISSION_COLUMN = 'raw_emission'

    state.BASELINE_COST_COLUMN = 'positive_baseline_cost'
    state.BASELINE_EMISSION_COLUMN = 'positive_baseline_emission'

    state.DENOMINATOR_TOLERANCE = 1e-10

    state.WEEK_CMAP = plt.get_cmap(
        'viridis'
    )


    # =============================================================================
    # Validate required columns
    # =============================================================================

    state.required_tradeoff_columns = [
        state.TRANSITION_COLUMN,
        state.CONTROLLER_COST_COLUMN,
        state.CONTROLLER_EMISSION_COLUMN,
        state.BASELINE_COST_COLUMN,
        state.BASELINE_EMISSION_COLUMN,
    ]

    state.missing_tradeoff_columns = [
        state.column
        for state.column in state.required_tradeoff_columns
        if state.column not in state.df.columns
    ]

    if state.missing_tradeoff_columns:
        raise KeyError(
            'The weekly cost-carbon trade-off figure cannot be generated '
            'because the following columns are missing:\n'
            f'{state.missing_tradeoff_columns}'
        )


    # =============================================================================
    # Validate numerical values
    # =============================================================================

    state.non_finite_tradeoff_columns = []

    for state.column in state.required_tradeoff_columns:

        state.column_values = state.df[state.column].to_numpy(
            dtype=float
        )

        if not np.isfinite(
            state.column_values
        ).all():
            state.non_finite_tradeoff_columns.append(
                state.column
            )

    if state.non_finite_tradeoff_columns:
        raise ValueError(
            'NaN or infinite values were detected in the following '
            'cost-carbon columns:\n'
            f'{state.non_finite_tradeoff_columns}'
        )


    if state.df[state.TRANSITION_COLUMN].duplicated().any():
        raise ValueError(
            f'The column "{state.TRANSITION_COLUMN}" contains duplicate values.'
        )


    if not state.df[state.TRANSITION_COLUMN].is_monotonic_increasing:
        raise ValueError(
            f'The column "{state.TRANSITION_COLUMN}" is not monotonically increasing.'
        )


    state.expected_transition_values = np.arange(
        len(state.df),
        dtype=int,
    )

    if not np.array_equal(
        state.df[state.TRANSITION_COLUMN].to_numpy(
            dtype=int
        ),
        state.expected_transition_values,
    ):
        raise ValueError(
            'The transition sequence is not continuous from zero to '
            f'{len(state.df) - 1}.'
        )


    # =============================================================================
    # Validate nonnegative cost and emission quantities
    # =============================================================================

    state.nonnegative_columns = [
        state.CONTROLLER_COST_COLUMN,
        state.CONTROLLER_EMISSION_COLUMN,
        state.BASELINE_COST_COLUMN,
        state.BASELINE_EMISSION_COLUMN,
    ]

    state.negative_value_report = {}

    for state.column in state.nonnegative_columns:

        state.negative_mask = (
            state.df[state.column]
            < -state.DENOMINATOR_TOLERANCE
        )

        if state.negative_mask.any():
            state.negative_value_report[state.column] = int(
                state.negative_mask.sum()
            )

    if state.negative_value_report:
        raise ValueError(
            'Substantial negative values were detected in quantities '
            'expected to be nonnegative:\n'
            f'{state.negative_value_report}'
        )


    # =============================================================================
    # Construct sequential evaluation weeks
    # =============================================================================

    state.tradeoff_df = state.df[
        state.required_tradeoff_columns
    ].copy()

    state.tradeoff_df['evaluation_week'] = (
        state.tradeoff_df[state.TRANSITION_COLUMN]
        // state.HOURS_PER_WEEK
    ) + 1


    state.weekly_transition_counts = (
        state.tradeoff_df.groupby(
            'evaluation_week'
        )
        .size()
        .rename(
            'number_of_transitions'
        )
    )


    state.complete_week_numbers = (
        state.weekly_transition_counts.loc[
            state.weekly_transition_counts
            == state.HOURS_PER_WEEK
        ]
        .index
        .tolist()
    )


    state.partial_week_numbers = (
        state.weekly_transition_counts.loc[
            state.weekly_transition_counts
            != state.HOURS_PER_WEEK
        ]
        .index
        .tolist()
    )


    if not state.complete_week_numbers:
        raise ValueError(
            'No complete 168-transition evaluation weeks were found.'
        )


    if state.partial_week_numbers:
        print(
            'Partial sequential weeks excluded from Plot 20:',
            state.partial_week_numbers,
        )


    # =============================================================================
    # Aggregate complete sequential weeks
    # =============================================================================

    state.weekly_tradeoff = (
        state.tradeoff_df.loc[
            state.tradeoff_df['evaluation_week']
            .isin(
                state.complete_week_numbers
            )
        ]
        .groupby(
            'evaluation_week',
            as_index=False,
        )[
            [
                state.CONTROLLER_COST_COLUMN,
                state.CONTROLLER_EMISSION_COLUMN,
                state.BASELINE_COST_COLUMN,
                state.BASELINE_EMISSION_COLUMN,
            ]
        ]
        .sum()
    )


    state.weekly_tradeoff = state.weekly_tradeoff.rename(
        columns={
            state.CONTROLLER_COST_COLUMN:
                'td3_electricity_cost',
            state.CONTROLLER_EMISSION_COLUMN:
                'td3_carbon_emission_kg',
            state.BASELINE_COST_COLUMN:
                'baseline_electricity_cost',
            state.BASELINE_EMISSION_COLUMN:
                'baseline_carbon_emission_kg',
        }
    )


    # =============================================================================
    # Validate weekly denominators
    # =============================================================================

    state.invalid_cost_denominator_mask = (
        state.weekly_tradeoff[
            'baseline_electricity_cost'
        ]
        <= state.DENOMINATOR_TOLERANCE
    )

    state.invalid_emission_denominator_mask = (
        state.weekly_tradeoff[
            'baseline_carbon_emission_kg'
        ]
        <= state.DENOMINATOR_TOLERANCE
    )


    if state.invalid_cost_denominator_mask.any():
        state.invalid_weeks = (
            state.weekly_tradeoff.loc[
                state.invalid_cost_denominator_mask,
                'evaluation_week',
            ]
            .astype(int)
            .tolist()
        )

        raise ValueError(
            'The baseline weekly electricity cost is zero or near zero '
            'for the following weeks:\n'
            f'{state.invalid_weeks}'
        )


    if state.invalid_emission_denominator_mask.any():
        state.invalid_weeks = (
            state.weekly_tradeoff.loc[
                state.invalid_emission_denominator_mask,
                'evaluation_week',
            ]
            .astype(int)
            .tolist()
        )

        raise ValueError(
            'The baseline weekly carbon emission is zero or near zero '
            'for the following weeks:\n'
            f'{state.invalid_weeks}'
        )


    # =============================================================================
    # Calculate weekly differences, ratios, and percentage changes
    # =============================================================================

    state.weekly_tradeoff[
        'electricity_cost_difference'
    ] = (
        state.weekly_tradeoff[
            'td3_electricity_cost'
        ]
        - state.weekly_tradeoff[
            'baseline_electricity_cost'
        ]
    )


    state.weekly_tradeoff[
        'carbon_emission_difference_kg'
    ] = (
        state.weekly_tradeoff[
            'td3_carbon_emission_kg'
        ]
        - state.weekly_tradeoff[
            'baseline_carbon_emission_kg'
        ]
    )


    state.weekly_tradeoff[
        'electricity_cost_ratio'
    ] = (
        state.weekly_tradeoff[
            'td3_electricity_cost'
        ]
        / state.weekly_tradeoff[
            'baseline_electricity_cost'
        ]
    )


    state.weekly_tradeoff[
        'carbon_emission_ratio'
    ] = (
        state.weekly_tradeoff[
            'td3_carbon_emission_kg'
        ]
        / state.weekly_tradeoff[
            'baseline_carbon_emission_kg'
        ]
    )


    # Negative percentage change means improvement relative to baseline.
    state.weekly_tradeoff[
        'electricity_cost_change_percent'
    ] = (
        100.0
        * state.weekly_tradeoff[
            'electricity_cost_difference'
        ]
        / state.weekly_tradeoff[
            'baseline_electricity_cost'
        ]
    )


    state.weekly_tradeoff[
        'carbon_emission_change_percent'
    ] = (
        100.0
        * state.weekly_tradeoff[
            'carbon_emission_difference_kg'
        ]
        / state.weekly_tradeoff[
            'baseline_carbon_emission_kg'
        ]
    )


    # =============================================================================
    # Classify weekly outcomes
    # =============================================================================

    state.weekly_tradeoff[
        'cost_improved'
    ] = (
        state.weekly_tradeoff[
            'electricity_cost_change_percent'
        ]
        < 0.0
    )

    state.weekly_tradeoff[
        'emission_improved'
    ] = (
        state.weekly_tradeoff[
            'carbon_emission_change_percent'
        ]
        < 0.0
    )


    def classify_weekly_outcome(
        row,
    ):
        """Classify the weekly cost-carbon outcome."""

        if (
            row['cost_improved']
            and row['emission_improved']
        ):
            return 'Both improved'

        if (
            row['cost_improved']
            and not row['emission_improved']
        ):
            return 'Cost improved, emissions worsened'

        if (
            not row['cost_improved']
            and row['emission_improved']
        ):
            return 'Cost worsened, emissions improved'

        return 'Both worsened'
    state.classify_weekly_outcome = classify_weekly_outcome


    state.weekly_tradeoff[
        'outcome'
    ] = state.weekly_tradeoff.apply(
        state.classify_weekly_outcome,
        axis=1,
    )


    # =============================================================================
    # Prepare week colors
    # =============================================================================

    state.minimum_week = int(
        state.weekly_tradeoff[
            'evaluation_week'
        ].min()
    )

    state.maximum_week = int(
        state.weekly_tradeoff[
            'evaluation_week'
        ].max()
    )


    def get_week_color(
        week_number,
    ):
        """Map sequential week number to the selected color map."""

        if state.maximum_week == state.minimum_week:
            normalized_week = 0.5
        else:
            normalized_week = (
                float(
                    week_number
                    - state.minimum_week
                )
                / float(
                    state.maximum_week
                    - state.minimum_week
                )
            )

        return state.WEEK_CMAP(
            normalized_week
        )
    state.get_week_color = get_week_color


    state.weekly_colors = [
        state.get_week_color(
            week_number
        )
        for week_number in state.weekly_tradeoff[
            'evaluation_week'
        ]
    ]


    # =============================================================================
    # Calculate overall episode-level results
    # =============================================================================

    state.overall_td3_cost = float(
        state.df[
            state.CONTROLLER_COST_COLUMN
        ].sum()
    )

    state.overall_baseline_cost = float(
        state.df[
            state.BASELINE_COST_COLUMN
        ].sum()
    )

    state.overall_td3_emission = float(
        state.df[
            state.CONTROLLER_EMISSION_COLUMN
        ].sum()
    )

    state.overall_baseline_emission = float(
        state.df[
            state.BASELINE_EMISSION_COLUMN
        ].sum()
    )


    state.overall_cost_change_percent = (
        100.0
        * (
            state.overall_td3_cost
            - state.overall_baseline_cost
        )
        / state.overall_baseline_cost
    )


    state.overall_emission_change_percent = (
        100.0
        * (
            state.overall_td3_emission
            - state.overall_baseline_emission
        )
        / state.overall_baseline_emission
    )


    state.both_improved_count = int(
        (
            state.weekly_tradeoff['outcome']
            == 'Both improved'
        ).sum()
    )

    state.cost_only_count = int(
        (
            state.weekly_tradeoff['outcome']
            == 'Cost improved, emissions worsened'
        ).sum()
    )

    state.emission_only_count = int(
        (
            state.weekly_tradeoff['outcome']
            == 'Cost worsened, emissions improved'
        ).sum()
    )

    state.both_worsened_count = int(
        (
            state.weekly_tradeoff['outcome']
            == 'Both worsened'
        ).sum()
    )


    # =============================================================================
    # Create two-panel figure
    # =============================================================================

    state.fig, state.axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(14.5, 6.0),
        gridspec_kw={
            'width_ratios': [
                1.15,
                1.0,
            ],
        },
    )


    # =============================================================================
    # Panel (a): Absolute weekly baseline-to-TD3 movement
    # =============================================================================

    for state.row_index, state.row in state.weekly_tradeoff.iterrows():

        state.week_color = state.weekly_colors[
            state.row_index
        ]

        state.baseline_cost = float(
            state.row[
                'baseline_electricity_cost'
            ]
        )

        state.baseline_emission = float(
            state.row[
                'baseline_carbon_emission_kg'
            ]
        )

        state.td3_cost = float(
            state.row[
                'td3_electricity_cost'
            ]
        )

        state.td3_emission = float(
            state.row[
                'td3_carbon_emission_kg'
            ]
        )

        # Arrow from baseline to TD3.
        state.axes[0].annotate(
            '',
            xy=(
                state.td3_cost,
                state.td3_emission,
            ),
            xytext=(
                state.baseline_cost,
                state.baseline_emission,
            ),
            arrowprops={
                'arrowstyle': '->',
                'color': state.week_color,
                'linewidth': 0.9,
                'alpha': 0.65,
                'shrinkA': 4,
                'shrinkB': 4,
            },
            zorder=2,
        )


    # Baseline weekly points
    state.baseline_scatter = state.axes[0].scatter(
        state.weekly_tradeoff[
            'baseline_electricity_cost'
        ],
        state.weekly_tradeoff[
            'baseline_carbon_emission_kg'
        ],
        c=state.weekly_tradeoff[
            'evaluation_week'
        ],
        cmap=state.WEEK_CMAP,
        vmin=state.minimum_week,
        vmax=state.maximum_week,
        marker='s',
        s=52,
        alpha=0.55,
        edgecolors='#555555',
        linewidth=0.5,
        label='Baseline without storage',
        zorder=3,
    )


    # TD3 weekly points
    state.axes[0].scatter(
        state.weekly_tradeoff[
            'td3_electricity_cost'
        ],
        state.weekly_tradeoff[
            'td3_carbon_emission_kg'
        ],
        c=state.weekly_tradeoff[
            'evaluation_week'
        ],
        cmap=state.WEEK_CMAP,
        vmin=state.minimum_week,
        vmax=state.maximum_week,
        marker='o',
        s=58,
        alpha=0.90,
        edgecolors='black',
        linewidth=0.55,
        label='TD3-controlled',
        zorder=4,
    )


    state.axes[0].set_xlabel(
        'Weekly electricity cost (monetary units)'
    )

    state.axes[0].set_ylabel(
        'Weekly grid-related carbon emissions (kgCO$_2$e)'
    )

    state.axes[0].set_title(
        '(a) Weekly Baseline-to-TD3 Movement',
        loc='left',
        pad=10,
    )


    state.axes[0].legend(
        loc='upper left',
        ncol=1,
        frameon=True,
        framealpha=0.94,
        facecolor='white',
        edgecolor='#CCCCCC',
        fontsize=8.5,
    )


    state.axes[0].xaxis.set_major_locator(
        ticker.MaxNLocator(
            nbins=6,
        )
    )

    state.axes[0].yaxis.set_major_locator(
        ticker.MaxNLocator(
            nbins=6,
        )
    )


    # Add sequential-week colorbar.
    state.week_colorbar = state.fig.colorbar(
        state.baseline_scatter,
        ax=state.axes[0],
        shrink=0.84,
        pad=0.02,
    )

    state.week_colorbar.set_label(
        'Sequential evaluation week'
    )

    state.week_colorbar.locator = (
        ticker.MaxNLocator(
            nbins=6,
            integer=True,
        )
    )

    state.week_colorbar.update_ticks()


    # =============================================================================
    # Panel (b): Weekly percentage changes relative to baseline
    # =============================================================================

    state.percentage_scatter = state.axes[1].scatter(
        state.weekly_tradeoff[
            'electricity_cost_change_percent'
        ],
        state.weekly_tradeoff[
            'carbon_emission_change_percent'
        ],
        c=state.weekly_tradeoff[
            'evaluation_week'
        ],
        cmap=state.WEEK_CMAP,
        vmin=state.minimum_week,
        vmax=state.maximum_week,
        marker='o',
        s=64,
        alpha=0.90,
        edgecolors='black',
        linewidth=0.55,
        zorder=4,
    )


    # Baseline-equivalent reference lines.
    state.axes[1].axvline(
        x=0.0,
        color='#333333',
        linewidth=0.9,
        linestyle='--',
        zorder=2,
    )

    state.axes[1].axhline(
        y=0.0,
        color='#333333',
        linewidth=0.9,
        linestyle='--',
        zorder=2,
    )


    # Lightly shade the desirable lower-left region.
    state.x_limits_before = state.axes[1].get_xlim()
    state.y_limits_before = state.axes[1].get_ylim()

    state.axes[1].axvspan(
        state.x_limits_before[0],
        0.0,
        ymin=0.0,
        ymax=1.0,
        color=state.GREEN,
        alpha=0.035,
        zorder=0,
    )


    # Mark the complete-episode percentage change.
    state.axes[1].scatter(
        state.overall_cost_change_percent,
        state.overall_emission_change_percent,
        marker='*',
        s=190,
        color=state.RED,
        edgecolor='black',
        linewidth=0.8,
        label='Complete episode',
        zorder=7,
    )


    state.axes[1].annotate(
        (
            f'Complete episode\n'
            f'Cost: {state.overall_cost_change_percent:+.1f}%\n'
            f'Emissions: '
            f'{state.overall_emission_change_percent:+.1f}%'
        ),
        xy=(
            state.overall_cost_change_percent,
            state.overall_emission_change_percent,
        ),
        xytext=(
            0.97,
            0.96,
        ),
        xycoords='data',
        textcoords='axes fraction',
        horizontalalignment='right',
        verticalalignment='top',
        fontsize=8.4,
        fontweight='bold',
        bbox={
            'boxstyle': 'round,pad=0.38',
            'facecolor': 'white',
            'edgecolor': state.RED,
            'linewidth': 1.0,
            'alpha': 0.95,
        },
        arrowprops={
            'arrowstyle': '-',
            'color': state.RED,
            'linewidth': 0.8,
        },
        zorder=8,
    )


    state.axes[1].set_xlabel(
        'Weekly electricity-cost change relative to baseline (%)'
    )

    state.axes[1].set_ylabel(
        'Weekly carbon-emission change relative to baseline (%)'
    )

    state.axes[1].set_title(
        '(b) Weekly Cost-Carbon Outcome',
        loc='left',
        pad=10,
    )


    state.axes[1].legend(
        loc='lower right',
        ncol=1,
        frameon=True,
        framealpha=0.94,
        facecolor='white',
        edgecolor='#CCCCCC',
        fontsize=8.5,
    )


    state.axes[1].xaxis.set_major_locator(
        ticker.MaxNLocator(
            nbins=6,
        )
    )

    state.axes[1].yaxis.set_major_locator(
        ticker.MaxNLocator(
            nbins=6,
        )
    )


    state.axes[1].xaxis.set_major_formatter(
        ticker.PercentFormatter(
            xmax=100.0,
            decimals=0,
        )
    )

    state.axes[1].yaxis.set_major_formatter(
        ticker.PercentFormatter(
            xmax=100.0,
            decimals=0,
        )
    )


    # Add quadrant labels.
    state.axes[1].text(
        0.02,
        0.03,
        'Both improved',
        transform=state.axes[1].transAxes,
        horizontalalignment='left',
        verticalalignment='bottom',
        fontsize=8.2,
        fontweight='bold',
        color=state.GREEN,
    )

    state.axes[1].text(
        0.98,
        0.03,
        'Cost worsened,\nemissions improved',
        transform=state.axes[1].transAxes,
        horizontalalignment='right',
        verticalalignment='bottom',
        fontsize=7.7,
        color='#555555',
    )

    state.axes[1].text(
        0.02,
        0.97,
        'Cost improved,\nemissions worsened',
        transform=state.axes[1].transAxes,
        horizontalalignment='left',
        verticalalignment='top',
        fontsize=7.7,
        color='#555555',
    )

    state.axes[1].text(
        0.98,
        0.78,
        'Both worsened',
        transform=state.axes[1].transAxes,
        horizontalalignment='right',
        verticalalignment='top',
        fontsize=7.7,
        color=state.RED,
    )


    # =============================================================================
    # Figure-level title and interpretation note
    # =============================================================================

    state.fig.suptitle(
        'Temporal Electricity-Cost and Carbon-Emission Outcomes',
        fontsize=13,
        fontweight='bold',
        y=0.99,
    )


    state.fig.text(
        0.5,
        0.012,
        (
            'Arrows in Panel (a) originate at the no-storage baseline and '
            'terminate at the TD3-controlled weekly outcome. Negative changes '
            'in Panel (b) indicate improvement relative to baseline. The figure '
            'uses complete sequential 168-hour weeks and does not represent '
            'multi-seed uncertainty or a Pareto front.'
        ),
        horizontalalignment='center',
        verticalalignment='bottom',
        fontsize=8.1,
        color='#444444',
    )


    state.fig.tight_layout(
        rect=[
            0.0,
            0.07,
            1.0,
            0.94,
        ],
        w_pad=2.3,
    )


    # =============================================================================
    # Save figure
    # =============================================================================

    state.save_figure(
        state.fig,
        'plot20_weekly_cost_carbon_tradeoff',
    )


    # =============================================================================
    # Create and save summary tables
    # =============================================================================

    state.plot20_episode_summary = pd.DataFrame([{
        'number_of_complete_weeks': len(
            state.weekly_tradeoff
        ),
        'partial_week_numbers': ', '.join([
            str(week_number)
            for week_number in state.partial_week_numbers
        ]),
        'overall_td3_electricity_cost':
            state.overall_td3_cost,
        'overall_baseline_electricity_cost':
            state.overall_baseline_cost,
        'overall_electricity_cost_change_percent':
            state.overall_cost_change_percent,
        'overall_td3_carbon_emission_kg':
            state.overall_td3_emission,
        'overall_baseline_carbon_emission_kg':
            state.overall_baseline_emission,
        'overall_carbon_emission_change_percent':
            state.overall_emission_change_percent,
        'weeks_both_improved':
            state.both_improved_count,
        'weeks_cost_only_improved':
            state.cost_only_count,
        'weeks_emission_only_improved':
            state.emission_only_count,
        'weeks_both_worsened':
            state.both_worsened_count,
    }])


    state.plot20_weekly_path = (
        state.TABLES_DIR
        / 'plot20_weekly_cost_carbon_tradeoff.csv'
    )

    state.plot20_episode_path = (
        state.TABLES_DIR
        / 'plot20_weekly_cost_carbon_tradeoff_episode_summary.csv'
    )


    state.weekly_tradeoff.to_csv(
        state.plot20_weekly_path,
        index=False,
    )

    state.plot20_episode_summary.to_csv(
        state.plot20_episode_path,
        index=False,
    )


    display(
        state.plot20_episode_summary
    )

    display(
        state.weekly_tradeoff.head(
            10
        )
    )


    print(
        'Weekly cost-carbon table saved to:\n'
        f'{state.plot20_weekly_path}'
    )

    print(
        'Episode-level cost-carbon summary saved to:\n'
        f'{state.plot20_episode_path}'
    )

def cell_14(state):
    """Cell 14: empty placeholder cell."""
    pass



# ---------------------------------------------------------------------------
# Ordered list of cell functions for run_all().
# ---------------------------------------------------------------------------
_CELL_FUNCTIONS = [

    cell_00,
    cell_01,
    cell_02,
    cell_03,
    cell_04,
    cell_05,
    cell_06,
    cell_07,
    cell_08,
    cell_09,
    cell_10,
    cell_11,
    cell_12,
    cell_13,
    cell_14,

]

def run_all(state=None):
    """Run every cell function in notebook order on one shared state."""
    if state is None:
        state = create_state()
    for fn in _CELL_FUNCTIONS:
        fn(state)
    return state

__all__ = ['create_state', 'run_all']
