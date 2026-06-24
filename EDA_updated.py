#!/usr/bin/env python
# coding: utf-8

# # **Comprehensive Exploratory Data Analysis for the Grid2Op Thesis Dataset**
# 
# 

# In[1]:


# SETUP 01: IMPORTS, PATHS, AND GLOBAL CONFIGURATION
from pathlib import Path
from collections import Counter
from itertools import combinations
import hashlib
import json
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
LABEL_ORDER = ["normal", "overload", "line_trip", "cascade"]
LABEL_TO_INT = {label: index for index, label in enumerate(LABEL_ORDER)}
INT_TO_LABEL = {value: key for key, value in LABEL_TO_INT.items()}

# The notebook is intended to run from the repository root.
JSONL_PATH = Path("data/grid_dataset_neurips2020.jsonl")
META_PATH = Path("data/grid_dataset_neurips2020_meta.json")
OUTPUT_DIR = Path("eda_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sampling keeps visualization and diagnostic baselines practical for 300,000 rows.
VIS_SAMPLE_SIZE = 5_000
STAT_SAMPLE_SIZE = 20_000
MODEL_SAMPLE_SIZE = 40_000
EXPENSIVE_SAMPLE_SIZE = 3_000
SHAP_SAMPLE_SIZE = 500

# Scan all JSONL rows during the final integrity audit. Set a positive integer to cap the scan.
RAW_AUDIT_MAX_LINES = None

np.random.seed(RANDOM_STATE)

if not JSONL_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {JSONL_PATH.resolve()}\n"
        "Run this notebook from the thesis-repository root or update JSONL_PATH."
    )

if not META_PATH.exists():
    raise FileNotFoundError(
        f"Metadata file not found at: {META_PATH.resolve()}\n"
        "Run this notebook from the thesis-repository root or update META_PATH."
    )

with META_PATH.open("r", encoding="utf-8") as file:
    metadata = json.load(file)

print("Dataset path :", JSONL_PATH)
print("Metadata path:", META_PATH)
print("Environment  :", metadata.get("env_name"))
print("Records      :", f'{metadata.get("total_records", 0):,}')


# In[2]:


# SETUP 02: LOAD JSONL RECORDS AND DERIVE RAW-STATE EDA FEATURES
def _finite_array(record, key, dtype=np.float64):
    """Return a finite NumPy array for a JSONL field."""
    values = record.get(key, [])
    array = np.asarray(values, dtype=dtype)
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_ratio(numerator, denominator, default=0.0):
    return float(numerator / denominator) if abs(float(denominator)) > 1e-12 else float(default)


def infer_physical_label(max_rho, connected_lines, total_lines):
    """Reconstruct the exact frame-based priority rule used by the generator."""
    if float(max_rho) >= 1.0:
        return "overload"
    disconnected_lines = int(total_lines) - int(connected_lines)
    if disconnected_lines == 0:
        return "normal"
    if disconnected_lines == 1:
        return "line_trip"
    return "cascade"


def load_and_extract_features(filepath):
    """
    Stream the completed JSONL dataset and derive graph-relevant aggregate features.

    These tabular aggregates are used for EDA only. The GNN still receives node-level
    and edge-level graph tensors from scripts/pyg_data.py.
    """
    rows = []

    with Path(filepath).open("r", encoding="utf-8") as file:
        for record_index, line in enumerate(file):
            record = json.loads(line)

            rho = _finite_array(record, "rho")
            line_status = np.asarray(record.get("line_status", np.ones(len(rho))), dtype=np.int64)
            load_p = _finite_array(record, "load_p")
            gen_p = _finite_array(record, "gen_p")
            p_or = _finite_array(record, "p_or")
            q_or = _finite_array(record, "q_or")
            v_or = _finite_array(record, "v_or")
            v_ex = _finite_array(record, "v_ex")

            if rho.size == 0:
                continue

            total_lines = int(line_status.size)
            connected_lines = int(np.sum(line_status == 1))
            disconnected_lines = int(total_lines - connected_lines)

            total_load = float(np.sum(load_p))
            total_gen = float(np.sum(gen_p))
            max_rho = float(np.max(rho))
            min_rho = float(np.min(rho))
            mean_rho = float(np.mean(rho))

            rows.append({
                "record_index": int(record_index),
                "label": str(record.get("label", "unknown")),
                "label_int": int(record.get("label_int", LABEL_TO_INT.get(str(record.get("label")), -1))),
                "physical_label_reconstructed": infer_physical_label(max_rho, connected_lines, total_lines),
                "fault_loc": int(record.get("fault_loc", -1)) if record.get("fault_loc") is not None else -1,
                "timestep": int(record.get("timestep", -1)),
                "chronic_id": int(record.get("chronic_id", -1)),
                "reward": float(record.get("reward", 0.0)),
                "max_rho": max_rho,
                "min_rho": min_rho,
                "mean_rho": mean_rho,
                "median_rho": float(np.median(rho)),
                "std_rho": float(np.std(rho)),
                "rho_q75": float(np.quantile(rho, 0.75)),
                "rho_range": float(max_rho - min_rho),
                "rho_above_90_pct": int(np.sum(rho > 0.90)),
                "rho_above_100_pct": int(np.sum(rho >= 1.0)),
                "rho_at_clip_count": int(np.sum(rho >= float(metadata.get("rho_clip", 2.0)))),
                "total_load": total_load,
                "mean_load": float(np.mean(load_p)) if load_p.size else 0.0,
                "std_load": float(np.std(load_p)) if load_p.size else 0.0,
                "total_gen": total_gen,
                "mean_gen": float(np.mean(gen_p)) if gen_p.size else 0.0,
                "std_gen": float(np.std(gen_p)) if gen_p.size else 0.0,
                "load_gen_ratio": _safe_ratio(total_load, total_gen),
                "power_balance": float(total_gen - total_load),
                "abs_power_balance": float(abs(total_gen - total_load)),
                "mean_abs_p_or": float(np.mean(np.abs(p_or))) if p_or.size else 0.0,
                "mean_abs_q_or": float(np.mean(np.abs(q_or))) if q_or.size else 0.0,
                "mean_v_or": float(np.mean(v_or)) if v_or.size else 0.0,
                "mean_v_ex": float(np.mean(v_ex)) if v_ex.size else 0.0,
                "total_lines": total_lines,
                "connected_lines": connected_lines,
                "disconnected_lines": disconnected_lines,
                "connected_fraction": _safe_ratio(connected_lines, total_lines),
            })

    frame = pd.DataFrame(rows)
    frame["label"] = pd.Categorical(frame["label"], categories=LABEL_ORDER, ordered=True)
    return frame


df = load_and_extract_features(JSONL_PATH)

MODEL_FEATURES_FULL = [
    "max_rho", "mean_rho", "std_rho", "rho_range",
    "rho_above_90_pct", "rho_above_100_pct",
    "total_load", "total_gen", "load_gen_ratio",
    "power_balance", "abs_power_balance",
    "mean_abs_p_or", "mean_abs_q_or", "mean_v_or", "mean_v_ex",
    "connected_lines", "disconnected_lines", "connected_fraction",
]

# Excludes the most direct rule-defining signals for a stricter diagnostic baseline.
MODEL_FEATURES_REDUCED = [
    "mean_rho", "std_rho", "rho_range", "rho_above_90_pct",
    "total_load", "total_gen", "load_gen_ratio",
    "power_balance", "abs_power_balance",
    "mean_abs_p_or", "mean_abs_q_or", "mean_v_or", "mean_v_ex",
]

EDA_TABLES = {}
EDA_RESULTS = {}

print(f"Loaded rows: {len(df):,}")
print(f"Columns    : {len(df.columns)}")
display(df.head())


# In[3]:


# SETUP 03: REUSABLE SAMPLING, PLOTTING, AND DIAGNOSTIC-BASELINE HELPERS
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def store_table(name, table):
    """Store and display a report-ready table."""
    EDA_TABLES[name] = table.copy()
    display(table)
    return table


def stratified_sample(frame, n, label_col="label", random_state=RANDOM_STATE):
    """Sample rows while approximately preserving class representation."""
    if len(frame) <= n:
        return frame.copy()
    fractions = frame[label_col].value_counts(normalize=True, dropna=False)
    pieces = []
    for label, fraction in fractions.items():
        group = frame[frame[label_col] == label]
        take = max(1, min(len(group), int(round(n * float(fraction)))))
        pieces.append(group.sample(n=take, random_state=random_state))
    sampled = pd.concat(pieces, ignore_index=False)
    if len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=random_state)
    return sampled.copy()


def model_frame(feature_cols=MODEL_FEATURES_FULL, n=MODEL_SAMPLE_SIZE):
    """Return a finite diagnostic-modelling sample."""
    sample = stratified_sample(df, min(n, len(df)))
    X = sample[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = sample["label"].astype(str)
    return sample, X, y


def train_test_baseline(feature_cols=MODEL_FEATURES_FULL, n=MODEL_SAMPLE_SIZE, class_weight=None):
    """Train a reproducible tabular Random Forest for EDA diagnostics only."""
    sample, X, y = model_frame(feature_cols=feature_cols, n=n)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=140,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight=class_weight,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return {
        "sample": sample, "X": X, "y": y,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "model": model, "pred": pred,
    }


def plot_bar(series, title, xlabel="", ylabel="Count", rotation=0):
    plt.figure(figsize=(9, 5))
    series.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.show()


def plot_line(x, y, title, xlabel, ylabel):
    plt.figure(figsize=(9, 5))
    plt.plot(x, y, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


print("Reusable helpers are ready.")


# ## **Automatic Output Save**

# In[ ]:


# AUTOMATIC OUTPUT PATHS FOR ALL TABLES AND FIGURES

from pathlib import Path
from collections import Counter
import json
import re

# 1. Create organized output folders

OUTPUT_DIR = Path("eda_outputs")

TABLE_OUTPUT_DIR = OUTPUT_DIR / "tables"
FIGURE_OUTPUT_DIR = OUTPUT_DIR / "figures"
SUMMARY_OUTPUT_DIR = OUTPUT_DIR / "summaries"

for directory in [
    OUTPUT_DIR,
    TABLE_OUTPUT_DIR,
    FIGURE_OUTPUT_DIR,
    SUMMARY_OUTPUT_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

# 2. Track all saved outputs

SAVED_TABLE_PATHS = {}
SAVED_FIGURE_PATHS = []

_FIGURE_NAME_COUNTER = Counter()
_SAVED_FIGURE_OBJECTS = set()

# Preserve the original Matplotlib function before overriding it.
_ORIGINAL_PLT_SHOW = plt.show


def make_safe_filename(text):
    """Convert a chart title or table name into a filesystem-safe filename."""
    text = str(text).strip().lower()

    text = re.sub(
        r"[^a-z0-9]+",
        "_",
        text,
    )

    text = re.sub(
        r"_+",
        "_",
        text,
    )

    text = text.strip("_")

    return text or "unnamed_output"


# 3. Replace store_table() so that every table is saved immediately

def store_table(name, table):
    """
    Store, display, and immediately export a report-ready table.

    Every registered table is saved automatically as:
    eda_outputs/tables/<table_name>.csv
    """

    safe_name = make_safe_filename(
        name
    )

    output_path = (
        TABLE_OUTPUT_DIR
        / f"{safe_name}.csv"
    )

    table_to_save = table.copy()

    EDA_TABLES[
        name
    ] = table_to_save

    table_to_save.to_csv(
        output_path,
        index=True,
    )

    SAVED_TABLE_PATHS[
        name
    ] = str(
        output_path.resolve()
    )

    display(
        table_to_save
    )

    print(
        f"Saved table: {output_path.resolve()}"
    )

    return table_to_save


# 4. Override plt.show() so that every Matplotlib chart is saved

def save_and_show_all_open_figures(*args, **kwargs):
    """
    Save every newly created Matplotlib figure before displaying it.

    The figure title is used as the filename when available.
    Files are saved as:
    eda_outputs/figures/<chart_title>_<number>.png
    """

    for figure_number in plt.get_fignums():

        figure = plt.figure(
            figure_number
        )

        figure_object_id = id(
            figure
        )

        # Prevent saving the same figure repeatedly.
        if figure_object_id in _SAVED_FIGURE_OBJECTS:
            continue

        chart_title = ""

        for axis in figure.axes:
            if axis.get_title():
                chart_title = axis.get_title()
                break

        if not chart_title:
            chart_title = (
                f"figure_{figure_number}"
            )

        safe_title = make_safe_filename(
            chart_title
        )

        _FIGURE_NAME_COUNTER[
            safe_title
        ] += 1

        occurrence_number = (
            _FIGURE_NAME_COUNTER[
                safe_title
            ]
        )

        output_path = (
            FIGURE_OUTPUT_DIR
            / (
                f"{safe_title}_"
                f"{occurrence_number:02d}.png"
            )
        )

        figure.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        SAVED_FIGURE_PATHS.append(
            str(
                output_path.resolve()
            )
        )

        _SAVED_FIGURE_OBJECTS.add(
            figure_object_id
        )

        print(
            f"Saved figure: {output_path.resolve()}"
        )

    _ORIGINAL_PLT_SHOW(
        *args,
        **kwargs,
    )


plt.show = save_and_show_all_open_figures

print(
    "Automatic table and figure export is ready."
)

print(
    f"Table folder  : {TABLE_OUTPUT_DIR.resolve()}"
)

print(
    f"Figure folder : {FIGURE_OUTPUT_DIR.resolve()}"
)

print(
    f"Summary folder: {SUMMARY_OUTPUT_DIR.resolve()}"
)


# ## **A. Dataset Structure and Physical Consistency**
# 

# In[4]:


# EDA 01: DATAFRAME PREVIEW, SHAPE, AND SCHEMA VALIDATION
print(f"DataFrame type : {type(df)}")
print(f"Rows           : {len(df):,}")
print(f"Columns        : {len(df.columns)}")
print("\nColumn names:")
print(df.columns.tolist())
print("\nFirst 30 rows:")
display(df.head(30))


# In[5]:


# EDA 02: CLASS DISTRIBUTION ANALYSIS
class_counts = df["label"].value_counts().reindex(LABEL_ORDER, fill_value=0)
class_percentages = (100 * class_counts / len(df)).round(2)
class_distribution = pd.DataFrame({
    "records": class_counts.astype(int),
    "percentage": class_percentages,
})
store_table("02_class_distribution", class_distribution)
plot_bar(class_counts, "Operational-State Class Distribution", "Operational state", "Records")


# In[6]:


# EDA 03: FRAME-BASED PHYSICAL LABEL CONSISTENCY AUDIT
label_consistency = (
    df["label"].astype(str) == df["physical_label_reconstructed"].astype(str)
)
consistency_summary = pd.DataFrame({
    "metric": ["matching labels", "contradictory labels", "consistency percentage"],
    "value": [
        int(label_consistency.sum()),
        int((~label_consistency).sum()),
        round(100 * float(label_consistency.mean()), 6),
    ],
})
store_table("03_physical_label_consistency", consistency_summary)

if (~label_consistency).any():
    print("\nExample contradictions:")
    display(
        df.loc[
            ~label_consistency,
            ["record_index", "label", "physical_label_reconstructed",
             "max_rho", "connected_lines", "disconnected_lines"],
        ].head(30)
    )
else:
    print("\nNo label contradictions were found.")


# In[7]:


# EDA 04: TOPOLOGY AND LINE-STATUS ANALYSIS
topology_summary = (
    df.groupby("label", observed=False)
      .agg(
          records=("label", "size"),
          mean_connected_lines=("connected_lines", "mean"),
          min_connected_lines=("connected_lines", "min"),
          max_connected_lines=("connected_lines", "max"),
          mean_disconnected_lines=("disconnected_lines", "mean"),
          max_disconnected_lines=("disconnected_lines", "max"),
          mean_connected_fraction=("connected_fraction", "mean"),
      )
      .round(4)
)
store_table("04_topology_summary", topology_summary)

plot_bar(
    df.groupby("label", observed=False)["disconnected_lines"].mean(),
    "Average Number of Disconnected Lines by Class",
    "Operational state",
    "Average disconnected lines",
)


# In[8]:


# EDA 05: ZERO-VARIANCE AND NEAR-ZERO-VARIANCE FEATURE CHECK
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
variance_table = pd.DataFrame({
    "feature": numeric_cols,
    "variance": [float(df[column].var()) for column in numeric_cols],
    "unique_values": [int(df[column].nunique(dropna=False)) for column in numeric_cols],
})
variance_table["zero_variance"] = variance_table["variance"].fillna(0.0) == 0.0
variance_table["near_zero_variance"] = variance_table["variance"].fillna(0.0) < 1e-10
variance_table = variance_table.sort_values(
    ["zero_variance", "near_zero_variance", "variance"],
    ascending=[False, False, True],
)
store_table("05_variance_check", variance_table)


# In[9]:


# EDA 06: FEATURE CORRELATION MATRIX
corr_features = MODEL_FEATURES_FULL
correlation_matrix = df[corr_features].corr(numeric_only=True)
store_table("06_correlation_matrix", correlation_matrix.round(4))

plt.figure(figsize=(12, 10))
plt.imshow(correlation_matrix, aspect="auto")
plt.colorbar(label="Pearson correlation")
plt.xticks(range(len(corr_features)), corr_features, rotation=90)
plt.yticks(range(len(corr_features)), corr_features)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()

high_corr_pairs = []
for left, right in combinations(corr_features, 2):
    value = float(correlation_matrix.loc[left, right])
    if abs(value) >= 0.90:
        high_corr_pairs.append({"feature_1": left, "feature_2": right, "correlation": value})
store_table("06_high_correlation_pairs", pd.DataFrame(high_corr_pairs))


# In[10]:


# EDA 07: NODE-VERSUS-EDGE TOPOLOGY-SIGNAL AUDIT
# This EDA checks whether topology changes are visible in graph-relevant aggregates.
topology_signal = (
    df.groupby("label", observed=False)
      .agg(
          mean_connected_fraction=("connected_fraction", "mean"),
          mean_disconnected_lines=("disconnected_lines", "mean"),
          mean_max_rho=("max_rho", "mean"),
          mean_abs_active_flow=("mean_abs_p_or", "mean"),
          mean_abs_reactive_flow=("mean_abs_q_or", "mean"),
      )
      .round(5)
)
store_table("07_node_edge_topology_signals", topology_signal)

for feature in ["connected_fraction", "disconnected_lines", "max_rho"]:
    grouped = []
    for label in LABEL_ORDER:
        label_values = df.loc[df["label"].astype(str) == label, feature]
        grouped.append(
            label_values.sample(n=min(1500, len(label_values)), random_state=RANDOM_STATE).values
        )
    plt.figure(figsize=(9, 5))
    plt.boxplot(grouped, labels=LABEL_ORDER, showfliers=False)
    plt.title(f"Topology-Signal Audit: {feature}")
    plt.xlabel("Operational state")
    plt.ylabel(feature)
    plt.tight_layout()
    plt.show()


# In[11]:


# EDA 08: FAULT-LOCALIZATION BIAS CHECK
localized = df[df["fault_loc"] >= 0].copy()
fault_loc_counts = localized["fault_loc"].value_counts().sort_index()
fault_loc_summary = pd.DataFrame({
    "localized_records": [len(localized)],
    "unlocalized_records": [int((df["fault_loc"] < 0).sum())],
    "unique_localized_substations": [int(localized["fault_loc"].nunique())],
    "most_common_fault_substation": [int(fault_loc_counts.index[0]) if len(fault_loc_counts) else -1],
    "largest_substation_count": [int(fault_loc_counts.iloc[0]) if len(fault_loc_counts) else 0],
})
store_table("08_fault_localization_summary", fault_loc_summary)

if len(fault_loc_counts):
    plot_bar(
        fault_loc_counts,
        "Fault-Location Frequency by Origin-Side Substation",
        "Substation ID",
        "Localized records",
        rotation=90,
    )


# In[12]:


# EDA 09: GRAPH-POOLING SIGNAL HYPOTHESIS TEST
# Proxy statistics show why mean, max, and extreme-value pooling preserve different signals.
pooling_proxy = (
    df.groupby("label", observed=False)
      .agg(
          rho_mean_proxy=("mean_rho", "mean"),
          rho_max_proxy=("max_rho", "mean"),
          rho_spread_proxy=("rho_range", "mean"),
          active_edge_count_proxy=("connected_lines", "mean"),
          disconnected_edge_count_proxy=("disconnected_lines", "mean"),
      )
      .round(5)
)
store_table("09_pooling_signal_proxy", pooling_proxy)

for feature in ["mean_rho", "max_rho", "rho_range", "connected_lines"]:
    series = df.groupby("label", observed=False)[feature].mean()
    plot_bar(series, f"Pooling-Signal Proxy: Mean {feature} by Class", "Operational state", f"Mean {feature}")


# ## **B. Data Quality and Descriptive Exploration**
# 

# In[13]:


# EDA 10: BASIC DESCRIPTIVE STATISTICS
descriptive_statistics = df[MODEL_FEATURES_FULL + ["fault_loc", "timestep", "reward"]].describe().T
store_table("10_descriptive_statistics", descriptive_statistics.round(5))


# In[14]:


# EDA 11: MISSING-VALUE AND NON-FINITE-VALUE AUDIT
numeric_frame = df.select_dtypes(include=[np.number])
quality_table = pd.DataFrame({
    "missing_values": df.isna().sum(),
    "missing_percentage": (100 * df.isna().mean()).round(6),
})
quality_table["positive_infinity"] = 0
quality_table["negative_infinity"] = 0
for column in numeric_frame.columns:
    values = numeric_frame[column].to_numpy(dtype=float)
    quality_table.loc[column, "positive_infinity"] = int(np.isposinf(values).sum())
    quality_table.loc[column, "negative_infinity"] = int(np.isneginf(values).sum())

store_table("11_missing_and_nonfinite_values", quality_table)


# In[15]:


# EDA 12: DUPLICATE-RECORD ANALYSIS
aggregate_duplicate_cols = [
    "label", "max_rho", "mean_rho", "std_rho",
    "total_load", "total_gen", "connected_lines",
    "disconnected_lines", "fault_loc", "timestep", "chronic_id",
]
aggregate_duplicate_mask = df.duplicated(subset=aggregate_duplicate_cols, keep=False)
duplicate_summary = pd.DataFrame({
    "metric": ["rows in aggregate-level duplicate groups", "aggregate-level duplicate percentage"],
    "value": [
        int(aggregate_duplicate_mask.sum()),
        round(100 * float(aggregate_duplicate_mask.mean()), 6),
    ],
})
store_table("12_duplicate_summary", duplicate_summary)

if aggregate_duplicate_mask.any():
    display(df.loc[aggregate_duplicate_mask, aggregate_duplicate_cols].head(30))


# In[16]:


# EDA 13: DATA-TYPE AND CARDINALITY VERIFICATION
dtype_table = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "unique_values": df.nunique(dropna=False),
    "missing_values": df.isna().sum(),
})
store_table("13_dtype_cardinality", dtype_table)


# In[17]:


# EDA 14: FEATURE-RANGE AND METADATA-CONSISTENCY CHECK
range_table = pd.DataFrame({
    "minimum": df[MODEL_FEATURES_FULL].min(),
    "maximum": df[MODEL_FEATURES_FULL].max(),
    "mean": df[MODEL_FEATURES_FULL].mean(),
    "standard_deviation": df[MODEL_FEATURES_FULL].std(),
}).round(6)
store_table("14_feature_ranges", range_table)

metadata_check = pd.DataFrame({
    "metadata_field": [
        "environment", "total_records", "n_sub", "n_line", "n_load", "n_gen",
        "rho_clip", "node_feature_dim", "edge_feature_dim",
    ],
    "metadata_value": [
        metadata.get("env_name"), metadata.get("total_records"),
        metadata.get("n_sub"), metadata.get("n_line"), metadata.get("n_load"),
        metadata.get("n_gen"), metadata.get("rho_clip"),
        metadata.get("node_feature_dim"), metadata.get("edge_feature_dim"),
    ],
    "observed_or_expected_value": [
        "l2rpn_neurips_2020_track1_small", len(df),
        metadata.get("n_sub"), int(df["total_lines"].iloc[0]),
        metadata.get("n_load"), metadata.get("n_gen"),
        float(df["max_rho"].max()), 4,
        "Generator metadata stores 3 raw edge measurements; current pyg_data.py constructs 4 edge attributes including line_status.",
    ],
})
store_table("14_metadata_consistency", metadata_check)


# ## **C. Visualization of Class-Conditional Behaviour**
# 

# In[18]:


# EDA 15: BOX-PLOT VISUALIZATION BY CLASS
box_features = ["max_rho", "mean_rho", "std_rho", "total_load", "total_gen", "connected_fraction", "disconnected_lines"]
sample = stratified_sample(df, VIS_SAMPLE_SIZE)

for feature in box_features:
    grouped = [sample.loc[sample["label"].astype(str) == label, feature].values for label in LABEL_ORDER]
    plt.figure(figsize=(9, 5))
    plt.boxplot(grouped, labels=LABEL_ORDER, showfliers=False)
    plt.title(f"Box Plot by Class: {feature}")
    plt.xlabel("Operational state")
    plt.ylabel(feature)
    plt.tight_layout()
    plt.show()


# In[19]:


# EDA 16: VIOLIN-PLOT DISTRIBUTION BY CLASS
violin_features = ["max_rho", "mean_rho", "total_load", "total_gen", "connected_fraction"]
sample = stratified_sample(df, VIS_SAMPLE_SIZE)

for feature in violin_features:
    grouped = [sample.loc[sample["label"].astype(str) == label, feature].values for label in LABEL_ORDER]
    plt.figure(figsize=(9, 5))
    plt.violinplot(grouped, showmeans=True, showmedians=True)
    plt.xticks(range(1, len(LABEL_ORDER) + 1), LABEL_ORDER)
    plt.title(f"Violin Plot by Class: {feature}")
    plt.xlabel("Operational state")
    plt.ylabel(feature)
    plt.tight_layout()
    plt.show()


# In[20]:


# EDA 17: PAIRWISE SCATTER-MATRIX EQUIVALENT
# Separate charts are used instead of a subplot matrix to keep every figure readable.
scatter_pairs = [
    ("mean_rho", "max_rho"),
    ("total_load", "total_gen"),
    ("connected_fraction", "max_rho"),
    ("mean_abs_p_or", "mean_abs_q_or"),
]
sample = stratified_sample(df, min(3000, VIS_SAMPLE_SIZE))

for x_feature, y_feature in scatter_pairs:
    plt.figure(figsize=(8, 5))
    for label in LABEL_ORDER:
        group = sample[sample["label"].astype(str) == label]
        plt.scatter(group[x_feature], group[y_feature], s=12, alpha=0.45, label=label)
    plt.title(f"Pairwise Scatter Plot: {x_feature} vs. {y_feature}")
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.legend()
    plt.tight_layout()
    plt.show()


# In[21]:


# EDA 18: TWO-DIMENSIONAL DENSITY HEATMAP
sample = stratified_sample(df, min(20_000, len(df)))
plt.figure(figsize=(8, 6))
plt.hist2d(sample["mean_rho"], sample["max_rho"], bins=60)
plt.colorbar(label="Record count")
plt.axhline(1.0, linestyle="--", label="Overload threshold")
plt.title("Two-Dimensional Density: Mean Rho vs. Maximum Rho")
plt.xlabel("mean_rho")
plt.ylabel("max_rho")
plt.legend()
plt.tight_layout()
plt.show()


# ## **D. Statistical Testing**
# 

# In[22]:


# EDA 19: ONE-WAY ANOVA ACROSS OPERATIONAL CLASSES
from scipy.stats import f_oneway

anova_rows = []
for feature in MODEL_FEATURES_FULL:
    groups = []
    for label in LABEL_ORDER:
        values = df.loc[df["label"].astype(str) == label, feature].dropna()
        groups.append(values.sample(n=min(STAT_SAMPLE_SIZE // len(LABEL_ORDER), len(values)), random_state=RANDOM_STATE).values)
    statistic, p_value = f_oneway(*groups)
    anova_rows.append({"feature": feature, "f_statistic": statistic, "p_value": p_value})

anova_table = pd.DataFrame(anova_rows).sort_values("p_value")
store_table("19_anova_results", anova_table)


# In[23]:


# EDA 20: CHI-SQUARE TEST FOR LABEL ASSOCIATION WITH DISCRETE GRID STATES
from scipy.stats import chi2_contingency

chi_square_rows = []
discrete_candidates = {
    "disconnected_lines": pd.cut(df["disconnected_lines"], bins=[-1, 0, 1, 3, np.inf], labels=["0", "1", "2-3", "4+"]),
    "rho_above_100_pct": pd.cut(df["rho_above_100_pct"], bins=[-1, 0, 1, 3, np.inf], labels=["0", "1", "2-3", "4+"]),
    "rho_above_90_pct": pd.cut(df["rho_above_90_pct"], bins=[-1, 0, 1, 3, np.inf], labels=["0", "1", "2-3", "4+"]),
}

for feature, binned in discrete_candidates.items():
    contingency = pd.crosstab(df["label"], binned)
    statistic, p_value, dof, _ = chi2_contingency(contingency)
    chi_square_rows.append({
        "feature": feature,
        "chi_square": statistic,
        "degrees_of_freedom": dof,
        "p_value": p_value,
    })

chi_square_table = pd.DataFrame(chi_square_rows).sort_values("p_value")
store_table("20_chi_square_results", chi_square_table)


# In[24]:


# EDA 21: PAIRWISE KOLMOGOROV-SMIRNOV DISTRIBUTION TEST
from scipy.stats import ks_2samp

ks_rows = []
ks_features = ["max_rho", "mean_rho", "total_load", "total_gen", "connected_fraction"]
for feature in ks_features:
    for label_a, label_b in combinations(LABEL_ORDER, 2):
        values_a = df.loc[df["label"].astype(str) == label_a, feature].dropna()
        values_b = df.loc[df["label"].astype(str) == label_b, feature].dropna()
        values_a = values_a.sample(n=min(5000, len(values_a)), random_state=RANDOM_STATE)
        values_b = values_b.sample(n=min(5000, len(values_b)), random_state=RANDOM_STATE)
        statistic, p_value = ks_2samp(values_a, values_b)
        ks_rows.append({
            "feature": feature,
            "class_a": label_a,
            "class_b": label_b,
            "ks_statistic": statistic,
            "p_value": p_value,
        })

ks_table = pd.DataFrame(ks_rows).sort_values(["feature", "p_value"])
store_table("21_pairwise_ks_results", ks_table)


# In[25]:


# EDA 22: SHAPIRO-WILK NORMALITY TEST
from scipy.stats import shapiro

shapiro_rows = []
for feature in MODEL_FEATURES_REDUCED:
    values = df[feature].dropna()
    values = values.sample(n=min(5000, len(values)), random_state=RANDOM_STATE)
    statistic, p_value = shapiro(values)
    shapiro_rows.append({
        "feature": feature,
        "shapiro_statistic": statistic,
        "p_value": p_value,
        "approximately_normal_at_0_05": bool(p_value >= 0.05),
    })

shapiro_table = pd.DataFrame(shapiro_rows).sort_values("p_value")
store_table("22_shapiro_normality", shapiro_table)


# In[26]:


# EDA 23: LEVENE TEST FOR HOMOGENEITY OF VARIANCE
from scipy.stats import levene

levene_rows = []
for feature in MODEL_FEATURES_FULL:
    groups = []
    for label in LABEL_ORDER:
        values = df.loc[df["label"].astype(str) == label, feature].dropna()
        groups.append(values.sample(n=min(5000, len(values)), random_state=RANDOM_STATE).values)
    statistic, p_value = levene(*groups, center="median")
    levene_rows.append({
        "feature": feature,
        "levene_statistic": statistic,
        "p_value": p_value,
        "equal_variance_at_0_05": bool(p_value >= 0.05),
    })

levene_table = pd.DataFrame(levene_rows).sort_values("p_value")
store_table("23_levene_results", levene_table)


# ## **E. Feature Engineering and Representation Analysis**
# 

# In[27]:


# EDA 24: FEATURE-SCALING AND STANDARDIZATION ANALYSIS
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

sample = stratified_sample(df, min(STAT_SAMPLE_SIZE, len(df)))
X = sample[MODEL_FEATURES_FULL].fillna(0.0)

scalers = {
    "original": None,
    "standard": StandardScaler(),
    "min_max": MinMaxScaler(),
    "robust": RobustScaler(),
}

scaling_rows = []
for scaler_name, scaler in scalers.items():
    transformed = X.to_numpy() if scaler is None else scaler.fit_transform(X)
    scaling_rows.append({
        "scaler": scaler_name,
        "mean_absolute_mean": float(np.mean(np.abs(np.mean(transformed, axis=0)))),
        "mean_standard_deviation": float(np.mean(np.std(transformed, axis=0))),
        "global_minimum": float(np.min(transformed)),
        "global_maximum": float(np.max(transformed)),
    })

scaling_table = pd.DataFrame(scaling_rows)
store_table("24_scaling_comparison", scaling_table)


# In[28]:


# EDA 25: FEATURE BINNING AND DISCRETIZATION
binned = df.copy()
binned["max_rho_band"] = pd.cut(
    binned["max_rho"],
    bins=[-np.inf, 0.70, 0.90, 1.00, 1.20, 1.50, np.inf],
    labels=["<0.70", "0.70-0.90", "0.90-1.00", "1.00-1.20", "1.20-1.50", ">=1.50"],
    right=False,
)
rho_band_table = pd.crosstab(binned["max_rho_band"], binned["label"], margins=True)
store_table("25_rho_band_by_label", rho_band_table)

plot_bar(
    binned["max_rho_band"].value_counts().sort_index(),
    "Maximum-Rho Discretization",
    "Maximum-rho band",
    "Records",
    rotation=45,
)


# In[29]:


# EDA 26: POLYNOMIAL FEATURE-TRANSFORMATION ANALYSIS
from sklearn.preprocessing import PolynomialFeatures

poly_source = ["max_rho", "mean_rho", "total_load", "total_gen", "connected_fraction"]
sample = stratified_sample(df, min(5000, len(df)))
poly = PolynomialFeatures(degree=2, include_bias=False)
transformed = poly.fit_transform(sample[poly_source].fillna(0.0))
poly_names = poly.get_feature_names_out(poly_source)

poly_summary = pd.DataFrame({
    "original_feature_count": [len(poly_source)],
    "transformed_feature_count": [len(poly_names)],
    "generated_features": [", ".join(poly_names)],
})
store_table("26_polynomial_feature_summary", poly_summary)


# In[30]:


# EDA 27: PCA DIMENSIONALITY-REDUCTION ANALYSIS
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sample = stratified_sample(df, min(STAT_SAMPLE_SIZE, len(df)))
X_scaled = StandardScaler().fit_transform(sample[MODEL_FEATURES_FULL].fillna(0.0))
pca = PCA().fit(X_scaled)

pca_table = pd.DataFrame({
    "component": np.arange(1, len(pca.explained_variance_ratio_) + 1),
    "explained_variance_ratio": pca.explained_variance_ratio_,
    "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
})
store_table("27_pca_variance", pca_table.round(6))

plot_line(
    pca_table["component"],
    pca_table["cumulative_explained_variance"],
    "PCA Cumulative Explained Variance",
    "Principal component",
    "Cumulative explained variance",
)


# In[31]:


# EDA 28: T-SNE MANIFOLD VISUALIZATION
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

sample = stratified_sample(df, min(800, EXPENSIVE_SAMPLE_SIZE, len(df)))
X_scaled = StandardScaler().fit_transform(sample[MODEL_FEATURES_FULL].fillna(0.0))
embedding = TSNE(
    n_components=2,
    random_state=RANDOM_STATE,
    init="pca",
    learning_rate="auto",
    perplexity=25,
    max_iter=400,
).fit_transform(X_scaled)

plt.figure(figsize=(9, 6))
for label in LABEL_ORDER:
    mask = sample["label"].astype(str).to_numpy() == label
    plt.scatter(embedding[mask, 0], embedding[mask, 1], s=12, alpha=0.55, label=label)
plt.title("t-SNE Projection of Grid-State Aggregates")
plt.xlabel("t-SNE dimension 1")
plt.ylabel("t-SNE dimension 2")
plt.legend()
plt.tight_layout()
plt.show()


# In[32]:


# EDA 29: MUTUAL-INFORMATION FEATURE IMPORTANCE
from sklearn.feature_selection import mutual_info_classif

sample, X, y = model_frame(MODEL_FEATURES_FULL, n=MODEL_SAMPLE_SIZE)
mi_values = mutual_info_classif(X, y, random_state=RANDOM_STATE)
mi_table = pd.DataFrame({
    "feature": MODEL_FEATURES_FULL,
    "mutual_information": mi_values,
}).sort_values("mutual_information", ascending=False)
store_table("29_mutual_information", mi_table)

plot_bar(
    mi_table.set_index("feature")["mutual_information"],
    "Mutual-Information Feature Importance",
    "Feature",
    "Mutual information",
    rotation=90,
)


# In[33]:


# EDA 30: RANDOM-FOREST FEATURE IMPORTANCE
sample, X, y = model_frame(MODEL_FEATURES_FULL, n=MODEL_SAMPLE_SIZE)
rf_importance_model = RandomForestClassifier(
    n_estimators=180,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
rf_importance_model.fit(X, y)

rf_importance_table = pd.DataFrame({
    "feature": MODEL_FEATURES_FULL,
    "importance": rf_importance_model.feature_importances_,
}).sort_values("importance", ascending=False)
store_table("30_random_forest_importance", rf_importance_table)

plot_bar(
    rf_importance_table.set_index("feature")["importance"],
    "Random-Forest Feature Importance",
    "Feature",
    "Importance",
    rotation=90,
)


# In[34]:


# EDA 31: CORRELATION-BASED FEATURE-IMPORTANCE ANALYSIS
encoded_label = df["label"].astype(str).map(LABEL_TO_INT).astype(float)
corr_rows = []
for feature in MODEL_FEATURES_FULL:
    value = float(pd.Series(df[feature]).corr(encoded_label))
    corr_rows.append({
        "feature": feature,
        "correlation_with_encoded_label": value,
        "absolute_correlation": abs(value),
    })

label_corr_table = pd.DataFrame(corr_rows).sort_values("absolute_correlation", ascending=False)
store_table("31_label_correlation_importance", label_corr_table)
plot_bar(
    label_corr_table.set_index("feature")["absolute_correlation"],
    "Absolute Correlation with Encoded Operational State",
    "Feature",
    "Absolute correlation",
    rotation=90,
)


# ## **F. Dataset Imbalance, Clustering, Explainability, and Outliers**
# 

# In[35]:


# EDA 32: CLASS-IMBALANCE ANALYSIS
counts = df["label"].value_counts().reindex(LABEL_ORDER, fill_value=0)
largest = int(counts.max())
smallest = int(counts.min())
imbalance_table = pd.DataFrame({
    "records": counts,
    "percentage": (100 * counts / counts.sum()).round(4),
    "inverse_frequency_weight": (counts.sum() / (len(counts) * counts)).round(6),
    "sqrt_inverse_frequency_weight": np.sqrt(counts.sum() / (len(counts) * counts)).round(6),
})
store_table("32_class_imbalance", imbalance_table)

print(f"Largest-to-smallest class ratio: {largest / max(smallest, 1):.4f}")


# In[36]:


# EDA 33: K-MEANS CLUSTERING ANALYSIS
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

sample = stratified_sample(df, min(10_000, len(df)))
X_scaled = StandardScaler().fit_transform(sample[MODEL_FEATURES_FULL].fillna(0.0))
kmeans = KMeans(n_clusters=len(LABEL_ORDER), random_state=RANDOM_STATE, n_init=20)
clusters = kmeans.fit_predict(X_scaled)

cluster_table = pd.crosstab(
    pd.Series(clusters, name="cluster"),
    sample["label"].astype(str).reset_index(drop=True),
)
store_table("33_kmeans_cluster_by_label", cluster_table)

cluster_summary = pd.DataFrame({
    "metric": ["adjusted_rand_index", "silhouette_score"],
    "value": [
        adjusted_rand_score(sample["label"].astype(str), clusters),
        silhouette_score(X_scaled, clusters, sample_size=min(5000, len(sample)), random_state=RANDOM_STATE),
    ],
})
store_table("33_kmeans_summary", cluster_summary)


# In[37]:


# EDA 34: SHAP OR PERMUTATION-IMPORTANCE INTERPRETATION
from sklearn.inspection import permutation_importance

baseline = train_test_baseline(MODEL_FEATURES_FULL, n=min(MODEL_SAMPLE_SIZE, 30_000))
permutation = permutation_importance(
    baseline["model"],
    baseline["X_test"],
    baseline["y_test"],
    n_repeats=5,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    scoring="f1_macro",
)
permutation_table = pd.DataFrame({
    "feature": MODEL_FEATURES_FULL,
    "mean_permutation_importance": permutation.importances_mean,
    "std_permutation_importance": permutation.importances_std,
}).sort_values("mean_permutation_importance", ascending=False)
store_table("34_permutation_importance", permutation_table)

plot_bar(
    permutation_table.set_index("feature")["mean_permutation_importance"],
    "Permutation Importance for the Diagnostic Baseline",
    "Feature",
    "Macro-F1 decrease after permutation",
    rotation=90,
)

try:
    import shap
    shap_sample = baseline["X_test"].sample(n=min(SHAP_SAMPLE_SIZE, len(baseline["X_test"])), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(baseline["model"])
    shap_values = explainer.shap_values(shap_sample)
    print("SHAP values computed successfully for", len(shap_sample), "records.")
except Exception as error:
    print("Optional SHAP rendering was skipped:", error)
    print("Permutation importance above remains available without the optional SHAP package.")


# In[38]:


# EDA 35: OUTLIER DETECTION USING IQR, Z-SCORE, AND MAHALANOBIS DISTANCE
from scipy.stats import chi2, zscore

outlier_features = ["max_rho", "mean_rho", "total_load", "total_gen", "mean_abs_p_or", "mean_abs_q_or"]
sample = stratified_sample(df, min(STAT_SAMPLE_SIZE, len(df)))
X = sample[outlier_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)

q1 = X.quantile(0.25)
q3 = X.quantile(0.75)
iqr = q3 - q1
iqr_mask = ((X < (q1 - 1.5 * iqr)) | (X > (q3 + 1.5 * iqr))).any(axis=1)

z_values = np.abs(zscore(X, nan_policy="omit"))
z_mask = np.any(z_values > 3.0, axis=1)

covariance = np.cov(X.to_numpy(), rowvar=False)
inverse_covariance = np.linalg.pinv(covariance)
centered = X.to_numpy() - X.to_numpy().mean(axis=0)
mahalanobis_squared = np.einsum("ij,jk,ik->i", centered, inverse_covariance, centered)
mahalanobis_threshold = chi2.ppf(0.997, df=X.shape[1])
mahalanobis_mask = mahalanobis_squared > mahalanobis_threshold

outlier_summary = pd.DataFrame({
    "method": ["IQR", "absolute z-score > 3", "Mahalanobis distance"],
    "outlier_records": [int(iqr_mask.sum()), int(z_mask.sum()), int(mahalanobis_mask.sum())],
    "outlier_percentage": [
        100 * float(iqr_mask.mean()),
        100 * float(z_mask.mean()),
        100 * float(mahalanobis_mask.mean()),
    ],
})
store_table("35_outlier_summary", outlier_summary.round(5))


# In[39]:


# EDA 36: THREE-DIMENSIONAL PCA AND ANDREWS-CURVE ANALYSIS
from pandas.plotting import andrews_curves
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sample = stratified_sample(df, min(2500, len(df)))
X_scaled = StandardScaler().fit_transform(sample[MODEL_FEATURES_FULL].fillna(0.0))
projection = PCA(n_components=3).fit_transform(X_scaled)

figure = plt.figure(figsize=(9, 7))
axis = figure.add_subplot(111, projection="3d")
for label in LABEL_ORDER:
    mask = sample["label"].astype(str).to_numpy() == label
    axis.scatter(projection[mask, 0], projection[mask, 1], projection[mask, 2], s=10, alpha=0.45, label=label)
axis.set_title("Three-Dimensional PCA Projection")
axis.set_xlabel("Principal component 1")
axis.set_ylabel("Principal component 2")
axis.set_zlabel("Principal component 3")
axis.legend()
plt.tight_layout()
plt.show()

andrews_sample = stratified_sample(df, min(500, len(df)))
plt.figure(figsize=(11, 6))
andrews_curves(
    andrews_sample[["label", "max_rho", "mean_rho", "total_load", "total_gen", "connected_fraction"]],
    "label",
    alpha=0.35,
)
plt.title("Andrews Curves for Selected Grid-State Features")
plt.tight_layout()
plt.show()


# In[40]:


# EDA 37: PARALLEL-COORDINATES PLOT
from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import MinMaxScaler

sample = stratified_sample(df, min(800, len(df)))
parallel_features = ["max_rho", "mean_rho", "total_load", "total_gen", "connected_fraction", "disconnected_lines"]
scaled = pd.DataFrame(
    MinMaxScaler().fit_transform(sample[parallel_features]),
    columns=parallel_features,
)
scaled["label"] = sample["label"].astype(str).reset_index(drop=True)

plt.figure(figsize=(12, 6))
parallel_coordinates(scaled, "label", alpha=0.20)
plt.title("Parallel-Coordinates Plot of Normalized Grid-State Features")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ## **G. Diagnostic Baseline Modelling**
# 

# In[41]:


# EDA 38: BASIC LEARNING-CURVE ANALYSIS
from sklearn.model_selection import learning_curve

sample, X, y = model_frame(MODEL_FEATURES_REDUCED, n=min(MODEL_SAMPLE_SIZE, 30_000))
curve_model = RandomForestClassifier(n_estimators=80, random_state=RANDOM_STATE, n_jobs=-1)

train_sizes, train_scores, validation_scores = learning_curve(
    curve_model,
    X,
    y,
    cv=3,
    scoring="f1_macro",
    train_sizes=np.linspace(0.15, 1.0, 6),
    n_jobs=-1,
)

learning_curve_table = pd.DataFrame({
    "training_records": train_sizes,
    "training_macro_f1": train_scores.mean(axis=1),
    "validation_macro_f1": validation_scores.mean(axis=1),
    "validation_macro_f1_std": validation_scores.std(axis=1),
})
store_table("38_learning_curve", learning_curve_table.round(6))

plot_line(
    learning_curve_table["training_records"],
    learning_curve_table["training_macro_f1"],
    "Learning Curve: Training Macro-F1",
    "Training records",
    "Macro-F1",
)
plot_line(
    learning_curve_table["training_records"],
    learning_curve_table["validation_macro_f1"],
    "Learning Curve: Validation Macro-F1",
    "Training records",
    "Macro-F1",
)


# In[42]:


# EDA 39: CONFUSION-MATRIX ANALYSIS
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

baseline = train_test_baseline(MODEL_FEATURES_REDUCED, n=MODEL_SAMPLE_SIZE)
matrix = confusion_matrix(baseline["y_test"], baseline["pred"], labels=LABEL_ORDER)
matrix_table = pd.DataFrame(matrix, index=LABEL_ORDER, columns=LABEL_ORDER)
store_table("39_confusion_matrix", matrix_table)

display(pd.DataFrame(classification_report(
    baseline["y_test"],
    baseline["pred"],
    labels=LABEL_ORDER,
    output_dict=True,
    zero_division=0,
)).T.round(5))

display_object = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=LABEL_ORDER)
display_object.plot(values_format="d")
plt.title("Confusion Matrix: Reduced-Feature Diagnostic Baseline")
plt.tight_layout()
plt.show()


# In[43]:


# EDA 40: ENHANCED HISTOGRAM ANALYSIS
histogram_features = ["max_rho", "mean_rho", "std_rho", "total_load", "total_gen", "connected_fraction"]
sample = stratified_sample(df, min(STAT_SAMPLE_SIZE, len(df)))

for feature in histogram_features:
    plt.figure(figsize=(9, 5))
    for label in LABEL_ORDER:
        values = sample.loc[sample["label"].astype(str) == label, feature]
        plt.hist(values, bins=45, alpha=0.35, density=True, label=label)
    plt.title(f"Class-Conditional Histogram: {feature}")
    plt.xlabel(feature)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()


# In[44]:


# EDA 41: FREQUENCY-DOMAIN ANALYSIS OF CHRONIC-ORDERED GRID STRESS
# The largest available chronic is used so that the signal preserves temporal order.
largest_chronic = int(df["chronic_id"].value_counts().index[0])
series = (
    df[df["chronic_id"] == largest_chronic]
      .sort_values("timestep")["max_rho"]
      .to_numpy(dtype=float)
)
series = series - np.mean(series)
frequency = np.fft.rfftfreq(len(series), d=1.0)
magnitude = np.abs(np.fft.rfft(series))

plt.figure(figsize=(9, 5))
plt.plot(frequency[1:], magnitude[1:])
plt.title(f"Frequency-Domain View of Maximum Rho: Chronic {largest_chronic}")
plt.xlabel("Frequency")
plt.ylabel("FFT magnitude")
plt.tight_layout()
plt.show()

frequency_summary = pd.DataFrame({
    "chronic_id": [largest_chronic],
    "ordered_records": [len(series)],
    "dominant_nonzero_frequency": [float(frequency[1:][np.argmax(magnitude[1:])]) if len(frequency) > 1 else np.nan],
})
store_table("41_frequency_domain_summary", frequency_summary)


# In[45]:


# EDA 42: THREE-DIMENSIONAL RAW-FEATURE VISUALIZATION
sample = stratified_sample(df, min(3500, len(df)))

figure = plt.figure(figsize=(9, 7))
axis = figure.add_subplot(111, projection="3d")
for label in LABEL_ORDER:
    group = sample[sample["label"].astype(str) == label]
    axis.scatter(group["mean_rho"], group["connected_fraction"], group["total_load"], s=10, alpha=0.45, label=label)
axis.set_title("Three-Dimensional Grid-State Visualization")
axis.set_xlabel("mean_rho")
axis.set_ylabel("connected_fraction")
axis.set_zlabel("total_load")
axis.legend()
plt.tight_layout()
plt.show()


# In[46]:


# EDA 43: ADVANCED LEARNING-CURVE ANALYSIS WITH ACCURACY AND MACRO-F1
from sklearn.model_selection import learning_curve

sample, X, y = model_frame(MODEL_FEATURES_FULL, n=min(MODEL_SAMPLE_SIZE, 30_000))
curve_model = RandomForestClassifier(n_estimators=80, random_state=RANDOM_STATE, n_jobs=-1)
sizes = np.linspace(0.15, 1.0, 6)

rows = []
for metric in ["accuracy", "f1_macro"]:
    train_sizes, train_scores, validation_scores = learning_curve(
        curve_model, X, y, cv=3, scoring=metric, train_sizes=sizes, n_jobs=-1
    )
    for size, training, validation in zip(train_sizes, train_scores, validation_scores):
        rows.append({
            "metric": metric,
            "training_records": int(size),
            "training_score": float(training.mean()),
            "validation_score": float(validation.mean()),
            "validation_score_std": float(validation.std()),
        })

advanced_curve_table = pd.DataFrame(rows)
store_table("43_advanced_learning_curve", advanced_curve_table.round(6))

for metric in ["accuracy", "f1_macro"]:
    subset = advanced_curve_table[advanced_curve_table["metric"] == metric]
    plot_line(
        subset["training_records"],
        subset["validation_score"],
        f"Advanced Learning Curve: Validation {metric}",
        "Training records",
        metric,
    )


# In[47]:


# EDA 44: PRECISION-RECALL CURVE ANALYSIS
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import label_binarize

baseline = train_test_baseline(MODEL_FEATURES_REDUCED, n=MODEL_SAMPLE_SIZE)
probabilities = baseline["model"].predict_proba(baseline["X_test"])
class_order = baseline["model"].classes_.tolist()
binary_y = label_binarize(baseline["y_test"], classes=class_order)

pr_rows = []
plt.figure(figsize=(9, 6))
for index, label in enumerate(class_order):
    precision, recall, _ = precision_recall_curve(binary_y[:, index], probabilities[:, index])
    average_precision = average_precision_score(binary_y[:, index], probabilities[:, index])
    pr_rows.append({"class": label, "average_precision": average_precision})
    plt.plot(recall, precision, label=f"{label} (AP={average_precision:.4f})")
plt.title("Precision-Recall Curves: Reduced-Feature Diagnostic Baseline")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.tight_layout()
plt.show()

store_table("44_precision_recall_summary", pd.DataFrame(pr_rows))


# In[48]:


# EDA 45: CROSS-VALIDATION FOLD ANALYSIS
from sklearn.model_selection import StratifiedKFold, cross_validate

sample, X, y = model_frame(MODEL_FEATURES_REDUCED, n=min(MODEL_SAMPLE_SIZE, 30_000))
cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)

scores = cross_validate(
    model,
    X,
    y,
    cv=cross_validation,
    scoring={"accuracy": "accuracy", "macro_f1": "f1_macro", "weighted_f1": "f1_weighted"},
    n_jobs=-1,
)

fold_table = pd.DataFrame({
    "fold": np.arange(1, len(scores["test_accuracy"]) + 1),
    "accuracy": scores["test_accuracy"],
    "macro_f1": scores["test_macro_f1"],
    "weighted_f1": scores["test_weighted_f1"],
})
store_table("45_cross_validation_folds", fold_table.round(6))


# ## **H. Temporal, Localization, Robustness, and Physics Audits**
# 

# In[49]:


# EDA 46: FAULT-PROGRESSION AND STATE-TRANSITION ANALYSIS
ordered = df.sort_values(["chronic_id", "timestep", "record_index"]).copy()
ordered["next_label"] = ordered.groupby("chronic_id", observed=False)["label"].shift(-1)
transition_table = pd.crosstab(
    ordered["label"],
    ordered["next_label"],
    normalize="index",
).reindex(index=LABEL_ORDER, columns=LABEL_ORDER, fill_value=0.0)
store_table("46_state_transition_matrix", transition_table.round(6))

plt.figure(figsize=(8, 6))
plt.imshow(transition_table.to_numpy(), aspect="auto")
plt.colorbar(label="Transition probability")
plt.xticks(range(len(LABEL_ORDER)), LABEL_ORDER, rotation=45)
plt.yticks(range(len(LABEL_ORDER)), LABEL_ORDER)
plt.title("Operational-State Transition Matrix")
plt.xlabel("Next state")
plt.ylabel("Current state")
plt.tight_layout()
plt.show()


# In[50]:


# EDA 47: FEATURE-TO-FAULT-LOCATION ASSOCIATION ANALYSIS
localized = df[df["fault_loc"] >= 0].copy()
if localized.empty:
    print("No localized fault rows are available.")
else:
    fault_location_correlations = (
        localized[MODEL_FEATURES_FULL + ["fault_loc"]]
        .corr(numeric_only=True)["fault_loc"]
        .drop("fault_loc")
        .to_frame("correlation_with_fault_loc")
    )
    fault_location_correlations["absolute_correlation"] = fault_location_correlations["correlation_with_fault_loc"].abs()
    fault_location_correlations = fault_location_correlations.sort_values("absolute_correlation", ascending=False)
    store_table("47_fault_location_correlations", fault_location_correlations)


# In[51]:


# EDA 48: RANDOM-SPLIT STRATIFICATION ANALYSIS
from sklearn.model_selection import train_test_split

train_frame, test_frame = train_test_split(
    df[["label"]],
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=df["label"],
)
stratification_table = pd.DataFrame({
    "full_dataset_pct": (100 * df["label"].value_counts(normalize=True).reindex(LABEL_ORDER)).round(5),
    "train_pct": (100 * train_frame["label"].value_counts(normalize=True).reindex(LABEL_ORDER)).round(5),
    "test_pct": (100 * test_frame["label"].value_counts(normalize=True).reindex(LABEL_ORDER)).round(5),
})
store_table("48_stratification_analysis", stratification_table)


# In[52]:


# EDA 49: INPUT-PERTURBATION SENSITIVITY ANALYSIS
baseline = train_test_baseline(MODEL_FEATURES_REDUCED, n=min(MODEL_SAMPLE_SIZE, 30_000))
X_test = baseline["X_test"].copy()
reference_prediction = baseline["model"].predict(X_test)
reference_macro_f1 = f1_score(baseline["y_test"], reference_prediction, average="macro")

rng = np.random.default_rng(RANDOM_STATE)
sensitivity_rows = []
for noise_scale in [0.00, 0.01, 0.03, 0.05, 0.10]:
    feature_scale = X_test.std(axis=0).replace(0.0, 1.0).to_numpy()
    noise = rng.normal(0.0, noise_scale, size=X_test.shape) * feature_scale
    perturbed = X_test.to_numpy() + noise
    prediction = baseline["model"].predict(perturbed)
    macro_f1 = f1_score(baseline["y_test"], prediction, average="macro")
    sensitivity_rows.append({
        "noise_scale": noise_scale,
        "macro_f1": macro_f1,
        "prediction_change_rate": float(np.mean(prediction != reference_prediction)),
        "macro_f1_change_from_reference": macro_f1 - reference_macro_f1,
    })

sensitivity_table = pd.DataFrame(sensitivity_rows)
store_table("49_input_perturbation_sensitivity", sensitivity_table.round(6))
plot_line(sensitivity_table["noise_scale"], sensitivity_table["macro_f1"], "Perturbation Sensitivity", "Noise scale", "Macro-F1")


# In[53]:


# EDA 50: PROBABILITY-CALIBRATION ANALYSIS
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import label_binarize

baseline = train_test_baseline(MODEL_FEATURES_REDUCED, n=MODEL_SAMPLE_SIZE)
probabilities = baseline["model"].predict_proba(baseline["X_test"])
class_order = baseline["model"].classes_.tolist()
binary_y = label_binarize(baseline["y_test"], classes=class_order)

calibration_rows = []
plt.figure(figsize=(9, 6))
for index, label in enumerate(class_order):
    fraction_positive, mean_predicted = calibration_curve(
        binary_y[:, index], probabilities[:, index], n_bins=10, strategy="quantile"
    )
    brier = brier_score_loss(binary_y[:, index], probabilities[:, index])
    calibration_rows.append({"class": label, "brier_score": brier})
    plt.plot(mean_predicted, fraction_positive, marker="o", label=f"{label} (Brier={brier:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
plt.title("Calibration Curves: Reduced-Feature Diagnostic Baseline")
plt.xlabel("Mean predicted probability")
plt.ylabel("Observed positive fraction")
plt.legend()
plt.tight_layout()
plt.show()

store_table("50_calibration_summary", pd.DataFrame(calibration_rows))


# In[54]:


# EDA 51: PER-CLASS DETAILED STATISTICS
per_class_statistics = (
    df.groupby("label", observed=False)[MODEL_FEATURES_FULL]
      .agg(["mean", "median", "std", "min", "max"])
      .round(5)
)
store_table("51_per_class_statistics", per_class_statistics)


# In[55]:


# EDA 52: RHO-THRESHOLD SENSITIVITY ANALYSIS
threshold_rows = []
for threshold in np.arange(0.80, 1.21, 0.05):
    predicted_overload = df["max_rho"] >= threshold
    actual_overload = df["label"].astype(str) == "overload"
    true_positive = int((predicted_overload & actual_overload).sum())
    false_positive = int((predicted_overload & ~actual_overload).sum())
    false_negative = int((~predicted_overload & actual_overload).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    threshold_rows.append({
        "rho_threshold": round(float(threshold), 2),
        "predicted_overload_records": int(predicted_overload.sum()),
        "precision": precision,
        "recall": recall,
    })

rho_threshold_table = pd.DataFrame(threshold_rows)
store_table("52_rho_threshold_sensitivity", rho_threshold_table.round(6))
plot_line(rho_threshold_table["rho_threshold"], rho_threshold_table["recall"], "Overload Recall Across Rho Thresholds", "Rho threshold", "Recall")


# In[56]:


# EDA 53: SAMPLE-COMPLEXITY ANALYSIS
sample, X, y = model_frame(MODEL_FEATURES_REDUCED, n=min(MODEL_SAMPLE_SIZE, 40_000))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

complexity_rows = []
for fraction in [0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00]:
    subset_size = max(200, int(len(X_train) * fraction))
    subset_indices = X_train.sample(n=subset_size, random_state=RANDOM_STATE).index
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train.loc[subset_indices], y_train.loc[subset_indices])
    prediction = model.predict(X_test)
    complexity_rows.append({
        "training_fraction": fraction,
        "training_records": subset_size,
        "macro_f1": f1_score(y_test, prediction, average="macro"),
    })

complexity_table = pd.DataFrame(complexity_rows)
store_table("53_sample_complexity", complexity_table.round(6))
plot_line(complexity_table["training_records"], complexity_table["macro_f1"], "Sample Complexity", "Training records", "Macro-F1")


# In[57]:


# EDA 54: DATA-LEAKAGE AND SPLIT-RISK AUDIT
full_baseline = train_test_baseline(MODEL_FEATURES_FULL, n=min(MODEL_SAMPLE_SIZE, 35_000))
reduced_baseline = train_test_baseline(MODEL_FEATURES_REDUCED, n=min(MODEL_SAMPLE_SIZE, 35_000))

leakage_rows = [
    {
        "feature_setting": "full aggregate features",
        "macro_f1": f1_score(full_baseline["y_test"], full_baseline["pred"], average="macro"),
        "contains_direct_rule_signals": True,
    },
    {
        "feature_setting": "reduced aggregate features",
        "macro_f1": f1_score(reduced_baseline["y_test"], reduced_baseline["pred"], average="macro"),
        "contains_direct_rule_signals": False,
    },
]
store_table("54_direct_signal_leakage_check", pd.DataFrame(leakage_rows).round(6))

available_chronics = np.array(sorted(df["chronic_id"].unique()))
rng = np.random.default_rng(RANDOM_STATE)
rng.shuffle(available_chronics)
test_chronic_count = max(1, int(round(0.20 * len(available_chronics))))
test_chronics = set(available_chronics[:test_chronic_count])
train_chronics = set(available_chronics[test_chronic_count:])

chronic_train = df[df["chronic_id"].isin(train_chronics)]
chronic_test = df[df["chronic_id"].isin(test_chronics)]
chronic_overlap = len(train_chronics.intersection(test_chronics))

chronic_split_table = pd.DataFrame({
    "metric": ["training chronics", "testing chronics", "chronic overlap", "training records", "testing records"],
    "value": [len(train_chronics), len(test_chronics), chronic_overlap, len(chronic_train), len(chronic_test)],
})
store_table("54_chronic_split_audit", chronic_split_table)


# In[58]:


# EDA 55: TWO-DIMENSIONAL DECISION-BOUNDARY VISUALIZATION
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sample = stratified_sample(df, min(6000, len(df)))
X = sample[MODEL_FEATURES_REDUCED].fillna(0.0)
y = sample["label"].astype(str)

scaled = StandardScaler().fit_transform(X)
projection = PCA(n_components=2).fit_transform(scaled)
boundary_model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
boundary_model.fit(projection, y)

x_min, x_max = projection[:, 0].min() - 0.5, projection[:, 0].max() + 0.5
y_min, y_max = projection[:, 1].min() - 0.5, projection[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
grid_predictions = boundary_model.predict(np.column_stack([xx.ravel(), yy.ravel()]))
encoded_grid = pd.Series(grid_predictions).map(LABEL_TO_INT).to_numpy().reshape(xx.shape)

plt.figure(figsize=(9, 6))
plt.contourf(xx, yy, encoded_grid, alpha=0.20)
for label in LABEL_ORDER:
    mask = y.to_numpy() == label
    plt.scatter(projection[mask, 0], projection[mask, 1], s=10, alpha=0.45, label=label)
plt.title("Decision Boundary in Two-Dimensional PCA Space")
plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")
plt.legend()
plt.tight_layout()
plt.show()


# In[59]:


# EDA 56: FEATURE-INTERACTION ANALYSIS
interaction_frame = df.copy()
interaction_frame["rho_topology_interaction"] = interaction_frame["max_rho"] * (1.0 - interaction_frame["connected_fraction"])
interaction_frame["load_stress_interaction"] = interaction_frame["total_load"] * interaction_frame["max_rho"]
interaction_frame["flow_stress_interaction"] = interaction_frame["mean_abs_p_or"] * interaction_frame["max_rho"]
interaction_frame["imbalance_stress_interaction"] = interaction_frame["abs_power_balance"] * interaction_frame["max_rho"]

interaction_features = [
    "rho_topology_interaction",
    "load_stress_interaction",
    "flow_stress_interaction",
    "imbalance_stress_interaction",
]
interaction_summary = (
    interaction_frame.groupby("label", observed=False)[interaction_features]
                     .mean()
                     .round(6)
)
store_table("56_feature_interactions", interaction_summary)

for feature in interaction_features:
    plot_bar(
        interaction_frame.groupby("label", observed=False)[feature].mean(),
        f"Feature Interaction by Class: {feature}",
        "Operational state",
        f"Mean {feature}",
    )


# In[60]:


# EDA 57: CASCADE-PATTERN ANALYSIS
cascade = df[df["label"].astype(str) == "cascade"].copy()
cascade_summary = pd.DataFrame({
    "cascade_records": [len(cascade)],
    "cascade_percentage": [100 * len(cascade) / len(df)],
    "minimum_disconnected_lines": [int(cascade["disconnected_lines"].min()) if len(cascade) else np.nan],
    "median_disconnected_lines": [float(cascade["disconnected_lines"].median()) if len(cascade) else np.nan],
    "maximum_disconnected_lines": [int(cascade["disconnected_lines"].max()) if len(cascade) else np.nan],
    "mean_connected_fraction": [float(cascade["connected_fraction"].mean()) if len(cascade) else np.nan],
})
store_table("57_cascade_summary", cascade_summary.round(6))

if len(cascade):
    cascade_line_counts = cascade["disconnected_lines"].value_counts().sort_index()
    plot_bar(cascade_line_counts, "Cascade-State Distribution by Number of Disconnected Lines", "Disconnected lines", "Cascade records")


# In[61]:


# EDA 58: PHYSICS-BASED VALIDATION
physics_checks = {
    "normal_has_no_overload": ((df["label"].astype(str) != "normal") | (df["max_rho"] < 1.0)),
    "normal_has_intact_topology": ((df["label"].astype(str) != "normal") | (df["disconnected_lines"] == 0)),
    "overload_meets_rho_threshold": ((df["label"].astype(str) != "overload") | (df["max_rho"] >= 1.0)),
    "line_trip_has_exactly_one_disconnection": ((df["label"].astype(str) != "line_trip") | (df["disconnected_lines"] == 1)),
    "cascade_has_multiple_disconnections": ((df["label"].astype(str) != "cascade") | (df["disconnected_lines"] > 1)),
    "connected_fraction_in_unit_interval": df["connected_fraction"].between(0.0, 1.0),
    "stored_rho_respects_clip": df["max_rho"] <= float(metadata.get("rho_clip", 2.0)),
}

physics_table = pd.DataFrame([
    {
        "check": name,
        "passed_records": int(mask.sum()),
        "failed_records": int((~mask).sum()),
        "pass_percentage": 100 * float(mask.mean()),
    }
    for name, mask in physics_checks.items()
])
store_table("58_physics_validation", physics_table.round(6))


# In[62]:


# EDA 59: MULTICLASS ROC-AUC CURVE ANALYSIS
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

baseline = train_test_baseline(MODEL_FEATURES_REDUCED, n=MODEL_SAMPLE_SIZE)
probabilities = baseline["model"].predict_proba(baseline["X_test"])
class_order = baseline["model"].classes_.tolist()
binary_y = label_binarize(baseline["y_test"], classes=class_order)

roc_rows = []
plt.figure(figsize=(9, 6))
for index, label in enumerate(class_order):
    false_positive_rate, true_positive_rate, _ = roc_curve(binary_y[:, index], probabilities[:, index])
    area = auc(false_positive_rate, true_positive_rate)
    roc_rows.append({"class": label, "roc_auc": area})
    plt.plot(false_positive_rate, true_positive_rate, label=f"{label} (AUC={area:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
plt.title("Multiclass ROC Curves: Reduced-Feature Diagnostic Baseline")
plt.xlabel("False-positive rate")
plt.ylabel("True-positive rate")
plt.legend()
plt.tight_layout()
plt.show()

store_table("59_roc_auc_summary", pd.DataFrame(roc_rows))


# ## **I. Advanced Diagnostic and Repository-Consistency Audits**
# 

# In[71]:


# SETUP 04: HELPERS FOR EXTENDED GRAPH, LINE, SPLIT, AND TEMPORAL AUDITS
from collections import Counter
from itertools import combinations
from pathlib import Path
import json
import sys

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

# Run this notebook from the repository root.
sys.path.append(str(Path.cwd()))

from scripts.pyg_data import GridEnvMetadata, build_node_features, build_edges

PYG_META = GridEnvMetadata(metadata)
N_SUB = int(metadata["n_sub"])
N_LINE = int(metadata["n_line"])

LINE_OR_BUS = np.asarray(metadata["topology"]["line_or_bus"], dtype=int)
LINE_EX_BUS = np.asarray(metadata["topology"]["line_ex_bus"], dtype=int)

# These samples are large enough for reliable EDA while keeping runtime practical.
EXTENDED_SAMPLE_SIZE = min(8_000, len(df))
FEATURE_JUSTIFICATION_SAMPLE_SIZE = min(25_000, len(df))


def canonical_edge(node_a, node_b):
    """Return an undirected edge key."""
    return tuple(sorted((int(node_a), int(node_b))))


def safe_array(record, key, expected_length=None, dtype=float):
    """Read a JSONL array safely and optionally validate its length."""
    values = np.asarray(record.get(key, []), dtype=dtype)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    if expected_length is not None and len(values) != int(expected_length):
        raise ValueError(
            f"{key!r} has length {len(values)}, expected {expected_length}."
        )

    return values


def load_records_by_indices(filepath, indices):
    """
    Stream the JSONL file once and return only the requested records.

    The returned dictionary is keyed by the original zero-based
    JSONL row index.
    """
    requested = {int(index) for index in indices}
    records = {}

    with Path(filepath).open("r", encoding="utf-8") as file:
        for record_index, line in enumerate(file):
            if record_index in requested:
                records[record_index] = json.loads(line)

                if len(records) == len(requested):
                    break

    missing = requested.difference(records)

    if missing:
        raise IndexError(
            f"Could not load {len(missing)} requested JSONL records."
        )

    return records


def sampled_record_indices(frame, n=EXTENDED_SAMPLE_SIZE):
    """Return a label-aware sample of original JSONL row indices."""
    sample = stratified_sample(frame, min(int(n), len(frame)))
    return sample["record_index"].astype(int).tolist()


def sample_frame(frame, n, random_state=RANDOM_STATE):
    """Return a reproducible sample without exceeding the available rows."""
    if len(frame) <= int(n):
        return frame.copy()

    return frame.sample(
        n=int(n),
        random_state=random_state,
    ).copy()


print("Extended EDA helpers are ready.")


# In[63]:


# EDA 60: FEATURE-ENGINEERING IMPACT ANALYSIS
from sklearn.model_selection import StratifiedKFold, cross_val_score

sample, X_base, y = model_frame(MODEL_FEATURES_REDUCED, n=min(MODEL_SAMPLE_SIZE, 30_000))
X_engineered = X_base.copy()
X_engineered["rho_topology_interaction"] = df.loc[X_engineered.index, "max_rho"] * (1.0 - df.loc[X_engineered.index, "connected_fraction"])
X_engineered["load_stress_interaction"] = df.loc[X_engineered.index, "total_load"] * df.loc[X_engineered.index, "max_rho"]
X_engineered["flow_stress_interaction"] = df.loc[X_engineered.index, "mean_abs_p_or"] * df.loc[X_engineered.index, "max_rho"]

cross_validation = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)

base_scores = cross_val_score(model, X_base, y, cv=cross_validation, scoring="f1_macro", n_jobs=-1)
engineered_scores = cross_val_score(model, X_engineered, y, cv=cross_validation, scoring="f1_macro", n_jobs=-1)

engineering_table = pd.DataFrame({
    "feature_setting": ["baseline reduced aggregates", "with engineered interactions"],
    "feature_count": [X_base.shape[1], X_engineered.shape[1]],
    "mean_macro_f1": [base_scores.mean(), engineered_scores.mean()],
    "std_macro_f1": [base_scores.std(), engineered_scores.std()],
})
store_table("60_feature_engineering_impact", engineering_table.round(6))


# In[64]:


# EDA 61: DISTRIBUTION-FIT ANALYSIS
from scipy.stats import kstest, kurtosis, skew

distribution_rows = []
for feature in MODEL_FEATURES_REDUCED:
    values = df[feature].replace([np.inf, -np.inf], np.nan).dropna()
    values = values.sample(n=min(5000, len(values)), random_state=RANDOM_STATE)
    standard_deviation = float(values.std())
    standardized = (values - values.mean()) / standard_deviation if standard_deviation > 0 else values * 0.0
    ks_statistic, p_value = kstest(standardized, "norm")
    distribution_rows.append({
        "feature": feature,
        "skewness": skew(values, bias=False),
        "excess_kurtosis": kurtosis(values, fisher=True, bias=False),
        "normal_fit_ks_statistic": ks_statistic,
        "normal_fit_p_value": p_value,
    })

distribution_fit_table = pd.DataFrame(distribution_rows).sort_values("normal_fit_p_value")
store_table("61_distribution_fit", distribution_fit_table.round(6))


# In[65]:


# EDA 62: PREDICTION-ERROR ANALYSIS
baseline = train_test_baseline(MODEL_FEATURES_REDUCED, n=MODEL_SAMPLE_SIZE)
error_frame = baseline["X_test"].copy()
error_frame["actual_label"] = baseline["y_test"].astype(str)
error_frame["predicted_label"] = baseline["pred"].astype(str)
error_frame["prediction_correct"] = error_frame["actual_label"] == error_frame["predicted_label"]

error_summary = (
    error_frame.groupby(["actual_label", "predicted_label"])
               .size()
               .reset_index(name="records")
               .sort_values("records", ascending=False)
)
store_table("62_prediction_error_pairs", error_summary)

incorrect = error_frame[~error_frame["prediction_correct"]]
print(f"Incorrect predictions: {len(incorrect):,} / {len(error_frame):,}")
if len(incorrect):
    store_table(
        "62_misclassified_feature_profile",
        incorrect[MODEL_FEATURES_REDUCED].describe().T.round(6),
    )


# In[66]:


# EDA 63: STATIC GRAPH-PROPERTY ANALYSIS
import networkx as nx

line_or_bus = np.asarray(metadata["topology"]["line_or_bus"], dtype=int)
line_ex_bus = np.asarray(metadata["topology"]["line_ex_bus"], dtype=int)

graph = nx.MultiGraph()
graph.add_nodes_from(range(int(metadata["n_sub"])))
graph.add_edges_from(zip(line_or_bus, line_ex_bus))

simple_graph = nx.Graph(graph)
degree_values = np.asarray([simple_graph.degree(node) for node in simple_graph.nodes()], dtype=float)

graph_property_table = pd.DataFrame({
    "property": [
        "nodes", "transmission lines", "simple edges", "connected components",
        "is connected", "average degree", "minimum degree", "maximum degree",
        "density", "average clustering coefficient",
    ],
    "value": [
        graph.number_of_nodes(), graph.number_of_edges(), simple_graph.number_of_edges(),
        nx.number_connected_components(simple_graph), nx.is_connected(simple_graph),
        float(degree_values.mean()), float(degree_values.min()), float(degree_values.max()),
        nx.density(simple_graph), nx.average_clustering(simple_graph),
    ],
})
store_table("63_static_graph_properties", graph_property_table)

degree_counts = pd.Series(degree_values.astype(int)).value_counts().sort_index()
plot_bar(degree_counts, "Static Grid-Topology Degree Distribution", "Node degree", "Substations")


# In[67]:


# EDA 64: HYPERPARAMETER, CLASS-IMBALANCE, RAW-JSONL, AND METADATA INTEGRITY AUDIT
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# Part A: lightweight hyperparameter sensitivity.
sample, X, y = model_frame(MODEL_FEATURES_REDUCED, n=min(MODEL_SAMPLE_SIZE, 25_000))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

hyperparameter_rows = []
for n_estimators in [40, 80, 140]:
    for max_depth in [None, 12, 24]:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        hyperparameter_rows.append({
            "n_estimators": n_estimators,
            "max_depth": str(max_depth),
            "macro_f1": f1_score(y_test, prediction, average="macro"),
        })
store_table("64_hyperparameter_sensitivity", pd.DataFrame(hyperparameter_rows).round(6))

# Part B: compare no weighting with balanced class weighting.
imbalance_rows = []
for class_weight in [None, "balanced", "balanced_subsample"]:
    model = RandomForestClassifier(
        n_estimators=120,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight=class_weight,
    )
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    imbalance_rows.append({
        "class_weight": str(class_weight),
        "macro_f1": f1_score(y_test, prediction, average="macro"),
        "weighted_f1": f1_score(y_test, prediction, average="weighted"),
    })
store_table("64_class_weight_comparison", pd.DataFrame(imbalance_rows).round(6))

# Part C: stream the JSONL file to verify row counts, array lengths, and label counts.
expected_lengths = {
    "rho": int(metadata["n_line"]),
    "p_or": int(metadata["n_line"]),
    "q_or": int(metadata["n_line"]),
    "p_ex": int(metadata["n_line"]),
    "q_ex": int(metadata["n_line"]),
    "v_or": int(metadata["n_line"]),
    "v_ex": int(metadata["n_line"]),
    "line_status": int(metadata["n_line"]),
    "load_p": int(metadata["n_load"]),
    "load_q": int(metadata["n_load"]),
    "gen_p": int(metadata["n_gen"]),
    "gen_q": int(metadata["n_gen"]),
}
raw_label_counts = Counter()
length_mismatches = Counter()
nonfinite_values = Counter()
raw_records = 0

with JSONL_PATH.open("r", encoding="utf-8") as file:
    for raw_records, line in enumerate(file, start=1):
        record = json.loads(line)
        raw_label_counts[str(record.get("label"))] += 1
        for key, expected_length in expected_lengths.items():
            values = record.get(key, [])
            if len(values) != expected_length:
                length_mismatches[key] += 1
            try:
                array = np.asarray(values, dtype=float)
                nonfinite_values[key] += int((~np.isfinite(array)).sum())
            except Exception:
                nonfinite_values[key] += 1

        if RAW_AUDIT_MAX_LINES is not None and raw_records >= RAW_AUDIT_MAX_LINES:
            break

raw_integrity_table = pd.DataFrame({
    "metric": [
        "raw JSONL rows scanned",
        "metadata total_records",
        "DataFrame rows loaded",
        "array-length mismatch fields",
        "non-finite-value fields",
        "metadata edge_feature_dim",
        "current pyg_data.py edge attributes",
    ],
    "value": [
        raw_records,
        metadata.get("total_records"),
        len(df),
        dict(length_mismatches),
        dict(nonfinite_values),
        metadata.get("edge_feature_dim"),
        "4: rho, p_or, q_or, line_status",
    ],
})
store_table("64_raw_integrity_summary", raw_integrity_table)

label_count_comparison = pd.DataFrame({
    "metadata_count": pd.Series(metadata.get("label_counts", {})),
    "raw_jsonl_count": pd.Series(raw_label_counts),
    "loaded_dataframe_count": df["label"].value_counts(),
}).fillna(0).astype(int).reindex(LABEL_ORDER)
store_table("64_label_count_comparison", label_count_comparison)

print(
    "\nImportant repository-consistency note:\n"
    "- The generator metadata records edge_feature_dim = 3 for the raw measurements rho, p_or, and q_or.\n"
    "- The current scripts/pyg_data.py graph builder constructs 4 active-edge attributes by adding line_status.\n"
    "- Keep this distinction explicit when documenting the final graph representation."
)


# In[72]:


# EDA 65: RAW-TO-GRAPH PYTORCH GEOMETRIC TENSOR AUDIT

graph_audit_indices = sampled_record_indices(
    df,
    n=EXTENDED_SAMPLE_SIZE,
)

graph_audit_records = load_records_by_indices(
    JSONL_PATH,
    graph_audit_indices,
)

graph_audit_rows = []
shape_failure_rows = []

for record_index in graph_audit_indices:
    record = graph_audit_records[record_index]

    line_status = safe_array(
        record,
        "line_status",
        expected_length=N_LINE,
        dtype=int,
    )

    expected_active_edges = int(np.sum(line_status == 1))

    node_features = np.asarray(
        build_node_features(record, PYG_META),
        dtype=np.float32,
    )

    edge_index_list, edge_attr_list = build_edges(
        record,
        PYG_META,
    )

    edge_index = np.asarray(
        edge_index_list,
        dtype=np.int64,
    )

    if edge_index.size == 0:
        edge_index = np.empty((2, 0), dtype=np.int64)
    else:
        edge_index = edge_index.reshape(2, -1)

    edge_attr = np.asarray(
        edge_attr_list,
        dtype=np.float32,
    )

    if edge_attr.size == 0:
        edge_attr = np.empty((0, 4), dtype=np.float32)
    else:
        edge_attr = edge_attr.reshape(-1, 4)

    graph = Data(
        x=torch.tensor(
            node_features,
            dtype=torch.float,
        ),
        edge_index=torch.tensor(
            edge_index,
            dtype=torch.long,
        ),
        edge_attr=torch.tensor(
            edge_attr,
            dtype=torch.float,
        ),
        y=torch.tensor(
            int(record.get("label_int", -1)),
            dtype=torch.long,
        ),
    )

    active_pairs = {
        (int(source), int(target))
        for source, target in graph.edge_index.t().tolist()
    }

    reciprocal_edges = sum(
        (target, source) in active_pairs
        for source, target in active_pairs
    )

    reciprocal_edge_fraction = (
        reciprocal_edges / len(active_pairs)
        if active_pairs
        else np.nan
    )

    row = {
        "record_index": int(record_index),
        "label": str(record.get("label")),
        "node_rows": int(graph.x.shape[0]),
        "node_feature_dim": int(graph.x.shape[1]),
        "active_edges": int(graph.edge_index.shape[1]),
        "edge_attr_rows": int(graph.edge_attr.shape[0]),
        "edge_feature_dim": int(graph.edge_attr.shape[1]),
        "expected_active_edges_from_line_status": expected_active_edges,
        "dead_lines_removed": (
            int(graph.edge_index.shape[1])
            == expected_active_edges
        ),
        "edge_attr_matches_edge_index": (
            int(graph.edge_attr.shape[0])
            == int(graph.edge_index.shape[1])
        ),
        "node_shape_valid": (
            tuple(graph.x.shape)
            == (N_SUB, 4)
        ),
        "edge_feature_shape_valid": (
            int(graph.edge_attr.shape[1])
            == 4
        ),
        "all_retained_edge_status_values_are_one": (
            bool(
                np.allclose(
                    graph.edge_attr[:, 3].numpy(),
                    1.0,
                )
            )
            if graph.edge_attr.shape[0] > 0
            else True
        ),
        "reciprocal_edge_fraction": reciprocal_edge_fraction,
    }

    graph_audit_rows.append(row)

    if not all([
        row["dead_lines_removed"],
        row["edge_attr_matches_edge_index"],
        row["node_shape_valid"],
        row["edge_feature_shape_valid"],
        row["all_retained_edge_status_values_are_one"],
    ]):
        shape_failure_rows.append(row)


graph_audit_detail = pd.DataFrame(
    graph_audit_rows
)

store_table(
    "65_raw_to_graph_audit_detail",
    graph_audit_detail.head(50),
)

graph_audit_summary = pd.DataFrame({
    "metric": [
        "sampled graphs",
        "graphs with valid node shape: n_sub x 4",
        "graphs with valid edge_attr width: 4",
        "graphs where active edges match connected lines",
        "graphs where edge_attr rows match edge_index columns",
        "graphs where retained line_status values are all 1",
        "graphs with zero active edges",
        "mean active edges",
        "metadata edge_feature_dim",
        "current pyg_data.py edge_feature_dim",
        "mean reciprocal-edge fraction",
    ],
    "value": [
        len(graph_audit_detail),
        int(
            graph_audit_detail[
                "node_shape_valid"
            ].sum()
        ),
        int(
            graph_audit_detail[
                "edge_feature_shape_valid"
            ].sum()
        ),
        int(
            graph_audit_detail[
                "dead_lines_removed"
            ].sum()
        ),
        int(
            graph_audit_detail[
                "edge_attr_matches_edge_index"
            ].sum()
        ),
        int(
            graph_audit_detail[
                "all_retained_edge_status_values_are_one"
            ].sum()
        ),
        int(
            (
                graph_audit_detail[
                    "active_edges"
                ]
                == 0
            ).sum()
        ),
        float(
            graph_audit_detail[
                "active_edges"
            ].mean()
        ),
        metadata.get("edge_feature_dim"),
        4,
        float(
            graph_audit_detail[
                "reciprocal_edge_fraction"
            ]
            .dropna()
            .mean()
        ),
    ],
})

store_table(
    "65_raw_to_graph_audit_summary",
    graph_audit_summary,
)

graph_shape_failures = pd.DataFrame(
    shape_failure_rows
)

store_table(
    "65_raw_to_graph_failures",
    graph_shape_failures,
)

active_edges_by_class = (
    graph_audit_detail
    .groupby(
        "label",
        observed=False,
    )["active_edges"]
    .agg([
        "count",
        "mean",
        "median",
        "min",
        "max",
    ])
    .reindex(LABEL_ORDER)
    .round(4)
)

store_table(
    "65_active_edges_by_class",
    active_edges_by_class,
)

print(
    "\nInterpretation note:\n"
    "- The current graph builder should produce 4 node features and 4 edge attributes.\n"
    "- Disconnected lines should be absent from edge_index.\n"
    "- The reciprocal-edge fraction is reported for inspection only.\n"
    "  A low value means that the builder stores mostly one directed\n"
    "  orientation per physical transmission line.\n"
)


# ## **J. Transmission-Line and Substation Vulnerability Analysis**

# In[73]:


# EDA 66: LINE-LEVEL TRANSMISSION-STRESS AND DISCONNECTION ANALYSIS

line_rho_sum = np.zeros(
    N_LINE,
    dtype=np.float64,
)

line_rho_max = np.full(
    N_LINE,
    -np.inf,
    dtype=np.float64,
)

line_overload_frames = np.zeros(
    N_LINE,
    dtype=np.int64,
)

line_disconnect_frames = np.zeros(
    N_LINE,
    dtype=np.int64,
)

line_cascade_disconnect_frames = np.zeros(
    N_LINE,
    dtype=np.int64,
)

line_trip_disconnect_frames = np.zeros(
    N_LINE,
    dtype=np.int64,
)

line_class_rho_sum = {
    label: np.zeros(
        N_LINE,
        dtype=np.float64,
    )
    for label in LABEL_ORDER
}

line_class_record_count = Counter()

disconnected_pair_counter = Counter()
cascade_pair_counter = Counter()

raw_rows_scanned = 0

with JSONL_PATH.open(
    "r",
    encoding="utf-8",
) as file:

    for raw_rows_scanned, line in enumerate(
        file,
        start=1,
    ):
        record = json.loads(line)

        rho = safe_array(
            record,
            "rho",
            expected_length=N_LINE,
            dtype=float,
        )

        line_status = safe_array(
            record,
            "line_status",
            expected_length=N_LINE,
            dtype=int,
        )

        label = str(
            record.get("label")
        )

        disconnected_indices = np.flatnonzero(
            line_status == 0
        )

        overloaded_mask = (
            rho >= 1.0
        )

        line_rho_sum += rho

        line_rho_max = np.maximum(
            line_rho_max,
            rho,
        )

        line_overload_frames += overloaded_mask.astype(
            np.int64
        )

        line_disconnect_frames += (
            line_status == 0
        ).astype(
            np.int64
        )

        if label in line_class_rho_sum:
            line_class_rho_sum[label] += rho
            line_class_record_count[label] += 1

        if label == "line_trip":
            line_trip_disconnect_frames += (
                line_status == 0
            ).astype(
                np.int64
            )

        if label == "cascade":
            line_cascade_disconnect_frames += (
                line_status == 0
            ).astype(
                np.int64
            )

        if len(disconnected_indices) >= 2:
            for first_line, second_line in combinations(
                disconnected_indices.tolist(),
                2,
            ):
                pair_key = tuple(
                    sorted(
                        (
                            int(first_line),
                            int(second_line),
                        )
                    )
                )

                disconnected_pair_counter[
                    pair_key
                ] += 1

                if label == "cascade":
                    cascade_pair_counter[
                        pair_key
                    ] += 1


line_level_table = pd.DataFrame({
    "line_id": np.arange(
        N_LINE,
        dtype=int,
    ),
    "origin_substation": LINE_OR_BUS,
    "extremity_substation": LINE_EX_BUS,
    "mean_rho": (
        line_rho_sum
        / max(
            raw_rows_scanned,
            1,
        )
    ),
    "maximum_rho": line_rho_max,
    "overload_frames": line_overload_frames,
    "overload_frame_pct": (
        100
        * line_overload_frames
        / max(
            raw_rows_scanned,
            1,
        )
    ),
    "disconnected_frames": line_disconnect_frames,
    "disconnected_frame_pct": (
        100
        * line_disconnect_frames
        / max(
            raw_rows_scanned,
            1,
        )
    ),
    "line_trip_disconnect_frames": (
        line_trip_disconnect_frames
    ),
    "cascade_disconnect_frames": (
        line_cascade_disconnect_frames
    ),
})

line_level_table["endpoint_pair"] = (
    line_level_table[
        "origin_substation"
    ].astype(str)
    + " -> "
    + line_level_table[
        "extremity_substation"
    ].astype(str)
)

line_level_ranked = (
    line_level_table
    .sort_values(
        [
            "disconnected_frames",
            "overload_frames",
            "maximum_rho",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )
)

store_table(
    "66_line_level_transmission_profile",
    line_level_ranked,
)

top_disconnected = (
    line_level_table
    .nlargest(
        15,
        "disconnected_frames",
    )
    .set_index(
        "line_id"
    )[
        "disconnected_frames"
    ]
)

plot_bar(
    top_disconnected,
    "Most Frequently Disconnected Transmission Lines",
    "Transmission-line ID",
    "Disconnected frames",
)

top_overloaded = (
    line_level_table
    .nlargest(
        15,
        "overload_frames",
    )
    .set_index(
        "line_id"
    )[
        "overload_frames"
    ]
)

plot_bar(
    top_overloaded,
    "Most Frequently Overloaded Transmission Lines",
    "Transmission-line ID",
    "Frames with rho >= 1.0",
)

class_mean_rho_by_line = pd.DataFrame(
    {
        label: (
            line_class_rho_sum[label]
            / max(
                line_class_record_count[label],
                1,
            )
        )
        for label in LABEL_ORDER
    },
    index=pd.Index(
        np.arange(
            N_LINE
        ),
        name="line_id",
    ),
).T

store_table(
    "66_class_mean_rho_by_line",
    class_mean_rho_by_line.round(6),
)

plt.figure(
    figsize=(
        16,
        5,
    )
)

plt.imshow(
    class_mean_rho_by_line,
    aspect="auto",
)

plt.colorbar(
    label="Mean rho"
)

plt.yticks(
    range(
        len(
            LABEL_ORDER
        )
    ),
    LABEL_ORDER,
)

plt.xticks(
    range(
        N_LINE
    ),
    range(
        N_LINE
    ),
    rotation=90,
)

plt.xlabel(
    "Transmission-line ID"
)

plt.ylabel(
    "Operational state"
)

plt.title(
    "Mean Transmission-Line Loading Ratio by Class"
)

plt.tight_layout()
plt.show()


# In[74]:


# EDA 67: FREQUENTLY CO-DISCONNECTED LINE-PAIR ANALYSIS

def pair_counter_to_table(
    counter,
    table_name,
    top_n=50,
):
    rows = []

    for (
        first_line,
        second_line,
    ), count in counter.most_common(
        top_n
    ):
        rows.append({
            "first_line_id": int(
                first_line
            ),
            "second_line_id": int(
                second_line
            ),
            "first_line_path": (
                f"{LINE_OR_BUS[first_line]}"
                f" -> "
                f"{LINE_EX_BUS[first_line]}"
            ),
            "second_line_path": (
                f"{LINE_OR_BUS[second_line]}"
                f" -> "
                f"{LINE_EX_BUS[second_line]}"
            ),
            "co_disconnected_frames": int(
                count
            ),
        })

    table = pd.DataFrame(
        rows
    )

    store_table(
        table_name,
        table,
    )

    return table


top_all_pairs = pair_counter_to_table(
    disconnected_pair_counter,
    "67_top_co_disconnected_line_pairs_all_multiline_states",
    top_n=50,
)

top_cascade_pairs = pair_counter_to_table(
    cascade_pair_counter,
    "67_top_co_disconnected_line_pairs_cascade_only",
    top_n=50,
)

if len(
    top_cascade_pairs
):
    top_pairs_plot = (
        top_cascade_pairs
        .head(
            15
        )
        .copy()
    )

    top_pairs_plot["line_pair"] = (
        top_pairs_plot.apply(
            lambda row: (
                f"{int(row['first_line_id'])}"
                f"-"
                f"{int(row['second_line_id'])}"
            ),
            axis=1,
        )
    )

    plot_bar(
        top_pairs_plot
        .set_index(
            "line_pair"
        )[
            "co_disconnected_frames"
        ],
        "Most Frequent Co-Disconnected Line Pairs in Cascade States",
        "Line-pair IDs",
        "Cascade frames",
        rotation=45,
    )

else:
    print(
        "No co-disconnected line pairs were found."
    )


# In[75]:


# EDA 67: FREQUENTLY CO-DISCONNECTED LINE-PAIR ANALYSIS

def pair_counter_to_table(
    counter,
    table_name,
    top_n=50,
):
    rows = []

    for (
        first_line,
        second_line,
    ), count in counter.most_common(
        top_n
    ):
        rows.append({
            "first_line_id": int(
                first_line
            ),
            "second_line_id": int(
                second_line
            ),
            "first_line_path": (
                f"{LINE_OR_BUS[first_line]}"
                f" -> "
                f"{LINE_EX_BUS[first_line]}"
            ),
            "second_line_path": (
                f"{LINE_OR_BUS[second_line]}"
                f" -> "
                f"{LINE_EX_BUS[second_line]}"
            ),
            "co_disconnected_frames": int(
                count
            ),
        })

    table = pd.DataFrame(
        rows
    )

    store_table(
        table_name,
        table,
    )

    return table


top_all_pairs = pair_counter_to_table(
    disconnected_pair_counter,
    "67_top_co_disconnected_line_pairs_all_multiline_states",
    top_n=50,
)

top_cascade_pairs = pair_counter_to_table(
    cascade_pair_counter,
    "67_top_co_disconnected_line_pairs_cascade_only",
    top_n=50,
)

if len(
    top_cascade_pairs
):
    top_pairs_plot = (
        top_cascade_pairs
        .head(
            15
        )
        .copy()
    )

    top_pairs_plot["line_pair"] = (
        top_pairs_plot.apply(
            lambda row: (
                f"{int(row['first_line_id'])}"
                f"-"
                f"{int(row['second_line_id'])}"
            ),
            axis=1,
        )
    )

    plot_bar(
        top_pairs_plot
        .set_index(
            "line_pair"
        )[
            "co_disconnected_frames"
        ],
        "Most Frequent Co-Disconnected Line Pairs in Cascade States",
        "Line-pair IDs",
        "Cascade frames",
        rotation=45,
    )

else:
    print(
        "No co-disconnected line pairs were found."
    )


# In[76]:


# EDA 68: SUBSTATION-LEVEL HOTSPOT, CENTRALITY, AND EXPOSURE ANALYSIS

static_multigraph = nx.MultiGraph()

static_multigraph.add_nodes_from(
    range(
        N_SUB
    )
)

static_multigraph.add_edges_from(
    zip(
        LINE_OR_BUS.tolist(),
        LINE_EX_BUS.tolist(),
    )
)

static_graph = nx.Graph(
    static_multigraph
)

degree_map = dict(
    static_graph.degree()
)

betweenness_map = nx.betweenness_centrality(
    static_graph,
    normalized=True,
)

closeness_map = nx.closeness_centrality(
    static_graph
)

articulation_points = set(
    nx.articulation_points(
        static_graph
    )
)

load_counts = np.bincount(
    np.asarray(
        metadata[
            "topology"
        ][
            "load_to_sub"
        ],
        dtype=int,
    ),
    minlength=N_SUB,
)

generator_counts = np.bincount(
    np.asarray(
        metadata[
            "topology"
        ][
            "gen_to_sub"
        ],
        dtype=int,
    ),
    minlength=N_SUB,
)

substation_overload_endpoint_frames = np.zeros(
    N_SUB,
    dtype=np.float64,
)

substation_disconnect_endpoint_frames = np.zeros(
    N_SUB,
    dtype=np.float64,
)

substation_cascade_disconnect_endpoint_frames = np.zeros(
    N_SUB,
    dtype=np.float64,
)

np.add.at(
    substation_overload_endpoint_frames,
    LINE_OR_BUS,
    line_level_table[
        "overload_frames"
    ].to_numpy(
        dtype=float
    ),
)

np.add.at(
    substation_overload_endpoint_frames,
    LINE_EX_BUS,
    line_level_table[
        "overload_frames"
    ].to_numpy(
        dtype=float
    ),
)

np.add.at(
    substation_disconnect_endpoint_frames,
    LINE_OR_BUS,
    line_level_table[
        "disconnected_frames"
    ].to_numpy(
        dtype=float
    ),
)

np.add.at(
    substation_disconnect_endpoint_frames,
    LINE_EX_BUS,
    line_level_table[
        "disconnected_frames"
    ].to_numpy(
        dtype=float
    ),
)

np.add.at(
    substation_cascade_disconnect_endpoint_frames,
    LINE_OR_BUS,
    line_level_table[
        "cascade_disconnect_frames"
    ].to_numpy(
        dtype=float
    ),
)

np.add.at(
    substation_cascade_disconnect_endpoint_frames,
    LINE_EX_BUS,
    line_level_table[
        "cascade_disconnect_frames"
    ].to_numpy(
        dtype=float
    ),
)

fault_location_counts = (
    df
    .loc[
        df[
            "fault_loc"
        ]
        >= 0,
        "fault_loc",
    ]
    .astype(
        int
    )
    .value_counts()
    .reindex(
        range(
            N_SUB
        ),
        fill_value=0,
    )
)

substation_hotspot_table = pd.DataFrame({
    "substation_id": np.arange(
        N_SUB,
        dtype=int,
    ),
    "static_degree": [
        degree_map.get(
            node,
            0,
        )
        for node in range(
            N_SUB
        )
    ],
    "attached_load_count": load_counts,
    "attached_generator_count": generator_counts,
    "betweenness_centrality": [
        betweenness_map.get(
            node,
            0.0,
        )
        for node in range(
            N_SUB
        )
    ],
    "closeness_centrality": [
        closeness_map.get(
            node,
            0.0,
        )
        for node in range(
            N_SUB
        )
    ],
    "is_articulation_point": [
        node in articulation_points
        for node in range(
            N_SUB
        )
    ],
    "fault_location_records": (
        fault_location_counts
        .to_numpy(
            dtype=int
        )
    ),
    "overload_endpoint_frames": (
        substation_overload_endpoint_frames
    ),
    "disconnect_endpoint_frames": (
        substation_disconnect_endpoint_frames
    ),
    "cascade_disconnect_endpoint_frames": (
        substation_cascade_disconnect_endpoint_frames
    ),
})

substation_hotspot_table[
    "disconnect_exposure_per_attached_line"
] = (
    substation_hotspot_table[
        "disconnect_endpoint_frames"
    ]
    / substation_hotspot_table[
        "static_degree"
    ].replace(
        0,
        np.nan,
    )
)

substation_hotspot_ranked = (
    substation_hotspot_table
    .sort_values(
        [
            "disconnect_endpoint_frames",
            "cascade_disconnect_endpoint_frames",
            "overload_endpoint_frames",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )
)

store_table(
    "68_substation_hotspot_profile",
    substation_hotspot_ranked.round(
        6
    ),
)

plot_bar(
    substation_hotspot_table
    .nlargest(
        15,
        "disconnect_endpoint_frames",
    )
    .set_index(
        "substation_id"
    )[
        "disconnect_endpoint_frames"
    ],
    "Substations with the Highest Disconnection Exposure",
    "Substation ID",
    "Incident disconnected-line frames",
)

plt.figure(
    figsize=(
        8,
        5,
    )
)

plt.scatter(
    substation_hotspot_table[
        "static_degree"
    ],
    substation_hotspot_table[
        "fault_location_records"
    ],
    s=60,
)

for _, row in substation_hotspot_table.iterrows():
    plt.annotate(
        str(
            int(
                row[
                    "substation_id"
                ]
            )
        ),
        (
            row[
                "static_degree"
            ],
            row[
                "fault_location_records"
            ],
        ),
        fontsize=8,
    )

plt.xlabel(
    "Static substation degree"
)

plt.ylabel(
    "Fault-location records"
)

plt.title(
    "Substation Degree versus Recorded Fault-Location Frequency"
)

plt.tight_layout()
plt.show()


# ## **K. Dynamic Connectivity and Temporal Behaviour**

# In[77]:


# EDA 69: DYNAMIC GRAPH-CONNECTIVITY AND ISLANDING ANALYSIS

dynamic_indices = sampled_record_indices(
    df,
    n=EXTENDED_SAMPLE_SIZE,
)

dynamic_records = load_records_by_indices(
    JSONL_PATH,
    dynamic_indices,
)

static_pair_multiplicity = Counter(
    canonical_edge(
        origin,
        extremity,
    )
    for origin, extremity in zip(
        LINE_OR_BUS,
        LINE_EX_BUS,
    )
)

static_simple_graph = nx.Graph()

static_simple_graph.add_nodes_from(
    range(
        N_SUB
    )
)

static_simple_graph.add_edges_from(
    canonical_edge(
        origin,
        extremity,
    )
    for origin, extremity in zip(
        LINE_OR_BUS,
        LINE_EX_BUS,
    )
)

# A bridge pair is counted only when one physical transmission line
# represents the endpoint pair.
static_bridge_pairs = {
    canonical_edge(
        origin,
        extremity,
    )
    for origin, extremity in nx.bridges(
        static_simple_graph
    )
    if static_pair_multiplicity[
        canonical_edge(
            origin,
            extremity,
        )
    ]
    == 1
}

dynamic_rows = []

for record_index in dynamic_indices:
    record = dynamic_records[
        record_index
    ]

    line_status = safe_array(
        record,
        "line_status",
        expected_length=N_LINE,
        dtype=int,
    )

    active_line_ids = np.flatnonzero(
        line_status == 1
    )

    disconnected_line_ids = np.flatnonzero(
        line_status == 0
    )

    graph = nx.Graph()

    graph.add_nodes_from(
        range(
            N_SUB
        )
    )

    graph.add_edges_from(
        (
            int(
                LINE_OR_BUS[
                    line_id
                ]
            ),
            int(
                LINE_EX_BUS[
                    line_id
                ]
            ),
        )
        for line_id in active_line_ids
    )

    components = list(
        nx.connected_components(
            graph
        )
    )

    component_sizes = sorted(
        (
            len(
                component
            )
            for component in components
        ),
        reverse=True,
    )

    disconnected_static_bridge_lines = sum(
        canonical_edge(
            LINE_OR_BUS[
                line_id
            ],
            LINE_EX_BUS[
                line_id
            ],
        )
        in static_bridge_pairs
        for line_id in disconnected_line_ids
    )

    dynamic_rows.append({
        "record_index": int(
            record_index
        ),
        "label": str(
            record.get(
                "label"
            )
        ),
        "active_lines": int(
            len(
                active_line_ids
            )
        ),
        "disconnected_lines": int(
            len(
                disconnected_line_ids
            )
        ),
        "connected_components": int(
            len(
                components
            )
        ),
        "largest_component_size": (
            int(
                component_sizes[
                    0
                ]
            )
            if component_sizes
            else 0
        ),
        "largest_component_fraction": (
            float(
                component_sizes[
                    0
                ]
                / N_SUB
            )
            if component_sizes
            else 0.0
        ),
        "isolated_substations": int(
            len(
                list(
                    nx.isolates(
                        graph
                    )
                )
            )
        ),
        "dynamic_density": float(
            nx.density(
                graph
            )
        ),
        "disconnected_static_bridge_lines": int(
            disconnected_static_bridge_lines
        ),
        "is_fragmented": bool(
            len(
                components
            )
            > 1
        ),
    })


dynamic_connectivity_detail = pd.DataFrame(
    dynamic_rows
)

store_table(
    "69_dynamic_connectivity_detail",
    dynamic_connectivity_detail.head(
        50
    ),
)

dynamic_connectivity_by_class = (
    dynamic_connectivity_detail
    .groupby(
        "label",
        observed=False,
    )
    .agg(
        sampled_graphs=(
            "record_index",
            "size",
        ),
        mean_active_lines=(
            "active_lines",
            "mean",
        ),
        mean_disconnected_lines=(
            "disconnected_lines",
            "mean",
        ),
        mean_connected_components=(
            "connected_components",
            "mean",
        ),
        max_connected_components=(
            "connected_components",
            "max",
        ),
        mean_largest_component_fraction=(
            "largest_component_fraction",
            "mean",
        ),
        mean_isolated_substations=(
            "isolated_substations",
            "mean",
        ),
        fragmentation_rate_pct=(
            "is_fragmented",
            lambda values: (
                100
                * values.mean()
            ),
        ),
        mean_disconnected_static_bridge_lines=(
            "disconnected_static_bridge_lines",
            "mean",
        ),
    )
    .reindex(
        LABEL_ORDER
    )
    .round(
        6
    )
)

store_table(
    "69_dynamic_connectivity_by_class",
    dynamic_connectivity_by_class,
)

plot_bar(
    dynamic_connectivity_detail
    .groupby(
        "label",
        observed=False,
    )[
        "connected_components"
    ]
    .mean()
    .reindex(
        LABEL_ORDER
    ),
    "Mean Number of Connected Components by Operational State",
    "Operational state",
    "Connected components",
)

plot_bar(
    dynamic_connectivity_detail
    .groupby(
        "label",
        observed=False,
    )[
        "isolated_substations"
    ]
    .mean()
    .reindex(
        LABEL_ORDER
    ),
    "Mean Number of Isolated Substations by Operational State",
    "Operational state",
    "Isolated substations",
)


# In[78]:


# EDA 70: TEMPORAL, CHRONIC-LEVEL, STATE-PERSISTENCE, AND RECOVERY ANALYSIS

temporal = (
    df
    .sort_values(
        "record_index"
    )
    .copy()
)

temporal[
    "label_str"
] = temporal[
    "label"
].astype(
    str
)

previous_chronic = temporal[
    "chronic_id"
].shift(
    1
)

previous_timestep = temporal[
    "timestep"
].shift(
    1
)

new_episode = (
    temporal[
        "chronic_id"
    ].ne(
        previous_chronic
    )
    | temporal[
        "timestep"
    ].le(
        previous_timestep
    )
    | previous_timestep.isna()
)

temporal[
    "episode_id"
] = new_episode.cumsum().astype(
    int
)

chronic_summary = (
    temporal
    .groupby(
        "chronic_id",
        observed=False,
    )
    .agg(
        records=(
            "record_index",
            "size",
        ),
        observed_episodes=(
            "episode_id",
            "nunique",
        ),
        first_record_index=(
            "record_index",
            "min",
        ),
        last_record_index=(
            "record_index",
            "max",
        ),
        minimum_timestep=(
            "timestep",
            "min",
        ),
        maximum_timestep=(
            "timestep",
            "max",
        ),
        mean_max_rho=(
            "max_rho",
            "mean",
        ),
        mean_disconnected_lines=(
            "disconnected_lines",
            "mean",
        ),
    )
)

for label in LABEL_ORDER:
    chronic_summary[
        f"{label}_records"
    ] = (
        temporal
        .assign(
            is_label=temporal[
                "label_str"
            ].eq(
                label
            )
        )
        .groupby(
            "chronic_id"
        )[
            "is_label"
        ]
        .sum()
    )

store_table(
    "70_chronic_level_summary",
    chronic_summary.round(
        6
    ),
)

temporal[
    "next_episode_id"
] = temporal[
    "episode_id"
].shift(
    -1
)

temporal[
    "next_label"
] = temporal[
    "label_str"
].shift(
    -1
)

valid_transition_mask = temporal[
    "episode_id"
].eq(
    temporal[
        "next_episode_id"
    ]
)

transition_counts = pd.crosstab(
    temporal.loc[
        valid_transition_mask,
        "label_str",
    ],
    temporal.loc[
        valid_transition_mask,
        "next_label",
    ],
).reindex(
    index=LABEL_ORDER,
    columns=LABEL_ORDER,
    fill_value=0,
)

transition_probabilities = (
    transition_counts
    .div(
        transition_counts
        .sum(
            axis=1
        )
        .replace(
            0,
            np.nan,
        ),
        axis=0,
    )
    .fillna(
        0.0
    )
    .round(
        6
    )
)

store_table(
    "70_state_transition_counts",
    transition_counts,
)

store_table(
    "70_state_transition_probabilities",
    transition_probabilities,
)

plt.figure(
    figsize=(
        7,
        6,
    )
)

plt.imshow(
    transition_probabilities,
    aspect="auto",
    vmin=0.0,
    vmax=1.0,
)

plt.colorbar(
    label="Transition probability"
)

plt.xticks(
    range(
        len(
            LABEL_ORDER
        )
    ),
    LABEL_ORDER,
    rotation=45,
)

plt.yticks(
    range(
        len(
            LABEL_ORDER
        )
    ),
    LABEL_ORDER,
)

plt.xlabel(
    "Next stored state"
)

plt.ylabel(
    "Current stored state"
)

plt.title(
    "Observed State-Transition Probability Matrix"
)

plt.tight_layout()
plt.show()

run_boundary = (
    temporal[
        "episode_id"
    ].ne(
        temporal[
            "episode_id"
        ].shift(
            1
        )
    )
    | temporal[
        "label_str"
    ].ne(
        temporal[
            "label_str"
        ].shift(
            1
        )
    )
)

temporal[
    "run_id"
] = run_boundary.cumsum().astype(
    int
)

state_runs = (
    temporal
    .groupby(
        [
            "episode_id",
            "run_id",
            "label_str",
        ],
        observed=False,
    )
    .agg(
        stored_frames=(
            "record_index",
            "size",
        ),
        start_record_index=(
            "record_index",
            "min",
        ),
        end_record_index=(
            "record_index",
            "max",
        ),
        start_timestep=(
            "timestep",
            "min",
        ),
        end_timestep=(
            "timestep",
            "max",
        ),
        maximum_rho=(
            "max_rho",
            "max",
        ),
        maximum_disconnected_lines=(
            "disconnected_lines",
            "max",
        ),
    )
    .reset_index()
)

state_runs[
    "observed_timestep_span"
] = (
    state_runs[
        "end_timestep"
    ]
    - state_runs[
        "start_timestep"
    ]
    + 1
)

run_summary = (
    state_runs
    .groupby(
        "label_str",
        observed=False,
    )
    .agg(
        observed_runs=(
            "run_id",
            "size",
        ),
        median_stored_frames=(
            "stored_frames",
            "median",
        ),
        mean_stored_frames=(
            "stored_frames",
            "mean",
        ),
        median_observed_timestep_span=(
            "observed_timestep_span",
            "median",
        ),
        mean_observed_timestep_span=(
            "observed_timestep_span",
            "mean",
        ),
        maximum_observed_timestep_span=(
            "observed_timestep_span",
            "max",
        ),
    )
    .reindex(
        LABEL_ORDER
    )
    .round(
        6
    )
)

store_table(
    "70_state_run_duration_summary",
    run_summary,
)

state_runs[
    "next_episode_id"
] = state_runs[
    "episode_id"
].shift(
    -1
)

state_runs[
    "next_label"
] = state_runs[
    "label_str"
].shift(
    -1
)

state_runs[
    "next_start_timestep"
] = state_runs[
    "start_timestep"
].shift(
    -1
)

recovery_runs = state_runs[
    state_runs[
        "label_str"
    ].isin(
        [
            "overload",
            "line_trip",
            "cascade",
        ]
    )
    & state_runs[
        "episode_id"
    ].eq(
        state_runs[
            "next_episode_id"
        ]
    )
    & state_runs[
        "next_label"
    ].eq(
        "normal"
    )
].copy()

recovery_runs[
    "observed_recovery_gap"
] = (
    recovery_runs[
        "next_start_timestep"
    ]
    - recovery_runs[
        "end_timestep"
    ]
)

recovery_summary = (
    recovery_runs
    .groupby(
        "label_str",
        observed=False,
    )[
        "observed_recovery_gap"
    ]
    .agg([
        "count",
        "mean",
        "median",
        "min",
        "max",
    ])
    .reindex(
        [
            "overload",
            "line_trip",
            "cascade",
        ]
    )
    .round(
        6
    )
)

store_table(
    "70_observed_recovery_gap_summary",
    recovery_summary,
)

lag_pairs = temporal[
    temporal[
        "episode_id"
    ].eq(
        temporal[
            "episode_id"
        ].shift(
            -1
        )
    )
][[
    "max_rho"
]].copy()

lag_pairs[
    "next_max_rho"
] = temporal[
    "max_rho"
].shift(
    -1
)

lag1_autocorrelation = lag_pairs[
    "max_rho"
].corr(
    lag_pairs[
        "next_max_rho"
    ]
)

store_table(
    "70_lag1_rho_autocorrelation",
    pd.DataFrame({
        "metric": [
            "lag-1 max_rho autocorrelation within episodes"
        ],
        "value": [
            lag1_autocorrelation
        ],
    }).round(
        6
    ),
)


# ## **L. Split Integrity, Generator Bias, and Feature-Selection Justification**

# In[79]:


# EDA 71: ACTUAL SAVED SPLIT-FILE AUDIT

from scipy.stats import ks_2samp

SPLIT_PATHS = {
    "train": Path(
        "data/split_neurips2020_train_idx.npy"
    ),
    "validation": Path(
        "data/split_neurips2020_val_idx.npy"
    ),
    "test": Path(
        "data/split_neurips2020_test_idx.npy"
    ),
}

missing_split_files = [
    str(
        path
    )
    for path in SPLIT_PATHS.values()
    if not path.exists()
]

if missing_split_files:
    print(
        "The saved split files were not found:"
    )

    for missing_file in missing_split_files:
        print(
            " -",
            missing_file,
        )

    print(
        "\nGenerate them first with:\n"
        "python scripts/split.py "
        "--input data/grid_dataset_neurips2020.jsonl "
        "--output_prefix split_neurips2020"
    )

else:
    split_indices = {
        split_name: np.load(
            path
        ).astype(
            int
        )
        for split_name, path in SPLIT_PATHS.items()
    }

    indexed_df = df.set_index(
        "record_index",
        drop=False,
    )

    split_frames = {}
    missing_index_rows = []

    for split_name, indices in split_indices.items():
        available_mask = np.isin(
            indices,
            indexed_df.index.to_numpy(),
        )

        missing_count = int(
            (
                ~available_mask
            ).sum()
        )

        missing_index_rows.append({
            "split": split_name,
            "saved_indices": int(
                len(
                    indices
                )
            ),
            "indices_found_in_loaded_dataframe": int(
                available_mask.sum()
            ),
            "indices_missing_from_loaded_dataframe": (
                missing_count
            ),
            "unique_indices": int(
                len(
                    np.unique(
                        indices
                    )
                )
            ),
            "duplicate_indices_inside_split": int(
                len(
                    indices
                )
                - len(
                    np.unique(
                        indices
                    )
                )
            ),
        })

        split_frames[
            split_name
        ] = (
            indexed_df
            .reindex(
                indices
            )
            .dropna(
                subset=[
                    "record_index"
                ]
            )
            .copy()
        )

    store_table(
        "71_split_index_integrity",
        pd.DataFrame(
            missing_index_rows
        ),
    )

    split_distribution_rows = []

    for split_name, frame in split_frames.items():
        label_counts = (
            frame[
                "label"
            ]
            .astype(
                str
            )
            .value_counts()
            .reindex(
                LABEL_ORDER,
                fill_value=0,
            )
        )

        for label in LABEL_ORDER:
            split_distribution_rows.append({
                "split": split_name,
                "label": label,
                "records": int(
                    label_counts[
                        label
                    ]
                ),
                "percentage_within_split": (
                    100
                    * int(
                        label_counts[
                            label
                        ]
                    )
                    / max(
                        len(
                            frame
                        ),
                        1,
                    )
                ),
            })

    split_distribution_table = pd.DataFrame(
        split_distribution_rows
    )

    store_table(
        "71_actual_split_class_distribution",
        split_distribution_table.round(
            6
        ),
    )

    split_names = list(
        split_frames
    )

    overlap_rows = []

    for first_split, second_split in combinations(
        split_names,
        2,
    ):
        first_indices = set(
            split_indices[
                first_split
            ].tolist()
        )

        second_indices = set(
            split_indices[
                second_split
            ].tolist()
        )

        first_chronics = set(
            split_frames[
                first_split
            ][
                "chronic_id"
            ].astype(
                int
            )
        )

        second_chronics = set(
            split_frames[
                second_split
            ][
                "chronic_id"
            ].astype(
                int
            )
        )

        overlap_rows.append({
            "first_split": first_split,
            "second_split": second_split,
            "shared_record_indices": len(
                first_indices.intersection(
                    second_indices
                )
            ),
            "shared_chronics": len(
                first_chronics.intersection(
                    second_chronics
                )
            ),
            "first_split_chronics": len(
                first_chronics
            ),
            "second_split_chronics": len(
                second_chronics
            ),
        })

    store_table(
        "71_split_record_and_chronic_overlap",
        pd.DataFrame(
            overlap_rows
        ),
    )

    fingerprint_columns = [
        "label",
        "max_rho",
        "mean_rho",
        "std_rho",
        "total_load",
        "total_gen",
        "connected_lines",
        "disconnected_lines",
        "fault_loc",
        "timestep",
        "chronic_id",
    ]

    def aggregate_fingerprint_set(
        frame,
        decimals=None,
    ):
        fingerprint_frame = frame[
            fingerprint_columns
        ].copy()

        numeric_columns = (
            fingerprint_frame
            .select_dtypes(
                include=[
                    np.number
                ]
            )
            .columns
        )

        if decimals is not None:
            fingerprint_frame[
                numeric_columns
            ] = fingerprint_frame[
                numeric_columns
            ].round(
                decimals
            )

        hashed = pd.util.hash_pandas_object(
            fingerprint_frame,
            index=False,
        ).astype(
            str
        )

        return set(
            hashed.tolist()
        )

    exact_fingerprints = {
        split_name: aggregate_fingerprint_set(
            frame,
            decimals=None,
        )
        for split_name, frame in split_frames.items()
    }

    near_fingerprints = {
        split_name: aggregate_fingerprint_set(
            frame,
            decimals=3,
        )
        for split_name, frame in split_frames.items()
    }

    fingerprint_overlap_rows = []

    for first_split, second_split in combinations(
        split_names,
        2,
    ):
        fingerprint_overlap_rows.append({
            "first_split": first_split,
            "second_split": second_split,
            "exact_aggregate_fingerprint_overlap": len(
                exact_fingerprints[
                    first_split
                ].intersection(
                    exact_fingerprints[
                        second_split
                    ]
                )
            ),
            "rounded_near_duplicate_overlap": len(
                near_fingerprints[
                    first_split
                ].intersection(
                    near_fingerprints[
                        second_split
                    ]
                )
            ),
        })

    store_table(
        "71_split_duplicate_and_near_duplicate_overlap",
        pd.DataFrame(
            fingerprint_overlap_rows
        ),
    )

    drift_features = [
        "max_rho",
        "mean_rho",
        "total_load",
        "total_gen",
        "connected_lines",
        "disconnected_lines",
    ]

    drift_rows = []

    for first_split, second_split in combinations(
        split_names,
        2,
    ):
        for feature in drift_features:
            statistic, p_value = ks_2samp(
                split_frames[
                    first_split
                ][
                    feature
                ],
                split_frames[
                    second_split
                ][
                    feature
                ],
            )

            drift_rows.append({
                "first_split": first_split,
                "second_split": second_split,
                "feature": feature,
                "ks_statistic": statistic,
                "p_value": p_value,
            })

    store_table(
        "71_split_distribution_drift_ks_tests",
        pd.DataFrame(
            drift_rows
        ).round(
            8
        ),
    )


# In[ ]:


# EDA 72: GENERATOR-SAMPLING BIAS AND QUOTA-SATURATION ANALYSIS

ordered_generation = (
    df
    .sort_values(
        "record_index"
    )
    .copy()
)

ordered_generation[
    "label_str"
] = ordered_generation[
    "label"
].astype(
    str
)

generation_bin_labels = [
    f"{10 * i:02d}-{10 * (i + 1):02d}%"
    for i in range(
        10
    )
]

ordered_generation[
    "generation_bin"
] = pd.qcut(
    ordered_generation[
        "record_index"
    ].rank(
        method="first"
    ),
    q=10,
    labels=generation_bin_labels,
)

generation_distribution = (
    pd.crosstab(
        ordered_generation[
            "generation_bin"
        ],
        ordered_generation[
            "label_str"
        ],
        normalize="index",
    )
    .reindex(
        columns=LABEL_ORDER,
        fill_value=0.0,
    )
    * 100
)

store_table(
    "72_generation_decile_class_percentages",
    generation_distribution.round(
        6
    ),
)

generation_bin_summary = (
    ordered_generation
    .groupby(
        "generation_bin",
        observed=False,
    )
    .agg(
        records=(
            "record_index",
            "size",
        ),
        unique_chronics=(
            "chronic_id",
            "nunique",
        ),
        mean_max_rho=(
            "max_rho",
            "mean",
        ),
        mean_disconnected_lines=(
            "disconnected_lines",
            "mean",
        ),
        mean_connected_fraction=(
            "connected_fraction",
            "mean",
        ),
        first_record_index=(
            "record_index",
            "min",
        ),
        last_record_index=(
            "record_index",
            "max",
        ),
    )
    .round(
        6
    )
)

store_table(
    "72_generation_decile_summary",
    generation_bin_summary,
)

plt.figure(
    figsize=(
        11,
        6,
    )
)

for label in LABEL_ORDER:
    plt.plot(
        generation_distribution
        .index
        .astype(
            str
        ),
        generation_distribution[
            label
        ],
        marker="o",
        label=label,
    )

plt.xlabel(
    "Position within generated JSONL file"
)

plt.ylabel(
    "Class percentage within generation decile"
)

plt.title(
    "Class Composition across the Dataset-Generation Process"
)

plt.xticks(
    rotation=45
)

plt.legend()
plt.tight_layout()
plt.show()

target_records = int(
    metadata.get(
        "total_records",
        len(
            ordered_generation
        ),
    )
)

configured_caps = {
    "normal": int(
        0.35
        * target_records
    ),
    "line_trip": int(
        0.25
        * target_records
    ),
    "cascade": int(
        0.20
        * target_records
    ),
}

quota_rows = []

for label, configured_cap in configured_caps.items():
    cumulative_count = (
        ordered_generation[
            "label_str"
        ]
        .eq(
            label
        )
        .cumsum()
    )

    reached_positions = np.flatnonzero(
        cumulative_count.to_numpy()
        >= configured_cap
    )

    if len(
        reached_positions
    ):
        first_position = int(
            reached_positions[
                0
            ]
        )

        first_record_index = int(
            ordered_generation
            .iloc[
                first_position
            ][
                "record_index"
            ]
        )

        reached = True

    else:
        first_position = np.nan
        first_record_index = np.nan
        reached = False

    final_count = int(
        ordered_generation[
            "label_str"
        ]
        .eq(
            label
        )
        .sum()
    )

    quota_rows.append({
        "label": label,
        "configured_cap_records": configured_cap,
        "final_records": final_count,
        "cap_reached": reached,
        "first_dataset_position_where_cap_was_reached": (
            first_position
        ),
        "first_record_index_where_cap_was_reached": (
            first_record_index
        ),
    })

quota_table = pd.DataFrame(
    quota_rows
)

store_table(
    "72_generation_quota_saturation",
    quota_table,
)

plt.figure(
    figsize=(
        11,
        6,
    )
)

for label in LABEL_ORDER:
    cumulative_values = (
        ordered_generation[
            "label_str"
        ]
        .eq(
            label
        )
        .cumsum()
    )

    plt.plot(
        ordered_generation[
            "record_index"
        ],
        cumulative_values,
        label=label,
    )

for label, configured_cap in configured_caps.items():
    plt.axhline(
        configured_cap,
        linestyle="--",
        linewidth=1,
        label=f"{label} configured cap",
    )

plt.xlabel(
    "Stored JSONL record index"
)

plt.ylabel(
    "Cumulative stored records"
)

plt.title(
    "Cumulative Class Counts and Configured Sampling Caps"
)

plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


# EDA 73: RAW-FEATURE COVERAGE AND FEATURE-SELECTION JUSTIFICATION

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

feature_sample_indices = sampled_record_indices(
    df,
    n=FEATURE_JUSTIFICATION_SAMPLE_SIZE,
)

feature_sample_records = load_records_by_indices(
    JSONL_PATH,
    feature_sample_indices,
)

RAW_FIELD_SELECTION = {
    "load_p": True,
    "v_or": True,
    "rho": True,
    "line_status": True,
    "p_or": True,
    "q_or": True,
    "load_q": False,
    "gen_p": False,
    "gen_q": False,
    "p_ex": False,
    "q_ex": False,
    "v_ex": False,
    "topo_vect": False,
}

feature_rows = []

for record_index in feature_sample_indices:
    record = feature_sample_records[
        record_index
    ]

    row = {
        "record_index": int(
            record_index
        ),
        "label": str(
            record.get(
                "label"
            )
        ),
    }

    for field_name in RAW_FIELD_SELECTION:
        values = safe_array(
            record,
            field_name,
            dtype=float,
        )

        if values.size == 0:
            row[
                f"{field_name}__mean"
            ] = 0.0

            row[
                f"{field_name}__std"
            ] = 0.0

            row[
                f"{field_name}__minimum"
            ] = 0.0

            row[
                f"{field_name}__maximum"
            ] = 0.0

            row[
                f"{field_name}__sum"
            ] = 0.0

            row[
                f"{field_name}__abs_mean"
            ] = 0.0

            row[
                f"{field_name}__nonzero_fraction"
            ] = 0.0

            row[
                f"{field_name}__unique_count"
            ] = 0

            continue

        row[
            f"{field_name}__mean"
        ] = float(
            np.mean(
                values
            )
        )

        row[
            f"{field_name}__std"
        ] = float(
            np.std(
                values
            )
        )

        row[
            f"{field_name}__minimum"
        ] = float(
            np.min(
                values
            )
        )

        row[
            f"{field_name}__maximum"
        ] = float(
            np.max(
                values
            )
        )

        row[
            f"{field_name}__sum"
        ] = float(
            np.sum(
                values
            )
        )

        row[
            f"{field_name}__abs_mean"
        ] = float(
            np.mean(
                np.abs(
                    values
                )
            )
        )

        row[
            f"{field_name}__nonzero_fraction"
        ] = float(
            np.mean(
                values
                != 0
            )
        )

        row[
            f"{field_name}__unique_count"
        ] = int(
            np.unique(
                values
            ).size
        )

    feature_rows.append(
        row
    )

raw_feature_proxy_frame = pd.DataFrame(
    feature_rows
)

proxy_feature_columns = [
    column
    for column in raw_feature_proxy_frame.columns
    if column
    not in {
        "record_index",
        "label",
    }
]

X_proxy = (
    raw_feature_proxy_frame[
        proxy_feature_columns
    ]
    .replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )
    .fillna(
        0.0
    )
)

y_proxy = raw_feature_proxy_frame[
    "label"
].astype(
    str
)

mutual_information = mutual_info_classif(
    X_proxy,
    y_proxy,
    random_state=RANDOM_STATE,
)

proxy_mi_table = pd.DataFrame({
    "proxy_feature": proxy_feature_columns,
    "mutual_information": mutual_information,
})

proxy_mi_table[
    "raw_field"
] = (
    proxy_mi_table[
        "proxy_feature"
    ]
    .str
    .split(
        "__"
    )
    .str[
        0
    ]
)

proxy_mi_table[
    "selected_by_current_graph_builder"
] = (
    proxy_mi_table[
        "raw_field"
    ]
    .map(
        RAW_FIELD_SELECTION
    )
)

store_table(
    "73_proxy_feature_mutual_information",
    proxy_mi_table
    .sort_values(
        "mutual_information",
        ascending=False,
    )
    .round(
        8
    ),
)

raw_field_justification_table = (
    proxy_mi_table
    .groupby(
        [
            "raw_field",
            "selected_by_current_graph_builder",
        ],
        observed=False,
    )
    .agg(
        maximum_proxy_mutual_information=(
            "mutual_information",
            "max",
        ),
        mean_proxy_mutual_information=(
            "mutual_information",
            "mean",
        ),
        proxy_feature_count=(
            "proxy_feature",
            "size",
        ),
    )
    .reset_index()
    .sort_values(
        "maximum_proxy_mutual_information",
        ascending=False,
    )
    .round(
        8
    )
)

store_table(
    "73_raw_field_selection_justification",
    raw_field_justification_table,
)

selected_proxy_columns = [
    "load_p__sum",
    "v_or__mean",
    "rho__maximum",
    "line_status__mean",
    "p_or__abs_mean",
    "q_or__abs_mean",
]

excluded_proxy_columns = [
    column
    for column in proxy_feature_columns
    if column.split(
        "__"
    )[
        0
    ]
    in {
        field_name
        for field_name, is_selected
        in RAW_FIELD_SELECTION.items()
        if not is_selected
    }
]

def proxy_baseline_score(
    feature_columns
):
    X = X_proxy[
        feature_columns
    ]

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y_proxy,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_proxy,
    )

    model = RandomForestClassifier(
        n_estimators=140,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    model.fit(
        X_train,
        y_train,
    )

    prediction = model.predict(
        X_test
    )

    return {
        "feature_count": len(
            feature_columns
        ),
        "macro_f1": f1_score(
            y_test,
            prediction,
            average="macro",
        ),
        "weighted_f1": f1_score(
            y_test,
            prediction,
            average="weighted",
        ),
    }


current_proxy_result = proxy_baseline_score(
    selected_proxy_columns
)

expanded_proxy_result = proxy_baseline_score(
    selected_proxy_columns
    + excluded_proxy_columns
)

feature_coverage_comparison = pd.DataFrame([
    {
        "diagnostic_setting": (
            "current graph-source proxy fields only"
        ),
        **current_proxy_result,
    },
    {
        "diagnostic_setting": (
            "current proxy fields plus excluded raw fields"
        ),
        **expanded_proxy_result,
    },
])

store_table(
    "73_proxy_feature_coverage_comparison",
    feature_coverage_comparison.round(
        8
    ),
)

print(
    "\nInterpretation note:\n"
    "- This cell helps identify which raw fields deserve deeper ablation experiments.\n"
    "- The Random-Forest results are diagnostic proxies only.\n"
    "- A final thesis claim about feature selection should be supported by a true GNN ablation run.\n"
)


# ## **Export Report-Ready Tables**
# 
# 

# In[82]:


# EXPORT: SAVE REPORT-READY TABLES, FIGURE PATHS, AND RUN SUMMARY

# 1. Re-export every registered table

for table_name, table in EDA_TABLES.items():

    safe_name = make_safe_filename(
        table_name
    )

    output_path = (
        TABLE_OUTPUT_DIR
        / f"{safe_name}.csv"
    )

    table.to_csv(
        output_path,
        index=True,
    )

    SAVED_TABLE_PATHS[
        table_name
    ] = str(
        output_path.resolve()
    )


# 2. Export the table manifest

table_manifest_path = (
    SUMMARY_OUTPUT_DIR
    / "eda_table_manifest.json"
)

with table_manifest_path.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        SAVED_TABLE_PATHS,
        file,
        indent=2,
    )


# 3. Export the figure manifest

figure_manifest_path = (
    SUMMARY_OUTPUT_DIR
    / "eda_figure_manifest.json"
)

with figure_manifest_path.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        SAVED_FIGURE_PATHS,
        file,
        indent=2,
    )


# 4. Export the complete EDA-run summary

run_summary = {
    "environment": metadata.get(
        "env_name"
    ),
    "records_loaded": int(
        len(
            df
        )
    ),
    "labels": LABEL_ORDER,
    "generated_table_count": int(
        len(
            SAVED_TABLE_PATHS
        )
    ),
    "generated_figure_count": int(
        len(
            SAVED_FIGURE_PATHS
        )
    ),
    "table_output_directory": str(
        TABLE_OUTPUT_DIR.resolve()
    ),
    "figure_output_directory": str(
        FIGURE_OUTPUT_DIR.resolve()
    ),
    "summary_output_directory": str(
        SUMMARY_OUTPUT_DIR.resolve()
    ),
    "generated_tables": sorted(
        SAVED_TABLE_PATHS
    ),
    "generated_figures": (
        SAVED_FIGURE_PATHS
    ),
}

summary_path = (
    SUMMARY_OUTPUT_DIR
    / "eda_run_summary.json"
)

with summary_path.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        run_summary,
        file,
        indent=2,
    )


print(
    f"Saved {len(SAVED_TABLE_PATHS)} tables to:"
)

print(
    TABLE_OUTPUT_DIR.resolve()
)

print(
    f"\nSaved {len(SAVED_FIGURE_PATHS)} figures to:"
)

print(
    FIGURE_OUTPUT_DIR.resolve()
)

print(
    "\nSaved the run summary to:"
)

print(
    summary_path.resolve()
)

print(
    "\nSaved the table manifest to:"
)

print(
    table_manifest_path.resolve()
)

print(
    "\nSaved the figure manifest to:"
)

print(
    figure_manifest_path.resolve()
)

