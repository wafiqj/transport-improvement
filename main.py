import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Universal Cube Explorer (Minimal)", layout="wide")


# ----------------------------------
# Utilities
# ----------------------------------

@st.cache_data
def load_file(uploaded_file):
    if uploaded_file is None:
        return None
    suffix = uploaded_file.name.lower().split(".")[-1]
    if suffix in ["xlsx", "xls"]:
        df = pd.read_excel(uploaded_file, engine="openpyxl")
    else:
        df = pd.read_csv(uploaded_file)
    df.columns = [c.strip() for c in df.columns]
    return df


def auto_parse_types(df):
    df = df.copy()
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in ["date", "time", "tgl", "waktu"]):
            try:
                df[c] = pd.to_datetime(df[c], errors="ignore")
            except:
                pass

    # Auto numeric detection
    for c in df.columns:
        if df[c].dtype == object:
            sample = df[c].dropna().astype(str).str.replace(",", "", regex=False)
            if (sample.str.match(r"^-?\d+(\.\d+)?$")).mean() > 0.7:
                df[c] = pd.to_numeric(sample, errors="coerce")

    return df


def ensure_lane(df, origin="originPostcode", dest="destPostcode"):
    df = df.copy()
    if origin in df.columns and dest in df.columns:
        if "lane" not in df.columns:
            df["lane"] = df[origin].astype(str).str.strip() + "→" + df[dest].astype(str).str.strip()
    return df


def build_agg(df, x_col, y_col, measure, aggfunc, time_grain):
    dfx = df.copy()

    # Time grain
    if np.issubdtype(dfx[x_col].dtype, np.datetime64):
        if time_grain == "Month":
            dfx["_X"] = dfx[x_col].dt.to_period("M").dt.to_timestamp()
        elif time_grain == "Week":
            dfx["_X"] = dfx[x_col].dt.to_period("W").apply(lambda p: p.start_time)
        elif time_grain == "Day":
            dfx["_X"] = dfx[x_col].dt.date
        elif time_grain == "Year":
            dfx["_X"] = dfx[x_col].dt.to_period("Y").dt.to_timestamp()
    else:
        dfx["_X"] = dfx[x_col].astype(str)

    dfx["_Y"] = dfx[y_col].astype(str)

    if aggfunc == "count":
        pivot = dfx.pivot_table(index="_X", columns="_Y", values=measure,
                                aggfunc="count", fill_value=0, observed=True)
        z_label = f"COUNT({measure})"
    elif aggfunc == "nunique":
        pivot = dfx.groupby(["_X", "_Y"])[measure].nunique().unstack(fill_value=0)
        z_label = f"NUnique({measure})"
    else:
        pivot = dfx.pivot_table(index="_X", columns="_Y", values=measure,
                                aggfunc=aggfunc, fill_value=0, observed=True)
        z_label = f"{aggfunc.upper()}({measure})"

    pivot = pivot.sort_index()
    return pivot, z_label


def plot_3d_scatter_from_pivot(pivot, title, z_label):
    times = list(pivot.index)
    cats = list(pivot.columns)

    x_map = {t: i for i, t in enumerate(times)}
    y_map = {c: i for i, c in enumerate(cats)}

    X, Y, Z, H = [], [], [], []

    vals = pivot.values
    for i, t in enumerate(times):
        for j, c in enumerate(cats):
            v = float(vals[i, j])
            X.append(i)
            Y.append(j)
            Z.append(v)

            # label
            if hasattr(t, "strftime"):
                t_label = t.strftime("%Y-%m")
            else:
                t_label = str(t)

            H.append(f"<b>{t_label}</b><br>{c}<br>{z_label}: {v:,.2f}")

    z_arr = np.array(Z)
    if np.nanmax(z_arr) > 0:
        size = 6 + 14 * (z_arr / (z_arr.max() + 1e-9))
    else:
        size = np.full_like(z_arr, 6, dtype=float)

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=X, y=Y, z=Z, mode="markers",
                marker=dict(size=size, color=Z, colorscale="Viridis",
                            opacity=0.9, colorbar=dict(title=z_label)),
                text=H, hovertemplate="%{text}<extra></extra>"
            )
        ]
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        scene=dict(
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title=z_label),
            aspectmode="cube"
        )
    )
    return fig


# ----------------------------------
# UI
# ----------------------------------

st.title("🧊 Universal Cube Explorer — Minimal + Dynamic Filters")

with st.sidebar:
    st.header("1) Data Source")
    uploaded = st.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx", "xls"])
    use_dummy = st.checkbox("Use dummy dataset", value=False)

# Load data
df = None
if use_dummy:
    rng = pd.date_range("2024-01-01", "2024-06-30", freq="D")
    np.random.seed(1)
    sample = []
    lanes = ["10110→40115", "15119→60221", "40213→10110"]
    for d in rng:
        for lane in np.random.choice(lanes, size=3, replace=True):
            origin, dest = lane.split("→")
            gw = np.random.gamma(5, 150)
            sample.append({
                "loadingDate": d,
                "originPostcode": origin,
                "destPostcode": dest,
                "grossWeight": round(gw, 2),
                "transporter": np.random.choice(["A", "B", "C"]),
                "serviceLevel": np.random.choice(["REG", "EXP"])
            })
    df = pd.DataFrame(sample)
else:
    df = load_file(uploaded)

if df is None:
    st.stop()

df = auto_parse_types(df)
df = ensure_lane(df)

st.subheader("Data Preview")
st.dataframe(df.head())

# ----------------------------------
# NEW: Dynamic Filters
# ----------------------------------

st.sidebar.header("2) Dynamic Filters")

if "filters" not in st.session_state:
    st.session_state["filters"] = []  # list of dict: {"col":..., "values":...}

if st.sidebar.button("Tambah Filter"):
    st.session_state["filters"].append({"col": None, "values": None})

numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
date_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
cat_cols = list(set(df.columns) - set(numeric_cols) - set(date_cols))

# Render filter widgets
new_df = df.copy()
to_remove = []

for idx, flt in enumerate(st.session_state["filters"]):
    st.sidebar.markdown(f"### Filter {idx+1}")

    col = st.sidebar.selectbox(
        "Kolom", options=df.columns, key=f"filter_col_{idx}",
        index=(list(df.columns).index(flt["col"]) if flt["col"] in df.columns else 0)
    )
    flt["col"] = col

    # Numerik
    if col in numeric_cols:
        min_v = float(df[col].min())
        max_v = float(df[col].max())
        val = st.sidebar.slider(
            "Range", min_value=min_v, max_value=max_v,
            value=(min_v, max_v), key=f"filter_val_{idx}"
        )
        flt["values"] = val
        new_df = new_df[(new_df[col] >= val[0]) & (new_df[col] <= val[1])]

    # Datetime
    elif col in date_cols:
        min_d = df[col].min()
        max_d = df[col].max()
        val = st.sidebar.date_input(
            "Date Range", value=(min_d.date(), max_d.date()),
            key=f"filter_val_{idx}"
        )
        if isinstance(val, (tuple, list)) and len(val) == 2:
            start, end = pd.to_datetime(val[0]), pd.to_datetime(val[1])
            flt["values"] = val
            new_df = new_df[(new_df[col] >= start) & (new_df[col] <= end)]

    # Categorical
    else:
        options = sorted(df[col].dropna().astype(str).unique().tolist())
        val = st.sidebar.multiselect(
            "Values", options=options, key=f"filter_val_{idx}"
        )
        flt["values"] = val
        if len(val) > 0:
            new_df = new_df[new_df[col].astype(str).isin(val)]

    if st.sidebar.button(f"Hapus Filter {idx+1}"):
        to_remove.append(idx)

# Remove filters clicked
for r in reversed(to_remove):
    st.session_state["filters"].pop(r)

df_filtered = new_df.copy()

# ----------------------------------
# Dimensions & Measure
# ----------------------------------

st.sidebar.header("3) Cube Dimensions")
datetime_cols = date_cols
numeric_cols  = numeric_cols
categorical_cols = sorted(list(set(df.columns)))

x_col = st.sidebar.selectbox("X", options=datetime_cols + categorical_cols)
y_col = st.sidebar.selectbox("Y", options=categorical_cols)
measure_col = st.sidebar.selectbox("Measure", options=numeric_cols)

aggfunc = st.sidebar.selectbox("Aggregation", ["sum", "mean", "count", "max", "min", "nunique"])

time_grain = "Month"
if x_col in datetime_cols:
    time_grain = st.sidebar.selectbox("Time Grain", ["Year", "Month", "Week", "Day"])

top_n = st.sidebar.slider("Top-N kategori Y", 3, 50, 12)

st.subheader("🔎 Filtered Data Preview")
st.dataframe(df_filtered.head())
st.write(f"Filtered rows: {len(df_filtered):,}")

# ----------------------------------
# Build cube
# ----------------------------------

pivot, z_label = build_agg(df_filtered, x_col, y_col, measure_col, aggfunc, time_grain)

# Limit top-N
totals = pivot.sum(axis=0).sort_values(ascending=False)
keep = totals.head(top_n).index
pivot_top = pivot[keep]

# ----------------------------------
# Visual + Table
# ----------------------------------

fig = plot_3d_scatter_from_pivot(
    pivot_top,
    title=f"3D Cube: {x_col} × {y_col} × {z_label}",
    z_label=z_label,
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Pivot Table (Top-N)")
st.dataframe(pivot_top)

st.download_button(
    "Download Pivot CSV",
    data=pivot_top.to_csv().encode("utf-8"),
    file_name="cube_pivot.csv",
    mime="text/csv"
)
