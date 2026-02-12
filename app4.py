import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Universal Cube Explorer (3D Dimensions)", layout="wide")

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


# --- NEW: 3D aggregation (X,Y,Z as dimensions; measure as color + size) ---

def prep_dim(series, time_grain=None):
    if np.issubdtype(series.dtype, np.datetime64) and time_grain:
        s = pd.to_datetime(series, errors="coerce")
        if time_grain == "Month":
            return s.dt.to_period("M").dt.to_timestamp()
        elif time_grain == "Week":
            return s.dt.to_period("W").apply(lambda p: p.start_time)
        elif time_grain == "Day":
            return s.dt.date
        elif time_grain == "Year":
            return s.dt.to_period("Y").dt.to_timestamp()
    return series.astype(str)


def build_agg_3d(df, x_col, y_col, z_col, measure_col, aggfunc, x_time_grain):
    dfx = df.copy()

    # X with optional time grain
    if np.issubdtype(dfx[x_col].dtype, np.datetime64):
        dfx["_X"] = prep_dim(dfx[x_col], x_time_grain)
    else:
        dfx["_X"] = dfx[x_col].astype(str)

    dfx["_Y"] = dfx[y_col].astype(str)
    dfx["_Z"] = dfx[z_col].astype(str)

    # Measure can be numeric col or row-count
    if measure_col == "__rows__":
        out = dfx.groupby(["_X", "_Y", "_Z"]).size().reset_index(name="_V")
        color_label = "COUNT(rows)"
        return out, color_label

    if aggfunc == "count":
        out = dfx.groupby(["_X", "_Y", "_Z"])[measure_col].count().reset_index(name="_V")
        color_label = f"COUNT({measure_col})"
    elif aggfunc == "nunique":
        out = dfx.groupby(["_X", "_Y", "_Z"])[measure_col].nunique().reset_index(name="_V")
        color_label = f"NUnique({measure_col})"
    else:
        out = dfx.groupby(["_X", "_Y", "_Z"])[measure_col].agg(aggfunc).reset_index(name="_V")
        color_label = f"{aggfunc.upper()}({measure_col})"

    return out, color_label


def plot_3d_scatter_from_3dims(
    df3d,
    title,
    color_label,
    x_title,
    y_title,
    z_title,
    size_range=(6, 20),   # <- kamu bisa ubah di sini kalau mau
):
    if df3d.empty:
        return None

    # levels
    x_levels = list(pd.unique(df3d["_X"]))
    y_levels = list(pd.unique(df3d["_Y"]))
    z_levels = list(pd.unique(df3d["_Z"]))

    x_map = {v: i for i, v in enumerate(x_levels)}
    y_map = {v: i for i, v in enumerate(y_levels)}
    z_map = {v: i for i, v in enumerate(z_levels)}

    X = df3d["_X"].map(x_map).to_list()
    Y = df3d["_Y"].map(y_map).to_list()
    Z = df3d["_Z"].map(z_map).to_list()

    V = df3d["_V"].astype(float).to_numpy()

    # ---- NEW: size scaling based on measure ----
    v = np.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)
    v_min, v_max = float(np.min(v)), float(np.max(v))

    s_min, s_max = size_range
    if v_max > v_min:
        # linear normalization
        size = s_min + (s_max - s_min) * ((v - v_min) / (v_max - v_min + 1e-9))
    else:
        size = np.full_like(v, s_min, dtype=float)

    # hover text
    H = []
    for a, b, c, val in zip(df3d["_X"], df3d["_Y"], df3d["_Z"], V):
        if hasattr(a, "strftime"):
            a_lbl = a.strftime("%Y-%m")
        else:
            a_lbl = str(a)
        H.append(f"<b>{a_lbl}</b><br>Y: {b}<br>Z: {c}<br>{color_label}: {val:,.2f}")

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=X, y=Y, z=Z,
                mode="markers",
                marker=dict(
                    size=size,          # <- size pakai measure
                    color=V,            # <- color pakai measure juga
                    colorscale="Viridis",
                    opacity=0.9,
                    colorbar=dict(title=color_label),
                ),
                text=H,
                hovertemplate="%{text}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        scene=dict(
            xaxis=dict(
                title=x_title,
                tickmode="array",
                tickvals=list(range(len(x_levels))),
                ticktext=[str(v) for v in x_levels],
            ),
            yaxis=dict(
                title=y_title,
                tickmode="array",
                tickvals=list(range(len(y_levels))),
                ticktext=y_levels,
            ),
            zaxis=dict(
                title=z_title,
                tickmode="array",
                tickvals=list(range(len(z_levels))),
                ticktext=z_levels,
            ),
            aspectmode="cube",
        ),
    )
    return fig



def cube_table_xyz(df3d):
    if df3d.empty:
        return pd.DataFrame()
    t = df3d.pivot_table(index=["_X", "_Y"], columns="_Z", values="_V", aggfunc="sum", fill_value=0)
    return t.sort_index()


# ----------------------------------
# UI
# ----------------------------------

st.title("🧊 Universal Cube Explorer — 3D Dimensions + Color Measure")

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
# Dynamic Filters  (UNCHANGED - as requested)
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
# Dimensions & Measure (UPDATED: X,Y,Z are dimensions; measure = color)
# ----------------------------------

st.sidebar.header("3) Cube Dimensions")

datetime_cols = date_cols
numeric_cols = numeric_cols
categorical_cols = sorted(cat_cols)  # IMPORTANT: only categorical

x_col = st.sidebar.selectbox("X", options=datetime_cols + categorical_cols)
y_col = st.sidebar.selectbox("Y", options=categorical_cols)
z_dim_col = st.sidebar.selectbox("Z (dimension)", options=categorical_cols)

measure_mode = st.sidebar.selectbox("Measure source (color)", ["Row count", "Numeric column"], index=0)
if measure_mode == "Numeric column" and len(numeric_cols) > 0:
    measure_col = st.sidebar.selectbox("Measure column", options=numeric_cols)
else:
    measure_col = "__rows__"

aggfunc = st.sidebar.selectbox("Aggregation", ["sum", "mean", "count", "max", "min", "nunique"])

time_grain = "Month"
if x_col in datetime_cols:
    time_grain = st.sidebar.selectbox("Time Grain (X)", ["Year", "Month", "Week", "Day"])

top_y = st.sidebar.slider("Top-N kategori Y", 3, 50, 12)
top_z = st.sidebar.slider("Top-N kategori Z", 3, 50, 12)

st.subheader("🔎 Filtered Data Preview")
st.dataframe(df_filtered.head())
st.write(f"Filtered rows: {len(df_filtered):,}")

# ----------------------------------
# Build cube (3D dims)
# ----------------------------------

df3d, color_label = build_agg_3d(df_filtered, x_col, y_col, z_dim_col, measure_col, aggfunc, time_grain)

if df3d.empty:
    st.warning("No data after filtering.")
    st.stop()

# Limit top-N for Y and Z by total |V|
df3d["_absV"] = df3d["_V"].abs()
keep_y = df3d.groupby("_Y")["_absV"].sum().sort_values(ascending=False).head(top_y).index
keep_z = df3d.groupby("_Z")["_absV"].sum().sort_values(ascending=False).head(top_z).index
df3d_top = df3d[df3d["_Y"].isin(keep_y) & df3d["_Z"].isin(keep_z)].copy()
df3d_top.drop(columns=["_absV"], inplace=True, errors="ignore")

# ----------------------------------
# Visual + Table
# ----------------------------------

fig = plot_3d_scatter_from_3dims(
    df3d_top,
    title=f"3D Cube: {x_col} × {y_col} × {z_dim_col} | Color: {color_label}",
    color_label=color_label,
    x_title=x_col,
    y_title=y_col,
    z_title=z_dim_col,
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Cube Table (X,Y) × Z")
cube_tbl = cube_table_xyz(df3d_top)
st.dataframe(cube_tbl)

st.download_button(
    "Download Cube Table CSV",
    data=cube_tbl.to_csv().encode("utf-8"),
    file_name="cube_table_xyz.csv",
    mime="text/csv"
)

st.subheader("Aggregated Points (X,Y,Z,Value)")
st.dataframe(df3d_top[["_X", "_Y", "_Z", "_V"]].head(200))

st.download_button(
    "Download Aggregated Points CSV",
    data=df3d_top[["_X", "_Y", "_Z", "_V"]].to_csv(index=False).encode("utf-8"),
    file_name="cube_points_xyz.csv",
    mime="text/csv"
)
