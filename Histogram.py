"""
NE111 Project – Histogram app by Morgan Wong
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scipy.stats import (
    norm,
    gamma,
    weibull_min,
    lognorm,
    beta,
    expon,
    rayleigh,
    uniform,
    chi2,
    triang,
)

# --------- Page Config ---------
st.set_page_config(
    page_title="Histogram Distribution Fitter",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------- Custom Styling ---------
st.markdown(
    """
    <style>
    html, body, [class*="css"]  {
        font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif;
    }

    /* App header */
    .app-header {
        padding: 1.6rem 1.2rem;
        margin-bottom: 1.5rem;
        border-radius: 14px;
        background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
        color: white;
        box-shadow: 0 4px 25px rgba(0,0,0,0.18);
    }
    .app-header h1 {
        margin: 0;
        font-size: 2.1rem;
        font-weight: 700;
    }
    .app-header p {
        margin-top: 0.3rem;
        font-size: 0.95rem;
        opacity: 0.95;
    }

    /* Glass cards */
    .glass-card {
        background: rgba(255,255,255,0.8);
        backdrop-filter: blur(10px);
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        border: 1px solid rgba(148,163,184,0.5);
        margin-bottom: 1.1rem;
    }

    /* Metric boxes */
    .metric-box {
        padding: 0.8rem;
        border-radius: 10px;
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    .metric-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        color: #64748b;
        letter-spacing: 0.05em;
    }

    /* Tabs */
    button[role="tab"] {
        border-radius: 999px !important;
        padding: 6px 18px !important;
        font-weight: 600 !important;
    }

    /* JSON containers */
    .stJson {
        border-radius: 10px !important;
        background: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
        padding: 0.8rem !important;
    }

    /* Error cards */
    .error-card {
        padding: 0.75rem 0.9rem;
        border-radius: 0.75rem;
        border: 1px dashed rgba(148,163,184,0.8);
        background-color: rgba(249, 250, 251, 0.9);
        margin-top: 0.5rem;
    }
    .error-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #4b5563;
        margin-bottom: 0.25rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .error-line {
        font-size: 0.8rem;
        color: #111827;
        margin-bottom: 0.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------- Helper Functions ---------
DISTRIBUTIONS = {
    "Normal (norm)": norm,
    "Gamma (gamma)": gamma,
    "Weibull (weibull_min)": weibull_min,
    "Lognormal (lognorm)": lognorm,
    "Beta (beta)": beta,
    "Exponential (expon)": expon,
    "Rayleigh (rayleigh)": rayleigh,
    "Uniform (uniform)": uniform,
    "Chi-squared (chi2)": chi2,
    "Triangular (triang)": triang,
}


def parse_manual_data(text: str) -> np.ndarray:
    if not text.strip():
        return np.array([])
    clean = text.replace(",", " ")
    tokens = [t for t in clean.split() if t.strip() != ""]
    values = []
    for t in tokens:
        try:
            values.append(float(t))
        except ValueError:
            # Ignore non-numeric tokens and keep going
            continue
    return np.array(values, dtype=float)


def get_histogram(data: np.ndarray, bins="auto"):
    counts, edges = np.histogram(data, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts, edges


def compute_errors(hist_y: np.ndarray, pdf_y: np.ndarray) -> dict:
    diff = pdf_y - hist_y
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    max_err = float(np.max(np.abs(diff)))
    return {"MSE": mse, "MAE": mae, "Max Error": max_err}


def split_params(params):
    params = tuple(params)
    if len(params) <= 2:
        shapes = ()
        loc, scale = params
    else:
        shapes = params[:-2]
        loc, scale = params[-2:]
    return shapes, float(loc), float(scale)


def stringify_params(params):
    shapes, loc, scale = split_params(params)
    result = {}
    for i, s in enumerate(shapes, start=1):
        result[f"shape{i}"] = float(s)
    result["loc"] = loc
    result["scale"] = scale
    return result


# --------- Header ---------
st.markdown(
    """
    <div class="app-header">
        <h1>📊 Distribution Fitting Tool</h1>
        <p>Fit probability distributions to your dataset and compare automatic vs manual fits.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------- Sidebar: Data Input ---------
st.sidebar.markdown("### 1️⃣ Data Input")

input_mode = st.sidebar.radio(
    "Data source",
    ["Paste numbers", "Upload CSV"],
)

data = np.array([])

if input_mode == "Paste numbers":
    example_text = "1.2 1.5 1.9 2.0 2.1 2.3 2.8 3.0 3.1 3.2"
    text = st.sidebar.text_area(
        "Enter numbers (spaces, commas, or newlines):",
        value=example_text,
        height=160,
    )
    data = parse_manual_data(text)

else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload a CSV file", type=["csv"]
    )
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, header=None)
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if not numeric_cols:
                st.sidebar.error("No numeric columns found in the CSV.")
            else:
                st.sidebar.write("Numeric columns:", numeric_cols)
                selected_cols = st.sidebar.multiselect(
                    "Choose column(s) to use as data:",
                    numeric_cols,
                    default=numeric_cols[:1],
                )
                if selected_cols:
                    data = df[selected_cols].to_numpy().ravel()
        except Exception as e:
            st.sidebar.error(f"Error reading CSV: {e}")

# --------- Data Overview ---------
if data.size == 0:
    st.warning("No valid numeric data yet. Please enter or upload data to proceed.")
    st.stop()

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Dataset Overview")

c1, c2, c3, c4 = st.columns(4)
c1.markdown(
    f"""
    <div class='metric-box'>
        <div class='metric-value'>{data.size}</div>
        <div class='metric-label'>Count</div>
    </div>
    """,
    unsafe_allow_html=True,
)
c2.markdown(
    f"""
    <div class='metric-box'>
        <div class='metric-value'>{np.mean(data):.4g}</div>
        <div class='metric-label'>Mean</div>
    </div>
    """,
    unsafe_allow_html=True,
)
c3.markdown(
    f"""
    <div class='metric-box'>
        <div class='metric-value'>{np.std(data, ddof=1):.4g}</div>
        <div class='metric-label'>Std Dev</div>
    </div>
    """,
    unsafe_allow_html=True,
)
c4.markdown(
    f"""
    <div class='metric-box'>
        <div class='metric-value'>{np.min(data):.4g} / {np.max(data):.4g}</div>
        <div class='metric-label'>Min / Max</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("👀 Preview first 20 values"):
    st.write(pd.Series(data[:20]))

st.markdown('</div>', unsafe_allow_html=True)

# --------- Sidebar: Distribution & Fit Options ---------
st.sidebar.markdown("---")
st.sidebar.markdown("### 2️⃣ Distribution & Fit")

dist_name = st.sidebar.selectbox(
    "Distribution",
    list(DISTRIBUTIONS.keys()),
    index=0,
)
dist_class = DISTRIBUTIONS[dist_name]

num_bins = st.sidebar.slider(
    "Histogram bins",
    min_value=5,
    max_value=100,
    value=30,
)

show_auto_fit = st.sidebar.checkbox("Show automatic fit", value=True)
enable_manual = st.sidebar.checkbox("Enable manual sliders", value=True)

# --------- Histogram + x-grid ---------
centers, hist_y, bin_edges = get_histogram(data, bins=num_bins)

data_min, data_max = float(np.min(data)), float(np.max(data))
data_range = max(data_max - data_min, 1e-6)
x_plot = np.linspace(data_min - 0.1 * data_range, data_max + 0.1 * data_range, 400)

# --------- Automatic Fit ---------
auto_fit_params = None
auto_pdf_at_centers = None
auto_errors = None
auto_pdf_for_plot = None

if show_auto_fit:
    try:
        auto_fit_params = dist_class.fit(data)
        auto_shapes, auto_loc, auto_scale = split_params(auto_fit_params)
        dist_auto = dist_class(*auto_fit_params)
        auto_pdf_at_centers = dist_auto.pdf(centers)
        auto_pdf_for_plot = dist_auto.pdf(x_plot)
        auto_errors = compute_errors(hist_y, auto_pdf_at_centers)
    except Exception as e:
        st.error(f"Automatic fit failed for {dist_name}: {e}")
        auto_fit_params = None

# --------- Manual Fit Sliders ---------
manual_params = None
manual_pdf_for_plot = None
manual_errors = None

if enable_manual:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 3️⃣ Manual Tuning")

    if auto_fit_params is None:
        fallback_loc = float(np.mean(data))
        fallback_scale = float(np.std(data, ddof=1) or 1.0)
        n_shapes = 1
        shapes_default = (1.0,) * n_shapes
    else:
        shapes_default, fallback_loc, fallback_scale = split_params(auto_fit_params)
        n_shapes = len(shapes_default)

    shape_values = []
    for i in range(n_shapes):
        default_val = float(shapes_default[i])
        shape_val = st.sidebar.slider(
            f"shape{i+1}",
            min_value=0.01,
            max_value=20.0,
            value=float(default_val),
            step=0.01,
        )
        shape_values.append(shape_val)

    loc_min = data_min - data_range
    loc_max = data_max + data_range
    loc_val = st.sidebar.slider(
        "loc",
        min_value=float(loc_min),
        max_value=float(loc_max),
        value=float(fallback_loc),
    )

    scale_max = max(data_range * 3.0, 1e-3)
    scale_val = st.sidebar.slider(
        "scale",
        min_value=1e-3,
        max_value=float(scale_max),
        value=float(fallback_scale if fallback_scale > 0 else 1.0),
    )

    manual_params = tuple(shape_values + [loc_val, scale_val])

    try:
        dist_manual = dist_class(*manual_params)
        manual_pdf_for_plot = dist_manual.pdf(x_plot)
        manual_pdf_at_centers = dist_manual.pdf(centers)
        manual_errors = compute_errors(hist_y, manual_pdf_at_centers)
    except Exception as e:
        st.sidebar.error(f"Manual parameters are invalid for {dist_name}: {e}")
        manual_params = None
        manual_pdf_for_plot = None

# --------- Main Content: Tabs ---------
tab_fit, tab_params = st.tabs(["📈 Fit & Plot", "⚙️ Parameters & Errors"])

with tab_fit:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader(f"Histogram & Fitted Curve – {dist_name}")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.hist(
        data,
        bins=num_bins,
        density=True,
        alpha=0.6,
        edgecolor="black",
        label="Data histogram",
    )

    if show_auto_fit and auto_fit_params is not None and auto_pdf_for_plot is not None:
        ax.plot(x_plot, auto_pdf_for_plot, linewidth=2, label="Automatic fit")

    if enable_manual and manual_pdf_for_plot is not None:
        ax.plot(
            x_plot,
            manual_pdf_for_plot,
            linestyle="--",
            linewidth=2,
            label="Manual fit",
        )

    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title(dist_name)
    ax.legend()

    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

with tab_params:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Fit Parameters & Error Metrics")

    col_auto, col_manual = st.columns(2)

    with col_auto:
        st.write("### Automatic fit")
        if auto_fit_params is None:
            st.info("Automatic fit not available.")
        else:
            st.json(stringify_params(auto_fit_params))
            if auto_errors is not None:
                st.markdown(
                    """
                    <div class="error-card">
                        <div class="error-title">Error vs histogram</div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""
                        <div class="error-line">MSE: <code>{auto_errors['MSE']:.4e}</code></div>
                        <div class="error-line">MAE: <code>{auto_errors['MAE']:.4e}</code></div>
                        <div class="error-line">Max Error: <code>{auto_errors['Max Error']:.4e}</code></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with col_manual:
        st.write("### Manual fit")
        if not enable_manual or manual_params is None:
            st.info("Manual parameters not available or sliders disabled.")
        else:
            st.json(stringify_params(manual_params))
            if manual_errors is not None:
                st.markdown(
                    """
                    <div class="error-card">
                        <div class="error-title">Error vs histogram</div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""
                        <div class="error-line">MSE: <code>{manual_errors['MSE']:.4e}</code></div>
                        <div class="error-line">MAE: <code>{manual_errors['MAE']:.4e}</code></div>
                        <div class="error-line">Max Error: <code>{manual_errors['Max Error']:.4e}</code></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown('</div>', unsafe_allow_html=True)

st.caption(
    "Try switching distributions, compare error metrics, and nudge the manual sliders to build intuition for what each parameter does."
)
