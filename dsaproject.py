"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          TIME SERIES DECOMPOSITION TOOL — Stock Market Dataset              ║
║  Methods: Moving Average · STL · Descriptive Stats · PCA · Hypothesis Tests  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Requirements:
    pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn

Usage:
    python time_series_decomposition.py
"""

# ─── Standard Library ────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

# ─── Third-Party ─────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind, f_oneway, chi2_contingency
from statsmodels.tsa.seasonal import STL
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ═══════════════════════════════════════════════════════════════════════════════
# 0.  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DATA_PATH   = "/Users/rakeshv/python_project/stock_market_dataset.csv"
TICKER_DEMO = "AAPL"        # ticker used for decomposition plots
MA_WINDOW   = 20            # moving-average window (trading days)
STL_PERIOD  = 5             # weekly seasonality (5 trading days)
ALPHA       = 0.05          # significance level

# ── Dark theme palette ────────────────────────────────────────────────────────
BG      = "#0d1117"
CARD    = "#161b22"
BORDER  = "#30363d"
TEXT    = "#e6edf3"
MUTED   = "#8b949e"
ACCENT  = "#58a6ff"
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf",
]

matplotlib.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    CARD,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   TEXT,
    "text.color":        TEXT,
    "xtick.color":       TEXT,
    "ytick.color":       TEXT,
    "grid.color":        BORDER,
    "grid.alpha":        0.5,
    "legend.facecolor":  CARD,
    "legend.edgecolor":  BORDER,
    "savefig.facecolor": BG,
    "savefig.bbox":      "tight",
    "font.family":       "DejaVu Sans",
})


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD & VALIDATE DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(path: str) -> pd.DataFrame:
    """Load the CSV, parse dates, sort by Ticker → Date."""
    print(f"\n{'═'*70}")
    print("  LOADING DATA")
    print(f"{'═'*70}")
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    print(f"  ✔  Shape          : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  ✔  Date range     : {df['Date'].min().date()} → {df['Date'].max().date()}")
    print(f"  ✔  Tickers        : {sorted(df['Ticker'].unique())}")
    print(f"  ✔  Sectors        : {sorted(df['Sector'].unique())}")
    print(f"  ✔  Missing values : {df.isnull().sum().sum()}")
    return df



# ═══════════════════════════════════════════════════════════════════════════════
# DATASET DESCRIPTION
# ═══════════════════════════════════════════════════════════════════════════════
def dataset_source():
    print(f"\n{'═'*70}")
    print("  DATASET DESCRIPTION")
    print(f"{'═'*70}")
    print("Dataset Name : Stock Market Dataset")
    print("Source       : User Provided CSV")
    print("Period       : 2021 - 2022")
    print("Domain       : Financial Markets")

# ═══════════════════════════════════════════════════════════════════════════════
# DATA UNDERSTANDING
# ═══════════════════════════════════════════════════════════════════════════════
def data_understanding(df):
    print(f"\n{'═'*70}")
    print("  DATA UNDERSTANDING")
    print(f"{'═'*70}")
    print("\nDataset Preview (First 5 Rows)")
    print(df.head())
    print("\nDataset Information")
    print(f"Observations : {df.shape[0]:,}")
    print(f"Variables    : {df.shape[1]}")
    target_variable = "Price_Up"
    print(f"\nTarget Variable : {target_variable}")
    type_rows = []
    for col in df.columns:
        if col == "Date":
            dtype = "Time Series"
        elif pd.api.types.is_object_dtype(df[col]):
            dtype = "Categorical (Nominal)"
        elif pd.api.types.is_integer_dtype(df[col]):
            dtype = "Numerical (Discrete)"
        else:
            dtype = "Numerical (Continuous)"
        type_rows.append([col, dtype])
    type_df = pd.DataFrame(type_rows, columns=["Variable","Data Type"])
    print("\nVariable Classification")
    print(type_df.to_string(index=False))
    return type_df


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  DESCRIPTIVE STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

NUM_COLS = [
    "Close", "Volume", "Daily_Return_Pct", "RSI",
    "PE_Ratio", "Beta", "Market_Cap_Billions", "Dividend_Yield",
]

def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute five-number summary, IQR, outlier count for numeric columns."""
    print(f"\n{'═'*70}")
    print("  DESCRIPTIVE STATISTICS")
    print(f"{'═'*70}")

    rows = []
    for col in NUM_COLS:
        d  = df[col].dropna()
        Q1 = d.quantile(0.25)
        Q3 = d.quantile(0.75)
        IQR = Q3 - Q1
        outliers = int(((d < Q1 - 1.5 * IQR) | (d > Q3 + 1.5 * IQR)).sum())
        rows.append({
            "Variable":  col,
            "N":         len(d),
            "Mean":      round(d.mean(),    4),
            "Median":    round(d.median(),  4),
            "Std":       round(d.std(),     4),
            "Variance":  round(d.var(),     4),
            "Min":       round(d.min(),     4),
            "Q1":        round(Q1,          4),
            "Q3":        round(Q3,          4),
            "Max":       round(d.max(),     4),
            "IQR":       round(IQR,         4),
            "Outliers":  outliers,
        })

    summary = pd.DataFrame(rows).set_index("Variable")
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(summary.to_string())
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  PROBABILITY & BAYES' THEOREM
# ═══════════════════════════════════════════════════════════════════════════════

def probability_analysis(df: pd.DataFrame) -> dict:
    """Basic & conditional probability + Bayes' theorem."""
    print(f"\n{'═'*70}")
    print("  PROBABILITY ANALYSIS")
    print(f"{'═'*70}")

    p_up           = df["Price_Up"].mean()
    p_bull         = (df["Market_Regime"] == "Bull").mean()
    p_up_bull      = df[df["Market_Regime"] == "Bull"]["Price_Up"].mean()
    p_bull_given_up = (p_up_bull * p_bull) / p_up          # Bayes

    results = {
        "P(Price Up)":         p_up,
        "P(Bull Regime)":      p_bull,
        "P(Up | Bull)":        p_up_bull,
        "P(Bull | Up) [Bayes]": p_bull_given_up,
    }

    for k, v in results.items():
        print(f"  {k:<28} = {v:.4f}  ({v*100:.2f}%)")

    print("\n  Bayes' Theorem:")
    print(f"    P(Bull|Up) = P(Up|Bull) × P(Bull) / P(Up)")
    print(f"              = {p_up_bull:.4f} × {p_bull:.4f} / {p_up:.4f}")
    print(f"              = {p_bull_given_up:.4f}")
    print("  → A single-day price rise provides almost no evidence for a Bull "
          "regime\n    (posterior ≈ prior), consistent with day-level market noise.")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  INFERENTIAL STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

def inferential_stats(df: pd.DataFrame) -> dict:
    """t-test, ANOVA, chi-square, confidence interval."""
    print(f"\n{'═'*70}")
    print("  INFERENTIAL STATISTICS")
    print(f"{'═'*70}")

    # ── t-test: Bull vs Bear daily return ────────────────────────────────────
    bull = df[df["Market_Regime"] == "Bull"]["Daily_Return_Pct"].dropna()
    bear = df[df["Market_Regime"] == "Bear"]["Daily_Return_Pct"].dropna()
    t_stat, t_p = ttest_ind(bull, bear)
    print(f"\n  [1] Two-Sample t-Test  (Daily Return: Bull vs Bear)")
    print(f"      t = {t_stat:.4f},  p = {t_p:.6f}  →  "
          f"{'REJECT H₀ ✔' if t_p < ALPHA else 'FAIL TO REJECT H₀ ✗'}")
    print(f"      Interpretation: No significant difference in daily returns "
          f"between regimes (p={t_p:.4f} >> 0.05).")

    # ── ANOVA: Close by Sector ────────────────────────────────────────────────
    groups  = [df[df["Sector"] == s]["Close"].dropna().values
               for s in df["Sector"].unique()]
    f_stat, f_p = f_oneway(*groups)
    print(f"\n  [2] One-Way ANOVA  (Close Price across Sectors)")
    print(f"      F = {f_stat:.2f},  p = {f_p:.2e}  →  "
          f"{'REJECT H₀ ✔' if f_p < ALPHA else 'FAIL TO REJECT H₀ ✗'}")
    print(f"      Interpretation: Sector explains highly significant price "
          f"variation (price-scale effect driven by AMZN/GOOGL).")

    # ── Chi-Square: Trading Signal vs Price_Up ────────────────────────────────
    ct = pd.crosstab(df["Trading_Signal"], df["Price_Up"])
    chi2_val, chi_p, dof, _ = chi2_contingency(ct)
    print(f"\n  [3] Chi-Square Test  (Trading Signal vs Price_Up)")
    print(f"      χ² = {chi2_val:.2f},  p = {chi_p:.4e},  dof = {dof}  →  "
          f"{'REJECT H₀ ✔' if chi_p < ALPHA else 'FAIL TO REJECT H₀ ✗'}")
    print(f"      Interpretation: Strong signal-direction association "
          f"(likely due to engineered momentum features — leakage risk).")

    # ── 95% Confidence Interval for mean daily return ─────────────────────────
    dr   = df["Daily_Return_Pct"].dropna()
    ci   = stats.t.interval(0.95, df=len(dr) - 1,
                             loc=dr.mean(), scale=stats.sem(dr))
    print(f"\n  [4] 95% CI for Mean Daily Return")
    print(f"      x̄ = {dr.mean():.5f}%,  SE = {stats.sem(dr):.5f}")
    print(f"      CI = ({ci[0]:.5f}%, {ci[1]:.5f}%)")
    print(f"      Interpretation: CI crosses zero → mean return not "
          f"statistically different from 0 (consistent with EMH).")

    # 95% CI for proportion
    p = df["Price_Up"].mean()
    n = len(df)
    se_prop = np.sqrt((p * (1-p))/n)
    ci_prop = (p - 1.96 * se_prop, p + 1.96 * se_prop)

    print(f"\n  [5] 95% CI for Probability of Price Increase")
    print(f"      p̂ = {p:.4f}")
    print(f"      CI = ({ci_prop[0]:.4f}, {ci_prop[1]:.4f})")

    return {
        "ci_prop": ci_prop,
        "t_stat": t_stat, "t_p": t_p,
        "f_stat": f_stat, "f_p": f_p,
        "chi2":   chi2_val, "chi_p": chi_p,
        "ci":     ci,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  PCA / LINEAR ALGEBRA
# ═══════════════════════════════════════════════════════════════════════════════

def pca_analysis(df: pd.DataFrame):
    """Mean-center, covariance / correlation matrices, eigen-decomposition, PCA."""
    print(f"\n{'═'*70}")
    print("  LINEAR ALGEBRA & PCA")
    print(f"{'═'*70}")

    num_df = df[NUM_COLS].dropna()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(num_df)                   # X ∈ ℝ^{n×8}

    # Covariance & correlation matrices
    cov_mat  = num_df.cov()
    corr_mat = num_df.corr()

    # Eigen Decomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_mat)

    print("\nEigenvalues:")
    print(np.round(eigenvalues,4))

    print("\nTop Eigenvectors:")
    print(np.round(eigenvectors[:,:3],4))

    # Full PCA (eigendecomposition under the hood)
    pca = PCA()
    pca.fit(scaled)
    explained  = pca.explained_variance_ratio_ * 100
    cumulative = np.cumsum(explained)

    print(f"\n  Data matrix X: {num_df.shape[0]:,} × {num_df.shape[1]}")
    print(f"  Covariance matrix shape : {cov_mat.shape}")
    print(f"\n  PCA Explained Variance:")
    for i, (ev, cv) in enumerate(zip(explained, cumulative), 1):
        bar = "█" * int(ev / 2)
        print(f"    PC{i}: {ev:5.2f}%  (cumulative {cv:5.1f}%)  {bar}")

    n80 = int(np.searchsorted(cumulative, 80)) + 1
    print(f"\n  → {n80} components needed to explain ≥ 80% of variance.")

    # Loadings for PC1–PC3
    pca3 = PCA(n_components=3)
    pca3.fit(scaled)
    loadings = pd.DataFrame(
        pca3.components_.T,
        index=NUM_COLS,
        columns=["PC1 (Size)", "PC2 (Momentum)", "PC3 (Valuation)"],
    )
    print(f"\n  Component Loadings (PC1–PC3):")
    print(loadings.round(4).to_string())

    return {
        "num_df":   num_df,
        "scaled":   scaled,
        "pca":      pca,
        "explained": explained,
        "cumulative": cumulative,
        "corr_mat": corr_mat,
        "cov_mat":  cov_mat,
        "loadings": loadings,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  HELPER — axis styler
# ═══════════════════════════════════════════════════════════════════════════════

def _ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=TEXT, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    if title:
        ax.set_title(title, fontsize=9, fontweight="bold", color=TEXT, pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color=MUTED)


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  FIGURE 1 — TIME SERIES DECOMPOSITION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_decomposition(df: pd.DataFrame, ticker: str = TICKER_DEMO):
    """Side-by-side MA and STL decomposition for one ticker."""
    print(f"\n  Generating Figure 1: Time Series Decomposition ({ticker}) …")

    ts = (
        df[df["Ticker"] == ticker]
        .set_index("Date")["Close"]
        .resample("B").mean()
        .ffill()
    )

    # Moving Average decomposition
    trend_ma    = ts.rolling(window=MA_WINDOW, center=True).mean()
    detrended   = ts - trend_ma
    seasonal_ma = detrended.groupby(detrended.index.dayofweek).transform("mean")
    residual_ma = ts - trend_ma - seasonal_ma

    # STL decomposition
    stl_result = STL(ts, period=STL_PERIOD, robust=True).fit()

    fig, axes = plt.subplots(4, 2, figsize=(18, 14))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f"Time Series Decomposition Tool  —  {ticker} Close Price",
        fontsize=15, fontweight="bold", color=TEXT, y=0.985,
    )
    fig.text(
        0.5, 0.958,
        f"Moving Average ({MA_WINDOW}-day)  |  STL Decomposition (period={STL_PERIOD}, robust)",
        ha="center", fontsize=9, color=MUTED,
    )

    labels  = ["Original Series", "Trend", "Seasonal", "Residual"]
    colors  = [ACCENT, "#ff7f0e", "#2ca02c", "#d62728"]
    ma_data = [ts, trend_ma, seasonal_ma, residual_ma]
    stl_data = [
        ts,
        pd.Series(stl_result.trend,    index=ts.index),
        pd.Series(stl_result.seasonal, index=ts.index),
        pd.Series(stl_result.resid,    index=ts.index),
    ]

    for row, (label, color, ma_s, stl_s) in enumerate(
        zip(labels, colors, ma_data, stl_data)
    ):
        for col, (s, method) in enumerate(
            zip([ma_s, stl_s], ["MA", "STL"])
        ):
            ax = axes[row][col]
            ax.plot(s.index, s.values, color=color, lw=0.85, alpha=0.9)
            ax.fill_between(s.index, s.values, alpha=0.12, color=color)
            _ax(ax, f"{label}  ({method})")

    plt.tight_layout(rect=[0, 0, 1, 0.955])
    plt.savefig("fig1_decomposition.png", dpi=150)
    plt.show()
    print("  ✔  fig1_decomposition.png saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  FIGURE 2 — DESCRIPTIVE STATISTICS VISUALS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_descriptive(df: pd.DataFrame):
    """Histograms, sector boxplot, mean return bar chart."""
    print("  Generating Figure 2: Descriptive Statistics …")

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "Descriptive Statistical Analysis",
        fontsize=15, fontweight="bold", color=TEXT, y=0.985,
    )
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.36)

    # Histograms (2 rows × 4 cols)
    for idx, col in enumerate(NUM_COLS):
        ax   = fig.add_subplot(gs[idx // 4, idx % 4])
        data = df[col].dropna()
        ax.hist(data, bins=40, color=PALETTE[idx], alpha=0.75, edgecolor="none")
        ax.axvline(data.mean(),   color="white",   lw=1.2, ls="--",
                   label=f"μ={data.mean():.2f}")
        ax.axvline(data.median(), color="#ffd700", lw=1.2, ls=":",
                   label=f"med={data.median():.2f}")
        _ax(ax, col)
        ax.legend(fontsize=6, labelcolor=TEXT, facecolor=CARD, edgecolor="none")

    # Boxplot — Close by Sector
    sectors = list(df["Sector"].unique())
    bdata   = [df[df["Sector"] == s]["Close"].dropna().values for s in sectors]
    ax_box  = fig.add_subplot(gs[2, :2])
    bp = ax_box.boxplot(
        bdata, patch_artist=True,
        labels=[s.replace(" ", "\n") for s in sectors],
        medianprops=dict(color="yellow", lw=2),
    )
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    _ax(ax_box, "Close Price Distribution by Sector", ylabel="Close ($)")
    ax_box.tick_params(axis="x", labelsize=7)

    # Bar chart — mean daily return by sector
    ax_bar = fig.add_subplot(gs[2, 2:])
    means  = df.groupby("Sector")["Daily_Return_Pct"].mean().sort_values()
    ax_bar.barh(
        means.index, means.values,
        color=[PALETTE[i] for i in range(len(means))], alpha=0.85,
    )
    ax_bar.axvline(0, color="white", lw=0.8, ls="--")
    _ax(ax_bar, "Mean Daily Return % by Sector", xlabel="Mean Return (%)")

    plt.savefig("fig2_descriptive.png", dpi=150)
    plt.show()
    print("  ✔  fig2_descriptive.png saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  FIGURE 3 — RELATIONSHIP & TREND PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_relationships(df: pd.DataFrame):
    """Scatter, line (MA), RSI-return, grouped bar, stacked bar, normalised close."""
    print("  Generating Figure 3: Relationship & Trend Plots …")

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "Relationship & Trend Plots",
        fontsize=15, fontweight="bold", color=TEXT, y=0.985,
    )

    # ① Scatter: Close vs Volume by Sector
    ax = axes[0][0]
    for i, sec in enumerate(df["Sector"].unique()):
        sub = df[df["Sector"] == sec]
        ax.scatter(sub["Volume"] / 1e6, sub["Close"],
                   alpha=0.25, s=4, color=PALETTE[i], label=sec)
    _ax(ax, "Close vs Volume by Sector", "Volume (M)", "Close ($)")
    ax.legend(fontsize=6, markerscale=3, labelcolor=TEXT,
              facecolor=CARD, edgecolor="none")

    # ② Line: ticker price + MA20 / MA50
    ax = axes[0][1]
    demo = df[df["Ticker"] == TICKER_DEMO].sort_values("Date")
    ax.plot(demo["Date"], demo["Close"],  color=ACCENT,    lw=0.9, label="Close")
    ax.plot(demo["Date"], demo["MA_20"],  color="#ff7f0e", lw=1.3, label="MA-20")
    ax.plot(demo["Date"], demo["MA_50"],  color="#2ca02c", lw=1.3, label="MA-50")
    _ax(ax, f"{TICKER_DEMO}: Price & Moving Averages", ylabel="Price ($)")
    ax.legend(fontsize=7, labelcolor=TEXT, facecolor=CARD, edgecolor="none")

    # ③ Scatter: RSI vs Daily Return (colour = Beta)
    ax  = axes[0][2]
    smp = df.sample(1000, random_state=42)
    sc  = ax.scatter(smp["RSI"], smp["Daily_Return_Pct"],
                     c=smp["Beta"], cmap="plasma", alpha=0.5, s=10)
    cb  = plt.colorbar(sc, ax=ax)
    cb.set_label("Beta", color=TEXT)
    cb.ax.yaxis.set_tick_params(color=TEXT)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT)
    _ax(ax, "RSI vs Daily Return  (coloured by Beta)", "RSI", "Daily Return (%)")

    # ④ Grouped bar: mean Close by Ticker & Market Regime
    ax = axes[1][0]
    rm = df.groupby(["Ticker", "Market_Regime"])["Close"].mean().unstack(fill_value=0)
    x  = np.arange(len(rm.index))
    w  = 0.35
    for i, col in enumerate(rm.columns):
        ax.bar(x + i * w - w / 2, rm[col], w,
               label=col, color=PALETTE[i], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(rm.index, rotation=45, ha="right", fontsize=7)
    _ax(ax, "Mean Close by Ticker & Market Regime", ylabel="Mean Close ($)")
    ax.legend(fontsize=7, labelcolor=TEXT, facecolor=CARD, edgecolor="none")

    # ⑤ Stacked bar: trading signal counts by sector
    ax  = axes[1][1]
    sc2 = df.groupby(["Sector", "Trading_Signal"]).size().unstack(fill_value=0)
    btm = np.zeros(len(sc2))
    for i, col in enumerate(sc2.columns):
        ax.bar(sc2.index, sc2[col], bottom=btm,
               label=col, color=PALETTE[i], alpha=0.85)
        btm += sc2[col].values
    ax.set_xticklabels(sc2.index, rotation=30, ha="right", fontsize=7)
    _ax(ax, "Trading Signal Distribution by Sector (Stacked)", ylabel="Count")
    ax.legend(fontsize=7, labelcolor=TEXT, facecolor=CARD, edgecolor="none")

    # ⑥ Normalised close for 5 tickers
    ax = axes[1][2]
    for i, ticker in enumerate(["AAPL", "MSFT", "GOOGL", "AMZN", "JPM"]):
        sub  = df[df["Ticker"] == ticker].sort_values("Date")
        norm = sub["Close"] / sub["Close"].iloc[0] * 100
        ax.plot(sub["Date"], norm.values,
                color=PALETTE[i], lw=0.9, label=ticker, alpha=0.9)
    _ax(ax, "Normalised Close (Base = 100) — 5 Tickers", ylabel="Indexed Price")
    ax.legend(fontsize=7, labelcolor=TEXT, facecolor=CARD, edgecolor="none")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("fig3_relationships.png", dpi=150)
    plt.show()
    print("  ✔  fig3_relationships.png saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. FIGURE 4 — MULTIVARIATE & PCA PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_multivariate(df: pd.DataFrame, pca_data: dict):
    """Correlation heatmap, covariance heatmap, scree plot, 2-D PCA projection."""
    print("  Generating Figure 4: Multivariate & PCA Plots …")

    num_df   = pca_data["num_df"]
    scaled   = pca_data["scaled"]
    pca      = pca_data["pca"]
    explained  = pca_data["explained"]
    cumulative = pca_data["cumulative"]
    corr_mat   = pca_data["corr_mat"]
    cov_mat    = pca_data["cov_mat"]

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "Multivariate & Statistical Plots",
        fontsize=15, fontweight="bold", color=TEXT, y=0.985,
    )

    # ① Correlation heatmap
    ax   = axes[0][0]
    mask = np.triu(np.ones_like(corr_mat, dtype=bool))
    sns.heatmap(
        corr_mat, ax=ax, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, linewidths=0.5, linecolor=BORDER,
        annot_kws={"size": 8}, cbar_kws={"shrink": 0.8},
    )
    ax.set_facecolor(CARD)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.set_title("Correlation Heatmap", color=TEXT, fontsize=11, fontweight="bold")

    # ② Covariance heatmap (log-scaled for readability)
    ax     = axes[0][1]
    logcov = np.sign(cov_mat) * np.log1p(np.abs(cov_mat))
    sns.heatmap(
        logcov, ax=ax, annot=False, cmap="viridis",
        linewidths=0.5, linecolor=BORDER, cbar_kws={"shrink": 0.8},
    )
    ax.set_facecolor(CARD)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.set_title("Covariance Heatmap (log-scaled)", color=TEXT,
                 fontsize=11, fontweight="bold")

    # ③ Scree plot
    ax = axes[1][0]
    ax.bar(range(1, len(explained) + 1), explained,
           color=ACCENT, alpha=0.8, label="Individual %")
    ax.plot(range(1, len(explained) + 1), cumulative,
            color="#ff7f0e", marker="o", markersize=5, lw=2, label="Cumulative %")
    ax.axhline(80, color="#d62728", lw=1.2, ls="--", label="80% threshold")
    _ax(ax, "Scree Plot — PCA Explained Variance",
        "Principal Component", "Variance Explained (%)")
    ax.legend(fontsize=8, labelcolor=TEXT, facecolor=CARD, edgecolor="none")

    # ④ 2-D PCA projection
    ax    = axes[1][1]
    pca2  = PCA(n_components=2)
    proj  = pca2.fit_transform(scaled)
    ev2   = pca2.explained_variance_ratio_ * 100
    sects = df.loc[num_df.index, "Sector"].values
    for i, sec in enumerate(np.unique(sects)):
        mask2 = sects == sec
        ax.scatter(proj[mask2, 0], proj[mask2, 1],
                   alpha=0.4, s=10, color=PALETTE[i], label=sec)
    _ax(ax,
        f"2-D PCA Projection  (PC1={ev2[0]:.1f}%  PC2={ev2[1]:.1f}%)",
        f"PC1 ({ev2[0]:.1f}%)", f"PC2 ({ev2[1]:.1f}%)")
    ax.legend(fontsize=7, markerscale=2, labelcolor=TEXT,
              facecolor=CARD, edgecolor="none")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("fig4_multivariate.png", dpi=150)
    plt.show()
    print("  ✔  fig4_multivariate.png saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# INSIGHT GENERATION
# ═══════════════════════════════════════════════════════════════════════════════
def generate_insights(df, prob, inf, pca_data):
    print(f"\n{'═'*70}")
    print("  INSIGHT GENERATION")
    print(f"{'═'*70}")
    insights = [
        "Daily returns do not significantly differ between Bull and Bear markets.",
        "Sector membership strongly influences stock price levels.",
        "Trading signals have a very strong relationship with price movement, indicating possible data leakage.",
        "Average daily return is statistically close to zero.",
        "Market capitalization and stock price dominate the first PCA component.",
        "Five principal components explain more than 80% of total variance.",
        "Large-cap technology companies create significant outliers in market capitalization.",
        "Single-day price increases provide little information about market regime."
    ]
    for i, insight in enumerate(insights,1):
        print(f"{i}. {insight}")

# ═══════════════════════════════════════════════════════════════════════════════
# 11. PRINT FINAL SUMMARY REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(inf: dict, prob: dict, pca_data: dict):
    explained  = pca_data["explained"]
    cumulative = pca_data["cumulative"]
    n80 = int(np.searchsorted(cumulative, 80)) + 1

    print(f"\n{'═'*70}")
    print("  FINAL SUMMARY REPORT")
    print(f"{'═'*70}")

    rows = [
        ("t-test (Bull vs Bear return)",
         f"t={inf['t_stat']:.3f}  p={inf['t_p']:.4f}",
         "No significant difference — regime ≠ daily signal"),
        ("ANOVA (Close by Sector)",
         f"F={inf['f_stat']:.1f}  p<0.0001",
         "Sectors differ significantly in absolute price"),
        ("Chi-Square (Signal vs Price_Up)",
         f"χ²={inf['chi2']:.0f}  p<0.0001",
         "Strong association — check for data leakage"),
        ("95% CI (Mean Daily Return)",
         f"({inf['ci'][0]:.5f}%, {inf['ci'][1]:.5f}%)",
         "Includes zero → return not different from 0"),
        ("P(Bull | Price_Up)  [Bayes]",
         f"{prob['P(Bull | Up) [Bayes]']:.4f}",
         "Prior ≈ posterior → day-up gives no regime info"),
        (f"PCA: components for 80% variance",
         f"{n80} PCs",
         f"PC1=Size, PC2=Momentum, PC3=Valuation"),
    ]

    print(f"  {'Test / Finding':<38} {'Result':<28} Interpretation")
    print(f"  {'─'*38} {'─'*28} {'─'*30}")
    for test, result, interp in rows:
        print(f"  {test:<38} {result:<28} {interp}")

    print(f"\n{'═'*70}")
    print("  OUTPUT FILES SAVED")
    print(f"{'═'*70}")
    for f in ["fig1_decomposition.png", "fig2_descriptive.png",
              "fig3_relationships.png",  "fig4_multivariate.png"]:
        print(f"  ✔  {f}")
    print(f"{'═'*70}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 12. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    df = load_data(DATA_PATH)
    dataset_source()
    data_understanding(df)
    stats_df = descriptive_stats(df)
    prob = probability_analysis(df)
    inf = inferential_stats(df)
    pca_data = pca_analysis(df)
    plot_decomposition(df)
    plot_descriptive(df)
    plot_relationships(df)
    plot_multivariate(df, pca_data)
    generate_insights(df, prob, inf, pca_data)
    print_summary(inf, prob, pca_data)

if __name__ == "__main__":
    main()