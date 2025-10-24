# --- Étape 1 : Setup global + chargement des données ---

import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
from pathlib import Path
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.stattools import acf
import plotly.graph_objects as go

MODEL_DIR = Path(__file__).resolve().parents[1] / "outputs" / "models"
DATA_DIR = Path(__file__).resolve().parents[1] / "data_clean"

MODEL_FILE_MAP = {
    "ols_base": "ols.pkl",
    "ols_opt": "ols_opt.pkl",
    "its_base": "its.pkl",
    "its_opt": "its_v2.pkl",
    "sarimax_best": "sarimax_best.pkl",
}

MODEL_AUX_FILES = {
    "its_design": "its_long.pkl",
}


def load_saved_result(path: Path):
    """Charge un objet statsmodels picklé, gestion des erreurs."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            return sm.load(str(path))
    except Exception as exc:
        st.warning(f"Impossible de charger le modèle sauvegardé `{path.name}` ({exc}). Recalcul en cours.")
        return None


def load_saved_frame(path: Path):
    """Charge un DataFrame picklé."""
    try:
        return pd.read_pickle(path)
    except Exception as exc:
        st.warning(f"Impossible de charger les données auxiliaires `{path.name}` ({exc}).")
        return None


def keyify(df: pd.DataFrame, date_col: str = "date_monday") -> pd.DataFrame:
    """S'assure que les colonnes ISO (année, semaine) existent."""
    df = df.copy()
    if date_col not in df.columns:
        return df
    iso = pd.to_datetime(df[date_col]).dt.isocalendar()
    if "year_iso" not in df.columns:
        df["year_iso"] = iso["year"].astype(int)
    if "week_iso_num" not in df.columns:
        df["week_iso_num"] = iso["week"].astype(int)
    return df


def zscore(series: pd.Series) -> pd.Series:
    """Calcule un z-score stable (std=0 -> 0)."""
    std = series.std(ddof=0)
    if std is None or np.isclose(std, 0):
        return pd.Series(0, index=series.index)
    return (series - series.mean()) / std


def build_time_features(df: pd.DataFrame, period: int = 52) -> pd.DataFrame:
    """Ajoute les composantes temporelles t, sin(t), cos(t)."""
    df = df.copy()
    df["t"] = np.arange(len(df))
    df["sin52"] = np.sin(2 * np.pi * df["t"] / period)
    df["cos52"] = np.cos(2 * np.pi * df["t"] / period)
    return df


def compute_mnp_components(df: pd.DataFrame, mask_vars: list[str]) -> pd.DataFrame:
    """Construit les composantes MNP (work inversé + gestes barrières)."""
    df = df.copy()
    features = pd.DataFrame(index=df.index)
    features["work_red"] = -df["work"]
    features["work_red_z"] = zscore(features["work_red"])
    for var in mask_vars:
        features[f"{var}_z"] = zscore(df[var])
    z_cols = [f"{var}_z" for var in mask_vars] + ["work_red_z"]
    features["MNP_score"] = features[z_cols].mean(axis=1)
    return features


def create_lagged_features(df: pd.DataFrame, lags: tuple[int, int, int]) -> pd.DataFrame:
    """Crée les colonnes décalées (vaccin, MNP, travail) + harmonique saisonnière."""
    lag_vac, lag_mnp, lag_work = lags
    features = pd.DataFrame(index=df.index)
    features["cov12_lag"] = df["couv_complet"].shift(lag_vac)
    features["MNP_lag"] = df["MNP_score"].shift(lag_mnp)
    features["work_lag"] = df["work"].shift(lag_work)
    features = build_time_features(features)
    return features


def search_best_lags(
    rsv_series: pd.Series,
    base_df: pd.DataFrame,
    lag_ranges: tuple[range, range, range],
) -> tuple[tuple[int, int, int], float]:
    """Recherche brute des meilleurs lags via R² ajusté."""
    best_lags = (4, 8, 9)
    best_r2 = -np.inf
    for lv in lag_ranges[0]:
        for lm in lag_ranges[1]:
            for lw in lag_ranges[2]:
                try:
                    X_tmp = create_lagged_features(base_df, (lv, lm, lw))
                    tmp = rsv_series.to_frame("RSV").join(X_tmp, how="left").dropna()
                    if len(tmp) < 30:
                        continue
                    X = tmp[["cov12_lag", "MNP_lag", "work_lag", "sin52", "cos52"]]
                    mod = sm.OLS(tmp["RSV"], sm.add_constant(X, has_constant="add")).fit()
                    if mod.rsquared_adj > best_r2:
                        best_r2 = mod.rsquared_adj
                        best_lags = (lv, lm, lw)
                except Exception:
                    continue
    return best_lags, best_r2


def add_fourier_terms(df: pd.DataFrame, period: int = 52, K: int = 1) -> pd.DataFrame:
    """Ajoute des harmoniques sin/cos supplémentaires (ITS)."""
    df = df.copy()
    t = np.arange(len(df))
    for k in range(1, K + 1):
        df[f"sin{k}"] = np.sin(2 * np.pi * k * t / period)
        df[f"cos{k}"] = np.cos(2 * np.pi * k * t / period)
    return df


def make_its_design(
    df: pd.DataFrame,
    covid_date: pd.Timestamp,
    vacc_date: pd.Timestamp,
    fourier_K: int = 1,
    add_exog: bool = True,
) -> tuple[sm.regression.linear_model.RegressionResultsWrapper, pd.DataFrame, list[str], int]:
    """Construit et ajuste le modèle ITS avec HAC."""
    df_design = df.reset_index().rename(columns={"date_monday": "date"}).sort_values("date")
    df_design["date"] = pd.to_datetime(df_design["date"])
    df_design["t"] = np.arange(len(df_design))
    df_design["post_covid"] = (df_design["date"] >= covid_date).astype(int)
    df_design["post_vacc"] = (df_design["date"] >= vacc_date).astype(int)
    df_design["t_post_covid"] = df_design["t"] * df_design["post_covid"]
    df_design["t_post_vacc"] = df_design["t"] * df_design["post_vacc"]
    df_design = add_fourier_terms(df_design, K=fourier_K)

    y = df_design["RSV"].astype(float)
    base_cols = ["t", "post_covid", "t_post_covid", "post_vacc", "t_post_vacc"]
    fourier_cols = []
    for k in range(1, fourier_K + 1):
        fourier_cols.extend([f"sin{k}", f"cos{k}"])
    Xcols = base_cols + fourier_cols
    if add_exog:
        for col in ["cov12_lag", "MNP_lag", "work_lag"]:
            if col in df_design.columns and col not in Xcols:
                Xcols.append(col)

    X = df_design[Xcols].copy()
    hac_lags = int(np.clip(np.sqrt(len(df_design)), 8, 24))
    fit = sm.OLS(y, sm.add_constant(X, has_constant="add"), missing="drop").fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lags}
    )
    df_design["date_monday"] = df_design["date"]
    df_design = df_design.set_index("date_monday")
    return fit, df_design, Xcols, hac_lags


def fit_its_model(
    df_base: pd.DataFrame, covid_start: pd.Timestamp, vacc_start: pd.Timestamp
) -> dict:
    """Lance la grille de recherche ITS et renvoie le meilleur modèle."""
    steps = [pd.to_timedelta(0, unit="D")]
    best = {"aic": np.inf}
    for K in [2]:
        for delta_c in steps:
            for delta_v in steps:
                covid_date = covid_start + delta_c
                vacc_date = vacc_start + delta_v
                if vacc_date <= covid_date:
                    continue
                try:
                    fit, design, cols, hac = make_its_design(
                        df_base[["RSV", "cov12_lag", "MNP_lag", "work_lag"]],
                        covid_date=covid_date,
                        vacc_date=vacc_date,
                        fourier_K=K,
                        add_exog=True,
                    )
                except Exception:
                    continue
                if fit.aic < best["aic"]:
                    best = {
                        "aic": fit.aic,
                        "model": fit,
                        "design": design,
                        "cols": cols,
                        "K": K,
                        "covid_date": covid_date,
                        "vacc_date": vacc_date,
                        "hac_lags": hac,
                    }
    return best


def fit_sarimax_model(
    df_opt: pd.DataFrame, covid_start: pd.Timestamp, vacc_start: pd.Timestamp
) -> dict:
    """Recherche du meilleur SARIMAX (ordre restreint) avec exogènes."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    df_sx = df_opt.copy().sort_index()
    df_sx.index = pd.to_datetime(df_sx.index)
    try:
        df_sx = df_sx.asfreq("W-MON")
    except ValueError:
        df_sx = df_sx.resample("W-MON").interpolate(method="linear")
    df_sx["post_covid"] = (df_sx.index >= covid_start).astype(int)
    df_sx["post_vacc"] = (df_sx.index >= vacc_start).astype(int)
    df_sx["t"] = np.arange(len(df_sx))
    df_sx["t_post_covid"] = df_sx["t"] * df_sx["post_covid"]

    exog_cols = [
        "cov12_lag",
        "MNP_lag",
        "work_lag",
        "tmean_z",
        "vacc_x_mnp",
        "post_covid",
        "post_vacc",
        "t_post_covid",
        "t",
    ]
    exog = df_sx[exog_cols].astype(float).replace([np.inf, -np.inf], np.nan)
    y = df_sx["RSV"].astype(float)
    mask = (~y.isna()) & (~exog.isna().any(axis=1))
    y = y.loc[mask]
    exog = exog.loc[mask]

    candidate_pdq = [(1, 0, 0), (1, 0, 1), (2, 0, 0)]
    candidate_PDQ = [(1, 0, 1, 52), (1, 1, 0, 52)]
    best = {"aic": np.inf}
    for order in candidate_pdq:
        for seasonal_order in candidate_PDQ:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    mod = SARIMAX(
                        y,
                        exog=exog,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit(disp=False, maxiter=200)
                if not getattr(mod, "mle_retvals", {}).get("converged", True):
                    mod = mod.model.fit(disp=False, method="powell", maxiter=150)
            except Exception:
                continue
            if mod.aic < best["aic"]:
                best = {
                    "aic": mod.aic,
                    "model": mod,
                    "order": order,
                    "seasonal_order": seasonal_order,
                }

    if best.get("model") is None:
        return best

    y_fit = best["model"].fittedvalues.reindex(y.index)
    ss_res = float(((y - y_fit) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    pseudo_r2 = 1 - ss_res / ss_tot if ss_tot else np.nan
    best.update(
        {
            "frame": df_sx.loc[mask],
            "exog_cols": exog_cols,
            "pseudo_r2": pseudo_r2,
        }
    )
    return best

# Configuration de la page Streamlit
st.set_page_config(page_title="RSV Analysis Dashboard", layout="wide")
st.title("Analyse du Virus RSV - Tableau de Bord Interactif")

# Chargement et préparation des données (mise en cache)
@st.cache_resource(show_spinner=False)
def load_and_prepare_data():
    data_dir = DATA_DIR
    COVID_START = pd.Timestamp("2020-03-01")
    VACC_START = pd.Timestamp("2021-01-01")

    # === 1) Chargement des sources brutes ===
    common = keyify(pd.read_csv(data_dir / "ODISSEE/common_FR_long.csv", engine="python"))
    vacsi = pd.read_csv(data_dir / "VACSI/vacsi_fr_extended.csv", engine="python")
    mobility = pd.read_csv(data_dir / "GOOGLE/google_mobility_fr_weekly.csv", engine="python")
    coviprev = pd.read_csv(data_dir / "COVIPREV/coviprev_reg_weekly.csv", engine="python")
    meteo = pd.read_csv(data_dir / "METEO/meteo_fr_weekly.csv", engine="python")

    # === 2) Signal RSV national ===
    mask = (common["topic"] == "RSV") & (common["geo_level"] == "FR")
    age_preference = ["00-04 ans", "0-1 an", "Tous âges"]
    age_used = next(
        (age for age in age_preference if ((mask) & (common["classe_d_age"] == age)).any()),
        "Tous âges",
    )
    mask &= common["classe_d_age"] == age_used
    y_col = "taux_passages_urgences" if "taux_passages_urgences" in common.columns else "taux_sos"
    rsv = (
        common.loc[mask, ["date_monday", "year_iso", "week_iso_num", y_col]]
        .rename(columns={y_col: "RSV"})
        .copy()
    )
    rsv["date_monday"] = pd.to_datetime(rsv["date_monday"])
    rsv = rsv.sort_values("date_monday")
    rsv_series = rsv.set_index("date_monday")[["RSV"]]

    # === 3) Vaccination ===
    vacsi["date_monday"] = pd.to_datetime(vacsi.get("date", vacsi.get("date_monday")))
    vacsi = keyify(vacsi)
    vac = vacsi.query("geo_level=='FR' & geo_code=='FR'")[["year_iso", "week_iso_num", "couv_complet"]]

    # === 4) Mobilité Google ===
    mobility["date_monday"] = pd.to_datetime(mobility.get("date", mobility.get("date_monday")))
    mobility = keyify(mobility)
    mobility_fr = mobility.query("geo_level=='FR' & geo_code=='FR'")
    mob_wide = (
        mobility_fr.pivot_table(
            index=["year_iso", "week_iso_num"],
            columns="indicator",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    mobility_cols = [
        "workplaces",
        "residential",
        "retail_and_recreation",
        "grocery_and_pharmacy",
        "parks",
        "transit_stations",
    ]
    for col in mobility_cols:
        if col not in mob_wide.columns:
            mob_wide[col] = 0
    mob_wide["work"] = mob_wide["workplaces"]

    # === 5) CoviPrev (gestes barrières) ===
    mask_vars = [
        "port_du_masque",
        "lavage_des_mains",
        "aeration_du_logement",
        "saluer_sans_serrer_la_main",
    ]
    coviprev["date_monday"] = pd.to_datetime(coviprev.get("date", coviprev.get("date_monday")))
    coviprev = keyify(coviprev)
    cov_nat = (
        coviprev[coviprev["indicator"].isin(mask_vars)]
        .groupby(["year_iso", "week_iso_num", "indicator"])["value"]
        .mean()
        .unstack()
        .reset_index()
    )
    for var in mask_vars:
        if var not in cov_nat.columns:
            cov_nat[var] = 0

    # === 6) Météo ===
    meteo["date_monday"] = pd.to_datetime(meteo.get("date", meteo.get("date_monday")))
    meteo = keyify(meteo)
    meteo_df = meteo[["year_iso", "week_iso_num", "tmean"]]

    # === 7) Fusion multi-sources ===
    df_merge = (
        rsv.merge(vac, on=["year_iso", "week_iso_num"], how="left")
        .merge(mob_wide, on=["year_iso", "week_iso_num"], how="left")
        .merge(cov_nat, on=["year_iso", "week_iso_num"], how="left")
        .merge(meteo_df, on=["year_iso", "week_iso_num"], how="left")
        .sort_values("date_monday")
    )
    df_merge["couv_complet"] = df_merge["couv_complet"].fillna(0)
    for col in mobility_cols + ["work"]:
        df_merge[col] = df_merge.get(col, 0).fillna(0)
    for var in mask_vars:
        df_merge[var] = df_merge.get(var, 0).fillna(0)
    df_merge["tmean"] = df_merge["tmean"].ffill().bfill()
    df_merge["date_monday"] = pd.to_datetime(df_merge["date_monday"])
    df_merge = df_merge.set_index("date_monday")

    # === 8) Score MNP ===
    mnp_components = compute_mnp_components(df_merge[["work"] + mask_vars], mask_vars)
    df_merge = df_merge.join(mnp_components, how="left")

    # Base pour la recherche de lags
    lag_input = df_merge[["couv_complet", "MNP_score", "work"]].copy()

    # === 9) Base (lags par défaut) ===
    default_lags = (4, 8, 9)
    X_base_default = create_lagged_features(lag_input, default_lags)
    df_base = (
        rsv_series.join(X_base_default, how="left")
        .join(df_merge.drop(columns=["RSV"]), how="left")
        .dropna(subset=["cov12_lag", "MNP_lag", "work_lag"])
        .sort_index()
    )
    df_base["t"] = np.arange(len(df_base))
    df_base["sin52"] = np.sin(2 * np.pi * df_base["t"] / 52)
    df_base["cos52"] = np.cos(2 * np.pi * df_base["t"] / 52)
    df_base["post_covid"] = (df_base.index >= COVID_START).astype(int)
    df_base["post_vacc"] = (df_base.index >= VACC_START).astype(int)
    df_base["t_postcovid"] = df_base["t"] * df_base["post_covid"]
    df_base["t_postvacc"] = df_base["t"] * df_base["post_vacc"]

    # === 10) Recherche des meilleurs lags ===
    lag_search_ranges = (
        range(default_lags[0], default_lags[0] + 1),
        range(default_lags[1], default_lags[1] + 1),
        range(default_lags[2], default_lags[2] + 1),
    )
    best_lags, best_r2 = search_best_lags(
        rsv_series["RSV"],
        lag_input,
        lag_search_ranges,
    )
    X_best = create_lagged_features(lag_input, best_lags)
    df_opt = (
        rsv_series.join(X_best, how="left")
        .join(df_merge.drop(columns=["RSV"]), how="left")
        .dropna(subset=["cov12_lag", "MNP_lag", "work_lag"])
        .sort_index()
    )
    df_opt["t"] = np.arange(len(df_opt))
    df_opt["sin52"] = np.sin(2 * np.pi * df_opt["t"] / 52)
    df_opt["cos52"] = np.cos(2 * np.pi * df_opt["t"] / 52)
    df_opt["post_covid"] = (df_opt.index >= COVID_START).astype(int)
    df_opt["post_vacc"] = (df_opt.index >= VACC_START).astype(int)
    df_opt["t_postcovid"] = df_opt["t"] * df_opt["post_covid"]
    df_opt["t_postvacc"] = df_opt["t"] * df_opt["post_vacc"]
    df_opt["tmean_z"] = zscore(df_opt["tmean"])
    df_opt["vacc_x_mnp"] = df_opt["cov12_lag"] * df_opt["MNP_lag"]
    df_opt["RSV_lag1"] = df_opt["RSV"].shift(1)
    df_opt["RSV_lag2"] = df_opt["RSV"].shift(2)
    df_opt = df_opt.dropna(subset=["RSV_lag2"])
    df_opt["year_iso"] = df_opt.index.isocalendar().year
    df_opt["week_iso_num"] = df_opt.index.isocalendar().week

    # === 11) Ajustement des modèles ===
    base_features = ["cov12_lag", "MNP_lag", "work_lag", "sin52", "cos52"]
    ols_features = [
        "cov12_lag",
        "MNP_lag",
        "work_lag",
        "tmean_z",
        "vacc_x_mnp",
        "RSV_lag1",
        "RSV_lag2",
        "sin52",
        "cos52",
    ]
    ols_base = sm.OLS(
        df_base["RSV"],
        sm.add_constant(df_base[base_features], has_constant="add"),
        missing="drop",
    ).fit(cov_type="HC3")
    ols_opt = sm.OLS(
        df_opt["RSV"],
        sm.add_constant(df_opt[ols_features], has_constant="add"),
        missing="drop",
    ).fit(cov_type="HC3")

    models = {
        "ols_base": ols_base,
        "ols_opt": ols_opt,
        "its_best": None,
        "sarimax_best": None,
    }

    saved_models = {}
    saved_aux = {}
    if MODEL_DIR.exists():
        for key, filename in MODEL_FILE_MAP.items():
            path = MODEL_DIR / filename
            if path.exists():
                saved_models[key] = load_saved_result(path)
        for key, filename in MODEL_AUX_FILES.items():
            path = MODEL_DIR / filename
            if path.exists():
                saved_aux[key] = load_saved_frame(path)

    saved_models = {k: v for k, v in saved_models.items() if v is not None}
    saved_aux = {k: v for k, v in saved_aux.items() if v is not None}

    if saved_models.get("ols_base") is not None:
        models["ols_base"] = saved_models["ols_base"]
    if saved_models.get("ols_opt") is not None:
        models["ols_opt"] = saved_models["ols_opt"]
    if saved_models.get("its_opt") is not None:
        models["its_best"] = saved_models["its_opt"]
    elif saved_models.get("its_base") is not None:
        models["its_best"] = saved_models["its_base"]
    if saved_models.get("sarimax_best") is not None:
        models["sarimax_best"] = saved_models["sarimax_best"]

    meta = {
        "df_base": df_base,
        "df_opt": df_opt,
        "mask_vars": mask_vars,
        "ols_features": ols_features,
        "ols_base_features": base_features,
        "best_lags": best_lags,
        "best_lag_r2": best_r2,
        "covid_start": COVID_START,
        "vacc_start": VACC_START,
        "age_used": age_used,
        "tmean_stats": {
            "mean": float(df_opt["tmean"].mean()),
            "std": float(df_opt["tmean"].std(ddof=0)),
        },
        "sarimax_exog_cols": [
            "cov12_lag",
            "MNP_lag",
            "work_lag",
            "tmean_z",
            "vacc_x_mnp",
            "post_covid",
            "post_vacc",
            "t_post_covid",
            "t",
        ],
        "saved_models": saved_models,
        "saved_aux": saved_aux,
    }

    return df_opt, models, meta

# Chargement des données
with st.spinner("Chargement des données et initialisation des modèles..."):
    df_cached, models_bundle, meta_bundle = load_and_prepare_data()

df = df_cached.copy()
model = models_bundle["ols_opt"]
X_cols = meta_bundle["ols_features"]
df_base = meta_bundle["df_base"].copy()


def ensure_its_model(meta: dict):
    """Retourne le modèle ITS optimisé en le recalculant si nécessaire."""
    key = "its_cached_results"
    if key in st.session_state:
        return st.session_state[key]

    saved = meta.get("saved_models", {})
    aux = meta.get("saved_aux", {})
    saved_model = saved.get("its_opt") or saved.get("its_base")
    if saved_model is not None:
        design = aux.get("its_design")
        if design is None:
            design = meta["df_base"].copy()
        cols = [c for c in saved_model.model.exog_names if c != "const"]
        st.session_state[key] = {
            "model": saved_model,
            "design": design,
            "cols": cols,
            "hac_lags": None,
        }
        return st.session_state[key]

    st.session_state[key] = fit_its_model(
        meta["df_base"].copy(),
        meta["covid_start"],
        meta["vacc_start"],
    )
    return st.session_state[key]


def ensure_sarimax_model(meta: dict):
    """Retourne le modèle SARIMAX optimisé en le recalculant si nécessaire."""
    key = "sarimax_cached_results"
    if key in st.session_state:
        return st.session_state[key]

    saved = meta.get("saved_models", {})
    saved_model = saved.get("sarimax_best")
    if saved_model is not None:
        frame = meta["df_opt"].copy()
        exog_cols = meta.get("sarimax_exog_cols", [])
        y = frame["RSV"].astype(float)
        y_fit = saved_model.fittedvalues.reindex(frame.index)
        ss_res = float(((y - y_fit) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        pseudo_r2 = 1 - ss_res / ss_tot if ss_tot else np.nan
        st.session_state[key] = {
            "model": saved_model,
            "frame": frame,
            "exog_cols": exog_cols,
            "pseudo_r2": pseudo_r2,
            "order": getattr(getattr(saved_model, "model", None), "order", None),
            "seasonal_order": getattr(getattr(saved_model, "model", None), "seasonal_order", None),
        }
        return st.session_state[key]

    st.session_state[key] = fit_sarimax_model(
        meta["df_opt"].copy(),
        meta["covid_start"],
        meta["vacc_start"],
    )
    return st.session_state[key]


def clear_cached_models():
    """Permet de purger les modèles recalculés (utile pour le debug)."""
    for key in ["its_cached_results", "sarimax_cached_results"]:
        if key in st.session_state:
            del st.session_state[key]


if st.button("♻️ Recalculer les modèles ITS / SARIMAX"):
    clear_cached_models()
    st.success("Modèles invalidés — ils seront recalculés à la prochaine utilisation.")

rsv_series = df["RSV"]
fitted_series = model.fittedvalues

# Onglets
tab1,tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(["📊 Aperçu et exploration des données RSV", 
                                                               "Modélisation", "Scénarios", "Diagnostics", "Synthèse finale",
                                        "ERVISS RSV — Modélisation & Scénarios",
                                        "MNP — Détails", "Mobilité — Détails", 
                                        "Régions & Départements",
                                        ])
with tab1:
    st.header("📊 Aperçu et exploration des données RSV")
    st.markdown("""
Le virus respiratoire syncytial (**RSV**) provoque des épidémies hivernales récurrentes en France.  
Cette section présente une **vue d’ensemble interactive** des principales séries temporelles mobilisées dans l’étude :
- Évolution hebdomadaire du RSV
- Indicateurs de **mobilité**, **gestes barrières**, **température moyenne**
- Couverture **vaccinale COVID-19**

Les visualisations ci-dessous permettent de contextualiser les ruptures liées à la pandémie et de comparer les tendances.
""")

    # === 1️⃣ KPIs principaux ===
    col1, col2, col3, col4 = st.columns(4)
    kpi_mean = df["RSV"].mean()
    kpi_max = df["RSV"].max()
    kpi_peak_date = df["RSV"].idxmax().strftime("%Y-%m-%d")
    kpi_peak_year = df["RSV"].idxmax().year
    col1.metric("RSV moyen", f"{kpi_mean:.2f}")
    col2.metric("RSV maximum", f"{kpi_max:.2f}")
    col3.metric("Semaine du pic", kpi_peak_date)
    col4.metric("Année du pic", kpi_peak_year)

    # === 2️⃣ Graphique principal RSV (avec jalons COVID et Vaccin) ===
    COVID_START = pd.to_datetime("2020-03-01")
    VACC_START  = pd.to_datetime("2021-01-01")

    fig_rsv = go.Figure()
    fig_rsv.add_trace(go.Scatter(
        x=df.index, y=df["RSV"], mode="lines", name="RSV Observé",
        line=dict(color="firebrick", width=2)
    ))
    fig_rsv.add_vline(x=COVID_START, line_dash="dash", line_color="red")
    fig_rsv.add_vline(x=VACC_START,  line_dash="dash", line_color="green")
    fig_rsv.add_annotation(x=COVID_START, y=1.05 * df["RSV"].max(), text="COVID-19", showarrow=False, font=dict(color="red"))
    fig_rsv.add_annotation(x=VACC_START,  y=1.05 * df["RSV"].max(), text="Vaccination", showarrow=False, font=dict(color="green"))
    fig_rsv.update_layout(
        title="📈 Évolution du RSV en France (2014–2025)",
        xaxis_title="Date (lundi ISO)",
        yaxis_title="Taux RSV (pour 100k hab.)",
        template="plotly_white",
        height=500
    )
    st.plotly_chart(fig_rsv, use_container_width=True)

    # === 3️⃣ Corrélations simples ===
    st.subheader("🔗 Corrélations entre variables clés")
    vars_corr = ["RSV", "couv_complet", "MNP_score", "work", "tmean"]
    corr_df = df[vars_corr].corr().round(2)
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns,
        y=corr_df.index,
        colorscale="RdBu",
        zmin=-1, zmax=1,
        text=corr_df.values,
        texttemplate="%{text}",
        textfont={"size":12}
    ))
    fig_corr.update_layout(
        title="Matrice de corrélation (RSV vs variables explicatives)",
        height=400, template="plotly_white"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.caption("👉 On observe une corrélation positive entre la mobilité et le RSV, et négative avec les gestes barrières.")

    # === 4️⃣ Multi-séries temporelles synchronisées ===
    st.subheader("📉 Tendances temporelles comparées")
    fig_multi = go.Figure()
    fig_multi.add_trace(go.Scatter(x=df.index, y=df["RSV"]/df["RSV"].max(), name="RSV (normé)", line=dict(color="firebrick")))
    fig_multi.add_trace(go.Scatter(x=df.index, y=df["couv_complet"]/df["couv_complet"].max(), name="Vaccination", line=dict(color="green", dash="dot")))
    fig_multi.add_trace(go.Scatter(x=df.index, y=df["MNP_score"]/df["MNP_score"].max(), name="Gestes barrières", line=dict(color="royalblue", dash="dot")))
    fig_multi.add_trace(go.Scatter(x=df.index, y=(df["work_red_z"]-df["work_red_z"].min())/(df["work_red_z"].max()-df["work_red_z"].min()), 
                                   name="Mobilité inversée", line=dict(color="orange", dash="dot")))
    fig_multi.add_trace(go.Scatter(x=df.index, y=(df["tmean_z"]-df["tmean_z"].min())/(df["tmean_z"].max()-df["tmean_z"].min()), 
                                   name="Température (z)", line=dict(color="gray", dash="dot")))
    fig_multi.update_layout(
        title="Évolution comparée (échelles normalisées 0–1)",
        xaxis_title="Date",
        yaxis_title="Indice normalisé",
        template="plotly_white",
        height=550
    )
    st.plotly_chart(fig_multi, use_container_width=True)

    # === 5️⃣ Distribution saisonnière moyenne ===
    st.subheader("🕒 Saison moyenne du RSV (par semaine ISO)")
    df["week_iso"] = df.index.isocalendar().week
    mean_weekly = df.groupby("week_iso")["RSV"].mean().reset_index()
    fig_week = go.Figure()
    fig_week.add_trace(go.Scatter(
        x=mean_weekly["week_iso"], y=mean_weekly["RSV"],
        mode="lines+markers", name="RSV moyen par semaine ISO",
        line=dict(color="firebrick", width=3)
    ))
    fig_week.update_layout(
        title="Profil saisonnier moyen du RSV (2014–2025)",
        xaxis_title="Semaine ISO (1–52)",
        yaxis_title="RSV moyen (pour 100k hab.)",
        template="plotly_white", height=450
    )
    st.plotly_chart(fig_week, use_container_width=True)
    st.caption("👉 Le pic saisonnier RSV se situe classiquement entre les semaines 48 et 4, avec un décalage post-COVID visible en 2021.")

    # === 6️⃣ Tableau interactif filtrable ===
    st.subheader("📋 Aperçu tabulaire des données fusionnées")
    st.markdown("Filtrez ou explorez la base complète utilisée pour les modèles :")
    available_years = sorted(df.index.year.unique())
    default_years = [y for y in [2019, 2020, 2021, 2022, 2023] if y in available_years]
    years = st.multiselect(
    "Filtrer par année :",
    available_years,
    default=default_years if default_years else available_years[-3:]
)
    df_filtered = df[df.index.year.isin(years)].copy()
    st.dataframe(df_filtered.head(20), use_container_width=True)

    # === 7️⃣ Option d’export ===
    csv = df_filtered.reset_index().to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Télécharger les données fusionnées (CSV)",
        data=csv,
        file_name="RSV_dataset_filtered.csv",
        mime="text/csv"
    )

    st.info("""
    **Notes :**
    - `MNP_score` = moyenne des z-scores de gestes barrières + mobilité inversée.
    - `cov12_lag`, `MNP_lag`, `work_lag` = valeurs décalées (lags) utilisées dans les modèles.
    - Données hebdomadaires (lundi ISO) normalisées et harmonisées pour tous les flux (ODiSSEE, VAC-SI, CoviPrev, Google, Météo-France).
    """)

with tab7:
    st.header("🧼 Gestes barrières : détail par indicateur")
    mnp_vars = ["port_du_masque","lavage_des_mains","aeration_du_logement","saluer_sans_serrer_la_main"]
    # Si indisponibles, on les reconstruit depuis le calcul initial (à adapter si besoin)
    mnp_avail = [v for v in mnp_vars if v in df.columns]
    if not mnp_avail:
        st.info("Indicateurs MNP unitaires absents dans df — fournir `coviprev_reg_weekly` agrégé FR pour les afficher.")
    else:
        # Courbes individuelles
        fig = go.Figure()
        for v in mnp_avail:
            fig.add_trace(go.Scatter(x=df.index, y=df[v], mode="lines", name=v))
        fig.update_layout(template="plotly_white", title="CoviPrev — Indicateurs unitaires (FR)")
        st.plotly_chart(fig, use_container_width=True)

        # Contribution au score (z-normalisé)
        zcols = []
        for v in mnp_avail:
            z = (df[v] - df[v].mean()) / (df[v].std(ddof=0) if df[v].std(ddof=0)!=0 else 1)
            df[f"{v}_z"] = z; zcols.append(f"{v}_z")
        df["work_inv_z"] = ( -df["work"] - (-df["work"]).mean() ) / ((-df["work"]).std(ddof=0) if (-df["work"]).std(ddof=0)!=0 else 1)
        contrib = df[zcols + ["work_inv_z"]].mean().sort_values(ascending=False)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=contrib.index, y=contrib.values, text=np.round(contrib.values,2), textposition="outside"))
        fig2.update_layout(template="plotly_white", title="Contribution moyenne (z-score) au MNP_score")
        st.plotly_chart(fig2, use_container_width=True)

        # Heatmap année×semaine (score)
        df["_year"] = df.index.year; df["_week"] = df.index.isocalendar().week.astype(int)
        heat = df.pivot_table(index="_year", columns="_week", values="MNP_score", aggfunc="mean")
        fig3 = go.Figure(data=go.Heatmap(z=heat.values, x=heat.columns, y=heat.index, colorscale="RdBu"))
        fig3.update_layout(template="plotly_white", title="MNP_score — Heatmap (année × semaine)")
        st.plotly_chart(fig3, use_container_width=True)

        # Corrélation RSV vs sous-indicateurs
        corr = df[["RSV"] + mnp_avail].corr().round(2)
        fig4 = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1, colorscale="RdBu"))
        fig4.update_layout(template="plotly_white", title="Corrélations RSV ↔ gestes barrières")
        st.plotly_chart(fig4, use_container_width=True)

with tab8:
    st.header("🚶 Google Mobility : exploration complète")
    mobility_vars = ["workplaces","residential","retail_and_recreation","grocery_and_pharmacy","parks","transit_stations"]
    mob_avail = [v for v in mobility_vars if v in df.columns]
    if not mob_avail:
        st.info("Catégories Google Mobility manquantes dans df — charger `google_mobility_fr_weekly` enrichi.")
    else:
        # Courbes normalisées
        fig = go.Figure()
        for v in mob_avail:
            s = df[v]; s_norm = (s - s.min()) / (s.max() - s.min() + 1e-9)
            fig.add_trace(go.Scatter(x=df.index, y=s_norm, mode="lines", name=v))
        fig.update_layout(template="plotly_white", title="Séries mobilité (0–1 normalisées)")
        st.plotly_chart(fig, use_container_width=True)

        # Heatmap par catégorie
        year = st.selectbox("Année", sorted(df.index.year.unique()))
        sub = df[df.index.year == year].copy()
        heat = sub[mob_avail].T
        fig2 = go.Figure(data=go.Heatmap(z=heat.values, x=sub.index, y=heat.index, colorscale="Viridis"))
        fig2.update_layout(template="plotly_white", title=f"Mobilité — Heatmap hebdo ({year})")
        st.plotly_chart(fig2, use_container_width=True)

        # Corrélations partielles (RSV vs une catégorie)
        cat = st.selectbox("Catégorie à corréler au RSV", mob_avail, index=mob_avail.index("workplaces") if "workplaces" in mob_avail else 0)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df[cat], y=df["RSV"], mode="markers", name=f"{cat} vs RSV"))
        fig3.update_layout(template="plotly_white", xaxis_title=cat, yaxis_title="RSV", title=f"RSV vs {cat}")
        st.plotly_chart(fig3, use_container_width=True)

with tab9:
    st.header("🗺️ Pics saisonniers par région/département")
    reg = pd.read_csv(DATA_DIR/"ODISSEE/common_REG_long.csv", engine="python")
    dep = pd.read_csv(DATA_DIR/"ODISSEE/common_DEP_long.csv", engine="python")

    def prep(df, level_col):
        df = df.copy()
        df["date_monday"] = pd.to_datetime(df["date_monday"])
        df = df[(df["topic"]=="RSV")]
        # Choix d’un indicateur (adapté si `taux_passages_urgences` n’existe pas partout)
        y = "taux_passages_urgences" if "taux_passages_urgences" in df.columns else "taux_sos"
        df = df.rename(columns={y:"RSV"}).dropna(subset=["RSV"])
        df["season"] = df["date_monday"].dt.year.where(df["date_monday"].dt.month<9, df["date_monday"].dt.year+1)
        df["week"] = df["date_monday"].dt.isocalendar().week.astype(int)
        return df, level_col

    reg_prep, lvl_reg = prep(reg, "geo_code")
    dep_prep, lvl_dep = prep(dep, "geo_code")

    level = st.radio("Niveau", ["Région (REG)","Département (DEP)"], horizontal=True)
    if level == "Région (REG)":
        dfL = reg_prep; code_col = lvl_reg; label = "Région"
    else:
        dfL = dep_prep; code_col = lvl_dep; label = "Département"

    season = st.selectbox("Saison (année de fin)", sorted(dfL["season"].unique()))
    dfS = dfL[dfL["season"]==season]

    # Semaine du pic par zone
    peak = dfS.sort_values(["RSV"]).groupby(code_col).tail(1)[[code_col,"week","RSV"]].rename(columns={"week":"peak_week","RSV":"peak_value"})
    top_early = peak.sort_values("peak_week").head(15)
    top_late  = peak.sort_values("peak_week", ascending=False).head(15)

    col1,col2 = st.columns(2)
    with col1:
        st.subheader("⏱️ Pics les plus précoces")
        st.dataframe(top_early.rename(columns={code_col:label}), use_container_width=True)
    with col2:
        st.subheader("🐢 Pics les plus tardifs")
        st.dataframe(top_late.rename(columns={code_col:label}), use_container_width=True)


with tab2:
    st.header("Modélisation multi-stratégies")
    st.markdown("""
Plusieurs modèles statistiques ont été testés pour comprendre et anticiper la dynamique du RSV :

- **OLS** : modèle linéaire classique avec variables explicatives (température, vaccination, gestes barrières, etc.)
- **ITS** (Interrupted Time Series) : modèle de rupture capturant les changements post-COVID ou post-vaccination
- **SARIMAX** : modèle de série temporelle auto-régressif avec variables exogènes

Choisissez un modèle dans la liste pour explorer ses performances.
""")

    model_choice = st.selectbox("Sélectionnez un modèle :", [
        "OLS de base",
        "OLS optimisé",
        "ITS simple",
        "ITS optimisé (v2)",
        "SARIMAX baseline",
        "SARIMAX optimisé"
    ])

    if model_choice == "OLS de base":
        base_model = models_bundle.get("ols_base")
        if base_model is None:
            st.warning("Modèle OLS de base indisponible (échec du fit).")
            pred = pd.Series(np.nan, index=df.index)
            r2 = aic = bic = np.nan
        else:
            pred = base_model.fittedvalues.reindex(df.index)
            r2 = base_model.rsquared_adj
            aic, bic = base_model.aic, base_model.bic
        model_desc = "Modèle linéaire avec lags par défaut et harmonique saisonnière."

    elif model_choice == "OLS optimisé":
        pred = model.fittedvalues.reindex(df.index)
        r2 = model.rsquared_adj
        aic, bic = model.aic, model.bic
        model_desc = "Modèle OLS incluant exogènes, ruptures, température et lags optimisés (issus du notebook)."

    elif model_choice == "ITS simple":
        cols = ["t", "post_covid", "t_postcovid"]
        its_base = sm.OLS(
            df_base["RSV"],
            sm.add_constant(df_base[cols], has_constant="add"),
            missing="drop",
        ).fit()
        pred = its_base.fittedvalues.reindex(df.index)
        r2 = its_base.rsquared_adj
        aic, bic = its_base.aic, its_base.bic
        model_desc = "Modèle de série interrompue simple (rupture post-COVID)."

    elif model_choice == "ITS optimisé (v2)":
        its_results = ensure_its_model(meta_bundle)
        its_model = its_results.get("model")
        if its_model is None:
            st.warning("Modèle ITS optimisé indisponible (échec du fit).")
            pred = pd.Series(np.nan, index=df.index)
            r2 = aic = bic = np.nan
        else:
            design = its_results.get("design")
            pred = its_model.fittedvalues.reindex(design.index).reindex(df.index)
            r2 = getattr(its_model, "rsquared_adj", np.nan)
            aic, bic = its_model.aic, its_model.bic
        model_desc = (
            "ITS optimisé avec recherche des dates de rupture, harmoniques Fourier et lags (reproduit depuis le notebook)."
        )

    elif model_choice == "SARIMAX baseline":
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        sarima1 = SARIMAX(df["RSV"], order=(1, 0, 1), seasonal_order=(1, 0, 1, 52)).fit(disp=False)
        pred = sarima1.fittedvalues.reindex(df.index)
        r2 = 1 - np.sum((df["RSV"] - pred) ** 2) / np.sum((df["RSV"] - df["RSV"].mean()) ** 2)
        aic, bic = sarima1.aic, sarima1.bic
        model_desc = "SARIMA simple sans exogènes (ARIMA saisonnier 52 semaines)."

    elif model_choice == "SARIMAX optimisé":
        sarimax_results = ensure_sarimax_model(meta_bundle)
        sarimax_model = sarimax_results.get("model")
        if sarimax_model is None:
            st.warning("Modèle SARIMAX optimisé indisponible (échec du fit).")
            pred = pd.Series(np.nan, index=df.index)
            r2 = aic = bic = np.nan
        else:
            pred = sarimax_model.fittedvalues.reindex(df.index)
            r2 = sarimax_results.get("pseudo_r2", np.nan)
            aic, bic = sarimax_model.aic, sarimax_model.bic
        order = sarimax_results.get("order")
        seasonal = sarimax_results.get("seasonal_order")
        model_desc = (
            f"SARIMAX optimisé (ordre={order}, saisonnier={seasonal}) "
            "avec exogènes (vaccination, gestes, météo) issu du notebook."
        )

    # Affichage courbe Observé vs Prédit
    fig_mod = go.Figure()
    fig_mod.add_trace(go.Scatter(x=df.index, y=df["RSV"], mode='lines', name='RSV Observé', line=dict(color='black')))
    fig_mod.add_trace(go.Scatter(x=df.index, y=pred, mode='lines', name='RSV Prédit', line=dict(color='royalblue', dash='dot')))
    fig_mod.update_layout(title=f"Comparaison Observé vs Prédit — {model_choice}", 
                          xaxis_title="Date", yaxis_title="RSV (pour 100k hab.)")
    st.plotly_chart(fig_mod, use_container_width=True)

    st.markdown(f"""
**{model_choice}**
- {model_desc}
- $R^2$ ajusté = {r2:.3f}
- AIC = {aic:.1f} | BIC = {bic:.1f}
""")
with tab3:
    st.header("Scénarios contrefactuels et personnalisés")
    st.markdown("""
Cette section permet de **simuler la trajectoire du RSV** selon différents contextes :
- Suppression de la pandémie (No COVID)
- Absence de vaccination (No Vaccine)
- Maintien des gestes barrières (Keep MNP)
- Conditions personnalisées (curseurs)
""")

    # --- Choix du modèle ---
    model_type = st.selectbox(
        "Choisissez le modèle à utiliser pour la simulation :",
        ["OLS optimisé", "ITS optimisé (v2)", "SARIMAX optimisé"]
    )

    # --- Scénarios prédéfinis ---
    scenario_choice = st.radio(
        "Sélectionnez un scénario à simuler :",
        [
            "Scénario observé (réel)",
            "No COVID",
            "No Vaccine",
            "Keep MNP (gestes maintenus)",
            "Scénario personnalisé"
        ]
    )

    # Base de travail selon le modèle sélectionné
    if model_type == "OLS optimisé":
        base_df = df.copy()
        feature_cols = [c for c in model.model.exog_names if c != "const"]
        fitted_model = model
    elif model_type == "ITS optimisé (v2)":
        its_results = ensure_its_model(meta_bundle)
        base_df = its_results.get("design")
        feature_cols = its_results.get("cols", [])
        fitted_model = its_results.get("model")
        if base_df is None or fitted_model is None:
            st.warning("Le modèle ITS optimisé n'est pas disponible. Veuillez régénérer le notebook.")
            st.stop()
        base_df = base_df.copy()
    elif model_type == "SARIMAX optimisé":
        sarimax_results = ensure_sarimax_model(meta_bundle)
        base_df = sarimax_results.get("frame")
        feature_cols = sarimax_results.get("exog_cols", [])
        fitted_model = sarimax_results.get("model")
        if base_df is None or fitted_model is None:
            st.warning("Le modèle SARIMAX optimisé n'est pas disponible. Veuillez régénérer le notebook.")
            st.stop()
        base_df = base_df.copy()
    else:
        st.warning("Modèle inconnu pour la simulation.")
        st.stop()

    df_scen = base_df.copy()

    # =============================
    # 🔹 SCÉNARIOS PRÉDÉFINIS
    # =============================
    if scenario_choice == "Scénario observé (réel)":
        df_scen = base_df.copy()

    elif scenario_choice == "No COVID":
        for col in ["post_covid", "post_vacc"]:
            if col in df_scen.columns:
                df_scen[col] = 0
        for col in ["t_postcovid", "t_postvacc"]:
            if col in df_scen.columns:
                df_scen[col] = 0
        for col in ["cov12_lag", "MNP_lag", "work_lag"]:
            if col in df_scen.columns:
                df_scen[col] = 0

    elif scenario_choice == "No Vaccine":
        if "cov12_lag" in df_scen.columns:
            df_scen["cov12_lag"] = 0
        for col in ["post_vacc", "t_postvacc"]:
            if col in df_scen.columns:
                df_scen[col] = 0

    elif scenario_choice == "Keep MNP (gestes maintenus)":
        if "MNP_lag" in df_scen.columns:
            df_scen["MNP_lag"] = df_scen["MNP_lag"].median()
        for col in ["post_covid", "post_vacc"]:
            if col in df_scen.columns:
                df_scen[col] = 1

    # =============================
    # 🔹 SCÉNARIO PERSONNALISÉ
    # =============================
    elif scenario_choice == "Scénario personnalisé":
        st.subheader("Ajustez vos paramètres")
        col1, col2, col3 = st.columns(3)
        with col1:
            vacc_slider = st.slider("Taux de vaccination COVID (%)", 0, 100, 100)
            mnp_slider = st.slider("Adhésion gestes barrières (%)", 0, 100, 100)
        with col2:
            work_slider = st.slider("Mobilité (lieux de travail, % activité)", 0, 150, 100)
            temp_slider = st.slider("Anomalie de température (°C)", -5, 5, 0)
        with col3:
            no_covid = st.checkbox("Pas de pandémie COVID ?", False)

        df_scen = base_df.copy()

        if no_covid:
            for col in ["post_covid", "post_vacc"]:
                if col in df_scen.columns:
                    df_scen[col] = 0
            for col in ["t_postcovid", "t_postvacc"]:
                if col in df_scen.columns:
                    df_scen[col] = 0
            if "cov12_lag" in df_scen.columns:
                df_scen["cov12_lag"] = 0

        # Application des curseurs
        if "cov12_lag" in df_scen.columns:
            df_scen["cov12_lag"] = df_scen["cov12_lag"] * (vacc_slider / 100.0)
        if "MNP_lag" in df_scen.columns:
            df_scen["MNP_lag"] = df_scen["MNP_lag"] * (mnp_slider / 100.0)
        if "work_lag" in df_scen.columns:
            df_scen["work_lag"] = df_scen["work_lag"] * (work_slider / 100.0)

        tmean_stats = meta_bundle.get("tmean_stats", {})
        mu, sigma = tmean_stats.get("mean"), tmean_stats.get("std")
        if "tmean" in df_scen.columns:
            df_scen["tmean_adj"] = df_scen["tmean"] + temp_slider
            denom = sigma if sigma not in (None, 0) else 1
            df_scen["tmean_z"] = (df_scen["tmean_adj"] - (mu if mu is not None else df_scen["tmean_adj"].mean())) / denom
        elif "tmean_z" in df_scen.columns and sigma not in (None, 0):
            df_scen["tmean_z"] = df_scen["tmean_z"] + temp_slider / sigma

    # Recalcule les variables dérivées nécessaires
    if {"cov12_lag", "MNP_lag"}.issubset(df_scen.columns):
        df_scen["vacc_x_mnp"] = df_scen["cov12_lag"] * df_scen["MNP_lag"]

    # =============================
    # 🔸 PRÉDICTION SELON LE MODÈLE
    # =============================
    st.markdown("### Simulation en cours...")

    if model_type == "OLS optimisé":
        exog_names = [c for c in fitted_model.model.exog_names if c != "const"]
        X_scen = sm.add_constant(df_scen[exog_names], has_constant="add")
        # aligne les colonnes sur l'ordre du modèle
        X_scen = X_scen.reindex(columns=fitted_model.params.index, fill_value=0)
        y_pred = fitted_model.predict(X_scen)

    elif model_type == "ITS optimisé (v2)":
        exog_names = [c for c in fitted_model.model.exog_names if c != "const"]
        X_scen = sm.add_constant(df_scen[exog_names], has_constant="add")
        X_scen = X_scen.reindex(columns=fitted_model.params.index, fill_value=0)
        y_pred = fitted_model.predict(X_scen)

    elif model_type == "SARIMAX optimisé":
        exog_cols = feature_cols or sarimax_results.get("exog_cols", [])
        y_pred = fitted_model.predict(start=0, end=len(df_scen) - 1, exog=df_scen[exog_cols])
    else:
        y_pred = pd.Series(np.nan, index=df.index)

    y_pred = pd.Series(y_pred, index=df_scen.index)
    pred_full = y_pred.reindex(df.index)
    observed_full = df["RSV"]

    # =============================
    # 🔸 VISUALISATION DU SCÉNARIO
    # =============================
    fig_scen = go.Figure()
    fig_scen.add_trace(go.Scatter(x=df.index, y=observed_full, mode='lines', name='RSV Observé', line=dict(color='black')))
    fig_scen.add_trace(go.Scatter(x=df.index, y=pred_full, mode='lines', name='RSV Simulé', line=dict(color='firebrick', dash='dot')))
    fig_scen.add_vline(x=pd.Timestamp("2020-03-01"), line=dict(color="red", dash="dash"))
    fig_scen.add_vline(x=pd.Timestamp("2021-01-01"), line=dict(color="green", dash="dash"))
    fig_scen.update_layout(
        title=f"RSV — Simulation ({model_type} / {scenario_choice})",
        xaxis_title="Semaine ISO", yaxis_title="RSV (modélisé)",
        height=600
    )
    st.plotly_chart(fig_scen, use_container_width=True)

    # Résumé
    st.markdown(f"""
**Modèle :** {model_type}  
**Scénario :** {scenario_choice}  
**Observation :** la courbe rouge représente la trajectoire simulée du RSV selon vos conditions.  
Vous pouvez comparer visuellement l'effet de chaque hypothèse par rapport à la courbe noire observée.
""")

    # =============================
    # 🔸 COMPARAISON & DELTA SCÉNARIO
    # =============================
    st.markdown("### 📊 Analyse comparative du scénario simulé")

    delta_series = (pred_full - observed_full).dropna()
    delta_cum = float(delta_series.sum())
    delta_pct = 100 * delta_cum / observed_full.sum()

    fig_delta = go.Figure()
    fig_delta.add_trace(go.Scatter(
        x=delta_series.index, y=delta_series,
        mode="lines", name="Δ (Scénario - Observé)",
        line=dict(color="darkorange", width=2)
    ))
    fig_delta.add_hline(y=0, line_dash="dot")
    fig_delta.update_layout(
        title="Écart hebdomadaire entre scénario simulé et observation réelle",
        xaxis_title="Semaine ISO",
        yaxis_title="Δ RSV (scénario - observé)",
        template="plotly_white",
        height=400
    )
    st.plotly_chart(fig_delta, use_container_width=True)

    st.markdown(f"""
**Bilan cumulé du scénario**  
- Δ cumulé total : **{delta_cum:.1f}** unités RSV  
- Variation cumulée : **{delta_pct:+.1f}%** par rapport à la situation observée  
_(valeur positive = RSV plus fort dans le scénario que dans la réalité)_
""")

    # =============================
    # 🔸 COMPARAISON INTER-MODÈLES
    # =============================
    st.markdown("### ⚙️ Comparaison de performance entre modèles")

    results_summary = []
    if models_bundle.get("ols_opt"):
        results_summary.append({
            "Modèle": "OLS optimisé",
            "AIC": model.aic,
            "BIC": model.bic,
            "R2/Pseudo-R2": model.rsquared_adj
        })
    its_cache = st.session_state.get("its_cached_results")
    if its_cache and its_cache.get("model") is not None:
        results_summary.append({
            "Modèle": "ITS optimisé (v2)",
            "AIC": its_cache["model"].aic,
            "BIC": its_cache["model"].bic,
            "R2/Pseudo-R2": getattr(its_cache["model"], "rsquared_adj", np.nan)
        })
    sarimax_cache = st.session_state.get("sarimax_cached_results")
    if sarimax_cache and sarimax_cache.get("model") is not None:
        results_summary.append({
            "Modèle": "SARIMAX optimisé",
            "AIC": sarimax_cache["model"].aic,
            "BIC": sarimax_cache["model"].bic,
            "R2/Pseudo-R2": sarimax_cache.get("pseudo_r2", np.nan)
        })

    if results_summary:
        df_perf = pd.DataFrame(results_summary)
        fig_perf = go.Figure()
        fig_perf.add_trace(go.Bar(
            x=df_perf["Modèle"],
            y=df_perf["R2/Pseudo-R2"],
            name="R² / Pseudo-R²",
            text=df_perf["R2/Pseudo-R2"].round(3),
            textposition="outside"
        ))
        fig_perf.add_trace(go.Bar(
            x=df_perf["Modèle"],
            y=-df_perf["AIC"],
            name="-AIC (plus haut = mieux)",
            opacity=0.6
        ))
        fig_perf.add_trace(go.Bar(
            x=df_perf["Modèle"],
            y=-df_perf["BIC"],
            name="-BIC (plus haut = mieux)",
            opacity=0.6
        ))
        fig_perf.update_layout(
            barmode="group",
            title="Comparaison de performance entre modèles (R², AIC, BIC)",
            xaxis_title="Modèle",
            yaxis_title="Score (échelle normalisée)",
            template="plotly_white",
            height=500
        )
        st.plotly_chart(fig_perf, use_container_width=True)

        best_model_row = df_perf.loc[df_perf["R2/Pseudo-R2"].idxmax()]
        st.success(f"🏆 Le modèle présentant la meilleure performance globale est : **{best_model_row['Modèle']}**")
    else:
        st.info("Aucun modèle n'a pu être comparé (résultats manquants).")
with tab4:
    st.header("Diagnostics et validation des modèles")
    st.markdown("""
Cette section permet de **vérifier la validité statistique** de chaque modèle utilisé dans l’analyse.
Les tests et graphiques permettent de s’assurer que :
- Les résidus sont bien distribués de manière aléatoire (pas d’autocorrélation),
- Le modèle ne laisse pas d’information structurelle non captée,
- Et les performances sont cohérentes avec les hypothèses de modélisation.
""")

    # Sélecteur de modèle à diagnostiquer
    diag_model = st.selectbox(
        "Choisissez le modèle à inspecter :",
        ["OLS optimisé", "ITS optimisé (v2)", "SARIMAX optimisé"]
    )

    # Fonctions de diagnostic
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import acf

    def plot_residuals(y_true, y_fit, title):
        resid = y_true - y_fit
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_true.index, y=resid, mode="lines", name="Résidus"))
        fig.add_hline(y=0, line_dash="dot")
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Erreur (prédit - observé)",
            template="plotly_white",
            height=400
        )
        return resid, fig

    def plot_acf(resid, title):
        acf_vals = acf(resid, nlags=24, fft=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(range(len(acf_vals))), y=acf_vals))
        fig.update_layout(
            title=title,
            xaxis_title="Lag (semaines)",
            yaxis_title="Autocorrélation (ACF)",
            template="plotly_white",
            height=400
        )
        return fig

    # --- Génération selon modèle choisi
    if diag_model == "OLS optimisé":
        y_true = df["RSV"]
        y_fit = model.fittedvalues.reindex(df.index)
        resid, fig_resid = plot_residuals(y_true, y_fit, "Résidus du modèle OLS optimisé")
        fig_acf_resid = plot_acf(resid, "ACF des résidus — OLS")

    elif diag_model == "ITS optimisé (v2)":
        its_results = st.session_state.get("its_cached_results") or ensure_its_model(meta_bundle)
        its_model = its_results.get("model")
        if its_model is None:
            st.warning("Le modèle ITS optimisé n'est pas disponible.")
            st.stop()
        design = its_results.get("design")
        y_true = design["RSV"]
        y_fit = its_model.fittedvalues.reindex(design.index)
        resid, fig_resid = plot_residuals(y_true, y_fit, "Résidus du modèle ITS optimisé (v2)")
        fig_acf_resid = plot_acf(resid, "ACF des résidus — ITS")

    elif diag_model == "SARIMAX optimisé":
        sarimax_results = st.session_state.get("sarimax_cached_results") or ensure_sarimax_model(meta_bundle)
        sarimax_model = sarimax_results.get("model")
        if sarimax_model is None:
            st.warning("Le modèle SARIMAX optimisé n'est pas disponible.")
            st.stop()
        frame = sarimax_results.get("frame")
        y_true = frame["RSV"]
        y_fit = sarimax_model.fittedvalues.reindex(frame.index)
        resid, fig_resid = plot_residuals(y_true, y_fit, "Résidus du modèle SARIMAX optimisé")
        fig_acf_resid = plot_acf(resid, "ACF des résidus — SARIMAX")

    # --- Calcul des métriques de diagnostic
    dw_stat = sm.stats.stattools.durbin_watson(resid)
    lb_test = acorr_ljungbox(resid, lags=[8, 12, 24], return_df=True)
    pval_12 = lb_test.loc[12, "lb_pvalue"] if 12 in lb_test.index else lb_test["lb_pvalue"].iloc[1]

    # --- Affichage
    col1, col2 = st.columns(2)
    col1.plotly_chart(fig_resid, use_container_width=True)
    col2.plotly_chart(fig_acf_resid, use_container_width=True)

    # Résumé des tests
    st.markdown(f"""
**🧠 Tests statistiques principaux :**
- **Durbin–Watson :** {dw_stat:.2f} → valeur proche de 2 = pas d’autocorrélation.
- **Ljung–Box (lag=12) :** p-value = {pval_12:.3f} → si > 0.05, pas d’autocorrélation significative.
""")

    # --- Interprétation automatique
    if abs(dw_stat - 2) < 0.3 and pval_12 > 0.05:
        st.success("✅ Le modèle semble statistiquement sain : les résidus sont non autocorrélés et bien distribués.")
    elif pval_12 < 0.05:
        st.warning("⚠️ Attention : autocorrélation significative détectée dans les résidus (Ljung–Box < 0.05).")
    else:
        st.info("ℹ️ Résidus légèrement autocorrélés — possible effet de sous-spécification ou saisonnalité non captée.")
# =======================================
# 🧭 Onglet 5 — Synthèse finale
# =======================================

with tab5:
    st.header("Synthèse finale de l’étude")
    st.markdown("""
Ce dernier onglet résume l’ensemble des analyses réalisées :
- **Comparaison des modèles (OLS, ITS, SARIMAX)**  
- **Effets cumulatifs simulés (Δ cumulés par scénario)**  
- **Prévisions à moyen terme (2025–2027)**  
""")

    # =============================
    # 🔹 COMPARAISON DES MODÈLES
    # =============================
    st.subheader("⚙️ Comparaison de performance entre modèles")
    perf_rows = [
        ["OLS optimisé", model.aic, model.bic, model.rsquared_adj]
    ]
    its_cache = st.session_state.get("its_cached_results")
    if its_cache and its_cache.get("model") is not None:
        perf_rows.append([
            "ITS optimisé (v2)",
            its_cache["model"].aic,
            its_cache["model"].bic,
            getattr(its_cache["model"], "rsquared_adj", np.nan)
        ])
    sarimax_cache = st.session_state.get("sarimax_cached_results")
    if sarimax_cache and sarimax_cache.get("model") is not None:
        perf_rows.append([
            "SARIMAX optimisé",
            sarimax_cache["model"].aic,
            sarimax_cache["model"].bic,
            sarimax_cache.get("pseudo_r2", np.nan)
        ])

    perf_df = pd.DataFrame(perf_rows, columns=["Modèle", "AIC", "BIC", "R2/Pseudo-R2"])

    fig_perf_summary = go.Figure()
    fig_perf_summary.add_trace(go.Bar(
        x=perf_df["Modèle"], y=perf_df["R2/Pseudo-R2"],
        text=perf_df["R2/Pseudo-R2"].round(3),
        textposition="outside", name="R² / Pseudo-R²"
    ))
    fig_perf_summary.add_trace(go.Bar(
        x=perf_df["Modèle"], y=-perf_df["AIC"], name="-AIC (meilleur ↑)", opacity=0.6
    ))
    fig_perf_summary.add_trace(go.Bar(
        x=perf_df["Modèle"], y=-perf_df["BIC"], name="-BIC (meilleur ↑)", opacity=0.6
    ))
    fig_perf_summary.update_layout(
        barmode="group",
        title="Comparaison finale des performances des modèles",
        xaxis_title="Modèle", yaxis_title="Score normalisé",
        template="plotly_white", height=500
    )
    st.plotly_chart(fig_perf_summary, use_container_width=True)

    best_model = perf_df.loc[perf_df["R2/Pseudo-R2"].idxmax(), "Modèle"]
    st.success(f"🏆 **Modèle le plus performant globalement : {best_model}**")

    # =============================
    # 🔹 EFFETS CUMULÉS PAR SCÉNARIO
    # =============================
    st.subheader("📊 Effets cumulés par scénario (ITS v2)")
    try:
        delta_summary = pd.read_csv("../outputs/RSV_results/summary_scenarios_delta.csv", engine="python")
        fig_delta_summary = go.Figure()
        fig_delta_summary.add_trace(go.Bar(
            x=delta_summary["Scenario"],
            y=delta_summary["Δ RSV cumulé"],
            text=[f"{v} ({p}%)" for v, p in zip(delta_summary["Δ RSV cumulé"], delta_summary["Variation (%)"])],
            textposition="outside",
            marker_color=["red" if "No" in s else "green" for s in delta_summary["Scenario"]]
        ))
        fig_delta_summary.update_layout(
            title="Δ cumulés des scénarios contrefactuels (ITS v2)",
            xaxis_title="Scénario", yaxis_title="Δ RSV cumulé (vs Observé)",
            template="plotly_white", height=550
        )
        st.plotly_chart(fig_delta_summary, use_container_width=True)
    except Exception:
        st.info("ℹ️ Aucune table de scénarios exportée n’a encore été trouvée (exécution locale nécessaire).")

    # =============================
    # 🔹 PRÉVISION À MOYEN TERME
    # =============================
    st.subheader("🔮 Prévision RSV France (2025–2027) — SARIMAX ITS long")
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    # Modèle SARIMAX long (comme dans ta dernière section)
    exog_cols_long = ["cov12_lag", "MNP_lag", "work_lag", "post_covid", "post_vacc", "sin52", "cos52"]
    model_long = SARIMAX(df["RSV"], order=(1,0,1), seasonal_order=(1,0,1,52),
                         exog=df[exog_cols_long], enforce_stationarity=False,
                         enforce_invertibility=False).fit(disp=False)
    forecast_steps = 104  # 2 ans
    future_idx = pd.date_range(df.index.max() + pd.Timedelta(weeks=1),
                               periods=forecast_steps, freq="W-MON")
    exog_future = pd.DataFrame({
        c: [df[c].iloc[-1]] * forecast_steps for c in exog_cols_long
    }, index=future_idx)
    forecast = model_long.get_forecast(steps=forecast_steps, exog=exog_future)
    pred_mean = forecast.predicted_mean
    conf_int = forecast.conf_int(alpha=0.05)

    # Plot
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=df.index, y=df["RSV"], mode="lines", name="RSV Observé", line=dict(color="black")))
    fig_forecast.add_trace(go.Scatter(x=pred_mean.index, y=pred_mean, mode="lines", name="Prévision 2025–2027", line=dict(color="blue", dash="dash")))
    fig_forecast.add_trace(go.Scatter(x=conf_int.index, y=conf_int.iloc[:,0], mode="lines", line=dict(width=0), showlegend=False))
    fig_forecast.add_trace(go.Scatter(x=conf_int.index, y=conf_int.iloc[:,1], mode="lines", fill="tonexty", fillcolor="rgba(0,0,255,0.1)", line=dict(width=0), name="IC 95%"))
    fig_forecast.update_layout(
        title="Projection RSV France (SARIMAX ITS long, 2025–2027)",
        xaxis_title="Semaine", yaxis_title="RSV modélisé",
        template="plotly_white", width=1100, height=600
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

    # =============================
    # 🔹 CONCLUSION AUTOMATIQUE
    # =============================
    st.markdown("""
### 🧩 Interprétation synthétique
- **Rupture COVID-19 (2020)** → Chute massive du RSV, effet direct des MNP.  
- **Rebond 2021 (printemps)** → Décalage saisonnier lié à la levée des restrictions.  
- **Effet Vaccination COVID** → Relâchement des MNP → hausse indirecte du RSV.  
- **MNP** = principal facteur protecteur ; **No Vaccine** = hausse de +60–80 % de RSV cumulé.  
- **SARIMAX** projette un retour progressif à une saisonnalité hivernale, sans retrouver exactement le régime pré-COVID.  
""")
    st.success("✅ Synthèse complète générée avec succès — prête pour intégration dans ton manuscrit ou dashboard final.")

# =======================================
# 🧮 Onglet 6 — Scénarios – Modèles ERVISS (2014–2025)
# =======================================
with tab6:
    st.header("Scénarios – Modèles ERVISS (2014–2025)")
    st.markdown("""
    Cet onglet reproduit la modélisation basée sur **ERVISS** (détections virologiques RSV, France),
    avec **3 modèles** (OLS, ITS v2, SARIMAX v2) et des **scénarios contrefactuels complets** :
    COVID/NoCOVID × MNP *(real / none / maintained)* × Vaccine/NoVaccine.
    """)

    # =============================
    # 1) Chargement & préparation
    # =============================
    FILES = {
        "erviss_fr_weekly": DATA_DIR / "ERVISS/erviss_fr_weekly.csv",
        "vacsi_fr_extended": DATA_DIR / "VACSI/vacsi_fr_extended.csv",
        "google_mobility_fr_weekly": DATA_DIR / "GOOGLE/google_mobility_fr_weekly.csv",
        "coviprev_reg_weekly": DATA_DIR / "COVIPREV/coviprev_reg_weekly.csv",
    }
    for k, p in FILES.items():
        if not p.exists():
            st.error(f"Fichier manquant: {k} → {p}")
            st.stop()

    def safe_zscore(s: pd.Series) -> pd.Series:
        m, sd = s.mean(), s.std(ddof=0)
        if sd == 0 or np.isnan(sd):
            return pd.Series(0.0, index=s.index)
        return (s - m) / sd

    # --- ERVISS: RSV detections (FR)
    erv = pd.read_csv(FILES["erviss_fr_weekly"], engine="python")
    mask_fr = (erv.get("geo_level", "FR") == "FR") & (erv.get("geo_code", "FR") == "FR")
    mask_rsv = (erv.get("pathogen", "").astype(str).str.upper() == "RSV") | (
        erv.get("pathogentype", "").astype(str).str.upper().str.contains("RSV", na=False)
    )
    mask_det = erv.get("indicator", "").astype(str).str.lower() == "detections"
    erv_rsv = erv.loc[mask_fr & mask_rsv & mask_det,
                      ["date_monday", "year_iso", "week_iso_num", "value"]].copy()
    if not {"year_iso","week_iso_num"}.issubset(erv_rsv.columns):
        erv_rsv = keyify(erv_rsv)
    y_erv = (erv_rsv.groupby(["year_iso","week_iso_num","date_monday"], as_index=False)["value"]
                      .sum()
                      .rename(columns={"value":"RSV_det"}))
    y_erv["date_monday"] = pd.to_datetime(y_erv["date_monday"])
    y_erv = y_erv.sort_values(["year_iso","week_iso_num"])

    # --- Exogènes: VACSI (couv_complet), Google Mobility (workplaces), CoviPrev (masking, washing, aeration)
    vacsi = keyify(pd.read_csv(FILES["vacsi_fr_extended"], engine="python"))
    vac = vacsi.query("geo_level=='FR' & geo_code=='FR'")[["year_iso","week_iso_num","couv_complet"]]

    gm = keyify(pd.read_csv(FILES["google_mobility_fr_weekly"], engine="python"))
    work = (gm.query("geo_level=='FR' & geo_code=='FR' & indicator=='workplaces'")
              [["year_iso","week_iso_num","value"]]
              .rename(columns={"value":"work"}))

    cov = keyify(pd.read_csv(FILES["coviprev_reg_weekly"], engine="python"))
    mask_vars = ["port_du_masque","lavage_des_mains","aeration_du_logement"]
    cov_nat = (cov[cov["indicator"].isin(mask_vars)]
                .groupby(["year_iso","week_iso_num","indicator"])["value"]
                .mean().unstack())

    # --- Merge base
    base_erv = (y_erv[["date_monday","year_iso","week_iso_num","RSV_det"]]
                .merge(vac,  on=["year_iso","week_iso_num"], how="left")
                .merge(work, on=["year_iso","week_iso_num"], how="left")
                .merge(cov_nat, on=["year_iso","week_iso_num"], how="left")
                .set_index("date_monday")
                .sort_index())

    # --- MNP composite (z-score sur variables + work inversé)
    base_erv["work_red"] = safe_zscore(-base_erv["work"])
    for v in mask_vars:
        base_erv[v] = safe_zscore(base_erv[v])
    base_erv["MNP_score"] = base_erv[mask_vars + ["work_red"]].mean(axis=1)

    # --- Lags & saisonnalité
    lag_vac, lag_mnp, lag_work = 4, 8, 9
    X_erv = pd.DataFrame(index=base_erv.index).sort_index()
    X_erv["cov12_lag"] = base_erv["couv_complet"].shift(lag_vac)
    X_erv["MNP_lag"]   = base_erv["MNP_score"].shift(lag_mnp)
    X_erv["work_lag"]  = base_erv["work"].shift(lag_work)

    X_erv["t"]     = np.arange(len(X_erv))
    X_erv["sin52"] = np.sin(2*np.pi*X_erv["t"]/52)
    X_erv["cos52"] = np.cos(2*np.pi*X_erv["t"]/52)

    # --- Ruptures
    COVID_START = pd.Timestamp("2020-03-01")
    VACC_START  = pd.Timestamp("2021-01-01")
    X_erv["post_covid"]   = (X_erv.index >= COVID_START).astype(int)
    X_erv["post_vacc"]    = (X_erv.index >= VACC_START).astype(int)
    X_erv["t_postcovid"]  = X_erv["t"] * X_erv["post_covid"]

    # --- Dataframe final
    df_erv = (base_erv[["RSV_det"]].join(X_erv, how="left"))

    # Nettoyage minimal pour régressions
    df_model_all = df_erv.dropna(subset=["RSV_det","cov12_lag","MNP_lag","work_lag","sin52","cos52"]).copy()
    df_model_all = df_model_all.replace([np.inf,-np.inf], np.nan).dropna()

    # =============================
    # 2) Modèle OLS (HC3)
    # =============================
    Y_ols = df_model_all["RSV_det"]
    X_ols = sm.add_constant(df_model_all[["cov12_lag","MNP_lag","work_lag","sin52","cos52"]], has_constant="add")
    ols_erv = sm.OLS(Y_ols, X_ols, missing="drop").fit(cov_type="HC3")

    # =============================
    # 3) Modèle ITS v2 (OLS + HAC)
    # =============================
    X_cols_its = ["t","post_covid","t_postcovid","post_vacc","cov12_lag","MNP_lag","work_lag","sin52","cos52"]
    df_its = df_erv.dropna(subset=["RSV_det"] + X_cols_its).copy()
    Y_its  = df_its["RSV_det"]
    X_its  = sm.add_constant(df_its[X_cols_its], has_constant="add")
    its_v2 = sm.OLS(Y_its, X_its, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags":12})

    # =============================
    # 4) Modèle SARIMAX v2 (1,0,0)x(1,1,0,52) + exog
    # =============================
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.stats.diagnostic import acorr_ljungbox

    exog_cols_sx = ["cov12_lag","MNP_lag","work_lag","post_covid","post_vacc","t"]
    df_sx = df_erv.dropna(subset=["RSV_det"] + exog_cols_sx).copy()
    sarimax_v2 = SARIMAX(
        endog=df_sx["RSV_det"],
        exog=df_sx[exog_cols_sx],
        order=(1,0,0),
        seasonal_order=(1,1,0,52),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)

    # =============================
    # 5) Visualisation Observé vs Modèles
    # =============================
    fig_fit = go.Figure()
    fig_fit.add_trace(go.Scatter(x=df_erv.index, y=df_erv["RSV_det"], mode="lines", name="Observed (ERVISS)", line=dict(color="black")))
    fig_fit.add_trace(go.Scatter(x=df_model_all.index, y=ols_erv.fittedvalues, mode="lines", name="OLS (HC3)", line=dict(dash="dot", color="#1F77B4")))
    fig_fit.add_trace(go.Scatter(x=df_its.index, y=its_v2.fittedvalues, mode="lines", name="ITS v2 (HAC)", line=dict(dash="dash", color="#D62728")))
    fig_fit.add_trace(go.Scatter(x=df_sx.index, y=sarimax_v2.fittedvalues, mode="lines", name="SARIMAX v2", line=dict(color="#2CA02C")))
    fig_fit.add_vline(x=COVID_START, line=dict(color="red", dash="dot"))
    fig_fit.add_vline(x=VACC_START,  line=dict(color="green", dash="dot"))
    fig_fit.update_layout(
        title="ERVISS RSV detections — Observé vs Modèles",
        xaxis_title="Semaine", yaxis_title="Détections RSV",
        template="plotly_white", height=520
    )
    st.plotly_chart(fig_fit, use_container_width=True)

    # =============================
    # 6) Synthèse métriques
    # =============================
    def safe_metric(obj, name, default=np.nan):
        try: return getattr(obj, name)
        except: return default

    from statsmodels.stats.stattools import durbin_watson
    synth = pd.DataFrame([
        {
            "Modèle": "OLS (HC3)",
            "AIC": safe_metric(ols_erv, "aic"),
            "BIC": safe_metric(ols_erv, "bic"),
            "R²_adj": safe_metric(ols_erv, "rsquared_adj"),
            "DW": durbin_watson(ols_erv.resid),
            "β_cov12": ols_erv.params.get("cov12_lag", np.nan),
            "p_cov12": ols_erv.pvalues.get("cov12_lag", np.nan),
        },
        {
            "Modèle": "ITS v2 (HAC, maxlags=12)",
            "AIC": safe_metric(its_v2, "aic"),
            "BIC": safe_metric(its_v2, "bic"),
            "R²_adj": safe_metric(its_v2, "rsquared_adj"),
            "DW": durbin_watson(its_v2.resid),
            "β_cov12": its_v2.params.get("cov12_lag", np.nan),
            "p_cov12": its_v2.pvalues.get("cov12_lag", np.nan),
        },
        {
            "Modèle": "SARIMAX v2 (1,0,0)x(1,1,0,52)",
            "AIC": safe_metric(sarimax_v2, "aic"),
            "BIC": safe_metric(sarimax_v2, "bic"),
            "R²_adj": np.nan,
            "DW": durbin_watson(sarimax_v2.resid),
            "β_cov12": sarimax_v2.params.get("cov12_lag", np.nan),
            "p_cov12": sarimax_v2.pvalues.get("cov12_lag", np.nan),
        },
    ])
    synth["β_cov12 (+10pp)"] = synth["β_cov12"] * 0.1
    synth_display = synth.copy()
    synth_display[["AIC","BIC","R²_adj","DW","β_cov12","p_cov12","β_cov12 (+10pp)"]] = \
        synth_display[["AIC","BIC","R²_adj","DW","β_cov12","p_cov12","β_cov12 (+10pp)"]].round(3)
    st.dataframe(synth_display, use_container_width=True)

    # =============================
    # 7) Scénarios contrefactuels complets (ITS v2)
    # =============================
    st.subheader("🎯 Scénarios contrefactuels complets (modèle ITS v2)")
    st.caption("Combinaisons: COVID/NoCOVID × MNP: real/none/maintained × Vaccine/NoVaccine. Scénarios *illogiques* (ex. vaccin sans COVID) peuvent être filtrés.")

    coef_its = its_v2.params.copy()
    its_cols = ["const"] + X_cols_its

    def predict_with_params(dfX: pd.DataFrame, params: pd.Series) -> pd.Series:
        X = sm.add_constant(dfX[X_cols_its], has_constant="add")
        # réalignement robuste colonnes/params
        common = X.columns.intersection(params.index)
        return pd.Series(np.dot(X[common], params[common]), index=X.index)

    def simulate_its(df_ref: pd.DataFrame, *, covid: bool, mnp: str, vacc: bool) -> pd.Series:
        """
        covid: True/False -> post_covid & t_postcovid
        mnp: 'real' | 'none' | 'maintained'
        vacc: True/False -> post_vacc & cov12_lag
        """
        dfS = df_ref.copy()

        # COVID
        if not covid:
            dfS["post_covid"] = 0
            dfS["t_postcovid"] = 0
        else:
            # conservé tel quel
            pass

        # Vaccination
        if not vacc:
            dfS["post_vacc"] = 0
            dfS["cov12_lag"] = 0

        # MNP handling
        if mnp == "none":
            dfS["MNP_lag"] = 0
        elif mnp == "maintained":
            # niveau élevé stable = p90 des années 2020–2021 (si dispo), sinon max
            try:
                p90 = df_ref.loc[(df_ref.index>=pd.Timestamp("2020-01-01")) &
                                 (df_ref.index<=pd.Timestamp("2021-12-31")),"MNP_lag"].quantile(0.9)
                if np.isnan(p90): p90 = df_ref["MNP_lag"].max()
            except:
                p90 = df_ref["MNP_lag"].max()
            dfS["MNP_lag"] = p90
        # 'real' => inchangé

        yhat = predict_with_params(dfS, coef_its)
        yhat.name = f"sim_{covid}_{mnp}_{vacc}"
        return yhat

    # Fenêtre scénario = fenêtre d’estimation ITS (pour cohérence)
    df_cf = df_its.copy()

    # Génération de toutes les combinaisons
    options_covid = [True, False]
    options_mnp   = ["real","none","maintained"]
    options_vacc  = [True, False]

    scenarios_all = {}
    scenarios_all["Observed"] = df_cf["RSV_det"]
    scenarios_all["Fitted (ITS v2)"] = its_v2.fittedvalues

    for c in options_covid:
        for m in options_mnp:
            for v in options_vacc:
                name = f"{'COVID' if c else 'NoCOVID'} • MNP:{m} • {'Vaccine' if v else 'NoVaccine'}"
                scenarios_all[name] = simulate_its(df_cf, covid=c, mnp=m, vacc=v)

    # Filtre de logique (optionnel : vaccin sans COVID = illogique, MNP:real sans COVID = illogique)
    def is_logical(name: str) -> bool:
        if name in ["Observed","Fitted (ITS v2)"]:
            return True
        c, m, v = name.split(" • ")
        covid = (c == "COVID")
        mnp   = m.split(":")[1]
        vacc  = (v == "Vaccine")
        if not covid:
            # pas de vaccin sans COVID + MNP 'real' n'a pas de sens sans COVID
            if vacc: return False
            if mnp == "real": return False
        return True

    show_only_logical = st.checkbox("Afficher uniquement les scénarios cohérents (recommandé)", True)
    scenarios_kept = {k: v for k, v in scenarios_all.items() if (is_logical(k) if show_only_logical else True)}

    # Sélection à tracer
    default_sel = [k for k in scenarios_kept.keys() if k in [
        "Observed","Fitted (ITS v2)",
        "NoCOVID • MNP:none • NoVaccine",
        "NoCOVID • MNP:maintained • NoVaccine",
        "COVID • MNP:real • Vaccine",
        "COVID • MNP:none • Vaccine",
        "COVID • MNP:maintained • Vaccine",
        "COVID • MNP:real • NoVaccine",
    ]]
    to_plot = st.multiselect(
        "Choisir les séries à afficher",
        list(scenarios_kept.keys()),
        default=default_sel
    )

    # Figure scénarios
    fig_scen = go.Figure()
    # Observé complet pour contexte
    fig_scen.add_trace(go.Scatter(
        x=df_erv.index, y=df_erv["RSV_det"], mode="lines",
        name="Observed (ERVISS, full)", line=dict(color="black", width=2)
    ))
    palette = {
        "Fitted (ITS v2)": "#1F77B4",
        "NoCOVID • MNP:none • NoVaccine": "#D62728",
        "NoCOVID • MNP:maintained • NoVaccine": "#2CA02C",
        "COVID • MNP:real • Vaccine": "#9467BD",
        "COVID • MNP:none • Vaccine": "#FF7F0E",
        "COVID • MNP:maintained • Vaccine": "#17BECF",
        "COVID • MNP:real • NoVaccine": "#8C564B",
        "COVID • MNP:none • NoVaccine": "#BCBD22",
        "COVID • MNP:maintained • NoVaccine": "#7F7F7F",
    }
    for name in to_plot:
        if name == "Observed":
            fig_scen.add_trace(go.Scatter(
                x=scenarios_kept[name].index, y=scenarios_kept[name].values, mode="lines",
                name="Observed (ITS window)", line=dict(color="black", width=2, dash="dot")
            ))
            continue
        color = palette.get(name, None)
        fig_scen.add_trace(go.Scatter(
            x=scenarios_kept[name].index, y=scenarios_kept[name].values, mode="lines",
            name=name, line=dict(width=2, dash="dash") if name!="Fitted (ITS v2)" else dict(width=2, dash="dot", color=color),
        ))

    fig_scen.add_vline(x=COVID_START, line_dash="dash", line_color="red")
    fig_scen.add_vline(x=VACC_START,  line_dash="dash", line_color="green")
    fig_scen.update_layout(
        title="Scénarios contrefactuels (ITS v2) — fenêtre modèle",
        xaxis_title="Semaine", yaxis_title="Détections RSV (modèle ITS)",
        template="plotly_white", height=560
    )
    st.plotly_chart(fig_scen, use_container_width=True)

    # Tableau Δ cumulés (vs Observed sur la fenêtre modèle)
    observed_sum = scenarios_kept.get("Observed", df_cf["RSV_det"]).sum()
    rows = []
    for name, series in scenarios_kept.items():
        total = float(np.nansum(series))
        delta = total - observed_sum
        pct   = (delta / observed_sum) * 100 if observed_sum != 0 else np.nan
        rows.append({"Scenario": name, "Δ RSV cumulé": round(delta, 0), "Variation (%)": round(pct, 1)})
    df_delta = pd.DataFrame(rows).sort_values("Δ RSV cumulé", ascending=False).reset_index(drop=True)

    st.markdown("**Δ cumulés par scénario (vs Observed, fenêtre ITS)**")
    st.dataframe(df_delta, use_container_width=True)

    # Bar chart Δ cumulés (hors Observed/Fitted)
    df_bar = df_delta[~df_delta["Scenario"].isin(["Observed","Fitted (ITS v2)"])].copy()
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=df_bar["Scenario"], y=df_bar["Δ RSV cumulé"], text=df_bar["Variation (%)"].astype(str) + "%",
        textposition="outside"
    ))
    fig_bar.update_layout(
        title="Δ cumulés de RSV par scénario (vs Observed)",
        xaxis_tickangle=-30, template="plotly_white", height=520
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # =============================
    # 8) Diagnostics rapides
    # =============================
    st.subheader("🔎 Diagnostics rapides")
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb_its = acorr_ljungbox(its_v2.resid, lags=[8,12,24], return_df=True)["lb_pvalue"].round(4).to_dict()
    lb_sx  = acorr_ljungbox(sarimax_v2.resid, lags=[8,12,24], return_df=True)["lb_pvalue"].round(4).to_dict()
    st.write("**Ljung–Box p-values (ITS v2)** :", lb_its)
    st.write("**Ljung–Box p-values (SARIMAX v2)** :", lb_sx)

    st.info("""
    **Notes méthodo :**
    - `cov12_lag` = couverture vaccinale COVID (t-4), `MNP_lag` = score gestes barrières (t-8),
      `work_lag` = mobilité lieux de travail (t-9), `sin52`,`cos52` = saison 52 semaines.
    - ITS v2 inclut (niveau + pente) post-COVID, niveau post-vaccination.
    - SARIMAX v2 applique une **différenciation saisonnière (D=1, s=52)** pour stabiliser la saisonnalité.
    - Les scénarios "illogiques" (ex. vaccine sans COVID, MNP=real sans COVID) peuvent être filtrés.
    """)
