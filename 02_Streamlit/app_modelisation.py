# ==========================================
# 📊 APPLICATION STREAMLIT — RSV MODELS (Optimisée)
# ==========================================
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from itertools import product
import warnings
from statsmodels.stats.diagnostic import acorr_ljungbox

st.set_page_config(page_title="Analyse RSV - Modèles et Scénarios", layout="wide")
st.title("🧬 Analyse temporelle du RSV (France, 2018–2025)")

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 180)
np.random.seed(42)
px.defaults.template = "plotly_white"
px.defaults.width = 1000
px.defaults.height = 520

tabs = st.tabs([
    "🧱 Bloc 1 — Setup & Données",
    "📈 Bloc 2 — Modèle OLS",
    "⛔ Bloc 3 — ITS",
    "🔁 Bloc 4 — SARIMAX",
    "📊 Bloc 5 — Performances",
    "🧩 Bloc 6 — Synthèse",
    "🎭 Bloc 7 — Scénarios contrefactuels",
    "🔮 Bloc 8 — Prévisions 2025–2027",
    "📉 Bloc 9 — Prévisions univariées"
])
# ==========================================
# 🧱 BLOC 1 — Setup, Helpers & Chargement Données + Modèles
# ==========================================
with tabs[0]:
    st.header("🧱 BLOC 1 — Setup, Helpers & Chargement Données / Modèles")

    # --- Dates jalons ---
    COVID_START = pd.Timestamp("2020-03-01")
    VACC_START  = pd.Timestamp("2021-01-01")
    LAG_VACC, LAG_MNP, LAG_WORK = 4, 8, 9
    SEASON_PERIOD = 52
    st.markdown(f"⏱️ **COVID_START =** {COVID_START.date()}, **VACC_START =** {VACC_START.date()} — Lags: `{(LAG_VACC, LAG_MNP, LAG_WORK)}`")

    # --- Chemins robustes (essaie ./ puis ../) ---
    def first_existing(*paths):
        for p in paths:
            if p.exists():
                return p
        return paths[0]

    DATA = first_existing(Path("./data_clean"), Path("../data_clean"))
    MODELS_DIR = first_existing(Path("./models"), Path("../models"))

    FILES = {
        "common_FR_long": DATA / "ODISSEE/common_FR_long.csv",
        "vacsi_fr_extended": DATA / "VACSI/vacsi_fr_extended.csv",
        "google_mobility_fr_weekly": DATA / "GOOGLE/google_mobility_fr_weekly.csv",
        "coviprev_reg_weekly": DATA / "COVIPREV/coviprev_reg_weekly.csv",
        "meteo_fr_weekly": DATA / "METEO/meteo_fr_weekly.csv",
        "erviss_fr_weekly": DATA / "ERVISS/erviss_fr_weekly.csv",
    }

    # --- Helpers légers ---
    def keyify(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        iso = pd.to_datetime(df["date_monday"]).dt.isocalendar()
        df["year_iso"] = iso["year"].astype(int)
        df["week_iso_num"] = iso["week"].astype(int)
        return df

    def zscore(s):
        std = s.std(ddof=0)
        return (s - s.mean()) / std if std != 0 else s * 0

    def build_time_features(df, period=52):
        df = df.copy()
        df["t"] = np.arange(len(df))
        df["sin52"] = np.sin(2 * np.pi * df["t"] / period)
        df["cos52"] = np.cos(2 * np.pi * df["t"] / period)
        return df

    def merge_exog(rsv_df, vac_df, work_df, cov_df):
        merged = (
            rsv_df[["date_monday", "year_iso", "week_iso_num"]]
            .merge(vac_df, on=["year_iso", "week_iso_num"], how="left")
            .merge(work_df, on=["year_iso", "week_iso_num"], how="left")
            .merge(cov_df, on=["year_iso", "week_iso_num"], how="left")
            .set_index("date_monday")
            .sort_index()
        )
        return merged

    def build_model_matrix(df, lags=(4, 8, 9), mask_vars=None):
        df = df.copy()
        lag_vac, lag_mnp, lag_work = lags
        df["work_red"] = zscore(-df["work"])
        if mask_vars:
            for v in mask_vars:
                df[v] = zscore(df[v])
            df["MNP_score"] = df[mask_vars + ["work_red"]].mean(axis=1)
        else:
            df["MNP_score"] = zscore(df["work_red"])

        X = pd.DataFrame(index=df.index)
        X["cov12_lag"] = df["couv_complet"].shift(lag_vac)
        X["MNP_lag"]   = df["MNP_score"].shift(lag_mnp)
        X["work_lag"]  = df["work"].shift(lag_work)
        return build_time_features(X)

    # --- Cache : chargement datasets & base finale ---
    @st.cache_data(show_spinner=False)
    def load_datasets(files: dict) -> dict:
        data = {}
        for name, path in files.items():
            data[name] = pd.read_csv(path)
        return data

    @st.cache_data(show_spinner=False)
    def build_base_tables(data, LAG_VACC, LAG_MNP, LAG_WORK):
        common = keyify(data["common_FR_long"])
        mask = (common["topic"] == "RSV") & (common["geo_level"] == "FR")
        age_used = next(a for a in ["00-04 ans", "0-1 an", "Tous âges"] if ((mask) & (common["classe_d_age"] == a)).any())
        mask &= (common["classe_d_age"] == age_used)
        ycol = "taux_passages_urgences" if "taux_passages_urgences" in common.columns else "taux_sos"

        rsv = common.loc[mask, ["date_monday", "year_iso", "week_iso_num", ycol]].rename(columns={ycol: "RSV"})
        rsv["date_monday"] = pd.to_datetime(rsv["date_monday"])
        rsv = rsv.sort_values("date_monday")

        vac = keyify(data["vacsi_fr_extended"]).query("geo_level=='FR'")[["year_iso","week_iso_num","couv_complet"]]
        gm  = keyify(data["google_mobility_fr_weekly"])
        work = gm.query("geo_level=='FR' & indicator=='workplaces'")[["year_iso","week_iso_num","value"]].rename(columns={"value": "work"})

        cov = keyify(data["coviprev_reg_weekly"])
        mask_vars = ["port_du_masque","lavage_des_mains","aeration_du_logement","saluer_sans_serrer_la_main"]
        cov_nat = cov[cov["indicator"].isin(mask_vars)].groupby(["year_iso","week_iso_num","indicator"])["value"].mean().unstack()

        X_base = merge_exog(rsv, vac, work, cov_nat)
        X_full = build_model_matrix(X_base, lags=(LAG_VACC, LAG_MNP, LAG_WORK), mask_vars=mask_vars)
        df_base = rsv.set_index("date_monday")[["RSV"]].join(X_full, how="left").dropna().sort_index()

        # enrichissement météo pour df_opt
        meteo = keyify(data["meteo_fr_weekly"])[["year_iso","week_iso_num","tmean"]]
        df_opt = keyify(df_base.reset_index()).merge(meteo, on=["year_iso","week_iso_num"], how="left").set_index("date_monday").sort_index()
        df_opt["tmean_z"]    = zscore(df_opt["tmean"])
        df_opt["vacc_x_mnp"] = df_opt["cov12_lag"] * df_opt["MNP_lag"]
        df_opt["RSV_lag1"]   = df_opt["RSV"].shift(1)
        df_opt["RSV_lag2"]   = df_opt["RSV"].shift(2)
        df_opt = df_opt.dropna()

        return rsv, df_base, df_opt, X_base, mask_vars, age_used

    # --- Exécution (chargement) ---
    data_keys = ["rsv", "df_base", "df_opt", "X_base", "mask_vars", "age_used"]
    need_data_load = any(k not in st.session_state for k in data_keys)

    if need_data_load:
        missing = [k for k, p in FILES.items() if not p.exists()]
        if missing:
            st.error(f"❌ Fichiers manquants: {missing}")
            st.stop()

        with st.spinner("Chargement des datasets & préparation des bases..."):
            data = load_datasets(FILES)
            rsv, df_base, df_opt, X_base, mask_vars, age_used = build_base_tables(
                data, LAG_VACC, LAG_MNP, LAG_WORK
            )

        st.session_state["rsv"] = rsv
        st.session_state["df_base"] = df_base
        st.session_state["df_opt"] = df_opt
        st.session_state["X_base"] = X_base
        st.session_state["mask_vars"] = mask_vars
        st.session_state["age_used"] = age_used
        st.session_state["raw_data"] = data

        st.success(f"✅ RSV prêt ({age_used}) — {rsv.shape[0]} lignes")
    else:
        data = st.session_state.get("raw_data", {})
        st.caption("♻️ Données récupérées depuis la session (pas de rechargement).")

    rsv = st.session_state["rsv"]
    df_base = st.session_state["df_base"]
    df_opt = st.session_state["df_opt"]
    X_base = st.session_state["X_base"]
    mask_vars = st.session_state["mask_vars"]
    age_used = st.session_state["age_used"]

    st.caption(f"df_base: {df_base.shape}, df_opt: {df_opt.shape}")

    # --- Chargement des modèles enregistrés (sans recalcul) ---
    import joblib
    MODEL_FILES = {
        "ols_base": MODELS_DIR / "ols_base.pkl",
        "ols_opt": MODELS_DIR / "ols_opt.pkl",
        "ols_causal": MODELS_DIR / "ols_causal.pkl",
        "its": MODELS_DIR / "its_base.pkl",
        "its_best": MODELS_DIR / "its_best.pkl",
        "sarimax_base": MODELS_DIR / "sarimax_base.pkl",
        "sarimax_best": MODELS_DIR / "sarimax_best.pkl",
    }

    if "MODELS" not in st.session_state:
        loaded = {}
        with st.spinner("Chargement des modèles sauvegardés..."):
            for k, p in MODEL_FILES.items():
                if p.exists():
                    try:
                        loaded[k] = joblib.load(p)
                        st.success(f"✅ {k} chargé ({p.name})")
                    except Exception as e:
                        st.warning(f"⚠️ {k} illisible ({e})")
                else:
                    st.info(f"ℹ️ {k} non trouvé ({p})")

        st.session_state["MODELS"] = loaded
        st.success("📦 Chargement terminé.")
    else:
        st.caption("♻️ Modèles récupérés depuis la session.")
# ==========================================
# 📈 BLOC 2 — Modèle OLS (base + optimisé) — sans recalcul lourd
# ==========================================
with tabs[1]:
    st.header("📈 BLOC 2 — Modèle OLS (base + optimisé)")

    rsv      = st.session_state["rsv"]
    df_base  = st.session_state["df_base"]
    df_opt   = st.session_state["df_opt"]
    MODELS   = st.session_state["MODELS"]

    # 1) OLS base
    ols_base = MODELS.get("ols_base", None)
    if ols_base is None:
        # calcul léger si modèle manquant (pas de grid)
        Y = df_base["RSV"].astype(float)
        Xb = df_base[["cov12_lag","MNP_lag","work_lag","sin52","cos52"]]
        ols_base = sm.OLS(Y, sm.add_constant(Xb)).fit(cov_type="HC3")
        st.info("ℹ️ OLS base recalculé (modèle non trouvé).")
    st.markdown(f"**OLS base** — R²_adj = `{ols_base.rsquared_adj:.3f}` | AIC = `{ols_base.aic:.1f}`")

    # 2) OLS optimisé + causal (chargés)
    ols_opt   = MODELS.get("ols_opt", None)
    ols_caus  = MODELS.get("ols_causal", None)

    # Xo & Xo_causal (reconstruits à partir de df_opt — rapide)
    Xo = df_opt[["cov12_lag","MNP_lag","work_lag","tmean_z","vacc_x_mnp","RSV_lag1","RSV_lag2","sin52","cos52"]]
    if ols_opt is None:
        ols_opt = sm.OLS(df_opt["RSV"], sm.add_constant(Xo)).fit(cov_type="HC3")
        st.info("ℹ️ OLS optimisé recalculé (modèle non trouvé).")

    LAG_MNP_EFFECT = 3
    df_opt_c = df_opt.copy()
    df_opt_c["MNP_lag_causal"] = df_opt_c["MNP_lag"].shift(LAG_MNP_EFFECT)
    df_opt_c["vacc_x_mnp_causal"] = df_opt_c["cov12_lag"] * df_opt_c["MNP_lag_causal"]
    Xo_causal = df_opt_c[[
        "cov12_lag","MNP_lag_causal","work_lag","tmean_z",
        "vacc_x_mnp_causal","RSV_lag1","RSV_lag2","sin52","cos52"
    ]].dropna()

    if ols_caus is None and len(Xo_causal) > 0:
        ols_caus = sm.OLS(df_opt_c.loc[Xo_causal.index, "RSV"], sm.add_constant(Xo_causal)).fit(cov_type="HC3")
        st.info("ℹ️ OLS causal recalculé (modèle non trouvé).")

    # --- Plots (mis en cache) ---
    @st.cache_data(show_spinner=False)
    def make_ols_plot(rsv, _df_idx, y_fit, title):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rsv["date_monday"], y=rsv["RSV"], name="RSV observé", mode="lines"))
        fig.add_trace(go.Scatter(x=_df_idx, y=y_fit, name="Fitted", mode="lines", line=dict(dash="dot")))
        fig.add_vline(x=pd.Timestamp("2020-03-01"), line_dash="dash", line_color="red")
        fig.add_vline(x=pd.Timestamp("2021-01-01"), line_dash="dash", line_color="green")
        fig.update_layout(title=title, xaxis_title="Semaine", yaxis_title="RSV")
        return fig

    st.plotly_chart(
        make_ols_plot(rsv, df_base.index, ols_base.fittedvalues, "RSV — OLS de base"),
        use_container_width=True
    )

    st.plotly_chart(
        make_ols_plot(rsv, df_opt.index, ols_opt.fittedvalues, "RSV — OLS optimisé"),
        use_container_width=True
    )

    if ols_caus is not None:
        st.markdown(f"**OLS causal (lag MNP +3)** — R²_adj = `{ols_caus.rsquared_adj:.3f}` | AIC = `{ols_caus.aic:.1f}`")
# ==========================================
# ⛔ BLOC 3 — ITS (base + optimisé) — sans grid-search par défaut
# ==========================================
with tabs[2]:
    st.header("⛔ BLOC 3 — Séries interrompues (ITS)")

    df_base = st.session_state["df_base"]
    rsv     = st.session_state["rsv"]
    MODELS  = st.session_state["MODELS"]

    its      = MODELS.get("its", None)
    its_best = MODELS.get("its_best", None)

    # Si ITS manquants, on fait un fit simple (léger)
    if its is None:
        df_its = df_base.reset_index().sort_values("date_monday").copy()
        df_its["t"] = np.arange(len(df_its))
        df_its["post_covid"] = (df_its["date_monday"] >= pd.Timestamp("2020-03-01")).astype(int)
        df_its["post_vacc"]  = (df_its["date_monday"] >= pd.Timestamp("2021-01-01")).astype(int)
        df_its["t_post_covid"] = df_its["t"] * df_its["post_covid"]
        df_its["t_post_vacc"]  = df_its["t"] * df_its["post_vacc"]
        X = df_its[["t","sin52","cos52","post_covid","t_post_covid","post_vacc","t_post_vacc"]]
        y = df_its["RSV"].astype(float)
        its = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags":12})
        st.info("ℹ️ ITS base recalculé (modèle non trouvé).")

    st.markdown(f"**ITS (base)** — AIC = `{its.aic:.1f}` | BIC = `{its.bic:.1f}`")

    # tracé ITS base
    @st.cache_data(show_spinner=False)
    def make_its_plot(rsv, fitted, dates, title):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rsv["date_monday"], y=rsv["RSV"], mode="lines",
                                 name="RSV observé", line=dict(width=2)))
        fig.add_trace(go.Scatter(x=dates, y=fitted, mode="lines",
                                 name="ITS fitted", line=dict(dash="dot", width=2)))
        fig.add_vline(x=pd.Timestamp("2020-03-01"), line_dash="dash", line_color="red")
        fig.add_vline(x=pd.Timestamp("2021-01-01"), line_dash="dash", line_color="green")
        fig.update_layout(title=title, xaxis_title="Semaine", yaxis_title="RSV")
        return fig

    @st.cache_data(show_spinner=False)
    def cached_ljungbox(residuals):
        return acorr_ljungbox(residuals, lags=[8, 12, 24], return_df=True)[["lb_stat", "lb_pvalue"]]

    # dates alignées
    df_its_plot = df_base.reset_index().sort_values("date_monday")
    st.plotly_chart(
        make_its_plot(rsv, its.fittedvalues, df_its_plot["date_monday"], "ITS (base) — Observé vs Ajusté"),
        use_container_width=True
    )

    # ITS optimisé chargé (sinon on skip)
    if its_best is not None:
        dw = sm.stats.stattools.durbin_watson(its_best.resid)
        lb = cached_ljungbox(its_best.resid)
        st.markdown(f"**ITS (optimisé)** — AIC = `{its_best.aic:.1f}` | DW = `{dw:.3f}`")
        st.dataframe(lb)

        # On suppose que les fitted sont alignés sur df_base (même période)
        st.plotly_chart(
            make_its_plot(rsv, its_best.fittedvalues, df_base.index, "ITS (optimisé) — Observé vs Ajusté"),
            use_container_width=True
        )
    else:
        st.info("ℹ️ ITS optimisé non disponible (fichier .pkl manquant).")
# ==========================================
# 🔁 BLOC 4 — SARIMAX (base + optimisé) — pas de grid-search par défaut
# ==========================================
with tabs[3]:
    st.header("🔁 BLOC 4 — SARIMAX (base + optimisé)")

    from statsmodels.tsa.statespace.sarimax import SARIMAX

    df_opt = st.session_state["df_opt"]
    rsv    = st.session_state["rsv"]
    MODELS = st.session_state["MODELS"]

    # exogènes pour affichage / prévisions
    exog_cols = [
        "cov12_lag","MNP_lag","work_lag","tmean_z","vacc_x_mnp"
    ]
    # Variables ITS "structurelles"
    df_sx = df_opt.copy().sort_index()
    df_sx.index = pd.to_datetime(df_sx.index)
    df_sx["post_covid"] = (df_sx.index >= pd.Timestamp("2020-03-01")).astype(int)
    df_sx["post_vacc"]  = (df_sx.index >= pd.Timestamp("2021-01-01")).astype(int)
    df_sx["t"] = np.arange(len(df_sx))
    df_sx["t_post_covid"] = df_sx["t"] * df_sx["post_covid"]
    exog_cols_full = exog_cols + ["post_covid","post_vacc","t_post_covid","t"]

    y = df_sx["RSV"].astype(float)
    exog = df_sx[exog_cols_full].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    mask = (~y.isna()) & (~exog.isna().any(axis=1))
    y, exog = y.loc[mask], exog.loc[mask]

    sarimax_base = MODELS.get("sarimax_base", None)
    sarimax_best = MODELS.get("sarimax_best", None)

    if sarimax_base is None:
        # fit minimal si manquant (une config simple)
        sarimax_base = SARIMAX(y, exog=exog, order=(1,1,1), seasonal_order=(1,1,1,52),
                               enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        st.info("ℹ️ SARIMAX base recalculé (modèle non trouvé).")

    y_fit_base = sarimax_base.fittedvalues.reindex(y.index)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rsv["date_monday"], y=rsv["RSV"], name="RSV Observé", mode="lines", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=y_fit_base.index, y=y_fit_base, name="SARIMAX base (fitted)", mode="lines", line=dict(dash="dot")))
    fig.add_vline(x=pd.Timestamp("2020-03-01"), line_dash="dash", line_color="red")
    fig.add_vline(x=pd.Timestamp("2021-01-01"), line_dash="dash", line_color="green")
    fig.update_layout(title="SARIMAX base — Observé vs Ajusté", xaxis_title="Semaine", yaxis_title="RSV")
    st.plotly_chart(fig, use_container_width=True)

    if sarimax_best is not None:
        y_fit = sarimax_best.fittedvalues.reindex(y.index)
        resid = sarimax_best.resid
        dw = sm.stats.stattools.durbin_watson(resid)
        lb = cached_ljungbox(resid)
        st.markdown(f"**SARIMAX optimisé** — AIC = `{sarimax_best.aic:.1f}` | BIC = `{sarimax_best.bic:.1f}` | DW = `{dw:.3f}`")
        st.dataframe(lb)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=rsv["date_monday"], y=rsv["RSV"], name="RSV Observé", mode="lines", line=dict(width=2)))
        fig2.add_trace(go.Scatter(x=y_fit.index, y=y_fit, name="SARIMAX optimisé (fitted)", mode="lines", line=dict(dash="dot")))
        fig2.add_vline(x=pd.Timestamp("2020-03-01"), line_dash="dash", line_color="red")
        fig2.add_vline(x=pd.Timestamp("2021-01-01"), line_dash="dash", line_color="green")
        fig2.update_layout(title="SARIMAX optimisé — Observé vs Ajusté", xaxis_title="Semaine", yaxis_title="RSV")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("ℹ️ SARIMAX optimisé non disponible (fichier .pkl manquant).")
# ==========================================
# 📊 BLOC 5 — Récapitulatif des performances modèles (léger)
# ==========================================
with tabs[4]:
    st.header("📊 BLOC 5 — Récapitulatif des performances des modèles")

    MODELS = st.session_state["MODELS"]
    df_opt = st.session_state["df_opt"]
    rsv    = st.session_state["rsv"]

    # Récup sécurisée (fallback si manquant)
    ols_base     = MODELS.get("ols_base")
    ols_opt      = MODELS.get("ols_opt")
    its          = MODELS.get("its")
    its_best     = MODELS.get("its_best")
    sarimax_base = MODELS.get("sarimax_base")
    sarimax_best = MODELS.get("sarimax_best")

    # Pseudo-R² SARIMAX optimisé (si dispo)
    pseudo_r2 = np.nan
    if sarimax_best is not None:
        y = df_opt["RSV"].astype(float)
        y_fit = sarimax_best.fittedvalues.reindex(y.index)
        m = y.dropna().align(y_fit.dropna(), join="inner")
        if len(m[0]) > 5:
            ss_res = ((m[0] - m[1])**2).sum()
            ss_tot = ((m[0] - m[0].mean())**2).sum()
            pseudo_r2 = 1 - ss_res/ss_tot

    def safe(v, attr, default=np.nan):
        try:
            return getattr(v, attr)
        except Exception:
            return default

    model_perf = pd.DataFrame([
        {"Modèle":"OLS (base)","R²_adj":safe(ols_base,"rsquared_adj"),"AIC":safe(ols_base,"aic"),
         "BIC":safe(ols_base,"bic"),"Durbin-Watson":(sm.stats.stattools.durbin_watson(ols_base.resid) if ols_base else np.nan),"Type":"Régression"},
        {"Modèle":"OLS (optimisé)","R²_adj":safe(ols_opt,"rsquared_adj"),"AIC":safe(ols_opt,"aic"),
         "BIC":safe(ols_opt,"bic"),"Durbin-Watson":(sm.stats.stattools.durbin_watson(ols_opt.resid) if ols_opt else np.nan),"Type":"Régression"},
        {"Modèle":"ITS (base)","R²_adj":safe(its,"rsquared_adj"),"AIC":safe(its,"aic"),
         "BIC":safe(its,"bic"),"Durbin-Watson":(sm.stats.stattools.durbin_watson(its.resid) if its else np.nan),"Type":"Rupture"},
        {"Modèle":"ITS (optimisé)","R²_adj":safe(its_best,"rsquared_adj"),"AIC":safe(its_best,"aic"),
         "BIC":safe(its_best,"bic"),"Durbin-Watson":(sm.stats.stattools.durbin_watson(its_best.resid) if its_best else np.nan),"Type":"Rupture"},
        {"Modèle":"SARIMAX (base)","R²_adj":np.nan,"AIC":safe(sarimax_base,"aic"),
         "BIC":safe(sarimax_base,"bic"),"Durbin-Watson":(sm.stats.stattools.durbin_watson(sarimax_base.resid) if sarimax_base else np.nan),"Type":"Série temporelle"},
        {"Modèle":"SARIMAX (optimisé)","R²_adj":pseudo_r2,"AIC":safe(sarimax_best,"aic"),
         "BIC":safe(sarimax_best,"bic"),"Durbin-Watson":(sm.stats.stattools.durbin_watson(sarimax_best.resid) if sarimax_best else np.nan),"Type":"Série temporelle"},
    ]).round(3)

    st.dataframe(model_perf, use_container_width=True)
    st.success("✅ Tableau de performances généré.")

    # Barres (sans text_auto)
    fig_r2 = px.bar(model_perf.dropna(subset=["R²_adj"]), x="Modèle", y="R²_adj", color="Type",
                    title="Pouvoir explicatif (R² ajusté / pseudo-R²)")
    st.plotly_chart(fig_r2, use_container_width=True)

    fig_ic = go.Figure()
    fig_ic.add_trace(go.Bar(x=model_perf["Modèle"], y=model_perf["AIC"], name="AIC"))
    fig_ic.add_trace(go.Bar(x=model_perf["Modèle"], y=model_perf["BIC"], name="BIC"))
    fig_ic.update_layout(title="Critères d'information (AIC / BIC)", barmode="group")
    st.plotly_chart(fig_ic, use_container_width=True)

    fig_dw = px.bar(model_perf, x="Modèle", y="Durbin-Watson", color="Type",
                    title="Autocorrélation des résidus (Durbin–Watson)")
    fig_dw.add_hrect(y0=1.5, y1=2.5, fillcolor="lightgreen", opacity=0.3, line_width=0)
    st.plotly_chart(fig_dw, use_container_width=True)
# ==========================================
# 🧩 BLOC 6 — Synthèse visuelle
# ==========================================
with tabs[5]:
    st.header("🧩 BLOC 6 — Synthèse visuelle et interprétation")
    st.markdown("""
    **Pourquoi mobiliser plusieurs modèles ?**
    - **OLS (régression)** : mesure l'effet marginal des leviers (vaccination, gestes barrières, météo) en supposant une relation stable – pratique pour l'interprétation causale.
    - **ITS (Interrupted Time Series)** : capture explicitement les ruptures de tendance/niveau (Covid, vaccination) et aide à quantifier l'impact structurel de chaque jalon.
    - **SARIMAX** : combine l'inertie temporelle (composante ARIMA) avec les exogènes ; c'est le plus adapté pour les simulations dynamiques et les prévisions.

    Utiliser ces trois angles évite de confondre interprétation structurelle, estimation d'effets ponctuels et capacité prédictive. Quand les conclusions convergent, on renforce la confiance dans les résultats ; quand elles divergent, cela signale des hypothèses à vérifier.
    """)
# ==========================================
# 🎭 BLOC 7 — Scénarios contrefactuels dynamiques (OLS / ITS / SARIMAX)
# ==========================================
with tabs[6]:
    st.header("🎭 BLOC 7 — Scénarios contrefactuels dynamiques")

    MODELS = st.session_state["MODELS"]
    df_opt = st.session_state["df_opt"]
    rsv    = st.session_state["rsv"]

    model_choice = st.selectbox(
        "🧠 Modèle utilisé pour la simulation contrefactuelle",
        ["OLS optimisé", "ITS optimisé", "SARIMAX optimisé"],
        index=0
    )

    # Sliders (légers)
    vacc_factor = st.slider("Vaccination ×", 0.0, 2.0, 1.0, 0.1)
    mnp_factor  = st.slider("Gestes barrières (MNP) ×", 0.0, 2.0, 1.0, 0.1)
    scale_strength = st.slider("Amplitude effet dynamique", 0.0, 1.0, 0.5, 0.05)
    st.caption(f"Paramètres personnalisés — Vaccination ×{vacc_factor:.2f}, MNP ×{mnp_factor:.2f}, intensité = {scale_strength:.2f}")

    if model_choice == "OLS optimisé":
        ols_caus = MODELS.get("ols_causal")
        ols_opt  = MODELS.get("ols_opt")
        model_used = ols_caus if ols_caus is not None else ols_opt
        if model_used is None:
            st.error("❌ Aucun modèle OLS chargé. Va au Bloc 2.")
            st.stop()

        cols_needed = ["cov12_lag","MNP_lag","work_lag","tmean_z","vacc_x_mnp","RSV_lag1","RSV_lag2","sin52","cos52"]
        if ols_caus is not None:
            df_c = df_opt.copy()
            df_c["MNP_lag_causal"] = df_c["MNP_lag"].shift(3)
            df_c["vacc_x_mnp_causal"] = df_c["cov12_lag"] * df_c["MNP_lag_causal"]
            X_used = ["cov12_lag","MNP_lag_causal","work_lag","tmean_z","vacc_x_mnp_causal","RSV_lag1","RSV_lag2","sin52","cos52"]
            base = df_c.dropna().copy()
        else:
            X_used = cols_needed
            base = df_opt.copy()

        def simulate_ols(ols_fit, df_exog):
            X_sim = sm.add_constant(df_exog[X_used], has_constant='add')
            return ols_fit.predict(X_sim)

        def build_dynamic_scenario(df, vacc=1.0, mnp=1.0, scale=0.4):
            df_new = df.copy()
            if "MNP_lag_causal" in df_new.columns:
                df_new["cov12_lag"] *= vacc
                df_new["MNP_lag_causal"] *= mnp
                if "vacc_x_mnp_causal" in df_new.columns:
                    df_new["vacc_x_mnp_causal"] = df_new["cov12_lag"] * df_new["MNP_lag_causal"]
            else:
                df_new["cov12_lag"] *= vacc
                df_new["MNP_lag"]   *= mnp
                if "vacc_x_mnp" in df_new.columns:
                    df_new["vacc_x_mnp"] = df_new["cov12_lag"] * df_new["MNP_lag"]
            hat = simulate_ols(model_used, df_new)
            dyn = 1 + scale * (mnp - 1) * (df["RSV"] / df["RSV"].max())
            return (hat * dyn).reindex(df.index)

        y_real = simulate_ols(model_used, base).reindex(base.index)
        y_custom   = build_dynamic_scenario(base, vacc_factor, mnp_factor, scale_strength)
        y_no_vacc  = build_dynamic_scenario(base, 0.0, 1.0, scale_strength)
        y_no_mnp   = build_dynamic_scenario(base, 1.0, 0.0, scale_strength)
        y_high_mnp = build_dynamic_scenario(base, 1.0, 1.5, scale_strength)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=base.index, y=base["RSV"], name="RSV observé", line=dict(width=2)))
        fig.add_trace(go.Scatter(x=base.index, y=y_real, name="RSV (modèle)", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=base.index, y=y_custom, name="Scénario personnalisé", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=base.index, y=y_no_vacc, name="Sans vaccination", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=base.index, y=y_no_mnp, name="Sans gestes barrières", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=base.index, y=y_high_mnp, name="MNP +50%", line=dict(dash="dot")))
        fig.update_layout(title="🎛️ Scénarios OLS", xaxis_title="Semaine", yaxis_title="RSV simulé")
        st.plotly_chart(fig, use_container_width=True)

    elif model_choice == "ITS optimisé":
        its_best = MODELS.get("its_best")
        if its_best is None:
            st.error("❌ ITS optimisé non disponible.")
            st.stop()

        # On reconstruit les exogènes du modèle ITS
        df = st.session_state["df_base"].reset_index().sort_values("date_monday").copy()
        df["t"] = np.arange(len(df))
        df["post_covid"] = (df["date_monday"] >= pd.Timestamp("2020-03-01")).astype(int)
        df["post_vacc"]  = (df["date_monday"] >= pd.Timestamp("2021-01-01")).astype(int)
        df["t_post_covid"] = df["t"] * df["post_covid"]
        df["t_post_vacc"]  = df["t"] * df["post_vacc"]

        exog_cols = [c for c in its_best.model.exog_names if c != "const"]
        X_actual = df[exog_cols]

        def predict_its(X):
            return its_best.predict(sm.add_constant(X))

        # Modulations simples des colonnes rupture
        X_custom = X_actual.copy()
        for c in ["post_covid","t_post_covid","post_vacc","t_post_vacc"]:
            if c in X_custom.columns:
                if "post" in c and "t_post" not in c:
                    X_custom[c] *= mnp_factor  # simple mapping visuel
                else:
                    X_custom[c] *= vacc_factor

        pred_custom = predict_its(X_custom)
        fitted = its_best.predict(sm.add_constant(X_actual))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["date_monday"], y=df["RSV"], name="RSV observé", line=dict(width=2)))
        fig.add_trace(go.Scatter(x=df["date_monday"], y=fitted, name="ITS (ajusté)", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=df["date_monday"], y=pred_custom, name="Scénario personnalisé", line=dict(dash="dot")))
        fig.update_layout(title="🎭 Scénarios ITS", xaxis_title="Semaine", yaxis_title="RSV simulé")
        st.plotly_chart(fig, use_container_width=True)

    else:
        sarimax_best = MODELS.get("sarimax_best")
        if sarimax_best is None:
            st.error("❌ SARIMAX optimisé non disponible.")
            st.stop()

        base_exog = sarimax_best.model.data.orig_exog
        model_index = sarimax_best.model.data.row_labels
        exog_names = sarimax_best.model.exog_names

        if base_exog is None or model_index is None or exog_names is None:
            st.error("❌ Impossible de récupérer les exogènes du modèle SARIMAX.")
            st.stop()

        model_index = pd.Index(model_index)
        exog_df = pd.DataFrame(base_exog, index=model_index, columns=exog_names).astype(float)

        def build_ramp(idx, start_date, scale):
            if isinstance(idx, pd.PeriodIndex):
                idx_ts = idx.to_timestamp()
            else:
                idx_ts = pd.to_datetime(idx)
            ramp = np.zeros(len(idx_ts))
            mask = idx_ts >= start_date
            if mask.any():
                ramp_vals = np.linspace(0, 1, mask.sum())
                ramp[mask] = ramp_vals
            return ramp * scale, idx_ts

        ramp_mnp, index_ts = build_ramp(exog_df.index, COVID_START, scale_strength)
        ramp_vacc, _ = build_ramp(exog_df.index, VACC_START, scale_strength)

        vacc_scale = float(np.nanstd(exog_df["cov12_lag"])) if "cov12_lag" in exog_df.columns else 0.0
        mnp_scale = float(np.nanstd(exog_df["MNP_lag"])) if "MNP_lag" in exog_df.columns else 0.0
        if vacc_scale == 0.0:
            median_vacc = abs(exog_df.get("cov12_lag", pd.Series([1.0]))).median()
            if pd.isna(median_vacc):
                median_vacc = 1.0
            vacc_scale = float(max(median_vacc, 1.0))
        if mnp_scale == 0.0:
            mnp_scale = 1.0

        coeff_vacc = float(sarimax_best.params.get("cov12_lag", -1.0)) if "cov12_lag" in sarimax_best.params else -1.0
        coeff_mnp = float(sarimax_best.params.get("MNP_lag", -1.0)) if "MNP_lag" in sarimax_best.params else -1.0
        vacc_direction = -1.0 if coeff_vacc > 0 else 1.0
        mnp_direction = -1.0 if coeff_mnp > 0 else 1.0

        def make_exog(df, vacc=1.0, mnp=1.0):
            mod = df.copy()
            if "cov12_lag" in mod.columns:
                mod["cov12_lag"] = mod["cov12_lag"] + vacc_direction * (vacc - 1.0) * ramp_vacc * vacc_scale
                mod["cov12_lag"] = mod["cov12_lag"].clip(lower=0)
            if "MNP_lag" in mod.columns:
                mod["MNP_lag"] = mod["MNP_lag"] + mnp_direction * (mnp - 1.0) * ramp_mnp * mnp_scale
                mod["MNP_lag"] = mod["MNP_lag"].clip(lower=-100)
            if "vacc_x_mnp" in mod.columns:
                mod["vacc_x_mnp"] = mod["cov12_lag"] * mod["MNP_lag"]
            return mod

        scenario_defs = {
            "Base": (1.0, 1.0),
            "Personnalisé": (vacc_factor, mnp_factor),
            "Sans vaccination": (0.1, 1.0),
            "Sans gestes barrières": (1.0, 0.2),
            "Gestes barrières +50%": (1.0, 1.5),
        }

        predictions = {}
        for label, (vacc_val, mnp_val) in scenario_defs.items():
            exog_mod = make_exog(exog_df, vacc=vacc_val, mnp=mnp_val)
            pred = sarimax_best.get_prediction(
                start=0, end=len(exog_mod) - 1, exog=exog_mod
            ).predicted_mean
            predictions[label] = pd.Series(pred, index=index_ts, name=label)

        rsv_obs = st.session_state["df_opt"]["RSV"].astype(float)
        rsv_obs.index = pd.to_datetime(rsv_obs.index)
        rsv_obs = rsv_obs.reindex(index_ts)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=index_ts, y=rsv_obs, name="RSV observé", line=dict(width=2, color="#2c3e50")))

        style_map = {
            "Base": dict(color="#1f77b4", dash="dot"),
            "Personnalisé": dict(color="#d62728", dash="solid"),
            "Sans vaccination": dict(color="#ff7f0e", dash="dash"),
            "Sans gestes barrières": dict(color="#9467bd", dash="dashdot"),
            "Gestes barrières +50%": dict(color="#2ca02c", dash="dot"),
        }

        for label, series in predictions.items():
            style = style_map.get(label, {})
            fig.add_trace(go.Scatter(
                x=index_ts,
                y=series,
                name=label,
                line=dict(width=2, **style)
            ))

        fig.update_layout(title="🔁 Scénarios SARIMAX", xaxis_title="Semaine", yaxis_title="RSV simulé")
        st.plotly_chart(fig, use_container_width=True)

        base_mean = predictions["Base"].mean()
        impact_rows = []
        for label, series in predictions.items():
            impact_rows.append({
                "Scénario": label,
                "RSV moyen": series.mean(),
                "RSV max": series.max(),
                "Δ vs base": series.mean() - base_mean
            })
        impact = pd.DataFrame(impact_rows).round(2)
        st.dataframe(impact, use_container_width=True)

        custom_last = predictions["Personnalisé"].iloc[-1]
        base_last = predictions["Base"].iloc[-1]
        st.metric("Impact en fin de période (scénario personnalisé)",
                  f"{custom_last:.1f}", delta=f"{(custom_last - base_last):+,.1f}")
# ==========================================
# 🔮 BLOC 8 — Prévisions SARIMAX 2025–2027 (stabilisées)
# ==========================================
with tabs[7]:
    st.header("🔮 BLOC 8 — Prévisions SARIMAX 2025–2027 (stabilisées)")

    MODELS = st.session_state["MODELS"]
    sarimax_best = MODELS.get("sarimax_best")
    if sarimax_best is None:
        st.error("❌ SARIMAX optimisé non disponible.")
        st.stop()

    df_sx = st.session_state["df_opt"].copy().sort_index()
    df_sx.index = pd.to_datetime(df_sx.index)
    for col in ["post_covid","post_vacc","t","t_post_covid"]:
        if col not in df_sx.columns:
            df_sx[col] = 1

    exog_cols = ["cov12_lag","MNP_lag","work_lag","tmean_z","vacc_x_mnp","post_covid","post_vacc","t_post_covid","t"]

    future_end = pd.Timestamp("2027-12-27")
    future_start = df_sx.index[-1] + pd.Timedelta(weeks=1)
    future_weeks = pd.date_range(start=future_start, end=future_end, freq="W-MON")

    last = df_sx.iloc[-1]
    future_exog_base = pd.DataFrame(index=future_weeks)
    for c in ["cov12_lag","MNP_lag","work_lag","tmean_z","vacc_x_mnp"]:
        future_exog_base[c] = last[c]
    future_exog_base["t"] = last["t"]
    future_exog_base["t_post_covid"] = last["t_post_covid"]
    future_exog_base["post_covid"] = 1
    future_exog_base["post_vacc"]  = 1

    def make_future_exog(df_base, mnp_factor=1.0, vacc_factor=1.0):
        df = df_base.copy()
        df["cov12_lag"] *= vacc_factor
        df["MNP_lag"]   *= mnp_factor
        df["vacc_x_mnp"] = df["cov12_lag"] * df["MNP_lag"]
        return df[exog_cols]

    exog_relax   = make_future_exog(future_exog_base, mnp_factor=0.70, vacc_factor=1.00)
    exog_stable  = make_future_exog(future_exog_base, mnp_factor=1.00, vacc_factor=1.00)
    exog_strong  = make_future_exog(future_exog_base, mnp_factor=1.40, vacc_factor=1.10)

    def forecast_with(model, exog_future):
        pred = model.get_forecast(steps=len(exog_future), exog=exog_future)
        return pred.predicted_mean

    y_relax  = forecast_with(sarimax_best, exog_relax)
    y_stable = forecast_with(sarimax_best, exog_stable)
    y_strong = forecast_with(sarimax_best, exog_strong)

    # Stabilisation simple (décroissance progressive)
    decay = np.exp(-np.linspace(0, 2, len(future_weeks)))
    y_relax_stab   = y_relax * decay
    y_stable_stab  = y_stable * decay
    y_strong_stab  = y_strong * decay

    # Observé complet (pour contexte)
    try:
        rsv_full = st.session_state["rsv"].copy()
        rsv_full = rsv_full[["date_monday","RSV"]].rename(columns={"RSV":"RSV_full"}).set_index("date_monday")
    except Exception:
        rsv_full = st.session_state["df_opt"][["RSV"]].rename(columns={"RSV":"RSV_full"})

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rsv_full.index, y=rsv_full["RSV_full"], name="RSV observé (2018–2025)", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=future_weeks, y=y_relax_stab,  name="Relâchement (-30%)",  line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=future_weeks, y=y_stable_stab, name="Maintien (2024)",    line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=future_weeks, y=y_strong_stab, name="Renforcement (+40%)",line=dict(dash="dot")))
    fig.add_vline(x=pd.Timestamp("2020-03-01"), line_dash="dash", line_color="red")
    fig.add_vline(x=pd.Timestamp("2021-01-01"), line_dash="dash", line_color="green")
    fig.add_vline(x=future_weeks[0], line_dash="dash", line_color="gray")
    fig.update_layout(title="🔮 Prévisions RSV — SARIMAX (2018–2027, stabilisées)",
                      xaxis_title="Semaine ISO", yaxis_title="RSV simulé (taux hebdo)")
    st.plotly_chart(fig, use_container_width=True)

    forecast_summary = pd.DataFrame({
        "Scénario": ["Relâchement (-30%)","Maintien (2024)","Renforcement (+40%)"],
        "RSV_moyen": [y_relax_stab.mean(), y_stable_stab.mean(), y_strong_stab.mean()],
        "RSV_max":   [y_relax_stab.max(),  y_stable_stab.max(),  y_strong_stab.max()]
    }).round(1)
    st.dataframe(forecast_summary, use_container_width=True)

    st.success("✅ Prévisions stabilisées jusqu'à fin 2027 générées avec succès.")
with tabs[8]:
    st.header("📉 BLOC 9 — Prévisions univariées (ERVISS & ODISSEE)")

    raw_data = st.session_state.get("raw_data")
    if not raw_data:
        raw_data = load_datasets(FILES)
        st.session_state["raw_data"] = raw_data

    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    @st.cache_data(show_spinner=False)
    def hw_forecast(series: pd.Series, horizon: int = 156):
        series = series.sort_index()
        series = series.asfreq("W-MON")
        series = series.interpolate(method="linear", limit_direction="both")
        series = series.fillna(method="bfill").fillna(method="ffill")

        seasonal_periods = 52 if len(series) >= 104 else None
        try:
            model = ExponentialSmoothing(
                series,
                trend="add",
                seasonal="add" if seasonal_periods else None,
                seasonal_periods=seasonal_periods,
                initialization_method="estimated"
            )
            fit = model.fit(optimized=True)
        except Exception:
            model = ExponentialSmoothing(
                series,
                trend="add",
                seasonal=None,
                initialization_method="estimated"
            )
            fit = model.fit(optimized=True)

        fitted = fit.fittedvalues
        forecast = fit.forecast(horizon)
        return fitted, forecast

    col_odissee, col_erviss = st.columns(2)

    with col_odissee:
        st.subheader("ODISSEE — Série RSV (0-4 ans)")
        rsv_df = st.session_state["rsv"].copy()
        rsv_series = rsv_df.set_index("date_monday")["RSV"].astype(float)
        fitted, forecast = hw_forecast(rsv_series, horizon=156)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rsv_series.index, y=rsv_series, name="Observé", line=dict(width=2)))
        fig.add_trace(go.Scatter(x=fitted.index, y=fitted, name="Lissage", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=forecast.index, y=forecast, name="Prévision ", line=dict(dash="dash")))
        fig.update_layout(xaxis_title="Semaine", yaxis_title="RSV (ODISSEE)", title="Prévision univariée Exponential Smoothing")
        st.plotly_chart(fig, use_container_width=True)

        st.metric("Sommet prévisionnel ", f"{forecast.max():.1f}", delta=f"{forecast.max() - rsv_series.iloc[-1]:+,.1f}")

    with col_erviss:
        st.subheader("ERVISS — Détections RSV")
        erviss_df = raw_data.get("erviss_fr_weekly")
        if isinstance(erviss_df, pd.DataFrame):
            erviss = erviss_df.copy()
            erviss["date_monday"] = pd.to_datetime(erviss["date_monday"])
            mask_rsv = (
                (erviss["geo_level"] == "FR") &
                (erviss["pathogen"].str.upper() == "RSV") &
                (erviss["indicator"].str.lower() == "detections")
            )
            erviss_series = (
                erviss.loc[mask_rsv]
                .groupby("date_monday")["value"]
                .sum()
                .astype(float)
                .sort_index()
            )
            if len(erviss_series) == 0:
                st.error("❌ Impossible de trouver des détections RSV dans ERVISS.")
            else:
                fitted_e, forecast_e = hw_forecast(erviss_series, horizon=156)
                fig_e = go.Figure()
                fig_e.add_trace(go.Scatter(x=erviss_series.index, y=erviss_series, name="Observé", line=dict(width=2)))
                fig_e.add_trace(go.Scatter(x=fitted_e.index, y=fitted_e, name="Lissage", line=dict(dash="dot")))
                fig_e.add_trace(go.Scatter(x=forecast_e.index, y=forecast_e, name="Prévision ", line=dict(dash="dash")))
                fig_e.update_layout(xaxis_title="Semaine", yaxis_title="Détections RSV (ERVISS)", title="Prévision univariée Exponential Smoothing")
                st.plotly_chart(fig_e, use_container_width=True)

                st.metric("Sommet prévisionnel ", f"{forecast_e.max():.1f}", delta=f"{forecast_e.max() - erviss_series.iloc[-1]:+,.1f}")
        else:
            st.error("❌ Format inattendu pour le dataset ERVISS.")
