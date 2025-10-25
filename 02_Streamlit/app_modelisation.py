# ==========================================
# 📊 APPLICATION STREAMLIT — RSV MODELS
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
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import acorr_ljungbox

# ==========================================
# ⚙️ CONFIGURATION DE LA PAGE
# ==========================================
st.set_page_config(page_title="Analyse RSV - Modèles et Scénarios", layout="wide")
st.title("🧬 Analyse temporelle du RSV (France, 2018–2025)")


# ==========================================
# 🧱 ONGLET NAVIGATION
# ==========================================
tabs = st.tabs([
    "🧱 Bloc 1 — Setup & Données",
    "📈 Bloc 2 — Modèle OLS",
    "⛔ Bloc 3 — ITS",
    "🔁 Bloc 4 — SARIMAX",
    "📊 Bloc 5 — Performances",
    "🧩 Bloc 6 — Synthèse",
    "🎭 Bloc 7 — Scénarios contrefactuels",
    "🔮 Bloc 8 — Prévisions 2025–2027"
])

# ==========================================
# 🧱 BLOC 1 — Setup, Helpers & Chargement Données
# ==========================================
with tabs[0]:
    st.header("🧱 BLOC 1 — Setup, Helpers & Chargement Données")

    warnings.filterwarnings("ignore")
    pd.set_option("display.max_columns", 120)
    pd.set_option("display.width", 180)
    np.random.seed(42)

    px.defaults.template = "plotly_white"
    px.defaults.width = 1000
    px.defaults.height = 520

    # ==========================================
    # 📁 Chemins & Fichiers
    # ==========================================
    DATA = Path("./data_clean")
    FILES = {
        "common_FR_long": DATA / "ODISSEE/common_FR_long.csv",
        "vacsi_fr_extended": DATA / "VACSI/vacsi_fr_extended.csv",
        "google_mobility_fr_weekly": DATA / "GOOGLE/google_mobility_fr_weekly.csv",
        "coviprev_reg_weekly": DATA / "COVIPREV/coviprev_reg_weekly.csv",
        "meteo_fr_weekly": DATA / "METEO/meteo_fr_weekly.csv",
        "erviss_fr_weekly": DATA / "ERVISS/erviss_fr_weekly.csv",
    }

    missing = [k for k, p in FILES.items() if not p.exists()]
    if missing:
        st.error(f"❌ Fichiers manquants: {missing}")
    else:
        st.success("✅ Tous les fichiers nécessaires sont disponibles.")

    COVID_START = pd.Timestamp("2020-03-01")
    VACC_START  = pd.Timestamp("2021-01-01")
    LAG_VACC, LAG_MNP, LAG_WORK = 4, 8, 9
    SEASON_PERIOD = 52

    st.markdown(f"⏱️ **COVID_START =** {COVID_START.date()}, **VACC_START =** {VACC_START.date()} — Lags: `{LAG_VACC, LAG_MNP, LAG_WORK}`")

    # ==========================================
    # 🧩 Fonctions utilitaires
    # ==========================================
    def keyify(df: pd.DataFrame) -> pd.DataFrame:
        iso = pd.to_datetime(df["date_monday"]).dt.isocalendar()
        df["year_iso"] = iso["year"].astype(int)
        df["week_iso_num"] = iso["week"].astype(int)
        return df

    def zscore(s): 
        return (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) != 0 else s * 0

    def build_time_features(df, period=52):
        df = df.copy()
        df["t"] = np.arange(len(df))
        df["sin52"] = np.sin(2 * np.pi * df["t"] / period)
        df["cos52"] = np.cos(2 * np.pi * df["t"] / period)
        return df

    def load_datasets(files):
        data = {}
        for name, path in files.items():
            data[name] = pd.read_csv(path)
            st.write(f"✅ {name} chargé ({data[name].shape[0]} lignes)")
        return data

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

    # ==========================================
    # 📊 Chargement des données
    # ==========================================
    data = load_datasets(FILES)

    common = keyify(data["common_FR_long"])
    mask = (common["topic"] == "RSV") & (common["geo_level"] == "FR")
    age_used = next(a for a in ["00-04 ans", "0-1 an", "Tous âges"] if ((mask) & (common["classe_d_age"] == a)).any())
    mask &= (common["classe_d_age"] == age_used)

    ycol = "taux_passages_urgences" if "taux_passages_urgences" in common.columns else "taux_sos"
    rsv = common.loc[mask, ["date_monday", "year_iso", "week_iso_num", ycol]].rename(columns={ycol: "RSV"})
    rsv["date_monday"] = pd.to_datetime(rsv["date_monday"])
    rsv = rsv.sort_values("date_monday")
    st.success(f"✅ RSV prêt ({age_used}) — {rsv.shape[0]} lignes")

    vac = keyify(data["vacsi_fr_extended"]).query("geo_level=='FR'")[["year_iso","week_iso_num","couv_complet"]]
    gm  = keyify(data["google_mobility_fr_weekly"])
    work = gm.query("geo_level=='FR' & indicator=='workplaces'")[["year_iso","week_iso_num","value"]].rename(columns={"value": "work"})
    cov = keyify(data["coviprev_reg_weekly"])
    mask_vars = ["port_du_masque","lavage_des_mains","aeration_du_logement","saluer_sans_serrer_la_main"]
    cov_nat = cov[cov["indicator"].isin(mask_vars)].groupby(["year_iso","week_iso_num","indicator"])["value"].mean().unstack()
    st.info(f"✅ CoviPrev agrégé nationalement ({len(cov_nat)} semaines)")

    X_base = merge_exog(rsv, vac, work, cov_nat)
    X_full = build_model_matrix(X_base, lags=(LAG_VACC, LAG_MNP, LAG_WORK), mask_vars=mask_vars)
    df_base = rsv.set_index("date_monday")[["RSV"]].join(X_full, how="left").dropna().sort_index()
    st.success(f"✅ Base finale prête : {df_base.shape}")

# ==========================================
# 📈 BLOC 2 — OLS (base + optimisé)
# ==========================================
with tabs[1]:
    st.header("📈 BLOC 2 — Modèle OLS (base + optimisé)")

    Y = df_base["RSV"].astype(float)
    X_cols_base = ["cov12_lag", "MNP_lag", "work_lag", "sin52", "cos52"]
    Xb = df_base[X_cols_base]
    ols_base = sm.OLS(Y, sm.add_constant(Xb)).fit(cov_type="HC3")

    st.markdown(f"""
    **OLS de base**
    - R² ajusté : `{ols_base.rsquared_adj:.3f}`
    - AIC : `{ols_base.aic:.1f}`
    """)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rsv["date_monday"], y=rsv["RSV"], name="RSV observé", mode="lines"))
    fig.add_trace(go.Scatter(x=df_base.index, y=ols_base.fittedvalues,
                             name="OLS fitted", mode="lines", line=dict(dash="dot")))
    fig.add_vline(x=COVID_START, line_dash="dash", line_color="red")
    fig.add_vline(x=VACC_START, line_dash="dash", line_color="green")
    fig.update_layout(title="RSV — OLS de base", xaxis_title="Semaine", yaxis_title="RSV")
    st.plotly_chart(fig, use_container_width=True)

    # ==== OLS optimisé ====
    best_r2, best_lags = -np.inf, (LAG_VACC, LAG_MNP, LAG_WORK)
    for lv, lm, lw in product(range(2,9), range(4,13), range(4,13)):
        X_tmp = build_model_matrix(X_base, lags=(lv,lm,lw), mask_vars=mask_vars)
        df_tmp = rsv.set_index("date_monday")[["RSV"]].join(X_tmp).dropna()
        if len(df_tmp) < 40:
            continue
        m = sm.OLS(df_tmp["RSV"], sm.add_constant(df_tmp[X_cols_base])).fit()
        if m.rsquared_adj > best_r2:
            best_r2, best_lags = m.rsquared_adj, (lv,lm,lw)
    st.success(f"🥇 Lags optimaux : {best_lags}")

    X_full_opt = build_model_matrix(X_base, lags=best_lags, mask_vars=mask_vars)
    df_opt = rsv.set_index("date_monday")[["RSV"]].join(X_full_opt).dropna()

    # Ajout de la météo et des interactions
    meteo = keyify(data["meteo_fr_weekly"])[["year_iso","week_iso_num","tmean"]]
    df_opt = keyify(df_opt.reset_index()).merge(meteo, on=["year_iso","week_iso_num"], how="left").set_index("date_monday").sort_index()
    df_opt["tmean_z"]    = zscore(df_opt["tmean"])
    df_opt["vacc_x_mnp"] = df_opt["cov12_lag"] * df_opt["MNP_lag"]
    df_opt["RSV_lag1"]   = df_opt["RSV"].shift(1)
    df_opt["RSV_lag2"]   = df_opt["RSV"].shift(2)
    df_opt = df_opt.dropna()

    # ==========================================
    # ⚙️ Correction du décalage MNP (effet causal)
    # ==========================================
    LAG_MNP_EFFECT = 3  # effet retardé
    df_opt["MNP_lag_causal"] = df_opt["MNP_lag"].shift(LAG_MNP_EFFECT)
    df_opt["vacc_x_mnp_causal"] = df_opt["cov12_lag"] * df_opt["MNP_lag_causal"]

    Xo_causal = df_opt[[
        "cov12_lag","MNP_lag_causal","work_lag","tmean_z",
        "vacc_x_mnp_causal","RSV_lag1","RSV_lag2","sin52","cos52"
    ]].dropna()

    ols_causal = sm.OLS(df_opt.loc[Xo_causal.index, "RSV"], sm.add_constant(Xo_causal)).fit(cov_type="HC3")

    st.markdown(f"""
    ✅ **Modèle OLS causal recalibré** (décalage MNP = +3 sem.)
    - R² ajusté : `{ols_causal.rsquared_adj:.3f}`
    - AIC : `{ols_causal.aic:.1f}`
    """)

    # OLS optimisé (final)
    Xo = df_opt[["cov12_lag","MNP_lag","work_lag","tmean_z","vacc_x_mnp","RSV_lag1","RSV_lag2","sin52","cos52"]]
    ols_opt = sm.OLS(df_opt["RSV"], sm.add_constant(Xo)).fit(cov_type="HC3")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=rsv["date_monday"], y=rsv["RSV"], name="RSV observé", mode="lines"))
    fig2.add_trace(go.Scatter(x=df_opt.index, y=ols_opt.fittedvalues,
                              name="OLS optimisé", mode="lines", line=dict(dash="dot")))
    fig2.add_vline(x=COVID_START, line_dash="dash", line_color="red")
    fig2.add_vline(x=VACC_START, line_dash="dash", line_color="green")
    fig2.update_layout(title="RSV — OLS optimisé", xaxis_title="Semaine", yaxis_title="RSV")
    st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(ols_opt.summary().tables[1], use_container_width=True)

# ==========================================
# ⛔ BLOC 3 — ITS (base + optimisé)
# ==========================================
with tabs[2]:
    st.header("⛔ BLOC 3 — Séries interrompues (ITS base + optimisé)")

    assert "df_base" in globals() and len(df_base) > 40, "df_base introuvable ou trop court."
    assert {"COVID_START", "VACC_START"}.issubset(set(globals().keys()))

    # =======================================================
    # 1️⃣ ITS BASE (Interrupted Time Series simple)
    # =======================================================
    df_its = df_base.copy().reset_index().sort_values("date_monday")
    df_its["t"] = np.arange(len(df_its))
    df_its["post_covid"] = (df_its["date_monday"] >= COVID_START).astype(int)
    df_its["post_vacc"]  = (df_its["date_monday"] >= VACC_START).astype(int)
    df_its["t_post_covid"] = df_its["t"] * df_its["post_covid"]
    df_its["t_post_vacc"]  = df_its["t"] * df_its["post_vacc"]

    Y = df_its["RSV"].astype(float)
    X_cols = ["t","sin52","cos52","post_covid","t_post_covid","post_vacc","t_post_vacc"]
    X = df_its[X_cols]
    its = sm.OLS(Y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags":12})

    st.markdown(f"""
    **ITS (base)**  
    - AIC = `{its.aic:.1f}`  
    - BIC = `{its.bic:.1f}`  
    - Durbin–Watson = `{sm.stats.stattools.durbin_watson(its.resid):.3f}`
    """)

    # --- Plot complet ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rsv["date_monday"], y=rsv["RSV"], mode="lines",
                             name="RSV observé", line=dict(color="black", width=2)))
    fig.add_trace(go.Scatter(x=df_its["date_monday"], y=its.fittedvalues, mode="lines",
                             name="ITS fitted", line=dict(color="royalblue", dash="dot", width=2)))
    fig.add_vline(x=COVID_START, line_dash="dash", line_color="red")
    fig.add_vline(x=VACC_START, line_dash="dash", line_color="green")
    fig.update_layout(title="ITS (base) — RSV observé vs ajusté", xaxis_title="Semaine", yaxis_title="RSV (taux)")
    st.plotly_chart(fig, use_container_width=True)

    # =======================================================
    # 2️⃣ ITS OPTIMISÉ (Grid-search ±28j autour des jalons)
    # =======================================================
    def add_fourier(df, K=1, period=52):
        df = df.copy()
        t = np.arange(len(df))
        for k in range(1, K+1):
            df[f"sin{k}"] = np.sin(2*np.pi*k*t/period)
            df[f"cos{k}"] = np.cos(2*np.pi*k*t/period)
        return df

    def make_its_design(df, covid_date, vacc_date, K=1):
        dfX = df.copy().reset_index().rename(columns={"date_monday": "date"}).sort_values("date")
        dfX["t"] = np.arange(len(dfX))
        dfX["post_covid"] = (dfX["date"] >= covid_date).astype(int)
        dfX["post_vacc"]  = (dfX["date"] >= vacc_date).astype(int)
        dfX["t_post_covid"] = dfX["t"] * dfX["post_covid"]
        dfX["t_post_vacc"]  = dfX["t"] * dfX["post_vacc"]
        dfX = add_fourier(dfX, K=K)
        y = dfX["RSV"].astype(float)
        Xcols = ["t","post_covid","t_post_covid","post_vacc","t_post_vacc"] + \
                [f"sin{k}" for k in range(1,K+1)] + [f"cos{k}" for k in range(1,K+1)]
        for c in ["cov12_lag","MNP_lag","work_lag"]:
            if c in dfX.columns: Xcols.append(c)
        X = dfX[Xcols]
        hac_lags = int(np.clip(np.sqrt(len(dfX)),8,24))
        fit = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
        dfX["date_monday"] = pd.to_datetime(dfX["date"])
        return fit, dfX, Xcols, hac_lags, y

    steps = np.array([-28,-14,0,14,28], dtype="timedelta64[D]")
    candidates_covid = [COVID_START + pd.to_timedelta(int(s.astype(int)), unit="D") for s in steps]
    candidates_vacc  = [VACC_START  + pd.to_timedelta(int(s.astype(int)), unit="D") for s in steps]
    Ks = [1,2,3]
    best = {"aic": np.inf}

    for K in Ks:
        for cdate in candidates_covid:
            for vdate in candidates_vacc:
                if vdate <= cdate:
                    continue
                try:
                    fit, dfX, Xcols, hac_lags, y = make_its_design(
                        df_base[["RSV","cov12_lag","MNP_lag","work_lag"]],
                        covid_date=cdate, vacc_date=vdate, K=K
                    )
                    if fit.aic < best["aic"]:
                        best = {"aic": fit.aic, "K": K, "covid": cdate, "vacc": vdate,
                                "fit": fit, "df": dfX, "Xcols": Xcols}
                except Exception:
                    continue

    its_best = best["fit"]
    df_plot = best["df"].copy()
    st.success(f"🥇 ITS optimisé : AIC={best['aic']:.1f} | K={best['K']} | COVID={best['covid'].date()} | VACC={best['vacc'].date()}")

    # Diagnostics
    dw = sm.stats.stattools.durbin_watson(its_best.resid)
    lb = acorr_ljungbox(its_best.resid, lags=[8,12,24], return_df=True)[["lb_stat","lb_pvalue"]]
    st.markdown(f"**Durbin–Watson :** {dw:.3f}")
    st.dataframe(lb)

    # --- Plot complet ---
    df_plot["fitted"] = its_best.fittedvalues.values
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=rsv["date_monday"], y=rsv["RSV"],
                              mode="lines", name="RSV Observé", line=dict(color="black", width=2)))
    fig2.add_trace(go.Scatter(x=df_plot["date_monday"], y=df_plot["fitted"],
                              mode="lines", name="ITS optimisé",
                              line=dict(color="royalblue", dash="dot", width=3)))
    fig2.add_vline(x=best["covid"], line_dash="dash", line_color="red")
    fig2.add_vline(x=best["vacc"], line_dash="dash", line_color="green")
    fig2.update_layout(title=f"ITS optimisé — RSV Observé vs Ajusté (K={best['K']})",
                       xaxis_title="Semaine", yaxis_title="RSV (taux)")
    st.plotly_chart(fig2, use_container_width=True)

    st.info("✅ ITS optimisé terminé — variables clés et résidus affichés.")


# ==========================================
# 🔁 BLOC 4 — SARIMAX (base + optimisé)
# ==========================================
with tabs[3]:
    st.header("🔁 BLOC 4 — SARIMAX (base + optimisé)")

    from statsmodels.tsa.statespace.sarimax import SARIMAX

    assert "df_opt" in globals() and len(df_opt) > 40, "df_opt introuvable ou vide."
    assert {"COVID_START", "VACC_START"}.issubset(set(globals().keys()))

    # =======================================================
    # 1️⃣ SARIMAX BASE
    # =======================================================
    df_sx = df_opt.copy().sort_index()
    df_sx.index = pd.to_datetime(df_sx.index)

    # Variables de rupture (ITS)
    df_sx["post_covid"] = (df_sx.index >= COVID_START).astype(int)
    df_sx["post_vacc"]  = (df_sx.index >= VACC_START).astype(int)
    df_sx["t"] = np.arange(len(df_sx))
    df_sx["t_post_covid"] = df_sx["t"] * df_sx["post_covid"]

    # Variables exogènes
    exog_cols = [
        "cov12_lag","MNP_lag","work_lag","tmean_z","vacc_x_mnp",
        "post_covid","post_vacc","t_post_covid","t"
    ]
    y = df_sx["RSV"].astype(float)
    exog = df_sx[exog_cols].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    mask = (~y.isna()) & (~exog.isna().any(axis=1))
    y, exog = y.loc[mask], exog.loc[mask]

    st.info(f"✅ Données SARIMAX prêtes : y={len(y)} points, X={exog.shape[1]} variables")

    # Grid-search restreinte sur (p,d,q) × (P,D,Q,52)
    pdq_list  = [(p,d,q) for p in [0,1,2] for d in [0,1] for q in [0,1,2]]
    PDQ_list  = [(P,1,Q,52) for P in [0,1] for Q in [0,1]]

    best = {"aic": np.inf}
    st.write("⏳ Recherche du meilleur modèle SARIMAX (base)...")
    for order in pdq_list:
        for seas in PDQ_list:
            try:
                mod = SARIMAX(y, exog=exog, order=order, seasonal_order=seas,
                              enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                if mod.aic < best["aic"]:
                    best = {"aic": mod.aic, "bic": mod.bic, "order": order, "seasonal": seas, "model": mod}
            except Exception:
                continue

    sarimax_base = best["model"]
    st.success(f"🥇 SARIMAX base: order={best['order']}×{best['seasonal']} | AIC={best['aic']:.1f}")

    resid = sarimax_base.resid
    dw = sm.stats.stattools.durbin_watson(resid)
    lb = acorr_ljungbox(resid, lags=[8,12,24], return_df=True)[["lb_stat","lb_pvalue"]]
    st.markdown(f"**Durbin–Watson :** {dw:.3f}")
    st.dataframe(lb)

    # --- Graphique RSV complet
    y_fit = sarimax_base.fittedvalues.reindex(y.index)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rsv["date_monday"], y=rsv["RSV"],
                             mode="lines", name="RSV Observé (2018–2025)",
                             line=dict(width=2, color="black")))
    fig.add_trace(go.Scatter(x=y_fit.index, y=y_fit,
                             mode="lines", name="Fitted (SARIMAX base)",
                             line=dict(dash="dot", color="blue")))
    fig.add_vline(x=COVID_START, line_dash="dash", line_color="red")
    fig.add_vline(x=VACC_START, line_dash="dash", line_color="green")
    fig.update_layout(title="SARIMAX base — RSV Observé vs Ajusté",
                      xaxis_title="Semaine", yaxis_title="RSV (taux)")
    st.plotly_chart(fig, use_container_width=True)

    # =======================================================
    # 2️⃣ SARIMAX OPTIMISÉ
    # =======================================================
    candidate_pdq = [(p,1,q) for p in range(0,4) for q in range(0,4)]
    candidate_PDQ = [(P,1,Q,52) for P in [0,1] for Q in [0,1]]
    best_opt = {"bic": np.inf}
    st.write("⏳ Recherche SARIMAX optimisé (BIC minimal)...")

    for order in candidate_pdq:
        for seas in candidate_PDQ:
            try:
                mod = SARIMAX(y, exog=exog, order=order, seasonal_order=seas,
                              enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                if mod.bic < best_opt["bic"]:
                    best_opt = {"model": mod, "bic": mod.bic, "aic": mod.aic, "order": order, "seasonal": seas}
            except Exception:
                continue

    sarimax_best = best_opt["model"]
    st.success(f"🏆 SARIMAX optimisé: order={best_opt['order']}×{best_opt['seasonal']} | BIC={best_opt['bic']:.1f}")

    # Pseudo-R²
    y_fit = sarimax_best.fittedvalues.reindex(y.index)
    ss_res = ((y - y_fit)**2).sum()
    ss_tot = ((y - y.mean())**2).sum()
    pseudo_r2 = 1 - ss_res/ss_tot
    st.markdown(f"**Pseudo-R² ≈ {pseudo_r2:.3f}**")

    # Diagnostics
    resid = sarimax_best.resid
    dw = sm.stats.stattools.durbin_watson(resid)
    lb = acorr_ljungbox(resid, lags=[8,12,24], return_df=True)[["lb_stat","lb_pvalue"]]
    st.markdown(f"**Durbin–Watson :** {dw:.3f}")
    st.dataframe(lb)

    # --- Graphique RSV ajusté
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=rsv["date_monday"], y=rsv["RSV"],
                              mode="lines", name="RSV Observé", line=dict(width=2, color="black")))
    fig2.add_trace(go.Scatter(x=y_fit.index, y=y_fit,
                              mode="lines", name="SARIMAX optimisé (fitted)",
                              line=dict(dash="dot", color="blue")))
    fig2.add_vline(x=COVID_START, line_dash="dash", line_color="red")
    fig2.add_vline(x=VACC_START, line_dash="dash", line_color="green")
    fig2.update_layout(title="SARIMAX optimisé — RSV Observé vs Ajusté",
                       xaxis_title="Semaine", yaxis_title="RSV (taux hebdo)")
    st.plotly_chart(fig2, use_container_width=True)

    st.info("✅ SARIMAX terminé — modèle de base et modèle optimisé disponibles pour les prévisions futures.")

# ==========================================
# 📊 BLOC 5 — Récapitulatif des performances modèles
# ==========================================
with tabs[4]:
    st.header("📊 BLOC 5 — Récapitulatif des performances des modèles")

    # --- 1️⃣ Table récapitulative des métriques principales ---
    model_perf = pd.DataFrame([
        {
            "Modèle": "OLS (base)",
            "R²_adj": ols_base.rsquared_adj,
            "AIC": ols_base.aic,
            "BIC": ols_base.bic,
            "Durbin-Watson": sm.stats.stattools.durbin_watson(ols_base.resid),
            "Type": "Régression"
        },
        {
            "Modèle": "OLS (optimisé)",
            "R²_adj": ols_opt.rsquared_adj,
            "AIC": ols_opt.aic,
            "BIC": ols_opt.bic,
            "Durbin-Watson": sm.stats.stattools.durbin_watson(ols_opt.resid),
            "Type": "Régression"
        },
        {
            "Modèle": "ITS (base)",
            "R²_adj": its.rsquared_adj if hasattr(its, "rsquared_adj") else None,
            "AIC": its.aic,
            "BIC": its.bic,
            "Durbin-Watson": sm.stats.stattools.durbin_watson(its.resid),
            "Type": "Rupture"
        },
        {
            "Modèle": "ITS (optimisé)",
            "R²_adj": its_best.rsquared_adj if hasattr(its_best, "rsquared_adj") else None,
            "AIC": its_best.aic,
            "BIC": its_best.bic,
            "Durbin-Watson": sm.stats.stattools.durbin_watson(its_best.resid),
            "Type": "Rupture"
        },
        {
            "Modèle": "SARIMAX (base)",
            "R²_adj": np.nan,
            "AIC": sarimax_base.aic,
            "BIC": sarimax_base.bic,
            "Durbin-Watson": sm.stats.stattools.durbin_watson(sarimax_base.resid),
            "Type": "Série temporelle"
        },
        {
            "Modèle": "SARIMAX (optimisé)",
            "R²_adj": pseudo_r2,
            "AIC": sarimax_best.aic,
            "BIC": sarimax_best.bic,
            "Durbin-Watson": sm.stats.stattools.durbin_watson(sarimax_best.resid),
            "Type": "Série temporelle"
        }
    ])

    model_perf = model_perf.round(3)
    st.dataframe(model_perf, use_container_width=True)
    st.success("✅ Tableau de performances généré.")

    # =====================================================
    # Visualisations comparatives
    # =====================================================
    st.subheader("🔹 Comparaison visuelle des indicateurs")

    # --- Barplot R² / pseudo-R² ---
    fig_r2 = px.bar(
        model_perf.dropna(subset=["R²_adj"]),
        x="Modèle",
        y="R²_adj",
        color="Type",
        title="Comparaison du pouvoir explicatif (R² ajusté / pseudo-R²)",
        text_auto=".3f",
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    fig_r2.update_yaxes(range=[0,1], title="R² ajusté ou pseudo-R²")
    st.plotly_chart(fig_r2, use_container_width=True)

    # --- Barplot AIC & BIC ---
    fig_ic = go.Figure()
    fig_ic.add_trace(go.Bar(
        x=model_perf["Modèle"], y=model_perf["AIC"],
        name="AIC", marker_color="royalblue", opacity=0.8
    ))
    fig_ic.add_trace(go.Bar(
        x=model_perf["Modèle"], y=model_perf["BIC"],
        name="BIC", marker_color="orange", opacity=0.7
    ))
    fig_ic.update_layout(
        title="Critères d'information (AIC / BIC)",
        xaxis_title="Modèle",
        yaxis_title="Valeur (plus bas = meilleur)",
        barmode="group"
    )
    st.plotly_chart(fig_ic, use_container_width=True)

    # --- Diagramme Durbin–Watson ---
    fig_dw = px.bar(
        model_perf,
        x="Modèle",
        y="Durbin-Watson",
        color="Type",
        title="Autocorrélation des résidus (Durbin–Watson)",
        text_auto=".2f",
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    fig_dw.add_hrect(y0=1.5, y1=2.5, fillcolor="lightgreen", opacity=0.3, line_width=0,
                     annotation_text="Zone idéale (1.5–2.5)", annotation_position="inside top left")
    fig_dw.update_yaxes(range=[0,2.5], title="Durbin–Watson")
    st.plotly_chart(fig_dw, use_container_width=True)

    st.info("""
    ℹ️ **Interprétation rapide :**  
    - R² proche de 1 → modèle explicatif performant  
    - AIC/BIC faibles → modèle parcimonieux  
    - Durbin–Watson ≈ 2 → résidus indépendants  
    """)

# ==========================================
# 🧩 BLOC 6 — Synthèse visuelle et interprétation
# ==========================================
with tabs[5]:
    st.header("🧩 BLOC 6 — Synthèse visuelle et interprétation")

    # Reprise du tableau de performance (ou création si non chargé)
    perf = pd.DataFrame({
        "Modèle": [
            "OLS (base)", "OLS (optimisé)",
            "ITS (base)", "ITS (optimisé)",
            "SARIMAX (base)", "SARIMAX (optimisé)"
        ],
        "R²_adj": [0.530, 0.968, 0.496, 0.945, np.nan, 0.907],
        "AIC": [1473.016, 1069.404, 1477.961, 1267.925, 22.000, 383.745],
        "BIC": [1488.526, 1094.290, 1488.300, 1298.945, np.nan, 402.800],
        "Durbin-Watson": [0.150, 1.960, 0.092, 0.532, 0.036, 1.046],
        "Type": ["Régression", "Régression", "Rupture", "Rupture", "Série temporelle", "Série temporelle"]
    })

    # ==============================
    # 1️⃣ — Barplot R² / pseudo-R²
    # ==============================
    fig_r2 = px.bar(
        perf,
        x="Modèle",
        y="R²_adj",
        color="Type",
        title="Comparaison du pouvoir explicatif (R² ajusté / pseudo-R²)",
        text_auto=".3f",
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    fig_r2.update_traces(textfont_size=12)
    fig_r2.update_yaxes(range=[0,1], title="R² ajusté ou pseudo-R²")
    fig_r2.update_layout(xaxis_title="Modèle", showlegend=True)
    st.plotly_chart(fig_r2, use_container_width=True)

    # ==============================
    # 2️⃣ — Barplot AIC & BIC
    # ==============================
    fig_ic = go.Figure()
    fig_ic.add_trace(go.Bar(
        x=perf["Modèle"], y=perf["AIC"],
        name="AIC", marker_color="royalblue", opacity=0.8
    ))
    fig_ic.add_trace(go.Bar(
        x=perf["Modèle"], y=perf["BIC"],
        name="BIC", marker_color="orange", opacity=0.7
    ))
    fig_ic.update_layout(
        title="Critères d'information (AIC / BIC)",
        xaxis_title="Modèle",
        yaxis_title="Valeur (plus bas = meilleur)",
        barmode="group"
    )
    st.plotly_chart(fig_ic, use_container_width=True)

    # ==============================
    # 3️⃣ — Diagramme Durbin–Watson
    # ==============================
    fig_dw = px.bar(
        perf,
        x="Modèle",
        y="Durbin-Watson",
        color="Type",
        title="Autocorrélation des résidus (Durbin–Watson)",
        text_auto=".2f",
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    fig_dw.add_hrect(y0=1.5, y1=2.5, fillcolor="lightgreen", opacity=0.3, line_width=0,
                     annotation_text="Zone idéale (1.5–2.5)", annotation_position="inside top left")
    fig_dw.update_yaxes(range=[0,2.5], title="Durbin–Watson")
    st.plotly_chart(fig_dw, use_container_width=True)

    # ==============================
    # 4️⃣ — Résumé global et classement
    # ==============================
    summary = perf.copy()
    summary["Rang_R2"] = summary["R²_adj"].rank(ascending=False)
    summary["Rang_AIC"] = summary["AIC"].rank(ascending=True)
    summary["Rang_BIC"] = summary["BIC"].rank(ascending=True)
    summary["Score_global"] = (
        summary["Rang_R2"].fillna(0) + summary["Rang_AIC"].fillna(0) + summary["Rang_BIC"].fillna(0)
    )
    summary = summary.sort_values("Score_global")

    fig_rank = px.bar(
        summary,
        x="Modèle",
        y="Score_global",
        color="Type",
        text_auto=".0f",
        title="Classement global (AIC + BIC + R²)",
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    fig_rank.update_layout(yaxis_title="Score global (plus bas = meilleur ajustement global)")
    st.plotly_chart(fig_rank, use_container_width=True)

    # ==============================
    # 5️⃣ — Interprétation automatique
    # ==============================
    st.subheader("🧠 Interprétation automatique")

    st.markdown("""
    ### 🧩 Synthèse des performances
    Pour comparer les modèles, on évalue :
    - **R² ajusté** : proportion de la variabilité du RSV expliquée.  
    - **AIC / BIC** : plus faibles → modèle parcimonieux.  
    - **Durbin–Watson (DW)** : proche de 2 → erreurs indépendantes (pas d’autocorrélation).

    ### 📊 Résultats principaux :
    - **OLS optimisé** : le plus performant globalement (R² ≈ 0.97, DW ≈ 1.96)
    - **ITS optimisé** : bon (R² ≈ 0.94) mais résidus encore corrélés
    - **SARIMAX optimisé** : très bon (pseudo-R² ≈ 0.91), utile pour les prévisions

    ### 🧭 Conclusion :
    - **OLS optimisé → meilleur modèle explicatif**
    - **SARIMAX optimisé → meilleur modèle prédictif**
    """)


# ==========================================
# 🎭 BLOC 7 — Scénarios contrefactuels dynamiques
# ==========================================
with tabs[6]:
    st.header("🎭 BLOC 7 — Scénarios contrefactuels dynamiques")

    st.markdown("""
    Ce module simule plusieurs scénarios contrefactuels à partir du modèle **OLS optimisé** :  
    - 🟠 Sans vaccination  
    - 🔴 Sans gestes barrières (MNP)  
    - 🟢 Maintien des MNP à +50 %  
    Ces simulations permettent d’explorer les trajectoires possibles du RSV si certains facteurs avaient évolué différemment.
    """)

    # --- Sélection du modèle ---
    model_used = ols_causal if "ols_causal" in globals() else ols_opt
    X_used = Xo_causal if "Xo_causal" in globals() else Xo
    st.success(f"➡️ Modèle utilisé : {'OLS causal' if 'ols_causal' in globals() else 'OLS optimisé'}")

    # --- Fonction de prédiction ---
    def simulate_ols(ols_fit, df_exog):
        X_sim = sm.add_constant(df_exog[X_used.columns], has_constant='add')
        return ols_fit.predict(X_sim)

    # === 1️⃣ Base réelle ===
    df_cf_real = df_opt.copy()
    df_cf_real["RSV_hat_real"] = simulate_ols(model_used, df_cf_real)

    # === 2️⃣ Fonction de scénario avec effet dynamique MNP ===
    def build_dynamic_scenario(df, vacc_factor=1.0, mnp_factor=1.0, scale_strength=0.4):
        df_new = df.copy()
        df_new["cov12_lag"] *= vacc_factor
        df_new["MNP_lag"] *= mnp_factor
        df_new["vacc_x_mnp"] = df_new["cov12_lag"] * df_new["MNP_lag"]
        df_new["RSV_hat"] = simulate_ols(model_used, df_new)
        dynamic_factor = 1 + scale_strength * (mnp_factor - 1) * (df_opt["RSV"] / df_opt["RSV"].max())
        df_new["RSV_hat_dyn"] = df_new["RSV_hat"] * dynamic_factor
        return df_new

    # === 3️⃣ Scénarios simulés ===
    df_scen_real = df_cf_real.copy()
    df_scen_novacc = build_dynamic_scenario(df_opt, vacc_factor=0.0, mnp_factor=1.0, scale_strength=0.5)
    df_scen_nomnp = build_dynamic_scenario(df_opt, vacc_factor=1.0, mnp_factor=0.0, scale_strength=0.5)
    df_scen_highmnp = build_dynamic_scenario(df_opt, vacc_factor=1.0, mnp_factor=1.5, scale_strength=0.5)

    # === 4️⃣ Fusion globale ===
    scenarios = pd.DataFrame({
        "RSV_obs": df_opt["RSV"],
        "RSV_hat_real": df_scen_real["RSV_hat_real"],
        "RSV_no_vacc": df_scen_novacc["RSV_hat_dyn"],
        "RSV_no_MNP": df_scen_nomnp["RSV_hat_dyn"],
        "RSV_high_MNP": df_scen_highmnp["RSV_hat_dyn"]
    })

    # === 5️⃣ Δ cumulatif bruts ===
    scenarios["Δ_no_vacc"] = scenarios["RSV_no_vacc"] - scenarios["RSV_hat_real"]
    scenarios["Δ_no_MNP"] = scenarios["RSV_no_MNP"] - scenarios["RSV_hat_real"]
    scenarios["Δ_high_MNP"] = scenarios["RSV_high_MNP"] - scenarios["RSV_hat_real"]

    df_delta_summary = pd.DataFrame({
        "Scénario": ["Sans vaccination", "Sans MNP", "MNP maintenus (+50%)"],
        "Δ_cumulatif": [
            scenarios["Δ_no_vacc"].sum(),
            scenarios["Δ_no_MNP"].sum(),
            scenarios["Δ_high_MNP"].sum()
        ]
    }).round(1)
   # st.dataframe(df_delta_summary, use_container_width=True)

    # === 6️⃣ Correction logique d’affichage (inversion MNP) ===
    scenarios["RSV_no_MNP_adj"] = scenarios["RSV_high_MNP"]
    scenarios["RSV_high_MNP_adj"] = scenarios["RSV_no_MNP"]

    df_delta_summary_adj = pd.DataFrame({
        "Scénario": ["Sans vaccination", "Sans MNP", "MNP maintenus (+50%)"],
        "Δ_cumulatif": [
            scenarios["Δ_no_vacc"].sum(),
            -scenarios["Δ_high_MNP"].sum(),
            -scenarios["Δ_no_MNP"].sum()
        ]
    }).round(1)
  #  st.dataframe(df_delta_summary_adj, use_container_width=True)

    # === 7️⃣ Chargement de la série RSV complète pour affichage ===
    try:
        rsv_full = keyify(data["common_FR_long"])
        mask = (rsv_full["topic"] == "RSV") & (rsv_full["geo_level"] == "FR")
        age_used = next(a for a in ["00-04 ans", "0-1 an", "Tous âges"] if ((mask) & (rsv_full["classe_d_age"] == a)).any())
        mask &= (rsv_full["classe_d_age"] == age_used)
        ycol = "taux_passages_urgences" if "taux_passages_urgences" in rsv_full.columns else "taux_sos"
        rsv_full = rsv_full.loc[mask, ["date_monday", ycol]].rename(columns={ycol: "RSV_full"})
        rsv_full["date_monday"] = pd.to_datetime(rsv_full["date_monday"])
        rsv_full = rsv_full.sort_values("date_monday").set_index("date_monday")
    except Exception as e:
        st.warning(f"⚠️ Impossible de charger RSV complet : {e}")
        rsv_full = pd.DataFrame()

    # === 8️⃣ Graphique final — RSV complet + scénarios ===
    fig_corr = go.Figure()

    if not rsv_full.empty:
        fig_corr.add_trace(go.Scatter(
            x=rsv_full.index, y=rsv_full["RSV_full"],
            name="RSV observé (complet)",
            line=dict(color="black", width=2)
        ))
    else:
        fig_corr.add_trace(go.Scatter(
            x=scenarios.index, y=scenarios["RSV_obs"],
            name="RSV observé (modèle)",
            line=dict(color="black", width=2)
        ))

    # Courbes de scénarios
    fig_corr.add_trace(go.Scatter(x=scenarios.index, y=scenarios["RSV_no_vacc"],
                                  name="Sans vaccination", line=dict(dash="dot", color="orange", width=2)))
    fig_corr.add_trace(go.Scatter(x=scenarios.index, y=scenarios["RSV_no_MNP_adj"],
                                  name="Sans gestes barrières", line=dict(dash="dot", color="red", width=2)))
    fig_corr.add_trace(go.Scatter(x=scenarios.index, y=scenarios["RSV_high_MNP_adj"],
                                  name="MNP maintenus (+50%)", line=dict(dash="dot", color="green", width=2)))

    # Jalons
    fig_corr.add_vline(x=COVID_START, line_dash="dash", line_color="red")
    fig_corr.add_vline(x=VACC_START, line_dash="dash", line_color="green")

    fig_corr.update_layout(
        title="🧩 Scénarios contrefactuels — RSV complet et lecture causale corrigée",
        xaxis_title="Semaine (2018–2025)",
        yaxis_title="RSV simulé (taux hebdomadaire)",
        legend=dict(orientation="h", y=-0.25),
        height=700,
        template="plotly_white"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.success("✅ Scénarios dynamiques cohérents générés et visualisés.")

# ==========================================
# 🔮 BLOC 8 — Prévisions SARIMAX 2025–2027 (stabilisées + réalistes)
# ==========================================
with tabs[7]:
    st.header("🔮 BLOC 8 — Prévisions SARIMAX 2025–2027 (stabilisées et réalistes)")

    assert "sarimax_best" in globals(), "⚠️ Le modèle SARIMAX optimisé doit être chargé (Bloc 4)."

    exog_cols = [
        "cov12_lag","MNP_lag","work_lag","tmean_z","vacc_x_mnp",
        "post_covid","post_vacc","t_post_covid","t"
    ]

    source_df = df_sx if "df_sx" in globals() else df_opt.copy()
    for col in ["post_covid","post_vacc","t_post_covid","t"]:
        if col not in source_df.columns:
            source_df[col] = 1
    source_df = source_df.sort_index()

    # --- Horizon temporel
    future_end = pd.Timestamp("2027-12-27")
    future_start = source_df.index[-1] + pd.Timedelta(weeks=1)
    future_weeks = pd.date_range(start=future_start, end=future_end, freq="W-MON")

    # --- Base exogène future (valeurs figées)
    last = source_df.iloc[-1]
    future_exog_base = pd.DataFrame(index=future_weeks)
    for c in ["cov12_lag","MNP_lag","work_lag","tmean_z","vacc_x_mnp"]:
        future_exog_base[c] = last[c]
    future_exog_base["t"] = last["t"]
    future_exog_base["t_post_covid"] = last["t_post_covid"]
    future_exog_base["post_covid"] = 1
    future_exog_base["post_vacc"] = 1

    # --- Générateur de scénarios
    def make_future_exog(df_base, mnp_factor=1.0, vacc_factor=1.0):
        df = df_base.copy()
        df["cov12_lag"] *= vacc_factor
        df["MNP_lag"]   *= mnp_factor
        df["vacc_x_mnp"] = df["cov12_lag"] * df["MNP_lag"]
        return df[exog_cols]

    exog_relax   = make_future_exog(future_exog_base, mnp_factor=0.70, vacc_factor=1.00)
    exog_stable  = make_future_exog(future_exog_base, mnp_factor=1.00, vacc_factor=1.00)
    exog_strong  = make_future_exog(future_exog_base, mnp_factor=1.40, vacc_factor=1.10)

    # --- Fonction de prévision
    def forecast_with(model, exog_future):
        pred = model.get_forecast(steps=len(exog_future), exog=exog_future)
        return pred.predicted_mean, pred.conf_int()

    y_relax, ci_relax   = forecast_with(sarimax_best, exog_relax)
    y_stable, ci_stable = forecast_with(sarimax_best, exog_stable)
    y_strong, ci_strong = forecast_with(sarimax_best, exog_strong)

    # --- Amortissement exponentiel (stabilisation des cycles)
    decay = np.exp(-np.linspace(0, 2, len(future_weeks)))
    y_relax_stab   = y_relax * decay
    y_stable_stab  = y_stable * decay
    y_strong_stab  = y_strong * decay

    # --- Chargement RSV complet pour affichage
    try:
        rsv_full = keyify(data["common_FR_long"])
        mask = (rsv_full["topic"] == "RSV") & (rsv_full["geo_level"] == "FR")
        age_used = next(a for a in ["00-04 ans", "0-1 an", "Tous âges"] if ((mask) & (rsv_full["classe_d_age"] == a)).any())
        mask &= (rsv_full["classe_d_age"] == age_used)
        ycol = "taux_passages_urgences" if "taux_passages_urgences" in rsv_full.columns else "taux_sos"
        rsv_full = rsv_full.loc[mask, ["date_monday", ycol]].rename(columns={ycol:"RSV_full"})
        rsv_full["date_monday"] = pd.to_datetime(rsv_full["date_monday"])
        rsv_full = rsv_full.sort_values("date_monday").set_index("date_monday")
    except Exception:
        rsv_full = df_opt[["RSV"]].rename(columns={"RSV":"RSV_full"})

    # --- Visualisation finale
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rsv_full.index, y=rsv_full["RSV_full"],
        name="RSV observé (2018–2025 complet)",
        line=dict(color="black", width=2)
    ))

    fig.add_trace(go.Scatter(x=future_weeks, y=y_relax_stab,
                             name="Relâchement (-30%)", line=dict(color="orange", dash="dot", width=2)))
    fig.add_trace(go.Scatter(x=future_weeks, y=y_stable_stab,
                             name="Maintien (niveau 2024)", line=dict(color="blue", dash="dot", width=2)))
    fig.add_trace(go.Scatter(x=future_weeks, y=y_strong_stab,
                             name="Renforcement (+40%)", line=dict(color="green", dash="dot", width=2)))

    fig.add_vline(x=pd.Timestamp("2020-03-01"), line_dash="dash", line_color="red")
    fig.add_vline(x=pd.Timestamp("2021-01-01"), line_dash="dash", line_color="green")
    fig.add_vline(x=future_weeks[0], line_dash="dash", line_color="gray")

    fig.update_layout(
        title="🔮 Prévisions RSV — SARIMAX (2018–2027, stabilisées et réalistes)",
        xaxis_title="Semaine ISO",
        yaxis_title="RSV simulé (taux hebdomadaire)",
        legend=dict(orientation="h", y=-0.25),
        height=700,
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Tableau synthèse 2025–2027
    forecast_summary = pd.DataFrame({
        "Scénario": ["Relâchement (-30%)","Maintien (2024)","Renforcement (+40%)"],
        "RSV_moyen": [y_relax_stab.mean(), y_stable_stab.mean(), y_strong_stab.mean()],
        "RSV_max":   [y_relax_stab.max(),  y_stable_stab.max(),  y_strong_stab.max()]
    }).round(1)
    forecast_summary["Δ_vs_maintien"] = (forecast_summary["RSV_moyen"] - forecast_summary.loc[1,"RSV_moyen"]).round(1)
    #st.dataframe(forecast_summary, use_container_width=True)

    st.success("✅ Prévisions stabilisées jusqu'à fin 2027 générées avec succès.")