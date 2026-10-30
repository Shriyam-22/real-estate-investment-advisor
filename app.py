"""
Real Estate Investment Advisor — Streamlit app
Run:  streamlit run app.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from features import AMENITY_LIST, BASE_GROWTH, HORIZON_YEARS, MODEL_FEATURES, add_features  # noqa: E402

st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏠", layout="wide")

ACCENT = "#0F766E"
st.markdown(f"""
<style>
.block-container {{padding-top: 1.6rem;}}
.verdict {{border-radius: 14px; padding: 18px 22px; color: white; margin-bottom: 8px;}}
.good {{background: linear-gradient(135deg, #0F766E, #14B8A6);}}
.bad  {{background: linear-gradient(135deg, #9F1239, #E11D48);}}
.verdict h2 {{margin: 0; color: white; font-size: 1.6rem;}}
.verdict p {{margin: 4px 0 0; opacity: .92;}}
div[data-testid="stMetricValue"] {{color: {ACCENT};}}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ loaders
def models_version() -> str:
    """Fingerprint of the model files, so the cache reloads after a retrain + push."""
    import hashlib
    h = hashlib.md5()
    for f in ["classifier.joblib", "regressor.joblib", "metrics.json"]:
        h.update((ROOT / "models" / f).read_bytes())
    return h.hexdigest()


@st.cache_resource
def load_models(version: str):
    clf = joblib.load(ROOT / "models" / "classifier.joblib")
    reg = joblib.load(ROOT / "models" / "regressor.joblib")
    city_med = json.loads((ROOT / "models" / "city_median_ppsf.json").read_text())
    metrics = json.loads((ROOT / "models" / "metrics.json").read_text())
    return clf, reg, city_med, metrics


@st.cache_data
def load_data():
    return pd.read_parquet(ROOT / "data" / "app_data.parquet")


clf, reg, CITY_MED, METRICS = load_models(models_version())
df = load_data()
CITY_STATE = df.groupby("City")["State"].first().to_dict()


def fmt_lakh(x):
    return f"₹{x/100:,.2f} Cr" if x >= 100 else f"₹{x:,.1f} L"


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🏠 Investment Advisor")
page = st.sidebar.radio("Go to", ["Predict a property", "Explore listings", "Market insights", "Model performance"])
st.sidebar.markdown("---")
st.sidebar.caption(f"Dataset: {len(df):,} properties · {df.City.nunique()} cities · {df.State.nunique()} states")
st.sidebar.caption(f"Models: {METRICS['best_classifier']} classifier · {METRICS['best_regressor']} regressor "
                   "(tracked in MLflow)")


# ================================================================== PAGE 1
if page == "Predict a property":
    st.title("Is this property a good investment?")
    st.caption("Enter property details to get an investment verdict and a 5-year price forecast.")

    with st.form("predict"):
        c1, c2, c3 = st.columns(3)
        with c1:
            city = st.selectbox("City", sorted(CITY_STATE), index=sorted(CITY_STATE).index("Bangalore"))
            ptype = st.selectbox("Property type", ["Apartment", "Independent House", "Villa"])
            bhk = st.slider("BHK", 1, 5, 3)
            size = st.number_input("Size (sq ft)", 300, 10000, 1500, step=50)
            price = st.number_input("Current price (₹ Lakhs)", 5.0, 1000.0, 120.0, step=5.0)
        with c2:
            year = st.slider("Year built", 1990, 2025, 2015)
            total_floors = st.slider("Total floors in building", 1, 30, 12)
            floor = st.slider("Floor number", 0, 30, 5)
            furnished = st.selectbox("Furnished status", ["Unfurnished", "Semi-furnished", "Furnished"])
            facing = st.selectbox("Facing", ["North", "East", "South", "West"])
        with c3:
            schools = st.slider("Nearby schools (1–10)", 1, 10, 6)
            hospitals = st.slider("Nearby hospitals (1–10)", 1, 10, 5)
            transport = st.select_slider("Public transport access", ["Low", "Medium", "High"], value="Medium")
            avail = st.radio("Availability", ["Ready_to_Move", "Under_Construction"], horizontal=True,
                             format_func=lambda s: s.replace("_", " "))
            owner = st.selectbox("Listed by", ["Owner", "Builder", "Broker"])
        c4, c5, c6 = st.columns([2, 1, 1])
        amenities = c4.multiselect("Amenities", AMENITY_LIST, default=["Gym", "Garden", "Playground"])
        parking = c5.toggle("Parking", True)
        security = c6.toggle("Security", True)
        submitted = st.form_submit_button("Analyse investment", type="primary", use_container_width=True)

    if submitted:
        row = pd.DataFrame([{
            "State": CITY_STATE[city], "City": city, "Locality": "User", "Property_Type": ptype,
            "BHK": bhk, "Size_in_SqFt": size, "Price_in_Lakhs": price, "Year_Built": year,
            "Furnished_Status": furnished, "Floor_No": min(floor, total_floors), "Total_Floors": total_floors,
            "Age_of_Property": 2025 - year, "Nearby_Schools": schools, "Nearby_Hospitals": hospitals,
            "Public_Transport_Accessibility": transport, "Parking_Space": "Yes" if parking else "No",
            "Security": "Yes" if security else "No", "Amenities": ", ".join(amenities) or "None",
            "Facing": facing, "Owner_Type": owner, "Availability_Status": avail,
        }])
        feats = add_features(row, CITY_MED)
        X = feats[MODEL_FEATURES]
        p_good = float(clf.predict_proba(X)[0, 1])
        is_good = p_good >= 0.5
        future = float(reg.predict(X)[0])
        cagr = (future / price) ** (1 / HORIZON_YEARS) - 1
        confidence = p_good if is_good else 1 - p_good

        left, right = st.columns([1.1, 1])
        with left:
            cls = "good" if is_good else "bad"
            title = "✅ Good Investment" if is_good else "⚠️ Not a Good Investment"
            st.markdown(f'<div class="verdict {cls}"><h2>{title}</h2>'
                        f'<p>Model confidence: <b>{confidence:.1%}</b> · P(good) = {p_good:.3f}</p></div>',
                        unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("Price today", fmt_lakh(price))
            m2.metric("Est. price in 5 yrs", fmt_lakh(future), f"+{future - price:,.1f} L")
            m3.metric("Implied annual growth", f"{cagr:.1%}", f"{(cagr - BASE_GROWTH)*100:+.1f} pts vs 8%")

            ppsf = feats["Price_per_SqFt"].iloc[0]
            rel = feats["PPSF_vs_City"].iloc[0]
            st.markdown("##### Why the model thinks so")
            checks = [
                (rel <= 1.0, f"Price/sq ft ₹{ppsf:,.0f} vs {city} median ₹{CITY_MED[city]:,.0f} ({rel:.2f}×)"),
                (bhk >= 3, f"{bhk} BHK (3+ preferred for resale)"),
                (avail == "Ready_to_Move", avail.replace("_", " ")),
                (transport != "Low", f"Public transport: {transport}"),
                (len(amenities) >= 3, f"{len(amenities)} amenities (3+ preferred)"),
                (security, "Security available" if security else "No security"),
                (cagr >= BASE_GROWTH, f"Projected growth {cagr:.1%} vs 8% baseline"),
            ]
            for ok, txt in checks:
                st.markdown(f"{'🟢' if ok else '🔴'} {txt}")

        with right:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=p_good * 100, number={"suffix": "%"},
                title={"text": "Probability of good investment"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": ACCENT if is_good else "#E11D48"},
                       "steps": [{"range": [0, 50], "color": "#FCE7EB"}, {"range": [50, 100], "color": "#DDF4F1"}],
                       "threshold": {"line": {"color": "black", "width": 2}, "value": 50}}))
            gauge.update_layout(height=250, margin=dict(t=50, b=10, l=20, r=20))
            st.plotly_chart(gauge, use_container_width=True)

            yrs = np.arange(0, HORIZON_YEARS + 1)
            traj = pd.DataFrame({"Year": 2025 + yrs,
                                 "Model forecast": price * (1 + cagr) ** yrs,
                                 "Flat 8% baseline": price * (1 + BASE_GROWTH) ** yrs})
            fig = px.line(traj.melt("Year", var_name="Scenario", value_name="₹ Lakhs"), x="Year", y="₹ Lakhs",
                          color="Scenario", markers=True,
                          color_discrete_map={"Model forecast": ACCENT, "Flat 8% baseline": "#94A3B8"})
            fig.update_layout(height=280, margin=dict(t=30, b=10), legend=dict(orientation="h", y=1.15))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### Similar properties in " + city)
        sim = df[(df.City == city) & (df.BHK == bhk) & (df.Property_Type == ptype)]
        sim = sim.assign(gap=(sim.Price_in_Lakhs - price).abs()).nsmallest(8, "gap")
        st.dataframe(sim[["Locality", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt", "Future_Price_5Y",
                          "Availability_Status", "Good_Investment"]], hide_index=True, use_container_width=True)


# ================================================================== PAGE 2
elif page == "Explore listings":
    st.title("Explore listings")
    f1, f2, f3 = st.columns(3)
    states = f1.multiselect("State", sorted(df.State.unique()))
    cities_pool = sorted(df[df.State.isin(states)].City.unique()) if states else sorted(df.City.unique())
    cities = f2.multiselect("City", cities_pool)
    ptypes = f3.multiselect("Property type", sorted(df.Property_Type.unique()))
    g1, g2, g3, g4 = st.columns(4)
    pr = g1.slider("Price (₹ L)", 10, 500, (50, 300))
    sz = g2.slider("Size (sq ft)", 500, 5000, (800, 3000), step=50)
    bhks = g3.multiselect("BHK", [1, 2, 3, 4, 5], default=[2, 3, 4])
    only_good = g4.toggle("Good investments only", False)

    q = df[df.Price_in_Lakhs.between(*pr) & df.Size_in_SqFt.between(*sz)]
    if states: q = q[q.State.isin(states)]
    if cities: q = q[q.City.isin(cities)]
    if ptypes: q = q[q.Property_Type.isin(ptypes)]
    if bhks: q = q[q.BHK.isin(bhks)]
    if only_good: q = q[q.Good_Investment == 1]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Matching properties", f"{len(q):,}")
    if len(q):
        k2.metric("Median price", fmt_lakh(q.Price_in_Lakhs.median()))
        k3.metric("Median ₹/sq ft", f"₹{q.Price_per_SqFt.median():,.0f}")
        k4.metric("Good-investment share", f"{q.Good_Investment.mean():.0%}")
        sort_by = st.selectbox("Sort by", ["Expected_Return_5Y_pct", "Price_in_Lakhs", "Price_per_SqFt",
                                           "Investment_Score"], format_func=lambda s: s.replace("_", " "))
        st.dataframe(q.sort_values(sort_by, ascending=sort_by in ("Price_in_Lakhs", "Price_per_SqFt")).head(500)[
            ["City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt",
             "Future_Price_5Y", "Expected_Return_5Y_pct", "Availability_Status", "Investment_Score",
             "Good_Investment"]], hide_index=True, use_container_width=True, height=420)
        st.download_button("Download filtered listings (CSV)", q.to_csv(index=False), "filtered_listings.csv")
    else:
        st.info("No properties match these filters.")


# ================================================================== PAGE 3
elif page == "Market insights":
    st.title("Market insights")
    metric = st.radio("Metric", ["Price_per_SqFt", "Price_in_Lakhs", "Expected_Return_5Y_pct", "Good_Investment"],
                      horizontal=True, format_func=lambda s: {"Price_per_SqFt": "₹ / sq ft",
                                                              "Price_in_Lakhs": "Avg price (₹ L)",
                                                              "Expected_Return_5Y_pct": "5-yr return %",
                                                              "Good_Investment": "% good investments"}[s])

    agg = df.groupby(["State", "City"], as_index=False).agg(value=(metric, "mean"), listings=("ID", "count"))
    fig = px.treemap(agg, path=["State", "City"], values="listings", color="value",
                     color_continuous_scale="Teal", title="Location heatmap — size = listings, colour = metric")
    fig.update_layout(height=480, margin=dict(t=40, l=0, r=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

    a, b = st.columns(2)
    hm = df.pivot_table(index="City", columns="BHK", values=metric, aggfunc="mean")
    hm = hm.loc[hm.mean(axis=1).sort_values(ascending=False).index[:15]]
    fig = px.imshow(hm, color_continuous_scale="Teal", aspect="auto", title="Top 15 cities × BHK")
    fig.update_layout(height=460)
    a.plotly_chart(fig, use_container_width=True)

    trend = df.groupby(["Year_Built", "Property_Type"], as_index=False)[metric].mean()
    fig = px.line(trend, x="Year_Built", y=metric, color="Property_Type", title="Trend by year built")
    fig.update_layout(height=460)
    b.plotly_chart(fig, use_container_width=True)

    c, d = st.columns(2)
    tp = df.groupby(["Public_Transport_Accessibility", "Availability_Status"], as_index=False)["Good_Investment"].mean()
    fig = px.bar(tp, x="Public_Transport_Accessibility", y="Good_Investment", color="Availability_Status",
                 barmode="group", category_orders={"Public_Transport_Accessibility": ["Low", "Medium", "High"]},
                 title="Good-investment share by transport & availability", labels={"Good_Investment": "Share"})
    fig.update_yaxes(tickformat=".0%")
    c.plotly_chart(fig, use_container_width=True)

    sample = df.sample(4000, random_state=3)
    fig = px.scatter(sample, x="Infrastructure_Score", y="Expected_Return_5Y_pct", color="Good_Investment",
                     opacity=.5, color_continuous_scale=["#E11D48", ACCENT],
                     title="Infrastructure score vs 5-yr expected return")
    d.plotly_chart(fig, use_container_width=True)


# ================================================================== PAGE 4
else:
    st.title("Model performance")
    st.caption("All runs tracked in MLflow — run `mlflow ui --backend-store-uri sqlite:///mlflow.db` to browse.")
    t1, t2 = st.tabs(["Classification — Good Investment", "Regression — Price in 5 years"])
    with t1:
        cm = pd.DataFrame(METRICS["classification"]).T.round(4)
        st.dataframe(cm.style.highlight_max(axis=0, color="#CCEFEA"), use_container_width=True)
        a, b = st.columns([1, 1.3])
        img = ROOT / "reports" / "models" / f"cm_{METRICS['best_classifier']}.png"
        if img.exists():
            a.image(str(img), caption="Confusion matrix (best model, test set)")
        imp = pd.Series(METRICS["clf_importance"]).head(15)[::-1]
        fig = px.bar(imp, orientation="h", title="Top 15 feature importances",
                     color_discrete_sequence=[ACCENT])
        fig.update_layout(showlegend=False, height=460, xaxis_title="importance", yaxis_title="")
        b.plotly_chart(fig, use_container_width=True)
    with t2:
        rm = pd.DataFrame(METRICS["regression"]).T.round(4)
        st.dataframe(rm.style.highlight_min(subset=["rmse", "mae", "mape_pct"], color="#CCEFEA")
                     .highlight_max(subset=["r2"], color="#CCEFEA"), use_container_width=True)
        a, b = st.columns([1, 1.3])
        img = ROOT / "reports" / "models" / f"avp_{METRICS['best_regressor']}.png"
        if img.exists():
            a.image(str(img), caption="Actual vs predicted (best model, test set)")
        imp = pd.Series(METRICS["reg_importance"]).head(15)[::-1]
        fig = px.bar(imp, orientation="h", title="Top 15 feature importances",
                     color_discrete_sequence=[ACCENT])
        fig.update_layout(showlegend=False, height=460, xaxis_title="importance", yaxis_title="")
        b.plotly_chart(fig, use_container_width=True)
    st.info("Targets are engineered from domain rules (see README → Target definitions), so near-perfect "
            "scores mean the models have learned those rules well — not that real-market prices are this predictable.")
