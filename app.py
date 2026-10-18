import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model_utils import FEATURES, train_bundle

st.set_page_config(page_title="Rain Forecast", page_icon="🌧️", layout="wide")

# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
@st.cache_resource
def load_bundle():
    """Load the saved model; retrain from rain.csv if the pickle is missing or
    was built with an incompatible scikit-learn version."""
    try:
        return joblib.load("rain_model.pkl")
    except Exception:
        return train_bundle("rain.csv")


B = load_bundle()
MODEL, RANGES, MEANS, STDS, COEFS, METRICS = (
    B["model"], B["ranges"], B["means"], B["stds"], B["coefs"], B["metrics"])

LABELS = {
    "Temperature": ("Temperature", "°C"),
    "Humidity": ("Humidity", "%"),
    "Pressure": ("Pressure", "hPa"),
    "WindSpeed": ("Wind speed", "km/h"),
    "CloudCover": ("Cloud cover", "%"),
    "RainToday": ("Rain today", ""),
}

# ----------------------------------------------------------------------------
# Themes
# ----------------------------------------------------------------------------
THEMES = {
    "Monsoon Night": dict(
        bg="radial-gradient(1200px 700px at 80% -10%, #1c3b57 0%, #0c1626 55%, #070d18 100%)",
        text="#e7f1f8", muted="#8ea6b8", card="rgba(18,34,52,.62)", edge="rgba(130,190,225,.18)",
        accent="#56c8e8", accent2="#2f7fb5", good="#7ad9a6", bad="#56c8e8", drop="190,225,245"),
    "Clear Morning": dict(
        bg="radial-gradient(900px 600px at 85% -5%, #ffe9b8 0%, #cfe6fb 55%, #b7d6f2 100%)",
        text="#18283a", muted="#56697c", card="rgba(255,255,255,.62)", edge="rgba(24,40,58,.12)",
        accent="#e8791a", accent2="#f3b13e", good="#2f9b63", bad="#2f6fa8", drop="47,111,168"),
    "Storm Front": dict(
        bg="radial-gradient(1000px 650px at 20% -10%, #3a2a63 0%, #17112b 55%, #0b0814 100%)",
        text="#ece8f7", muted="#a199c0", card="rgba(30,22,52,.66)", edge="rgba(190,170,255,.2)",
        accent="#ffd75e", accent2="#a98bff", good="#7ad9a6", bad="#c1a8ff", drop="205,190,255"),
}

PRESETS = {
    "Clear day": dict(Temperature=32, Humidity=35, Pressure=1020, WindSpeed=10, CloudCover=15, RainToday=0),
    "Humid evening": dict(Temperature=28, Humidity=68, Pressure=1010, WindSpeed=22, CloudCover=50, RainToday=0),
    "Monsoon spell": dict(Temperature=24, Humidity=92, Pressure=1003, WindSpeed=32, CloudCover=90, RainToday=1),
}

DEFAULTS = PRESETS["Humid evening"]
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def apply_preset(name):
    for k, v in PRESETS[name].items():
        st.session_state[k] = v


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Appearance")
    theme_choice = st.radio(
        "Theme", ["Auto", *THEMES.keys()], index=0,
        help="Auto switches with the forecast: clear skies, monsoon night, or storm front.")
    effects = st.toggle("Weather animation", value=True)
    st.markdown("### Quick scenarios")
    for name in PRESETS:
        st.button(name, on_click=apply_preset, args=(name,), width="stretch")

# ----------------------------------------------------------------------------
# Inputs -> prediction
# ----------------------------------------------------------------------------
def make_row(values):
    return pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)


def rain_probability(values):
    return float(MODEL.predict_proba(make_row(values))[0, 1])


# Prediction is needed before drawing the theme, so read widget state first.
values = {f: st.session_state[f] for f in FEATURES}
prob = rain_probability(values)

if theme_choice == "Auto":
    theme_name = "Clear Morning" if prob < 0.35 else ("Storm Front" if prob > 0.8 else "Monsoon Night")
else:
    theme_name = theme_choice
T = THEMES[theme_name]

# ----------------------------------------------------------------------------
# CSS + weather layer
# ----------------------------------------------------------------------------
def build_css(t):
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=DM+Sans:wght@400;500;600&display=swap');
html, body, [class*="st-"], .stApp {{ font-family: 'DM Sans', system-ui, sans-serif; }}
.stApp {{ background: {t['bg']}; background-attachment: fixed; color: {t['text']}; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2.2rem; max-width: 1180px; position: relative; z-index: 2; }}
h1, h2, h3, .display {{ font-family: 'Bricolage Grotesque', sans-serif !important; color: {t['text']}; letter-spacing: -0.01em; }}
p, label, span, div[data-testid="stMarkdownContainer"] {{ color: {t['text']}; }}
.muted, .muted * {{ color: {t['muted']} !important; }}

section[data-testid="stSidebar"] {{ background: {t['card']}; backdrop-filter: blur(14px); border-right: 1px solid {t['edge']}; }}
.st-key-inputs, .st-key-result, .st-key-why, .st-key-whatif, .st-key-perf {{
    background: {t['card']}; backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
    border: 1px solid {t['edge']} !important; border-radius: 22px; padding: 1.1rem 1.3rem;
}}
.st-key-result {{ border-radius: 30px; }}

div[data-baseweb="slider"] [role="slider"] {{ background: {t['accent']} !important; border-color: {t['accent']} !important; }}
div[data-baseweb="slider"] div[data-testid="stSliderThumbValue"] {{ color: {t['accent']} !important; }}
.stButton > button {{
    background: transparent; color: {t['text']}; border: 1px solid {t['edge']}; border-radius: 999px;
    transition: border-color .2s, background .2s;
}}
.stButton > button:hover {{ border-color: {t['accent']}; background: {t['accent']}22; color: {t['text']}; }}
.stButton > button:focus-visible {{ outline: 2px solid {t['accent']}; outline-offset: 2px; }}
div[data-testid="stToggle"] label p {{ color: {t['text']}; }}

.verdict {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: clamp(2.1rem, 4.6vw, 3.4rem);
    line-height: 1.02; margin: .1rem 0 .4rem; color: {t['text']}; }}
.chip {{ display:inline-block; padding:.28rem .8rem; border-radius:999px; font-weight:600; font-size:.85rem;
    border:1px solid {t['accent']}; color:{t['accent']}; margin-bottom:.6rem; }}
.advice {{ color:{t['muted']}; font-size:1.02rem; max-width:34ch; }}
.kpi {{ display:flex; gap:1.4rem; flex-wrap:wrap; margin-top:.4rem; }}
.kpi div b {{ font-family:'Bricolage Grotesque'; font-size:1.5rem; color:{t['text']}; display:block; }}
.kpi div span {{ color:{t['muted']}; font-size:.85rem; }}

/* weather layer */
.wx {{ position: fixed; inset: 0; pointer-events: none; z-index: 1; overflow: hidden; }}
.wx i {{ position:absolute; top:-12vh; width:1.5px; height:60px;
    background: linear-gradient(to bottom, rgba({t['drop']},0), rgba({t['drop']},.55));
    animation: fall linear infinite; }}
@keyframes fall {{ to {{ transform: translateY(125vh); }} }}
.wx .sun {{ position:absolute; top:-140px; right:-120px; width:460px; height:460px; border-radius:50%;
    background: radial-gradient(circle, #fff3c4 0%, #ffd27a 38%, rgba(255,210,122,0) 70%);
    animation: glow 7s ease-in-out infinite alternate; }}
@keyframes glow {{ from {{ opacity:.75; transform:scale(.96); }} to {{ opacity:1; transform:scale(1.04); }} }}
.wx .flash {{ position:absolute; inset:0; background: rgba(255,255,255,.0); animation: flash 9s infinite; }}
@keyframes flash {{ 0%,92%,96%,100% {{ background: rgba(255,255,255,0); }} 93% {{ background: rgba(235,225,255,.22); }} 94.5% {{ background: rgba(255,255,255,0); }} 95% {{ background: rgba(235,225,255,.14); }} }}
@media (prefers-reduced-motion: reduce) {{ .wx i, .wx .sun, .wx .flash {{ animation: none; }} }}
</style>
"""


def build_weather(p, theme, on):
    if not on:
        return ""
    if theme == "Clear Morning" and p < 0.5:
        return '<div class="wx"><div class="sun"></div></div>'
    rng = random.Random(7)  # stable layout between reruns
    n = int(25 + 110 * p)
    drops = "".join(
        f'<i style="left:{rng.uniform(0, 100):.1f}%;animation-duration:{rng.uniform(.7, 1.5):.2f}s;'
        f'animation-delay:-{rng.uniform(0, 1.5):.2f}s;opacity:{rng.uniform(.35, .9):.2f}"></i>'
        for _ in range(n))
    flash = '<div class="flash"></div>' if theme == "Storm Front" else ""
    return f'<div class="wx">{drops}{flash}</div>'


st.markdown(build_css(T) + build_weather(prob, theme_name, effects), unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Chart helpers
# ----------------------------------------------------------------------------
def style_fig(fig, height):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=30, b=8), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family="DM Sans", color=T["text"]),
        showlegend=False)
    return fig


def gauge(p):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100,
        number=dict(suffix="%", font=dict(size=54, family="Bricolage Grotesque", color=T["text"]), valueformat=".0f"),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=T["muted"], tickfont=dict(color=T["muted"])),
            bar=dict(color=T["accent"], thickness=.28), bgcolor="rgba(0,0,0,0)", borderwidth=0,
            steps=[dict(range=[0, 30], color="rgba(120,180,140,.18)"),
                   dict(range=[30, 70], color="rgba(240,180,70,.16)"),
                   dict(range=[70, 100], color="rgba(110,160,230,.2)")])))
    return style_fig(fig, 270)


def contributions(vals):
    rows = []
    for f in FEATURES:
        z = (vals[f] - MEANS[f]) / STDS[f]
        rows.append((LABELS[f][0], COEFS[f] * z))
    rows.sort(key=lambda r: r[1])
    fig = go.Figure(go.Bar(
        x=[r[1] for r in rows], y=[r[0] for r in rows], orientation="h",
        marker_color=[T["bad"] if r[1] > 0 else T["good"] for r in rows],
        hovertemplate="%{y}: %{x:+.1f}<extra></extra>"))
    fig.update_xaxes(zeroline=True, zerolinecolor=T["muted"], gridcolor="rgba(128,128,128,.15)",
                     title="← points to dry      points to rain →", title_font=dict(size=12, color=T["muted"]))
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    return style_fig(fig, 300)


def what_if(vals, feature):
    lo, hi = RANGES[feature]
    xs = np.linspace(lo, hi, 60)
    probs = []
    for x in xs:
        v = dict(vals)
        v[feature] = float(x)
        probs.append(rain_probability(v) * 100)
    fig = go.Figure(go.Scatter(x=xs, y=probs, mode="lines", line=dict(color=T["accent"], width=3),
                               fill="tozeroy", fillcolor="rgba(128,128,128,.10)",
                               hovertemplate="%{x:.0f} → %{y:.0f}%<extra></extra>"))
    fig.add_vline(x=vals[feature], line_dash="dot", line_color=T["text"])
    fig.add_hline(y=50, line_dash="dot", line_color=T["muted"], opacity=.5)
    fig.update_yaxes(range=[0, 100], title="Chance of rain (%)", gridcolor="rgba(128,128,128,.15)")
    fig.update_xaxes(title=f"{LABELS[feature][0]} ({LABELS[feature][1]})" if LABELS[feature][1] else LABELS[feature][0],
                     gridcolor="rgba(0,0,0,0)")
    return style_fig(fig, 300)


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
st.markdown("# Tomorrow's rain forecast")
st.markdown('<p class="muted">Move the sliders to describe today\'s conditions. The forecast updates as you go.</p>',
            unsafe_allow_html=True)

left, right = st.columns([5, 6], gap="large")

with left:
    with st.container(key="inputs"):
        st.markdown("#### Today's conditions")
        for f in ["Temperature", "Humidity", "Pressure", "WindSpeed", "CloudCover"]:
            lo, hi = RANGES[f]
            name, unit = LABELS[f]
            st.slider(f"{name} ({unit})", int(lo), int(hi), key=f, step=1)
        st.radio("Did it rain today?", [0, 1], key="RainToday", horizontal=True,
                 format_func=lambda v: "Yes" if v else "No")

with right:
    with st.container(key="result"):
        p = prob
        if p >= 0.7:
            chip, verdict, tip = "High chance", "Rain is likely tomorrow", "Pack an umbrella and plan indoor backups."
        elif p >= 0.4:
            chip, verdict, tip = "Uncertain", "Rain is a toss-up", "A light jacket and a small umbrella are a safe bet."
        else:
            chip, verdict, tip = "Low chance", "Mostly dry tomorrow", "Outdoor plans look fine."
        a, b = st.columns([6, 5])
        with a:
            st.markdown(f'<span class="chip">{chip}</span><div class="verdict">{verdict}</div>'
                        f'<p class="advice">{tip}</p>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="kpi"><div><b>{p*100:.0f}%</b><span>rain</span></div>'
                f'<div><b>{(1-p)*100:.0f}%</b><span>dry</span></div></div>', unsafe_allow_html=True)
        with b:
            st.plotly_chart(gauge(p), width="stretch", config={"displayModeBar": False})

st.write("")
c1, c2 = st.columns(2, gap="large")
with c1:
    with st.container(key="why"):
        st.markdown("#### What is driving this forecast")
        st.markdown('<p class="muted">Each bar shows how far a reading pushes the forecast toward rain or dry, '
                    'compared with typical conditions.</p>', unsafe_allow_html=True)
        st.plotly_chart(contributions(values), width="stretch", config={"displayModeBar": False})
with c2:
    with st.container(key="whatif"):
        st.markdown("#### What if this changed?")
        pick = st.selectbox("Vary", FEATURES[:5], format_func=lambda f: LABELS[f][0],
                            index=2, label_visibility="collapsed")
        st.plotly_chart(what_if(values, pick), width="stretch", config={"displayModeBar": False})

st.write("")
with st.expander("How reliable is the model?"):
    with st.container(key="perf"):
        m = METRICS
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Accuracy", f"{m['accuracy']*100:.1f}%")
        k2.metric("Precision", f"{m['precision']*100:.1f}%")
        k3.metric("Recall", f"{m['recall']*100:.1f}%")
        k4.metric("ROC AUC", f"{m['roc_auc']:.3f}")
        (tn, fp), (fn, tp) = m["confusion"]
        st.markdown(
            f"Scored on {m['n_test']:,} days the model never saw during training. It called {tp + tn:,} correctly, "
            f"missed {fn} rainy days and raised {fp} false alarms. Five-fold cross-validation on the training set gave "
            f"{m['cv_accuracy']*100:.2f}% ± {m['cv_std']*100:.2f}%.")
        st.caption("This dataset is highly regular, so scores are very high. Real-world weather data is noisier, "
                   "so expect lower accuracy if you retrain on live observations.")
