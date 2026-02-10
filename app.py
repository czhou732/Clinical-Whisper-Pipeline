"""
Clinical Intelligence Report
─────────────────────────────
Neuro-Systems Group | Clinical Audio Pipeline

Enterprise clinical-grade Streamlit dashboard for offline audio analysis.
Styled as a digitized medical report / academic paper.

Run:  streamlit run app.py
"""

import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Clinical Intelligence Report",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL CSS — "PAPER" AESTHETIC
# ═══════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
<style>
/* ── Hide Streamlit chrome ── */
#MainMenu, header, footer {visibility: hidden !important;}
div[data-testid="stToolbar"] {display: none !important;}

/* ── Global: Times New Roman, pure white ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"],
h1, h2, h3, h4, h5, h6, p, div, span, label, button, input, textarea, select,
[data-testid="stMarkdownContainer"], [data-testid="stMetric"] * {
    font-family: 'Times New Roman', Times, serif !important;
    color: #000000 !important;
}
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #FFFFFF !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #F9FAFB !important;
    border-right: 1px solid #000000 !important;
}

/* ── Metric tiles: no shadow, no card, just data fields ── */
div[data-testid="stMetric"] {
    background: transparent !important;
    border: none !important;
    border-bottom: 1px solid #000000 !important;
    border-radius: 0px !important;
    padding: 8px 4px 12px 4px !important;
    box-shadow: none !important;
}
div[data-testid="stMetric"] label {
    font-size: 11px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 28px !important;
    font-weight: 400 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    font-size: 11px !important;
}

/* ── Kill all rounded corners globally ── */
div, section, button, input, textarea, select,
[data-testid="stFileUploader"], [data-testid="stFileUploader"] *,
.stPlotlyChart, div[data-testid="stMetric"] {
    border-radius: 0px !important;
}

/* ── Section rule ── */
.section-rule {
    font-size: 11px;
    font-weight: 700;
    color: #000000;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-bottom: 4px;
    padding-bottom: 4px;
    border-bottom: 2px solid #000000;
}

/* ── Report header ── */
.report-title {
    text-align: center;
    font-size: 20px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    text-decoration: underline;
    color: #000000;
    margin-bottom: 2px;
    padding-top: 8px;
}
.report-subtitle {
    text-align: center;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #000000;
    margin-bottom: 16px;
}

/* ── Plotly: kill border, clean embed ── */
.stPlotlyChart {
    border: 1px solid #000000 !important;
    background: #FFFFFF !important;
    box-shadow: none !important;
}

/* ── Transcript document ── */
.transcript-doc {
    background: #FFFFFF;
    border: 1px solid #000000;
    padding: 28px 36px;
    margin-bottom: 12px;
    font-family: 'Times New Roman', Times, serif;
    font-size: 16px;
    line-height: 1.5;
    color: #000000;
}
.record-line {
    margin-bottom: 6px;
    color: #000000;
}
.record-ts {
    font-family: 'Courier New', Courier, monospace;
    font-size: 13px;
    color: #000000;
}
.record-speaker {
    font-weight: 700;
    text-transform: uppercase;
    color: #000000;
    font-family: 'Times New Roman', Times, serif;
}
.record-text {
    color: #000000;
    font-family: 'Times New Roman', Times, serif;
}

/* ── Footer ── */
.footer-text {
    font-size: 11px;
    color: #000000;
    text-align: center;
    padding: 8px 0;
    border-top: 1px solid #000000;
    margin-top: 16px;
}

/* ── Streamlit dividers ── */
hr { border-color: #000000 !important; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# MOCK DATA
# ═══════════════════════════════════════════════════════════════════════════

def generate_mock_data() -> list[dict]:
    """
    Simulates a ~3-minute de-identified clinical interview
    about sleep patterns and anhedonia (Patient 042).
    """
    return [
        {"start": 0.0,  "end": 5.2,  "speaker": "Clinician", "text": "Good morning. I'd like to start by checking in on your sleep patterns since our last session. How have the nights been?", "sentiment": 7.5},
        {"start": 5.5,  "end": 12.8, "speaker": "Patient",   "text": "Honestly, not great. I've been waking up around three or four in the morning and then I just can't get back to sleep. I end up lying there staring at the ceiling.", "sentiment": 3.2},
        {"start": 13.0, "end": 18.0, "speaker": "Clinician", "text": "That early-morning waking pattern is something we should track carefully. When you wake at three, what's going through your mind?", "sentiment": 6.8},
        {"start": 18.5, "end": 27.3, "speaker": "Patient",   "text": "It's mostly just this heavy feeling. Like dread, I guess. Not about anything specific. I just feel like the day ahead doesn't have anything I'm looking forward to.", "sentiment": 2.1},
        {"start": 27.5, "end": 34.0, "speaker": "Clinician", "text": "That sounds like it could be related to the anhedonia we discussed. The difficulty finding pleasure or motivation in activities you used to enjoy. Does that still feel accurate?", "sentiment": 6.5},
        {"start": 34.5, "end": 43.0, "speaker": "Patient",   "text": "Yeah, very much so. I used to play guitar every day. Now I see it in the corner and I don't even want to pick it up. It's like the connection just isn't there anymore.", "sentiment": 2.5},
        {"start": 43.5, "end": 52.0, "speaker": "Clinician", "text": "I appreciate you sharing that. The guitar example is helpful — it gives us a concrete marker to track. Are there any activities that still bring some sense of engagement, even small ones?", "sentiment": 7.2},
        {"start": 52.5, "end": 60.0, "speaker": "Patient",   "text": "Walking the dog, maybe. I don't feel excited about it, but once I'm outside, the fresh air helps a little. It doesn't fix anything but it's... less heavy.", "sentiment": 4.8},
        {"start": 60.5, "end": 68.0, "speaker": "Clinician", "text": "That's actually a meaningful observation. The fact that environmental change produces even a small shift in affect is a positive sign. It suggests your capacity for engagement is still there, just suppressed.", "sentiment": 8.0},
        {"start": 68.5, "end": 78.0, "speaker": "Patient",   "text": "I hadn't thought of it that way. I keep feeling like something is broken. But maybe it's more like it's... muted?", "sentiment": 5.5},
        {"start": 78.5, "end": 87.0, "speaker": "Clinician", "text": "Muted is a very good word for it. And importantly, muted is not permanent. Let's talk about the sleep hygiene adjustments we discussed. Have you been able to limit screen time before bed?", "sentiment": 7.8},
        {"start": 87.5, "end": 96.0, "speaker": "Patient",   "text": "Some nights, yes. But when I can't sleep, I end up reaching for my phone, which I know makes it worse. It's a cycle.", "sentiment": 3.8},
        {"start": 96.5, "end": 105.0,"speaker": "Clinician", "text": "Cycles are breakable. Let's think about what we could put in place of the phone. What about the breathing exercises we practiced? Even two or three minutes can help downregulate the nervous system.", "sentiment": 7.5},
        {"start": 105.5,"end": 115.0,"speaker": "Patient",   "text": "I tried them a couple of times. It felt awkward at first, but the second time I actually fell back asleep within twenty minutes. That was kind of surprising.", "sentiment": 6.2},
        {"start": 115.5,"end": 125.0,"speaker": "Clinician", "text": "That's significant progress. The fact that it worked even once shows your system is responsive to those interventions. I'd like you to try it consistently for the next two weeks. Can we make that a goal?", "sentiment": 8.5},
        {"start": 125.5,"end": 133.0,"speaker": "Patient",   "text": "Yeah, I think I can do that. It's a small enough thing that it doesn't feel overwhelming.", "sentiment": 6.8},
        {"start": 133.5,"end": 142.0,"speaker": "Clinician", "text": "Good. Small and sustainable is exactly the approach here. We're not trying to fix everything at once — just finding the levers that help. Let's also schedule a follow-up in ten days.", "sentiment": 8.2},
        {"start": 142.5,"end": 150.0,"speaker": "Patient",   "text": "Okay. I actually feel a bit better about things after talking through this. Not fixed, but... clearer, maybe.", "sentiment": 6.5},
        {"start": 150.5,"end": 158.0,"speaker": "Clinician", "text": "Clearer is exactly where we want to be heading. You're doing the work, and I want you to recognize that. Same time next week works?", "sentiment": 8.0},
        {"start": 158.5,"end": 162.0,"speaker": "Patient",   "text": "Yes, that works. Thank you.", "sentiment": 7.0},
    ]


def _ts(seconds: float) -> str:
    """Format seconds as MM:SS."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("#### Neuro-Systems Group")
    st.caption("Clinical Audio Pipeline · v2.1")
    st.divider()

    demo_mode = st.checkbox("Load Demo Patient (Patient 042)", value=True)

    st.divider()
    uploaded = st.file_uploader(
        "Upload Analysis",
        type=["json"],
        help="Upload a JSON export from ClinicalWhisper.",
    )

    st.divider()
    st.caption("All data processed locally. No network egress. HIPAA-compliant architecture.")

# ═══════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════

data: list[dict] = []

if uploaded is not None:
    try:
        data = json.load(uploaded)
        if not isinstance(data, list):
            data = data.get("segments", [])
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")
        data = []

if not data and demo_mode:
    data = generate_mock_data()

if not data:
    st.info("Upload a JSON analysis file or enable Demo Patient to begin.")
    st.stop()

df = pd.DataFrame(data)

# ═══════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown(
    '<p class="report-title">Clinical Intelligence Report</p>'
    '<p class="report-subtitle">Confidential // Neuro-Systems Group</p>',
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — SESSION VITALS
# ═══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-rule">I. Session Vitals</div>', unsafe_allow_html=True)

overall_sentiment = round(df["sentiment"].mean(), 1)
clinician_words = df.loc[df["speaker"] == "Clinician", "text"].str.split().str.len().sum()
patient_words = df.loc[df["speaker"] == "Patient", "text"].str.split().str.len().sum()
total_words = clinician_words + patient_words
dominance_pct = round(clinician_words / total_words * 100, 1) if total_words else 0
conflict_events = int((df["sentiment"] <= 3.0).sum())

v1, v2, v3 = st.columns(3)
v1.metric("Overall Sentiment", f"{overall_sentiment} / 10", delta="Positive" if overall_sentiment >= 5 else "At Risk")
v2.metric("Clinician Talk-Time", f"{dominance_pct}%", delta=f"{clinician_words} words")
v3.metric("Distress Segments", conflict_events, delta="flagged" if conflict_events >= 3 else "within range")

st.markdown("")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — EMOTIONAL ARC (FIGURE 1)
# ═══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-rule">II. Emotional Arc (Figure 1)</div>', unsafe_allow_html=True)

fig = px.line(
    df,
    x="start",
    y="sentiment",
    color="speaker",
    markers=True,
    color_discrete_map={"Clinician": "#000080", "Patient": "#800000"},
    labels={"start": "Time (s)", "sentiment": "Sentiment Score", "speaker": "Speaker"},
)

# Distress threshold
fig.add_hline(
    y=3.0,
    line_dash="dot",
    line_color="#555555",
    line_width=1,
    annotation_text="Distress Threshold",
    annotation_position="top left",
    annotation_font_color="#555555",
    annotation_font_size=11,
)

fig.update_layout(
    template="plotly_white",
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    margin=dict(l=50, r=20, t=24, b=50),
    height=300,
    font=dict(family="Times New Roman", size=14, color="black"),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(family="Times New Roman", size=12, color="black"),
    ),
    xaxis=dict(
        showgrid=False,
        zeroline=False,
        linecolor="#000000",
        linewidth=1,
        title_font=dict(family="Times New Roman", size=13, color="black"),
        tickfont=dict(family="Times New Roman", size=11, color="black"),
    ),
    yaxis=dict(
        showgrid=False,
        zeroline=False,
        range=[0, 10],
        dtick=2,
        linecolor="#000000",
        linewidth=1,
        title_font=dict(family="Times New Roman", size=13, color="black"),
        tickfont=dict(family="Times New Roman", size=11, color="black"),
    ),
)

fig.update_traces(
    line=dict(width=2),
    marker=dict(size=5),
)

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.markdown("")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — SESSION TRANSCRIPT
# ═══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-rule">III. Session Transcript</div>', unsafe_allow_html=True)

transcript_html = '<div class="transcript-doc">'

for _, row in df.iterrows():
    ts = _ts(row["start"])
    text = row["text"]
    speaker = row["speaker"]
    sentiment = row["sentiment"]

    flag = "[!] " if sentiment < 4.0 else ""

    transcript_html += (
        f'<div class="record-line">'
        f'<span class="record-ts">[{ts}]</span> '
        f'<span class="record-speaker">{speaker.upper()}:</span> '
        f'{flag}'
        f'<span class="record-text">{text}</span>'
        f'</div>'
    )

transcript_html += "</div>"

st.markdown(transcript_html, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown(
    '<div class="footer-text">'
    "Neuro-Systems Group · Clinical Audio Pipeline · "
    "All data processed locally · HIPAA-compliant architecture · "
    "ClinicalWhisper v2.1 · "
    "This document is confidential and intended solely for authorized clinical personnel."
    "</div>",
    unsafe_allow_html=True,
)
