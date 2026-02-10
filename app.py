"""
Clinical Intelligence Dashboard
────────────────────────────────
Neuro-Systems Group | Clinical Audio Pipeline

Enterprise clinical-grade Streamlit dashboard for offline audio analysis.
Reads JSON exports from ClinicalWhisper or generates demo data.

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
    page_title="Clinical Intelligence Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
# ENTERPRISE LIGHT-MODE CSS
# ═══════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
<style>
/* ── Hide Streamlit chrome ── */
#MainMenu, header, footer {visibility: hidden !important;}
div[data-testid="stToolbar"] {display: none !important;}

/* ── Force light mode everywhere ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #F8F9FA !important;
    color: #111827 !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E5E7EB !important;
}
[data-testid="stSidebar"] * { color: #374151 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {
    color: #111827 !important;
}

/* ── All text ── */
h1, h2, h3, h4, h5, h6 { color: #111827 !important; font-weight: 600 !important; }
p, span, label, div { color: #374151; }

/* ── Card containers ── */
div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    padding: 16px 20px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
div[data-testid="stMetric"] label {
    color: #6B7280 !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #111827 !important;
    font-weight: 700 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    font-size: 11px !important;
}

/* ── Section headers ── */
.section-head {
    font-size: 11px;
    font-weight: 600;
    color: #9CA3AF;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 8px;
    padding-bottom: 8px;
    border-bottom: 1px solid #E5E7EB;
}

/* ── White card wrapper ── */
.card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 12px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}

/* ── Page header ── */
.page-title {
    font-size: 22px;
    font-weight: 700;
    color: #111827;
    letter-spacing: -0.4px;
    margin: 0;
}
.page-badge {
    display: inline-block;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 4px;
    border: 1px solid #E5E7EB;
    color: #6B7280;
    background: #F9FAFB;
    margin-right: 6px;
}

/* ── Clinical transcript record ── */
.record-line {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 10px 14px;
    border-bottom: 1px solid #F3F4F6;
    font-size: 13px;
    line-height: 1.6;
    color: #374151;
    transition: background 0.15s;
}
.record-line:hover { background: #FAFBFC; }
.record-line.critical {
    background: #FEF2F2;
    border-left: 3px solid #FCA5A5;
    padding-left: 11px;
}
.record-line.critical:hover { background: #FEE2E2; }
.record-ts {
    font-family: 'SF Mono', 'Menlo', 'Consolas', monospace;
    font-size: 11px;
    color: #9CA3AF;
    white-space: nowrap;
    padding-top: 2px;
    min-width: 44px;
}
.record-speaker {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    white-space: nowrap;
    padding-top: 2px;
    min-width: 72px;
}
.record-speaker.clinician { color: #2563EB; }
.record-speaker.patient { color: #6B7280; }
.record-text { flex: 1; color: #1F2937; }

/* ── Plotly container ── */
.stPlotlyChart {
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    overflow: hidden;
    background: #FFFFFF;
}

/* ── Footer ── */
.footer-text {
    font-size: 11px;
    color: #9CA3AF;
    text-align: center;
    padding: 8px 0;
}
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
    '<p class="page-title">Clinical Intelligence Dashboard</p>'
    '<span class="page-badge">Privacy-Preserving</span>'
    '<span class="page-badge">Offline-First</span>'
    '<span class="page-badge">De-Identified</span>',
    unsafe_allow_html=True,
)
st.markdown("")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — SESSION VITALS
# ═══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-head">Session Vitals</div>', unsafe_allow_html=True)

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
# SECTION 2 — EMOTIONAL ARC
# ═══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-head">Emotional Arc</div>', unsafe_allow_html=True)

fig = px.line(
    df,
    x="start",
    y="sentiment",
    color="speaker",
    markers=True,
    color_discrete_map={"Clinician": "#2563EB", "Patient": "#9CA3AF"},
    labels={"start": "Time (s)", "sentiment": "Sentiment Score", "speaker": "Speaker"},
)

# Distress threshold
fig.add_hline(
    y=3.0,
    line_dash="dot",
    line_color="#EF4444",
    line_width=1,
    annotation_text="Distress Threshold",
    annotation_position="top left",
    annotation_font_color="#DC2626",
    annotation_font_size=10,
)

fig.update_layout(
    template="plotly_white",
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    margin=dict(l=40, r=20, t=24, b=40),
    height=300,
    font=dict(family="-apple-system, BlinkMacSystemFont, sans-serif", color="#6B7280", size=11),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=11, color="#374151"),
    ),
    xaxis=dict(
        gridcolor="#F3F4F6",
        zeroline=False,
        title_font=dict(size=11, color="#9CA3AF"),
        linecolor="#E5E7EB",
        linewidth=1,
    ),
    yaxis=dict(
        gridcolor="#F3F4F6",
        zeroline=False,
        range=[0, 10],
        dtick=2,
        title_font=dict(size=11, color="#9CA3AF"),
        linecolor="#E5E7EB",
        linewidth=1,
    ),
)

fig.update_traces(
    line=dict(width=2),
    marker=dict(size=5),
)

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.markdown("")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — CLINICAL TRANSCRIPT RECORD
# ═══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-head">Session Transcript</div>', unsafe_allow_html=True)

transcript_html = '<div class="card">'

for _, row in df.iterrows():
    ts = _ts(row["start"])
    text = row["text"]
    speaker = row["speaker"]
    sentiment = row["sentiment"]

    is_critical = sentiment < 4.0
    line_class = "record-line critical" if is_critical else "record-line"
    speaker_class = "clinician" if speaker == "Clinician" else "patient"

    transcript_html += f"""
    <div class="{line_class}">
        <span class="record-ts">{ts}</span>
        <span class="record-speaker {speaker_class}">{speaker}</span>
        <span class="record-text">{text}</span>
    </div>"""

transcript_html += "</div>"

st.markdown(transcript_html, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(
    '<div class="footer-text">'
    "Neuro-Systems Group · Clinical Audio Pipeline · "
    "All data processed locally · HIPAA-compliant architecture · "
    "ClinicalWhisper v2.1"
    "</div>",
    unsafe_allow_html=True,
)
