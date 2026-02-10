"""
Clinical Intelligence Dashboard
────────────────────────────────
Neuro-Systems Group | Clinical Audio Pipeline

A Streamlit dashboard for offline clinical-audio analysis.
Reads JSON exports from ClinicalWhisper or generates demo data.

Run:  streamlit run app.py
"""

import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG + CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Clinical Intelligence Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* ── Hide Streamlit chrome ── */
#MainMenu, header, footer {visibility: hidden;}
div[data-testid="stToolbar"] {display: none;}

/* ── Global ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0a0e14 !important;
    color: #d4d4d4;
    font-family: 'Inter', -apple-system, sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #0d1117 !important;
    border-right: 1px solid #1a2233;
}
[data-testid="stSidebar"] * { color: #c9d1d9; }

/* ── Typography ── */
h1, h2, h3, h4, h5, h6 { color: #e6edf3 !important; }

/* ── Metric cards ── */
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #111820 0%, #0d1117 100%);
    border: 1px solid #1a2233;
    border-radius: 10px;
    padding: 18px 22px;
}
div[data-testid="stMetric"] label { color: #7d8590 !important; font-size: 12px !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #58d5c1 !important; font-weight: 700 !important; }
div[data-testid="stMetric"] [data-testid="stMetricDelta"] { font-size: 12px !important; }

/* ── Section dividers ── */
hr { border-color: #1a2233 !important; }

/* ── Chat bubbles ── */
.chat-bubble {
    padding: 12px 16px;
    border-radius: 14px;
    margin-bottom: 8px;
    font-size: 14px;
    line-height: 1.55;
    max-width: 75%;
    position: relative;
}
.chat-bubble .ts {
    font-size: 10px;
    color: #5c6370;
    margin-top: 4px;
    display: block;
}
.chat-left {
    background: #112233;
    border: 1px solid #1a3350;
    color: #b8d4e8;
    margin-right: auto;
    border-bottom-left-radius: 4px;
}
.chat-right {
    background: #1a1d23;
    border: 1px solid #2a2d33;
    color: #c9cdd2;
    margin-left: auto;
    border-bottom-right-radius: 4px;
}
.chat-speaker {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 4px;
}
.chat-speaker.clinician { color: #58d5c1; }
.chat-speaker.patient { color: #8b949e; }

/* ── Plotly container ── */
.stPlotlyChart { border: 1px solid #1a2233; border-radius: 10px; overflow: hidden; }

/* ── Badge ── */
.badge {
    display: inline-block;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    border: 1px solid #1a3350;
    color: #58d5c1;
    background: rgba(88, 213, 193, 0.06);
    margin-left: 8px;
}

/* ── Section headers ── */
.section-head {
    font-size: 13px;
    font-weight: 600;
    color: #7d8590;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 6px;
    padding-bottom: 6px;
    border-bottom: 1px solid #1a2233;
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
    """Format seconds → [MM:SS]."""
    m, s = divmod(int(seconds), 60)
    return f"[{m:02d}:{s:02d}]"


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("#### 🧠  Neuro-Systems Group")
    st.caption("Clinical Audio Pipeline · v2.1")
    st.markdown("---")

    demo_mode = st.checkbox("Load De-Identified Sample Data (Patient 042)", value=True)

    st.markdown("---")
    uploaded = st.file_uploader(
        "Upload Analysis (.json)",
        type=["json"],
        help="Upload a JSON export from ClinicalWhisper.",
    )

    st.markdown("---")
    st.caption("All processing is local. No data leaves this machine.")

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
    st.markdown("## Upload a JSON file or enable Demo Mode to begin.")
    st.stop()

df = pd.DataFrame(data)

# ═══════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown(
    '<h2 style="margin-bottom:0;letter-spacing:-0.5px;">Clinical Intelligence Dashboard</h2>'
    '<span class="badge">Privacy-Preserving</span>'
    '<span class="badge">Offline-First</span>',
    unsafe_allow_html=True,
)
st.markdown("")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — VITALS
# ═══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-head">Session Vitals</div>', unsafe_allow_html=True)

overall_sentiment = round(df["sentiment"].mean(), 1)
clinician_words = df.loc[df["speaker"] == "Clinician", "text"].str.split().str.len().sum()
patient_words = df.loc[df["speaker"] == "Patient", "text"].str.split().str.len().sum()
total_words = clinician_words + patient_words
dominance_pct = round(clinician_words / total_words * 100, 1) if total_words else 0
conflict_events = int((df["sentiment"] <= 3.0).sum())

v1, v2, v3 = st.columns(3)
v1.metric("Overall Sentiment", f"{overall_sentiment} / 10", delta="Positive" if overall_sentiment >= 5 else "Concern")
v2.metric("Clinician Talk-Time", f"{dominance_pct}%", delta=f"{clinician_words} words")
v3.metric("Distress Segments", conflict_events, delta="flagged" if conflict_events >= 3 else "low")

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
    color_discrete_map={"Clinician": "#58d5c1", "Patient": "#5a6a7a"},
    labels={"start": "Time (seconds)", "sentiment": "Sentiment Score", "speaker": "Speaker"},
)

# Distress threshold
fig.add_hline(
    y=3.0,
    line_dash="dot",
    line_color="#ff4d4d",
    line_width=1,
    annotation_text="Distress Threshold",
    annotation_position="top left",
    annotation_font_color="#ff4d4d",
    annotation_font_size=10,
)

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    margin=dict(l=40, r=20, t=30, b=40),
    height=320,
    font=dict(family="Inter, sans-serif", color="#8b949e", size=11),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=11),
    ),
    xaxis=dict(
        gridcolor="#1a2233",
        zeroline=False,
        title_font=dict(size=11),
    ),
    yaxis=dict(
        gridcolor="#1a2233",
        zeroline=False,
        range=[0, 10],
        dtick=2,
        title_font=dict(size=11),
    ),
)

fig.update_traces(
    line=dict(width=2.5),
    marker=dict(size=6),
)

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — DIARIZED TRANSCRIPT
# ═══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-head">Diarized Transcript</div>', unsafe_allow_html=True)

transcript_html = ""
for _, row in df.iterrows():
    ts = _ts(row["start"])
    text = row["text"]
    speaker = row["speaker"]

    if speaker == "Clinician":
        transcript_html += f"""
        <div style="display:flex;justify-content:flex-start;margin-bottom:6px;">
            <div class="chat-bubble chat-left">
                <div class="chat-speaker clinician">Clinician</div>
                {text}
                <span class="ts">{ts}</span>
            </div>
        </div>"""
    else:
        transcript_html += f"""
        <div style="display:flex;justify-content:flex-end;margin-bottom:6px;">
            <div class="chat-bubble chat-right">
                <div class="chat-speaker patient">Patient</div>
                {text}
                <span class="ts">{ts}</span>
            </div>
        </div>"""

st.markdown(
    f'<div style="max-height:420px;overflow-y:auto;padding:12px 0;">{transcript_html}</div>',
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.caption(
    "Neuro-Systems Group | Clinical Audio Pipeline · "
    "All data processed locally · HIPAA-compliant architecture · "
    "ClinicalWhisper v2.1"
)
