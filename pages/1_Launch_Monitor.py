"""Launch Monitor — standalone Streamlit page for the FlightScope Mevo+.

Kept separate from the swing-video app for now. Connects to the device (or replays a
captured pcap when no hardware is present) and shows measured shot metrics live.
"""

import os
import threading

import pandas as pd
import streamlit as st

from mevo import MevoClient, MevoDevice, ShotMetrics
from mevo.discovery import discover

MEVOINFO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mevoinfo")

st.title("🎯 Launch Monitor")
st.caption("FlightScope Mevo+ — measured ball/club speed and spin per swing.")

# Shared, thread-safe shot buffer (the reader thread appends; the UI reads).
if "lm_shots" not in st.session_state:
    st.session_state.lm_shots = []
if "lm_lock" not in st.session_state:
    st.session_state.lm_lock = threading.Lock()
if "lm_client" not in st.session_state:
    st.session_state.lm_client = None
if "lm_status" not in st.session_state:
    st.session_state.lm_status = "Disconnected"


def _record_shot(shot: ShotMetrics) -> None:
    with st.session_state.lm_lock:
        st.session_state.lm_shots.append(shot)


def _shots_df() -> pd.DataFrame:
    with st.session_state.lm_lock:
        rows = [
            {
                "Shot": i + 1,
                "Ball (mph)": round(s.ball_speed_mph, 1) if s.ball_speed_mph else None,
                "Club (mph)": round(s.club_speed_mph, 1) if s.club_speed_mph else None,
                "Spin (rpm)": s.spin_rpm,
                "Launch dir (°)": round(s.launch_direction_deg, 1)
                if s.launch_direction_deg is not None else None,
            }
            for i, s in enumerate(st.session_state.lm_shots)
        ]
    return pd.DataFrame(rows)


# --- controls ----------------------------------------------------------------

with st.sidebar:
    st.header("Connection")
    mode = st.radio("Source", ["Live device", "Replay capture (offline)"])

    if mode == "Live device":
        host = st.text_input("Host (blank = auto-discover)", value="")
        broadcast = st.text_input("Broadcast address", value="255.255.255.255")
        if st.button("Connect", type="primary", disabled=st.session_state.lm_client is not None):
            try:
                dev = MevoDevice(host=host, port=5100) if host.strip() else None
                client = MevoClient(on_shot=_record_shot)
                if dev is None:
                    st.session_state.lm_status = "Discovering…"
                    dev = discover(timeout=3.0, broadcast_addr=broadcast.strip())
                    if dev is None:
                        raise ConnectionError("no device answered discovery")
                client.connect(device=dev)
                st.session_state.lm_client = client
                st.session_state.lm_status = f"Connected to {dev.host}:{dev.port}"
            except Exception as e:  # noqa: BLE001 — surface any connection failure to the UI
                st.session_state.lm_status = f"Error: {e}"
            st.rerun()

        if st.button("Disconnect", disabled=st.session_state.lm_client is None):
            st.session_state.lm_client.close()
            st.session_state.lm_client = None
            st.session_state.lm_status = "Disconnected"
            st.rerun()

    else:  # offline replay
        caps = sorted(f for f in os.listdir(MEVOINFO) if f.endswith(".pcapng")) \
            if os.path.isdir(MEVOINFO) else []
        if not caps:
            st.info("No .pcapng captures found in mevoinfo/ (they are gitignored).")
        else:
            cap = st.selectbox("Capture", caps, index=caps.index("mevo_p3_fullswing.pcapng")
                               if "mevo_p3_fullswing.pcapng" in caps else 0)
            if st.button("Replay", type="primary"):
                from mevo.pcap_source import iter_frames
                from mevo.metrics import parse_shot_frames
                frames = list(iter_frames(os.path.join(MEVOINFO, cap)))
                shots = [s for s in parse_shot_frames(frames) if not s.is_empty()]
                with st.session_state.lm_lock:
                    st.session_state.lm_shots = shots
                st.session_state.lm_status = f"Replayed {cap}: {len(shots)} shots"
                st.rerun()

    if st.button("Clear shots"):
        with st.session_state.lm_lock:
            st.session_state.lm_shots = []
        st.rerun()

st.write(f"**Status:** {st.session_state.lm_status}")

# --- display -----------------------------------------------------------------

df = _shots_df()
if df.empty:
    st.info("No shots yet. Connect to the device and take a swing, or replay a capture.")
else:
    latest = st.session_state.lm_shots[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ball speed", f"{latest.ball_speed_mph:.1f} mph" if latest.ball_speed_mph else "—")
    c2.metric("Club speed", f"{latest.club_speed_mph:.1f} mph" if latest.club_speed_mph else "—")
    c3.metric("Spin", f"{latest.spin_rpm} rpm" if latest.spin_rpm else "—")
    ldir = latest.launch_direction_deg
    c4.metric("Launch dir", f"{ldir:+.1f}° {'R' if ldir >= 0 else 'L'}" if ldir is not None else "—")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption("Ball/club/spin are measured on the wire. Launch dir is experimental "
               "(single-session calibration). Face angle, club path, AoA, spin axis are "
               "computed by the FlightScope app from raw club tracking and are not transmitted.")

if st.session_state.lm_client is not None:
    st.button("Refresh shots")  # live mode: rerun to pull new shots from the buffer
    st.caption("Click Refresh after a swing to pull new shots from the device.")
