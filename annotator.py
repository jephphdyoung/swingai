import cv2
import streamlit as st
from utils.annotations import load_annotations, save_annotations, load_view_angle

P_POSITIONS = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"]


def _get_frame(video_path: str, frame_num: int):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


def _get_video_info(video_path: str):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total


def _step_frame(slider_key, delta, max_frame):
    current = st.session_state.get(slider_key, 0)
    st.session_state[slider_key] = max(0, min(max_frame, current + delta))


def render_annotator(video_path: str, key_prefix: str):
    fps, total_frames = _get_video_info(video_path)
    if total_frames == 0:
        st.error(f"Cannot read video: {video_path}")
        return

    st.caption(f"{total_frames} frames, {fps:.1f} fps, {total_frames / fps:.1f}s")

    marks_key = f"{key_prefix}_marks"
    video_key = f"{key_prefix}_video_path"
    slider_key = f"{key_prefix}_slider"
    angle_key = f"{key_prefix}_view_angle"

    if st.session_state.get(video_key) != video_path:
        saved = load_annotations(video_path)
        st.session_state[marks_key] = saved if saved else {}
        st.session_state[video_key] = video_path
        st.session_state[slider_key] = 0
        saved_angle = load_view_angle(video_path)
        st.session_state[angle_key] = saved_angle or "Face On"

    marks = st.session_state[marks_key]

    view_angle = st.radio(
        "View angle:",
        ["Face On", "DTL"],
        horizontal=True,
        key=angle_key,
    )

    # Frame step buttons (before slider — callbacks modify state before render)
    max_frame = total_frames - 1
    col_back10, col_back1, col_fwd1, col_fwd10 = st.columns(4)
    with col_back10:
        st.button("-10", key=f"{key_prefix}_back10", use_container_width=True,
                  on_click=_step_frame, args=(slider_key, -10, max_frame))
    with col_back1:
        st.button("-1", key=f"{key_prefix}_back1", use_container_width=True,
                  on_click=_step_frame, args=(slider_key, -1, max_frame))
    with col_fwd1:
        st.button("+1", key=f"{key_prefix}_fwd1", use_container_width=True,
                  on_click=_step_frame, args=(slider_key, 1, max_frame))
    with col_fwd10:
        st.button("+10", key=f"{key_prefix}_fwd10", use_container_width=True,
                  on_click=_step_frame, args=(slider_key, 10, max_frame))

    frame_num = st.slider(
        "Frame",
        0, max_frame,
        key=slider_key,
    )
    timestamp_s = frame_num / fps
    st.caption(f"Frame {frame_num} / {max_frame}  |  {timestamp_s:.2f}s")

    frame_img = _get_frame(video_path, frame_num)
    if frame_img is not None:
        st.image(frame_img, use_container_width=True)

    st.markdown("**Mark P-positions** at the current frame:")
    for row_start in range(0, len(P_POSITIONS), 5):
        row = P_POSITIONS[row_start:row_start + 5]
        cols = st.columns(len(row))
        for col, pos in zip(cols, row):
            with col:
                current = marks.get(pos)
                label = f"{pos}: {current}" if current is not None else f"{pos}: --"
                if st.button(label, key=f"{key_prefix}_mark_{pos}", use_container_width=True):
                    marks[pos] = frame_num
                    st.session_state[marks_key] = marks
                    st.rerun()

    marked_count = sum(1 for p in P_POSITIONS if p in marks)
    st.markdown(f"**{marked_count}/{len(P_POSITIONS)}** positions marked")

    col_save, col_clear = st.columns(2)
    with col_save:
        if st.button("Save annotations", key=f"{key_prefix}_save",
                      use_container_width=True, type="primary",
                      disabled=marked_count == 0):
            angle_val = "face_on" if view_angle == "Face On" else "dtl"
            save_annotations(video_path, marks, fps, view_angle=angle_val)
            st.success(f"Saved {marked_count} positions ({view_angle})")
    with col_clear:
        if st.button("Clear all", key=f"{key_prefix}_clear",
                      use_container_width=True):
            st.session_state[marks_key] = {}
            st.rerun()
