import streamlit as st
import os
from analyzer import analyze_swing

SAMPLE_DIR = "sample_videos"
MY_DIR = "my_videos"

st.title("Golf Swing Analyzer")

user_video = st.selectbox("Choose your video:", os.listdir(MY_DIR))
sample_video = st.selectbox("Choose a sample video:", os.listdir(SAMPLE_DIR))

if st.button("Run Comparison"):
    user_path = os.path.join(MY_DIR, user_video)
    sample_path = os.path.join(SAMPLE_DIR, sample_video)

    with st.status("Processing...", expanded=True) as status:
        progress_bar = st.progress(0)

        def on_progress(step, total, message):
            st.write(message)
            progress_bar.progress(step / total)

        output_path = analyze_swing(user_path, sample_path, progress_callback=on_progress)
        progress_bar.progress(1.0)
        status.update(label="Complete!", state="complete", expanded=False)

    st.video(output_path)
    with open(output_path, "rb") as f:
        st.download_button(
            label="Download comparison video",
            data=f,
            file_name=os.path.basename(output_path),
            mime="video/mp4",
        )
