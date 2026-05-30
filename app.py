import streamlit as st
import os
from analyzer import analyze_swing
from annotator import render_annotator
from utils.annotations import has_annotations

SAMPLE_DIR = "sample_videos"
MY_DIR = "my_videos"


def _label_videos(directory):
    """Return video filenames with annotation status markers."""
    files = sorted(os.listdir(directory))
    labels = []
    for f in files:
        path = os.path.join(directory, f)
        marker = "[P] " if has_annotations(path) else ""
        labels.append(f"{marker}{f}")
    return files, labels


st.title("Golf Swing Analyzer")

tab_analyze, tab_annotate = st.tabs(["Analyze", "Annotate"])

with tab_analyze:
    user_files, user_labels = _label_videos(MY_DIR)
    sample_files, sample_labels = _label_videos(SAMPLE_DIR)

    user_idx = st.selectbox("Choose your video:", range(len(user_labels)),
                            format_func=lambda i: user_labels[i], key="analyze_user")
    sample_idx = st.selectbox("Choose a sample video:", range(len(sample_labels)),
                              format_func=lambda i: sample_labels[i], key="analyze_sample")

    user_video = user_files[user_idx]
    sample_video = sample_files[sample_idx]

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

with tab_annotate:
    source = st.selectbox("Video source:", ["Baseline / Sample", "User"],
                          key="annot_source")

    if source == "User":
        vid_dir = MY_DIR
    else:
        vid_dir = SAMPLE_DIR

    annot_files, annot_labels = _label_videos(vid_dir)
    annot_idx = st.selectbox("Select video:", range(len(annot_labels)),
                             format_func=lambda i: annot_labels[i], key="annot_video")
    vid = annot_files[annot_idx]
    if vid:
        render_annotator(os.path.join(vid_dir, vid), key_prefix="annot")
