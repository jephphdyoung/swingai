import streamlit as st
import streamlit.components.v1 as components
import os
import base64
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


def create_video_player_html(video_path: str, video_path_no_skeleton: str, p_timestamps: dict[str, float]) -> str:
    """
    Generate HTML/JS for video player with P-position navigation buttons and skeleton toggle.

    Args:
        video_path: Path to video file with skeleton
        video_path_no_skeleton: Path to video file without skeleton
        p_timestamps: Dict mapping P-names to timestamps in seconds
    """
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    video_b64 = base64.b64encode(video_bytes).decode()

    with open(video_path_no_skeleton, "rb") as f:
        video_no_skeleton_bytes = f.read()
    video_no_skeleton_b64 = base64.b64encode(video_no_skeleton_bytes).decode()

    # Generate button HTML for each P-position (sorted P1-P10)
    buttons_html = ""
    sorted_positions = sorted(p_timestamps.keys(), key=lambda x: int(x[1:]))
    for p_name in sorted_positions:
        timestamp = p_timestamps[p_name]
        buttons_html += f'''
            <button class="p-button" data-time="{timestamp}" onclick="seekTo({timestamp}, this)">
                {p_name}
            </button>
        '''

    # Convert p_timestamps to JS object string
    p_timestamps_js = ", ".join([f'"{k}": {v}' for k, v in p_timestamps.items()])

    html = f'''
    <style>
        .video-container {{
            width: 100%;
            max-width: 1200px;
            margin: 0 auto;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }}
        .video-player {{
            width: 100%;
            border-radius: 8px;
            background: #000;
        }}
        .p-buttons {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 16px;
            justify-content: center;
        }}
        .p-button {{
            padding: 12px 20px;
            font-size: 16px;
            font-weight: 600;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: transform 0.15s, box-shadow 0.15s, background 0.15s;
            min-width: 60px;
        }}
        .p-button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }}
        .p-button:active {{
            transform: translateY(0);
        }}
        .p-button.active {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            box-shadow: 0 4px 16px rgba(245, 87, 108, 0.5);
        }}
        .current-position {{
            text-align: center;
            margin-top: 12px;
            font-size: 14px;
            color: #666;
        }}
        .toggle-container {{
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 16px;
            margin-top: 12px;
            margin-bottom: 8px;
        }}
        .toggle-label {{
            font-size: 14px;
            font-weight: 500;
            color: #333;
            cursor: pointer;
            padding: 8px 16px;
            border-radius: 20px;
            transition: all 0.2s;
        }}
        .toggle-label:hover {{
            background: #f0f0f0;
        }}
        .toggle-label.active {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        input[type="radio"] {{
            display: none;
        }}
    </style>

    <div class="video-container">
        <div class="toggle-container">
            <label class="toggle-label active" id="labelSkeleton">
                <input type="radio" name="skeleton" value="on" checked onchange="toggleSkeleton(true)">
                Skeleton On
            </label>
            <label class="toggle-label" id="labelNoSkeleton">
                <input type="radio" name="skeleton" value="off" onchange="toggleSkeleton(false)">
                Skeleton Off
            </label>
        </div>
        <video id="swingVideo" class="video-player" controls>
            <source id="videoSource" src="data:video/mp4;base64,{video_b64}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        <video id="swingVideoNoSkeleton" class="video-player" style="display:none;">
            <source src="data:video/mp4;base64,{video_no_skeleton_b64}" type="video/mp4">
        </video>
        <div class="p-buttons">
            {buttons_html}
        </div>
        <div class="current-position" id="currentPosition">
            Click a P-position button to jump to that point in the swing
        </div>
    </div>

    <script>
        const videoSkeleton = document.getElementById('swingVideo');
        const videoNoSkeleton = document.getElementById('swingVideoNoSkeleton');
        let video = videoSkeleton;  // Current active video
        const buttons = document.querySelectorAll('.p-button');
        const positionDisplay = document.getElementById('currentPosition');
        const pTimestamps = {{{p_timestamps_js}}};
        const labelSkeleton = document.getElementById('labelSkeleton');
        const labelNoSkeleton = document.getElementById('labelNoSkeleton');

        function toggleSkeleton(showSkeleton) {{
            const currentTime = video.currentTime;
            const wasPlaying = !video.paused;

            if (showSkeleton) {{
                videoNoSkeleton.style.display = 'none';
                videoNoSkeleton.pause();
                videoSkeleton.style.display = 'block';
                videoSkeleton.currentTime = currentTime;
                video = videoSkeleton;
                labelSkeleton.classList.add('active');
                labelNoSkeleton.classList.remove('active');
            }} else {{
                videoSkeleton.style.display = 'none';
                videoSkeleton.pause();
                videoNoSkeleton.style.display = 'block';
                videoNoSkeleton.currentTime = currentTime;
                video = videoNoSkeleton;
                labelNoSkeleton.classList.add('active');
                labelSkeleton.classList.remove('active');
            }}

            if (wasPlaying) {{
                video.play();
            }}
        }}

        function seekTo(timestamp, button) {{
            video.currentTime = timestamp;
            video.play();

            // Update active button styling
            buttons.forEach(btn => btn.classList.remove('active'));
            if (button) {{
                button.classList.add('active');
            }}
        }}

        // Highlight button as video plays through positions
        function onTimeUpdate() {{
            const currentTime = video.currentTime;
            let closestP = null;
            let minDiff = Infinity;

            for (const [pName, timestamp] of Object.entries(pTimestamps)) {{
                const diff = Math.abs(currentTime - timestamp);
                if (diff < minDiff && diff < 3.0) {{
                    minDiff = diff;
                    closestP = pName;
                }}
            }}

            buttons.forEach(btn => {{
                const btnName = btn.textContent.trim();
                if (btnName === closestP) {{
                    btn.classList.add('active');
                }} else {{
                    btn.classList.remove('active');
                }}
            }});

            if (closestP) {{
                positionDisplay.textContent = `Currently at: ${{closestP}}`;
            }} else {{
                positionDisplay.textContent = `Time: ${{currentTime.toFixed(1)}}s`;
            }}
        }}

        videoSkeleton.addEventListener('timeupdate', onTimeUpdate);
        videoNoSkeleton.addEventListener('timeupdate', onTimeUpdate);

        // Keyboard navigation
        document.addEventListener('keydown', function(e) {{
            const positions = Object.entries(pTimestamps).sort((a, b) => a[1] - b[1]);
            const currentTime = video.currentTime;

            if (e.key === 'ArrowRight') {{
                // Jump to next P-position
                for (const [name, time] of positions) {{
                    if (time > currentTime + 0.5) {{
                        seekTo(time, document.querySelector(`[data-time="${{time}}"]`));
                        break;
                    }}
                }}
            }} else if (e.key === 'ArrowLeft') {{
                // Jump to previous P-position
                for (let i = positions.length - 1; i >= 0; i--) {{
                    if (positions[i][1] < currentTime - 0.5) {{
                        seekTo(positions[i][1], document.querySelector(`[data-time="${{positions[i][1]}}"]`));
                        break;
                    }}
                }}
            }}
        }});
    </script>
    '''

    return html


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

            result = analyze_swing(user_path, sample_path, progress_callback=on_progress)
            progress_bar.progress(1.0)
            status.update(label="Complete!", state="complete", expanded=False)

        # Custom video player with P-position navigation buttons and skeleton toggle
        player_html = create_video_player_html(
            result.output_path,
            result.output_path_no_skeleton,
            result.p_timestamps
        )
        components.html(player_html, height=750, scrolling=False)

        # Download button
        with open(result.output_path, "rb") as f:
            st.download_button(
                label="Download comparison video",
                data=f,
                file_name=os.path.basename(result.output_path),
                mime="video/mp4",
            )

        # Show P-position timestamps in expander
        with st.expander("P-Position Timestamps"):
            for p_name in sorted(result.p_timestamps.keys(), key=lambda x: int(x[1:])):
                timestamp = result.p_timestamps[p_name]
                st.write(f"**{p_name}**: {timestamp:.2f}s")

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
