"""Streamlit web application for YOLO11 blood-cell detection.

Supports two modes:
    * Image detection (existing behaviour, unchanged)
    * Video detection (frame-by-frame CPU inference, annotated video download)
"""

from io import BytesIO
from pathlib import Path

import streamlit as st

from detector import (
    PROJECT_ROOT,
    DEFAULT_WEIGHTS,
    CLASS_ORDER,
    load_model,
    open_image,
    detect,
    count_by_class,
    detections_to_csv,
)
from video_processor import (
    VIDEO_SUFFIXES,
    delete_quietly,
    process_video,
    save_upload_to_temp,
)


st.set_page_config(
    page_title="Blood Cell Detection | YOLO11",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Intelligent Blood Cell Detection")
st.caption("YOLO11n · White blood cells, red blood cells and platelets")

st.markdown(
    """
    Upload a blood-smear **image** or a prerecorded **video** to detect and
    count **WBCs, RBCs and Platelets**. The model runs locally on your CPU.
    Results are for academic demonstration, not clinical diagnosis.
    """
)

with st.sidebar:
    st.header("Detection settings")
    confidence = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="Higher values filter out lower-confidence detections.",
    )
    st.caption("Inference device: CPU")
    st.caption(f"Model: {DEFAULT_WEIGHTS.name}")
    st.caption("Classes: WBC · RBC · Platelets")

@st.cache_resource(show_spinner="Loading YOLO11 model...")
def get_model():
    return load_model(DEFAULT_WEIGHTS)


tab_image, tab_video = st.tabs(["📷 Image detection", "🎬 Video detection"])


# ---------------------------------------------------------------------------
# Image mode (existing behaviour, unchanged apart from living inside a tab).
# ---------------------------------------------------------------------------
with tab_image:
    uploaded_file = st.file_uploader(
        "Choose a blood-smear image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG and PNG.",
    )

    if uploaded_file is None:
        st.info("Upload an image to begin detection.")
    else:
        try:
            original_image = open_image(uploaded_file)
        except ValueError as exc:
            st.error(str(exc))
            original_image = None

        if original_image is not None:
            st.subheader("Uploaded image")
            st.image(original_image, caption="Original image", use_column_width=True)

            if st.button(
                "Run blood-cell detection",
                type="primary",
                use_container_width=False,
                key="run_image_detection",
            ):
                try:
                    model = get_model()
                    with st.spinner("Detecting blood cells on CPU..."):
                        detections, annotated_image = detect(
                            model,
                            original_image,
                            conf=confidence,
                        )

                    counts = count_by_class(detections)
                    st.session_state["detection_result"] = {
                        "annotated_image": annotated_image,
                        "detections": detections,
                        "counts": counts,
                        "csv": detections_to_csv(detections),
                        "source_name": uploaded_file.name,
                        "confidence": confidence,
                    }
                except Exception as exc:
                    st.error(f"Detection failed: {exc}")

            result = st.session_state.get("detection_result")

            # Do not display results from a different upload or confidence setting.
            if result is not None:
                if (
                    result["source_name"] != uploaded_file.name
                    or result["confidence"] != confidence
                ):
                    st.info(
                        "Click **Run blood-cell detection** to update the results."
                    )
                else:
                    st.divider()
                    st.subheader("Detection results")

                    left, right = st.columns(2)
                    with left:
                        st.image(
                            original_image,
                            caption="Original",
                            use_column_width=True,
                        )
                    with right:
                        st.image(
                            result["annotated_image"],
                            caption="YOLO11 annotated detections",
                        )

                    st.markdown("### Blood-cell counts")
                    count_cols = st.columns(len(CLASS_ORDER))
                    for col, class_name in zip(count_cols, CLASS_ORDER):
                        with col:
                            st.metric(class_name, result["counts"][class_name])

                    detections = result["detections"]
                    if not detections:
                        st.warning(
                            "No objects were detected at this confidence threshold. "
                            "Try lowering the threshold or uploading another image."
                        )
                    else:
                        st.markdown(f"### Individual detections ({len(detections)})")
                        table_rows = [
                            {
                                "Class": d.name,
                                "Confidence": round(d.confidence, 4),
                                "X1": d.x1,
                                "Y1": d.y1,
                                "X2": d.x2,
                                "Y2": d.y2,
                                "Width": d.width,
                                "Height": d.height,
                            }
                            for d in detections
                        ]
                        for i, row in enumerate(table_rows, start=1):
                            st.markdown(
                                f"**Detection {i}: {row['Class']}** — "
                                f"Confidence: {row['Confidence']:.4f} | "
                                f"Box: ({row['X1']}, {row['Y1']}) to "
                                f"({row['X2']}, {row['Y2']}) | "
                                f"Size: {row['Width']} × {row['Height']}"
                            )

                    st.markdown("### Download results")
                    download_cols = st.columns(2)

                    image_buffer = BytesIO()
                    result["annotated_image"].save(image_buffer, format="PNG")

                    with download_cols[0]:
                        st.download_button(
                            "Download annotated image",
                            data=image_buffer.getvalue(),
                            file_name="blood_cell_detections.png",
                            mime="image/png",
                        )

                    with download_cols[1]:
                        st.download_button(
                            "Download detection CSV",
                            data=result["csv"],
                            file_name="blood_cell_detections.csv",
                            mime="text/csv",
                        )

                    st.caption(
                        f"Source: {result['source_name']} · "
                        f"Confidence threshold: {result['confidence']:.2f} · "
                        f"Detections: {len(detections)}"
                    )


# ---------------------------------------------------------------------------
# Video mode (new).
# ---------------------------------------------------------------------------
with tab_video:
    st.markdown(
        """
        Upload a prerecorded blood-smear video (MP4, AVI or MOV). Frames are
        processed **sequentially on CPU** — the video is never loaded into
        memory at once — and an annotated video is produced for playback
        and download.
        """
    )

    video_file = st.file_uploader(
        "Choose a blood-smear video",
        type=list(VIDEO_SUFFIXES),
        help="Supported formats: MP4, AVI and MOV (subject to the local "
        "OpenCV/FFmpeg backend).",
        key="video_uploader",
    )

    if video_file is None:
        st.info("Upload a video to begin detection.")
    else:
        frame_interval = st.slider(
            "Process every Nth frame",
            min_value=1,
            max_value=10,
            value=1,
            step=1,
            help="1 = process every frame. Higher values skip frames to "
            "finish faster; skipped frames are written without bounding "
            "boxes and are not counted.",
            key="video_frame_interval",
        )
        st.caption(
            f"Confidence threshold: {confidence:.2f} (shared with image mode, "
            "adjust in the sidebar)"
        )

        if st.button(
            "Run video detection",
            type="primary",
            use_container_width=False,
            key="run_video_detection",
        ):
            input_path = None
            try:
                input_path = save_upload_to_temp(
                    video_file.getvalue(),
                    video_file.name,
                )

                model = get_model()

                status_text = st.empty()
                progress_bar = st.progress(0)
                last_pct = {"value": -1}

                def on_progress(done: int, total: int) -> None:
                    """Update the progress bar without flooding Streamlit."""
                    if total > 0:
                        pct = min(100, int(done / total * 100))
                        if pct != last_pct["value"]:
                            last_pct["value"] = pct
                            progress_bar.progress(pct)
                            status_text.text(
                                f"Processing frame {done} of {total}..."
                            )
                    else:
                        status_text.text(f"Processed {done} frames...")

                with st.spinner("Detecting blood cells in video frames on CPU..."):
                    video_run = process_video(
                        model,
                        input_path,
                        conf=confidence,
                        frame_interval=frame_interval,
                        progress_callback=on_progress,
                    )

                progress_bar.progress(100)
                status_text.text(
                    f"Finished: {video_run.processed_frames} frame(s) processed, "
                    f"{video_run.written_frames} frame(s) written."
                )

                # Remove the previous run's output so temp files do not pile up.
                previous = st.session_state.pop("video_result", None)
                if previous is not None:
                    delete_quietly(previous["output_path"])

                st.session_state["video_result"] = {
                    "output_path": video_run.output_path,
                    "source_name": video_file.name,
                    "confidence": confidence,
                    "frame_interval": frame_interval,
                    "run": video_run,
                }
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Video processing failed: {exc}")
            finally:
                # The uploaded copy is only needed during processing.
                delete_quietly(input_path)

        video_result = st.session_state.get("video_result")

        if video_result is None:
            st.info("Click **Run video detection** to process the video.")
        elif (
            video_result["source_name"] != video_file.name
            or video_result["confidence"] != confidence
            or video_result["frame_interval"] != frame_interval
        ):
            st.info("Click **Run video detection** to update the results.")
        else:
            run = video_result["run"]
            info = run.info

            st.divider()
            st.subheader("Processing status")
            st.success(
                f"Processed {run.processed_frames} of "
                f"{run.written_frames} frames using {run.output_label}."
            )

            st.markdown("### Video properties")
            prop_cols = st.columns(4)
            minutes, seconds = divmod(int(round(info.duration_seconds)), 60)
            with prop_cols[0]:
                st.metric(
                    "Duration",
                    f"{minutes}:{seconds:02d}"
                    if info.duration_seconds > 0
                    else "Unknown",
                )
            with prop_cols[1]:
                st.metric("Resolution", info.resolution)
            with prop_cols[2]:
                st.metric("Frame rate", f"{info.fps:.2f} FPS")
            with prop_cols[3]:
                st.metric("Processed frames", run.processed_frames)

            if run.note:
                st.caption(run.note)

            output_path = Path(run.output_path)
            try:
                output_bytes = output_path.read_bytes()
            except OSError:
                # Temp file cleaned up (e.g. reboot or temp-folder purge).
                st.session_state.pop("video_result", None)
                st.warning(
                    "The annotated video file is no longer available on "
                    "this machine. Please run video detection again."
                )
                output_bytes = None

            if output_bytes is not None:
                st.markdown("### Annotated output video")
                st.video(output_bytes)

            st.markdown("### Blood-cell counts")
            st.markdown(
                f"**Total detections across all {run.processed_frames} "
                "processed frames**"
            )
            count_cols = st.columns(len(CLASS_ORDER))
            for col, class_name in zip(count_cols, CLASS_ORDER):
                with col:
                    st.metric(class_name, run.counts[class_name])

            st.caption(
                "These totals count one detection per processed frame: the "
                "same cell visible in several frames is counted once in each "
                "frame, so totals are **not** unique-cell counts."
            )

            if run.total_detections == 0:
                st.warning(
                    "No objects were detected at this confidence threshold. "
                    "Try lowering the threshold."
                )

            if output_bytes is None:
                # Results block was already dismissed above; skip downloads.
                pass
            else:
                st.markdown("### Download results")
                output_suffix = output_path.suffix.lower()
                st.download_button(
                    "Download annotated video",
                    data=output_bytes,
                    file_name=f"annotated_{Path(video_file.name).stem}"
                    f"{output_suffix}",
                    mime="video/mp4"
                    if output_suffix == ".mp4"
                    else "video/x-msvideo",
                )

            st.caption(
                f"Source: {video_result['source_name']} · "
                f"Confidence threshold: {video_result['confidence']:.2f} · "
                f"Frame interval: {video_result['frame_interval']} · "
                f"Total detections: {run.total_detections}"
            )


st.divider()
st.caption(
    "Academic research prototype. Detection results can contain errors "
    "and must not be used as a substitute for professional medical review."
)
