
"""Streamlit web application for YOLO11 blood-cell detection."""

from io import BytesIO

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


st.set_page_config(
    page_title="Blood Cell Detection | YOLO11",
    page_icon="ðŸ”¬",
    layout="wide",
)

st.title("ðŸ”¬ Intelligent Blood Cell Detection")
st.caption("YOLO11n Â· White blood cells, red blood cells and platelets")

st.markdown(
    """
    Upload a blood-smear image to detect and count **WBCs, RBCs and Platelets**.
    The model runs locally on your CPU. Results are for academic demonstration,
    not clinical diagnosis.
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
    st.caption("Classes: WBC Â· RBC Â· Platelets")

@st.cache_resource(show_spinner="Loading YOLO11 model...")
def get_model():
    return load_model(DEFAULT_WEIGHTS)


uploaded_file = st.file_uploader(
    "Choose a blood-smear image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG and PNG.",
)

if uploaded_file is None:
    st.info("Upload an image to begin detection.")
    st.stop()

try:
    original_image = open_image(uploaded_file)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

st.subheader("Uploaded image")
st.image(original_image, caption="Original image", use_column_width=True)

if st.button("Run blood-cell detection", type="primary", use_container_width=False):
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
        st.stop()

result = st.session_state.get("detection_result")

# Do not display results from a different upload or confidence setting.
if result is not None:
    if (
        result["source_name"] != uploaded_file.name
        or result["confidence"] != confidence
    ):
        st.info("Click **Run blood-cell detection** to update the results.")
        st.stop()

    st.divider()
    st.subheader("Detection results")

    left, right = st.columns(2)
    with left:
        st.image(original_image, caption="Original", use_column_width=True)
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
        f"Source: {result['source_name']} Â· "
        f"Confidence threshold: {result['confidence']:.2f} Â· "
        f"Detections: {len(detections)}"
    )

st.divider()
st.caption(
    "Academic research prototype. Detection results can contain errors "
    "and must not be used as a substitute for professional medical review."
)

