"""
app.py
------
Streamlit demo for DeepVerify. Run after training a model:

    streamlit run app.py

Upload a document IMAGE (use public/synthetic samples only -- never a real
government-issued ID) and see every pipeline stage: preprocessing,
classification, OCR, field extraction, and anomaly/tamper scoring.
"""

import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from src.copy_move import detect_copy_move
from src.inference import DeepVerifyPipeline
from src.qr_detection import detect_qr
from src.risk_analysis import calculate_risk
from src.tamper_detection import ela_analysis

st.set_page_config(page_title="DeepVerify", layout="wide")
st.title("DeepVerify — Document Intelligence & Visual Fraud Detection")
st.caption(
    "Academic prototype. Trained on public/synthetic document data only. "
    "Not a production KYC or identity-verification system."
)

CHECKPOINT = "results/best_model.pt"
AUTOENCODER = "results/autoencoder.pt"
SEGMENTATION_MODEL = "results/best_segmentation_model_grouped.pth"


@st.cache_resource
def load_pipeline():
    ae_path = AUTOENCODER if Path(AUTOENCODER).exists() else None
    return DeepVerifyPipeline(
    checkpoint_path=CHECKPOINT,
    segmentation_model_path=SEGMENTATION_MODEL,
    autoencoder_path=ae_path
)

if not Path(CHECKPOINT).exists():
    st.warning(
        f"No trained model found at `{CHECKPOINT}`.\n\n"
        "Run the training pipeline first, e.g.:\n\n"
        "```\npython scripts/make_dummy_data.py   # or scripts/prepare_midv500.py for real data\n"
        "python src/train.py --epochs 10 --pretrained --augment\n```"
    )
    st.stop()

st.write("Loading pipeline...")

pipeline = load_pipeline()

st.write("Pipeline loaded successfully.")

uploaded = st.file_uploader("Upload a document image (public/synthetic only)",
                             type=["jpg", "jpeg", "png"])

if uploaded is not None:
    upload_bytes = uploaded.read()

    file_bytes = np.frombuffer(upload_bytes, np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image_bgr is None:
        st.error("The uploaded file is not a readable image.")
        st.stop()

    with st.spinner("Running pipeline..."):
        result = pipeline.run(image_bgr)
        # ELA and copy-move helpers currently accept a path. Use a unique
        # temporary file so concurrent users cannot overwrite one another.
        with NamedTemporaryFile(suffix=".jpg") as temp_file:
            temp_file.write(upload_bytes)
            temp_file.flush()
            ela_score, ela_image = ela_analysis(temp_file.name)

            copy_score = detect_copy_move(temp_file.name)

            qr_result = detect_qr(temp_file.name)

        risk_score = calculate_risk(
            ela_score,
            copy_score,
            bool(qr_result),
            result["classification_confidence"],
        )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Input")
        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col2:
        st.subheader("2. Preprocessed (perspective + denoise + CLAHE)")
        st.image(cv2.cvtColor(result["processed_image"], cv2.COLOR_BGR2RGB), use_container_width=True)

    st.subheader("3. Document Classification")
    st.metric("Predicted type", result["document_type"],
               f"{result['classification_confidence']*100:.1f}% confidence")
    st.bar_chart(result["class_probabilities"])

    st.subheader("4. OCR Text")
    st.text_area("Extracted text", result["ocr_text"], height=150)

    st.subheader("5. Extracted Fields")
    st.json(result["extracted_fields"])

    st.subheader("6. Tamper / Anomaly Score")
    a = result["anomaly"]
    c1, c2 = st.columns(2)
    c1.metric("ELA mean (higher = more suspicious edit energy)", f"{a['ela_mean']:.2f}")
    if a["autoencoder_recon_error"] is not None:
        c2.metric("Autoencoder reconstruction error", f"{a['autoencoder_recon_error']:.5f}")
    else:
        c2.info("Autoencoder not loaded — train scripts/train_autoencoder.py to enable this.")

    st.caption(
        "These are heuristic anomaly signals for an academic prototype, not a "
        "calibrated fraud verdict."
    )
    st.subheader("7. Forgery Analysis")

    st.write("ELA score:", ela_score)

    st.write("Copy-move score:", copy_score)

    st.write("QR result:", qr_result)

    st.metric("Risk score", f"{risk_score}%")

    st.image(ela_image)
else:
    st.info("Upload a document image to run the pipeline.")
