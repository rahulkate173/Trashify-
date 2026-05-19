"""
🚮 Trashify Object Detection — Streamlit App
Loads RT-DETRv2 fine-tuned model and runs inference on uploaded images.
"""

import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoImageProcessor, AutoModelForObjectDetection

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Trashify Object Detector",
    page_icon="🚮",
    layout="wide",
)

# ─────────────────────────────────────────────
# Custom CSS for a premium dark look
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #e0e0e0;
    }

    /* ── Header banner ── */
    .hero-banner {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem;
    }
    .hero-banner h1 {
        font-size: 2.8rem;
        background: linear-gradient(90deg, #00c9ff, #92fe9d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }
    .hero-banner p {
        color: #b0b0b0;
        font-size: 1.05rem;
        max-width: 700px;
        margin: 0 auto;
    }

    /* ── Cards ── */
    .glass-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(12px);
        margin-bottom: 1rem;
    }

    /* ── Result banner ── */
    .result-banner {
        padding: 1rem 1.5rem;
        border-radius: 12px;
        font-size: 1rem;
        line-height: 1.5;
        margin-top: 1rem;
    }
    .result-success {
        background: rgba(0,200,83,0.15);
        border: 1px solid rgba(0,200,83,0.4);
        color: #a5d6a7;
    }
    .result-warning {
        background: rgba(255,193,7,0.15);
        border: 1px solid rgba(255,193,7,0.4);
        color: #ffe082;
    }
    .result-info {
        background: rgba(33,150,243,0.15);
        border: 1px solid rgba(33,150,243,0.4);
        color: #90caf9;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* ── Hide default Streamlit branding ── */
    #MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Model loading (cached so it only happens once)
# ─────────────────────────────────────────────
MODEL_PATH = "RahulKate-173/rt_detrv2_finetuned_trashify_box_detector_v1"

@st.cache_resource(show_spinner="Loading model… this may take a minute on first run.")
def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForObjectDetection.from_pretrained(MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return processor, model, device

image_processor, model, device = load_model()
id2label = model.config.id2label

# Colour map for bounding boxes
COLOR_MAP = {
    "bin": "#00e676",
    "trash": "#2979ff",
    "hand": "#aa00ff",
    "trash_arm": "#ffd600",
    "not_trash": "#ff1744",
    "not_bin": "#ff1744",
    "not_hand": "#ff1744",
}
def get_image_dimensions_from_pil(image: Image.Image) -> torch.tensor:
    """
    Convert the dimensions of a PIL image to a PyTorch tensor in the order (height, width).

    Args:
        image (Image.Image): The input PIL image.

    Returns:
        torch.Tensor: A tensor containing the height and width of the image.
    """
    # Get (width, height) of image (PIL.Image.size returns width, height)
    width, height = image.size

    # Convert to a tensor in the order (height, width)
    image_dimensions_tensor = torch.tensor([height, width])

    return image_dimensions_tensor
# ─────────────────────────────────────────────
# Prediction function
# ─────────────────────────────────────────────
def predict(image: Image.Image, conf_threshold: float):
    with torch.no_grad():
        # do_pad=False prevents the processor from padding to a square,
        # which would skew the normalized box coordinates vs. the original image dims.
        inputs = image_processor(images=[image], return_tensors="pt", do_pad=False)
        outputs = model(**inputs.to(device))

        # target_sizes is (batch, 2) as (height, width)
        # target_sizes = torch.tensor([image.size[1], image.size[0]], device=device).unsqueeze(0)
        target_sizes = get_image_dimensions_from_pil(image=image).unsqueeze(0)
        results = image_processor.post_process_object_detection(
            outputs, threshold=conf_threshold, target_sizes=target_sizes
        )[0]

    # Move everything to CPU
    for k, v in results.items():
        try:
            results[k] = v.item().cpu()
        except Exception:
            results[k] = v.cpu()

    # Draw boxes on a copy of the image
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.load_default(size=20)
    except TypeError:
        font = ImageFont.load_default()

    detected_labels = []

    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        x1, y1, x2, y2 = box.tolist()
        label_name = id2label[label.item()]
        color = COLOR_MAP.get(label_name, "#ffffff")
        detected_labels.append(label_name)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text = f"{label_name} ({score.item():.3f})"
        draw.text((x1, y1), text, fill="white", font=font)

    del draw

    # Build status message
    target_items = {"trash", "bin", "hand"}
    detected_set = set(detected_labels)

    if not detected_set & target_items:
        status = "info"
        message = (
            f"No trash, bin or hand detected at confidence **{conf_threshold}**. "
            "Try another image or lower the confidence threshold."
        )
    elif (missing := target_items - detected_set):
        status = "warning"
        message = (
            f"Detected: **{sorted(detected_set & target_items)}**. "
            f"Missing for +1: **{sorted(missing)}**. "
            "Try altering the confidence threshold or using a different image."
        )
    else:
        status = "success"
        message = f"🎉 **+1!** Found: **{sorted(detected_set)}** — thank you for cleaning up!"

    return annotated, message, status, len(detected_labels)


# ─────────────────────────────────────────────
# UI — Hero
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <h1>🚮 Trashify Object Detector</h1>
    <p>Upload a photo and earn <strong>+1</strong> when the model detects <em>trash</em>, a <em>bin</em>, and a <em>hand</em> — all in one frame. Powered by a fine-tuned RT-DETRv2.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Sidebar — Settings
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    conf_threshold = st.slider(
        "Confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Only detections above this score will be shown.",
    )

    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown(
        "Model: **RT-DETRv2** fine-tuned on the "
        "[Trashify dataset](https://huggingface.co/datasets/mrdbourke/trashify_manual_labelled_images)."
    )
    st.markdown(
        f"Running on **`{device.upper()}`**"
    )
    st.markdown(
        "Source model: [`RahulKate-173/rt_detrv2_finetuned_trashify_box_detector_v1`]"
        "(https://huggingface.co/RahulKate-173/rt_detrv2_finetuned_trashify_box_detector_v1)"
    )

# ─────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────
col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### 📤 Upload an Image")
    uploaded_file = st.file_uploader(
        "Choose a JPEG / PNG image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )
    if uploaded_file:
        input_image = Image.open(uploaded_file).convert("RGB")
        st.image(input_image, caption="Original image", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_result:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### 🔍 Detection Results")

    if uploaded_file:
        with st.spinner("Running detection…"):
            annotated_img, message, status, det_count = predict(input_image, conf_threshold)

        st.image(annotated_img, caption=f"{det_count} detection(s)", use_container_width=True)

        css_class = f"result-{status}"
        st.markdown(
            f'<div class="result-banner {css_class}">{message}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("⬅️ Upload an image to get started.")

    st.markdown('</div>', unsafe_allow_html=True)
