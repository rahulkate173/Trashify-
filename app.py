import torch
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoImageProcessor, AutoModelForObjectDetection

# ----------------------------------------------------
# 1. Page Layout & Pure Black Theme Injection
# ----------------------------------------------------
st.set_page_config(layout="wide", page_title="Trashify Demo")

st.markdown("""
    <style>
        /* Force pure black background canvas */
        .stApp {
            background-color: #000000;
            color: #ffffff;
        }
        /* Lock UI text blocks, sliders, and headings to white text */
        h1, h2, h3, p, label, .stSlider {
            color: #ffffff !important;
        }
        /* Styled File upload block container */
        section[data-testid="stFileUploader"] {
            background-color: #0f0f0f;
            border: 1px dashed #2d2d2d;
            border-radius: 8px;
        }
        /* Custom UI blocks for text output alerts */
        .stAlert {
            background-color: #0f0f0f !important;
            color: #ffffff !important;
            border: 1px solid #2d2d2d !important;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>🚮 Trashify Object Detection Demo V4</h1>", unsafe_allow_html=True)

description = """
<div style='text-align: center; color: #cccccc; margin-bottom: 25px;'>
<p>Help clean up your local area! Upload an image and get +1 if there is all of the following items detected: trash, bin, hand.</p>
<p>Model is a fine-tuned version of <a href="https://huggingface.co/docs/transformers/main/en/model_doc/rt_detr_v2" target="_blank" style="color: #2979ff;">RT-DETRv2</a> on the <a href="https://huggingface.co/datasets/mrdbourke/trashify_manual_labelled_images" target="_blank" style="color: #2979ff;">Trashify dataset</a>.</p>
<p style='font-size: 13px; color: #777777;'>This version is v4 because the first three versions were using a different model and did not perform as well.</p>
</div>
"""
st.markdown(description, unsafe_allow_html=True)

# ----------------------------------------------------
# 2. Cached Resource Pipelines
# ----------------------------------------------------
@st.cache_resource
def load_detector_infrastructure():
    model_save_path = "RahulKate-173/rt_detrv2_finetuned_trashify_box_detector_v2"
    
    # Force 640x640 size adjustment patch to clear tensor division runtime crash
    image_processor = AutoImageProcessor.from_pretrained(model_save_path)
    image_processor.size = {"height": 640, "width": 640}
    
    model = AutoModelForObjectDetection.from_pretrained(model_save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    
    return image_processor, model, device

image_processor, model, device = load_detector_infrastructure()
id2label = model.config.id2label

# Hex equivalents matching your specified color tokens
color_dict = {   
    "bin": "#00e676",       # green
    "trash": "#2979ff",     # blue
    "hand": "#aa00ff",      # purple
    "trash_arm": "#ffd600", # yellow
    "not_trash": "#ff1744", # red
    "not_bin": "#ff1744",
    "not_hand": "#ff1744",
}

# ----------------------------------------------------
# 3. Main Operational Pipeline Engine
# ----------------------------------------------------
def predict_on_image(image, conf_threshold):
    model.eval()
    
    # Calculate box coordinates mapping against the actual original image scales
    target_sizes = torch.tensor([[image.size[1], image.size[0]]]).to(device)
    
    with torch.no_grad():
        inputs = image_processor(images=[image], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        model_outputs = model(**inputs)
        
        results = image_processor.post_process_object_detection(
            model_outputs,
            threshold=conf_threshold,
            target_sizes=target_sizes
        )[0]

    # Safely unpack tensor results down to execution lists
    boxes = results["boxes"].cpu().tolist()
    scores = results["scores"].cpu().tolist()
    labels = results["labels"].cpu().tolist()

    # Create editable image duplicate
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    
    try:
        font = ImageFont.load_default(size=20)
    except:
        font = ImageFont.load_default()

    detected_class_name_text_labels = []

    for box, score, label in zip(boxes, scores, labels):
        label_name = id2label[label]
        targ_color = color_dict.get(label_name, "#ff1744")
        detected_class_name_text_labels.append(label_name)

        # Draw the rectangle bounding tracking window
        draw.rectangle(xy=box, outline=targ_color, width=3)
        
        # Build text description 
        text_string_to_show = f"{label_name} ({round(score, 3)})"

        # Apply a dark filled background accent capsule underneath the text label for visibility
        text_w, text_h = draw.textbbox((0, 0), text_string_to_show, font=font)[2:]
        draw.rectangle([box[0], box[1] - text_h - 4, box[0] + text_w + 6, box[1]], fill=targ_color)
        draw.text(xy=(box[0] + 3, box[1] - text_h - 2), text=text_string_to_show, fill="black", font=font)
    
    del draw

    # --- Gamification Evaluation Rule Engine ---
    target_items = {"trash", "bin", "hand"} 
    detected_items = set(detected_class_name_text_labels)

    if not detected_items & target_items:
        return_string = (
            f"No trash, bin or hand detected at confidence threshold {conf_threshold}. "
            "Try another image or lowering the confidence threshold."
        )
        return annotated_image, return_string, "info"

    missing_items = target_items - detected_items
    if missing_items:
        return_string = (
            f"Detected the following items: {sorted(detected_items & target_items)}. But missing the following in order to get +1: {sorted(missing_items)}. \n\n"
            "If this is an error, try another image or altering the confidence threshold. "
            "Otherwise, the model may need to be updated with better data."
        )
        return annotated_image, return_string, "warning"

    return_string = f"🎉 +1! Found the following items: {sorted(detected_items)}, thank you for cleaning up the area!"
    return annotated_image, return_string, "success"

# ----------------------------------------------------
# 4. Streamlit Interactive Layout Interface
# ----------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📸 Target Image")
    uploaded_file = st.file_uploader("Upload Canvas File Link", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    # Store dynamic target state reference
    input_image = None
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert("RGB")
        st.image(input_image, use_container_width=True)

    conf_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
    submit_btn = st.button("Submit", type="primary", use_container_width=True)

with col2:
    st.subheader("🖼️ Image Output")
    
    if input_image is not None and submit_btn:
        # Run standard inference mapping sequence
        out_img, message, alert_type = predict_on_image(input_image, conf_threshold)
        
        st.image(out_img, use_container_width=True)
        
        st.subheader("📝 Text Output")
        if alert_type == "success":
            st.success(message)
        elif alert_type == "warning":
            st.warning(message)
        else:
            st.info(message)
    else:
        st.info("Upload an image on the left canvas block and press 'Submit' to parse predictions.")

# ----------------------------------------------------
# 5. Core Setup App Examples Array Checkpoints
# ----------------------------------------------------
st.write("---")
st.subheader("📋 Examples")
st.caption("Click on any preset image column configuration target frame below to explore typical predictions.")

ex_col1, ex_col2, ex_col3 = st.columns(3)

# Mocked asset placeholders tracking your Gradio assets architecture layout locally or via remote url
with ex_col1:
    st.image("https://images.unsplash.com/photo-1532996122724-e3c354a0b15b?w=400", caption="Example 1 (Threshold: 0.3)")
with ex_col2:
    st.image("https://images.unsplash.com/photo-1611284446314-60a58ac0deb9?w=400", caption="Example 2 (Threshold: 0.3)")
with ex_col3:
    st.image("https://images.unsplash.com/photo-1516996087931-5ae40242528e?w=400", caption="Example 3 (Threshold: 0.3)")