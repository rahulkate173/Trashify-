import torch
import torchvision 
from PIL import Image
# Use the explicit RTDetrV2 classes to resolve model head loading issues
from transformers import AutoImageProcessor, AutoModelForObjectDetection

MODEL_PATH = "RahulKate-173/rt_detrv2_finetuned_trashify_box_detector_v1"

# 1. Load processor and force a strictly compatible, divisible size (640x640)
image_processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
image_processor.size = {"height": 640, "width": 640}
print('[INFO] Image Processor loaded ')

# 2. Load the model
model = AutoModelForObjectDetection.from_pretrained(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {device}")
print('[INFO] Model  loaded ')
model = model.to(device).eval()

COLOR_MAP = {
    "bin": "#00e676",
    "trash": "#2979ff",
    "hand": "#aa00ff",
    "trash_arm": "#ffd600",
    "not_trash": "#ff1744",
    "not_bin": "#ff1744",
    "not_hand": "#ff1744",
}

def create_target_image(image: Image.Image):
    width, height = image.size
    # Target size is used for scaling the bounding boxes back to original sizing
    print(f'[INFO] TARGET SIZE:{height,width}')
    return torch.tensor([height, width])
    
def predict(image_path: str, conf_threshold: float):
    image = Image.open(image_path).convert("RGB")
    
    # Keep original dimensions for accurate post-processing box restoration
    target_size = create_target_image(image).unsqueeze(0).to(device)
    print(f'[INFO] target size shape:{target_size.shape}')
    model.eval()
    with torch.no_grad():
        # 3. Process the image using our locked 640x640 dimensions
        inputs = image_processor(images=image, return_tensors="pt")
        print(f'[INFO] Inputs after image processor\n{inputs}')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        print(f'[INFO] Inputs after image processor ++ \n{inputs}')
        # 4. Run through the model without alignment errors
        outputs = model(**inputs)
        print(f'[INFO] model outputs:\n{outputs}\n[INFO] outputs keys:\n{outputs.keys()}')
        # 5. Map boxes cleanly back onto original image dimension thresholds
        output_processed = image_processor.post_process_object_detection(
            outputs, threshold=conf_threshold, target_sizes=target_size
        )[0]
        print(f'[INFO] processed output:\n{output_processed}\n[INFO] processed output keys:\n{output_processed.keys()}')
        scores = output_processed["scores"].tolist()
        labels = output_processed["labels"].tolist()
        boxes = output_processed["boxes"].tolist()
        
        # Map labels from IDs to names using the model configuration mapping
        label_names = [model.config.id2label[lbl] for lbl in labels]
        
        print(f"[INFO] Detected {len(boxes)} items.")
        for score, label_name, box in zip(scores, label_names, boxes):
            print(f" - Found {label_name} ({score:.2f}) at bounding box: {[round(b, 1) for b in box]}")
            
        return scores, label_names, boxes

# Run prediction
scores, label_names, boxes = predict("sample1.jpg", 0.2)
print(f'[INFO] THRESHOLD :\n{0.35}')
print(f'[INFO] Scores:\n{scores}\n[INFO] Scores shape:\n{scores.__len__()}')
print(f'[INFO] labels:\n{label_names}\n[INFO] labels shape:\n{label_names.__len__()}')
print(f'[INFO] boxes:\n{boxes}\n[INFO] boxes shape:\n{boxes.__len__()}')
