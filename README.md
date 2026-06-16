# 🚮 Trashify Object Detection Demo

### streamlit link (hosted) : https://ry438wpt4hzqbn4app33cux.streamlit.app/

Trashify is an object detection application aimed at helping clean up local areas! Upload an image, and the system evaluates if it contains a `trash`, `bin`, and `hand`. If all three are detected, you receive a +1!

## 🚀 Features
- **Gamified Object Detection:** Earn points by proving you picked up trash and disposed of it correctly.
- **State-of-the-Art Model:** Powered by a fine-tuned version of [RT-DETRv2](https://huggingface.co/docs/transformers/main/en/model_doc/rt_detr_v2) trained on the [Trashify dataset](https://huggingface.co/datasets/mrdbourke/trashify_manual_labelled_images).
- **Interactive UI:** Built using Streamlit, featuring an intuitive drag-and-drop interface, confidence threshold sliders, and real-time inference with bounding box annotations.
- **Model Hosting:** The model is hosted on Hugging Face: [`RahulKate-173/rt_detrv2_finetuned_trashify_box_detector_v2`](https://huggingface.co/RahulKate-173/rt_detrv2_finetuned_trashify_box_detector_v2).

## 📁 Repository Structure
- **`app.py`**: The main Streamlit application script containing the UI layout, inference pipeline, and the gamification rule engine.
- **`notebooks/`**: A collection of Jupyter Notebooks detailing the end-to-end process of building Trashify:
  - `Trashify_2_model.ipynb`: Loading the dataset, preprocessing in COCO format, training/fine-tuning the `RT-DETRv2` model, and pushing to Hugging Face.
  - `Building_trashify_with_augmentation.ipynb`: Experiments utilizing data augmentation.
  - `Non_Mamixum_Supperission (1).ipynb`: Deep dives into object detection concepts like Non-Maximum Suppression (NMS).
  - Other notebooks detailing the iterative process of building the model and interface.
- **`requirements.txt`**: The required Python packages to run the application and notebooks.

## 🛠️ How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd trashify
   ```

2. **Install the dependencies:**
   Ensure you have Python installed. It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: Core dependencies include `torch`, `streamlit`, `Pillow`, and `transformers`)*.

3. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

4. **Open in Browser:**
   The application should automatically open in your default browser at `http://localhost:8501`.

## 🧠 Model Classes & Colors
The model detects the following classes, which are annotated with specific colors during inference:
- **Bin**: Green (`#00e676`)
- **Trash**: Blue (`#2979ff`)
- **Hand**: Purple (`#aa00ff`)
- **Trash Arm**: Yellow (`#ffd600`)
- **Not Trash / Not Bin / Not Hand**: Red (`#ff1744`)
