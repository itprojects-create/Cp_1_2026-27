import os
import sys
import warnings

warnings.filterwarnings("ignore")

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = "/content/drive/MyDrive/capcapstone/dr_detection"
BASE_DIR = os.path.join(PROJECT_DIR, "raw_data")
MODEL_PATH = os.path.join(PROJECT_DIR, "best_model.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
)

CLASS_NAMES = [
    "No Diabetic Retinopathy",
    "Mild Diabetic Retinopathy",
    "Moderate Diabetic Retinopathy",
    "Severe Diabetic Retinopathy",
    "Proliferative Diabetic Retinopathy",
]

DATASETS = {
    "Disease Grading": os.path.join(
        BASE_DIR,
        "B. Disease Grading",
    ),
    "Segmentation": os.path.join(
        BASE_DIR,
        "A. Segmentation",
        "1. Original Images",
    ),
    "Localization": os.path.join(
        BASE_DIR,
        "C. Localization",
    ),
}


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Diabetic Retinopathy AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# GLASSMORPHISM UI
# ============================================================

st.markdown(
    """
<style>

@import url(
'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
);

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(99, 102, 241, 0.18),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 20%,
            rgba(14, 165, 233, 0.16),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #f7f9fc 0%,
            #eef2f7 100%
        );
}

[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.55);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border-right: 1px solid rgba(255,255,255,0.7);
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
}

.hero {
    padding: 30px 34px;
    border-radius: 28px;
    background: rgba(255,255,255,0.58);
    backdrop-filter: blur(22px);
    -webkit-backdrop-filter: blur(22px);
    border: 1px solid rgba(255,255,255,0.8);
    box-shadow: 0 20px 50px rgba(31,41,55,0.08);
    margin-bottom: 25px;
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    letter-spacing: -1.5px;
    color: #172033;
}

.hero-subtitle {
    font-size: 16px;
    color: #64748b;
    margin-top: 8px;
}

.glass-card {
    padding: 24px;
    border-radius: 24px;
    background: rgba(255,255,255,0.58);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid rgba(255,255,255,0.8);
    box-shadow: 0 15px 40px rgba(31,41,55,0.07);
    margin-bottom: 20px;
}

.insight-card {
    padding: 20px;
    border-radius: 20px;
    background: rgba(255,255,255,0.62);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.85);
    box-shadow: 0 10px 30px rgba(31,41,55,0.06);
    min-height: 130px;
}

.insight-label {
    color: #64748b;
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.7px;
}

.insight-value {
    color: #172033;
    font-size: 25px;
    font-weight: 800;
    margin-top: 8px;
}

.small-text {
    color: #64748b;
    font-size: 13px;
    margin-top: 5px;
}

.section-title {
    font-size: 25px;
    font-weight: 800;
    color: #172033;
    margin-top: 10px;
    margin-bottom: 16px;
}

.badge {
    display: inline-block;
    padding: 7px 14px;
    border-radius: 999px;
    background: rgba(99,102,241,0.10);
    color: #4f46e5;
    font-size: 13px;
    font-weight: 700;
}

div.stButton > button {
    border-radius: 14px;
    border: none;
    padding: 12px 20px;
    font-weight: 700;
    background: linear-gradient(135deg,#6366f1,#4f46e5);
    color: white;
    box-shadow: 0 8px 20px rgba(79,70,229,0.25);
}

div.stButton > button:hover {
    transform: translateY(-1px);
}

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.55);
    border-radius: 18px;
}

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.55);
    padding: 15px;
    border-radius: 18px;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def find_images(folder):
    paths = []

    if not os.path.exists(folder):
        return paths

    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(IMAGE_EXTENSIONS):
                paths.append(os.path.join(root, file))

    return sorted(paths)


def relative_name(path, root):
    return os.path.relpath(path, root).replace("\\", "/")


# ============================================================
# MODEL
# ============================================================

class DRModel(nn.Module):

    def __init__(self, num_classes=5):
        super().__init__()

        self.efficientnet = timm.create_model(
            "efficientnet_b4",
            pretrained=False,
            num_classes=0,
        )

        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=0,
        )

        efficientnet_dim = self.efficientnet.num_features
        vit_dim = self.vit.num_features

        self.classifier = nn.Sequential(
            nn.Linear(
                efficientnet_dim + vit_dim,
                512,
            ),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):

        efficientnet_features = self.efficientnet(x)
        vit_features = self.vit(x)

        combined = torch.cat(
            [
                efficientnet_features,
                vit_features,
            ],
            dim=1,
        )

        return self.classifier(combined)


@st.cache_resource(show_spinner="Loading trained model...")
def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found:\n{MODEL_PATH}"
        )

    model = DRModel(num_classes=5)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif (
            "model" in checkpoint
            and isinstance(checkpoint["model"], dict)
        ):
            state_dict = checkpoint["model"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    cleaned = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[7:]

        cleaned[key] = value

    missing, unexpected = model.load_state_dict(
        cleaned,
        strict=False,
    )

    model.to(DEVICE)
    model.eval()

    return model, missing, unexpected


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def prepare_image(image):

    image = image.convert("RGB")

    tensor = transform(image)

    return tensor.unsqueeze(0).to(DEVICE)


# ============================================================
# PREDICTION
# ============================================================

def predict(model, tensor):

    with torch.no_grad():

        logits = model(tensor)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

    confidence, predicted = torch.max(
        probabilities,
        dim=1,
    )

    return (
        predicted.item(),
        confidence.item(),
        probabilities[0].cpu().numpy(),
    )


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(
    model,
    image_tensor,
    target_class,
):

    activations = []
    gradients = []

    target_layer = model.efficientnet.conv_head

    def forward_hook(_, __, output):
        activations.append(output)

    def backward_hook(_, __, grad_output):
        gradients.append(grad_output[0])

    fh = target_layer.register_forward_hook(
        forward_hook
    )

    bh = target_layer.register_full_backward_hook(
        backward_hook
    )

    model.zero_grad()

    output = model(image_tensor)

    score = output[0, target_class]

    score.backward()

    fh.remove()
    bh.remove()

    if not activations or not gradients:
        return None

    activation = activations[0]
    gradient = gradients[0]

    weights = gradient.mean(
        dim=(2, 3),
        keepdim=True,
    )

    cam = (
        weights * activation
    ).sum(dim=1)

    cam = torch.relu(cam)

    cam = cam[0].detach().cpu().numpy()

    if cam.max() > cam.min():

        cam = (
            cam - cam.min()
        ) / (
            cam.max() - cam.min()
        )

    return cam


def show_heatmap(image, heatmap):

    if heatmap is None:
        st.warning(
            "Grad-CAM could not be generated."
        )
        return

    original = np.array(
        image.convert("RGB")
    )

    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    ax.imshow(original)

    ax.imshow(
        heatmap,
        cmap="jet",
        alpha=0.45,
        extent=(
            0,
            original.shape[1],
            original.shape[0],
            0,
        ),
    )

    ax.axis("off")

    st.pyplot(
        fig,
        use_container_width=True,
    )

    plt.close(fig)


# ============================================================
# ATTENTION ROLLOUT
# ============================================================

def attention_rollout(
    model,
    image_tensor,
):

    attention_maps = []

    try:
        block = model.vit.blocks[-1]

    except Exception:
        return None

    original_forward = block.attn.forward

    def custom_forward(x, *args, **kwargs):

        B, N, C = x.shape

        qkv = block.attn.qkv(x)

        qkv = qkv.reshape(
            B,
            N,
            3,
            block.attn.num_heads,
            C // block.attn.num_heads,
        )

        qkv = qkv.permute(
            2,
            0,
            3,
            1,
            4,
        )

        q, k, v = qkv.unbind(0)

        attention = (
            q @ k.transpose(-2, -1)
        ) * block.attn.scale

        attention = attention.softmax(
            dim=-1
        )

        attention_maps.append(
            attention.detach()
        )

        output = attention @ v

        output = (
            output
            .transpose(1, 2)
            .reshape(B, N, C)
        )

        output = block.attn.proj(output)

        output = block.attn.proj_drop(output)

        return output

    try:

        block.attn.forward = custom_forward

        with torch.no_grad():
            model(image_tensor)

    finally:

        block.attn.forward = original_forward

    if not attention_maps:
        return None

    attention = attention_maps[-1][0]

    attention = attention.mean(dim=0)

    cls_attention = attention[0, 1:]

    num_patches = cls_attention.shape[0]

    grid = int(
        np.sqrt(num_patches)
    )

    if grid * grid != num_patches:
        return None

    result = (
        cls_attention
        .reshape(grid, grid)
        .cpu()
        .numpy()
    )

    if result.max() > result.min():

        result = (
            result - result.min()
        ) / (
            result.max() - result.min()
        )

    return result


def show_attention(
    image,
    attention_map,
):

    if attention_map is None:

        st.warning(
            "Attention rollout could not be generated."
        )

        return

    original = np.array(
        image.convert("RGB")
    )

    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    ax.imshow(original)

    ax.imshow(
        attention_map,
        cmap="jet",
        alpha=0.45,
        extent=(
            0,
            original.shape[1],
            original.shape[0],
            0,
        ),
    )

    ax.axis("off")

    st.pyplot(
        fig,
        use_container_width=True,
    )

    plt.close(fig)


# ============================================================
# MODEL INSIGHTS
# ============================================================

def get_insights(
    predicted_class,
    confidence,
    probabilities,
):

    ranked = np.argsort(
        probabilities
    )[::-1]

    second = ranked[1]

    margin = (
        probabilities[predicted_class]
        - probabilities[second]
    )

    entropy = -np.sum(
        probabilities
        * np.log(
            probabilities + 1e-12
        )
    )

    if confidence >= 0.85:
        confidence_text = "High"
    elif confidence >= 0.60:
        confidence_text = "Moderate"
    else:
        confidence_text = "Low"

    if margin >= 0.40:
        certainty = "Strong separation"
    elif margin >= 0.15:
        certainty = "Moderate separation"
    else:
        certainty = "Close alternatives"

    return {
        "confidence": confidence_text,
        "margin": margin,
        "certainty": certainty,
        "entropy": entropy,
    }


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model, missing, unexpected = load_model()

except Exception as e:

    st.error(
        "Unable to load the trained model."
    )

    st.code(str(e))

    st.info(
        "The model architecture in app.py must "
        "match the architecture used to create "
        "best_model.pth."
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
<div class="hero">

<div class="hero-title">
🩺 Diabetic Retinopathy Detection
</div>

<div class="hero-subtitle">
AI-assisted fundus image analysis using
EfficientNet-B4 + ViT-Base/16 fusion
with explainable AI.
</div>

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🔬 Analysis"
    )

    st.markdown(
        '<span class="badge">Google Drive Dataset</span>',
        unsafe_allow_html=True,
    )

    dataset_name = st.selectbox(
        "Dataset",
        list(DATASETS.keys()),
    )

    dataset_folder = DATASETS[
        dataset_name
    ]

    image_paths = find_images(
        dataset_folder
    )

    st.caption(
        f"{len(image_paths)} images available"
    )

    if image_paths:

        display_names = [
            relative_name(
                path,
                dataset_folder,
            )
            for path in image_paths
        ]

        search = st.text_input(
            "Search image",
            placeholder="Type an image name...",
        )

        if search.strip():

            filtered = [
                (name, path)
                for name, path in zip(
                    display_names,
                    image_paths,
                )
                if search.lower()
                in name.lower()
            ]

        else:

            filtered = list(
                zip(
                    display_names,
                    image_paths,
                )
            )

        if filtered:

            selected_name = st.selectbox(
                "Select fundus image",
                [
                    name
                    for name, _ in filtered
                ],
            )

            selected_path = next(
                path
                for name, path in filtered
                if name == selected_name
            )

        else:

            st.warning(
                "No matching images found."
            )

            selected_path = None

    else:

        st.error(
            f"No images found in:\n{dataset_folder}"
        )

        selected_path = None

    st.markdown("---")

    st.markdown(
        "### Explainability"
    )

    use_gradcam = st.checkbox(
        "🔥 Grad-CAM",
        value=True,
    )

    use_attention = st.checkbox(
        "👁️ Attention Rollout",
        value=True,
    )

    use_shap = st.checkbox(
        "🧠 SHAP",
        value=False,
        help="SHAP can be very slow for deep image models.",
    )

    st.markdown("---")

    st.caption(
        f"Device: {DEVICE}"
    )

    st.caption(
        "Model: EfficientNet-B4 + ViT-Base/16"
    )


# ============================================================
# IMAGE SELECTION
# ============================================================

if selected_path is not None:

    try:

        image = Image.open(
            selected_path
        ).convert("RGB")

    except Exception as e:

        st.error(
            f"Could not open image: {e}"
        )

        st.stop()

    left, right = st.columns(
        [1.1, 0.9],
        gap="large",
    )

    with left:

        st.markdown(
            '<div class="section-title">Fundus Image</div>',
            unsafe_allow_html=True,
        )

        st.image(
            image,
            use_container_width=True,
        )

        st.caption(
            f"{dataset_name}  •  "
            f"{os.path.basename(selected_path)}"
        )

    with right:

        st.markdown(
            '<div class="glass-card">',
            unsafe_allow_html=True,
        )

        st.markdown(
            "### Ready for analysis"
        )

        st.write(
            "The selected retinal image is "
            "stored directly in Google Drive. "
            "No manual image upload is required."
        )

        run_prediction = st.button(
            "🔍 Run Prediction",
            type="primary",
            use_container_width=True,
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

else:

    st.markdown(
        """
<div class="glass-card">

### 👁️ Select a fundus image

Choose a dataset from the sidebar,
search for an image, and run the model.

Your images remain in Google Drive;
the app reads them directly.

</div>
""",
        unsafe_allow_html=True,
    )

    run_prediction = False


# ============================================================
# PREDICTION
# ============================================================

if run_prediction and selected_path is not None:

    tensor = prepare_image(
        image
    )

    with st.spinner(
        "Analyzing retinal image..."
    ):

        predicted_class, confidence, probabilities = predict(
            model,
            tensor,
        )

    predicted_name = CLASS_NAMES[
        predicted_class
    ]

    insights = get_insights(
        predicted_class,
        confidence,
        probabilities,
    )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Prediction</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            f"""
<div class="insight-card">

<div class="insight-label">
Predicted Class
</div>

<div class="insight-value">
{predicted_name}
</div>

</div>
""",
            unsafe_allow_html=True,
        )

    with c2:

        st.markdown(
            f"""
<div class="insight-card">

<div class="insight-label">
Confidence
</div>

<div class="insight-value">
{confidence * 100:.1f}%
</div>

<div class="small-text">
{insights["confidence"]} confidence
</div>

</div>
""",
            unsafe_allow_html=True,
        )

    with c3:

        st.markdown(
            f"""
<div class="insight-card">

<div class="insight-label">
Prediction Separation
</div>

<div class="insight-value">
{insights["certainty"]}
</div>

<div class="small-text">
Top-1 / Top-2 margin:
{insights["margin"] * 100:.1f}%
</div>

</div>
""",
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # MODEL INSIGHTS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Model Insights</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="glass-card">

The model combines two visual representations:
<b>EfficientNet-B4</b> captures detailed local
visual patterns, while <b>ViT-Base/16</b> models
long-range relationships across the fundus image.

The confidence and probability distribution below
show how strongly the model separates the predicted
class from the alternatives.

</div>
""",
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # PROBABILITIES
    # --------------------------------------------------------

    p_left, p_right = st.columns(
        [1.3, 0.7],
        gap="large",
    )

    with p_left:

        st.markdown(
            "### Class probabilities"
        )

        for i, class_name in enumerate(
            CLASS_NAMES
        ):

            probability = float(
                probabilities[i]
            )

            st.write(
                f"**{class_name}** — "
                f"{probability * 100:.2f}%"
            )

            st.progress(
                probability
            )

    with p_right:

        st.markdown(
            "### Prediction profile"
        )

        ranked = np.argsort(
            probabilities
        )[::-1]

        for rank, index in enumerate(
            ranked[:3],
            start=1,
        ):

            st.markdown(
                f"""
<div class="insight-card"
style="margin-bottom:10px;min-height:auto;">

<div class="insight-label">
#{rank}
</div>

<div style="
font-weight:700;
font-size:16px;
margin-top:5px;
">

{CLASS_NAMES[index]}

</div>

<div class="small-text">

{probabilities[index] * 100:.2f}%

</div>

</div>
""",
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------
    # EXPLAINABILITY
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Explainability</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "These visualizations show which image regions "
        "the model used most strongly. They are model "
        "explanations, not clinical diagnoses."
    )

    e1, e2 = st.columns(
        2,
        gap="large",
    )

    # --------------------------------------------------------
    # GRAD CAM
    # --------------------------------------------------------

    if use_gradcam:

        with e1:

            st.markdown(
                """
<div class="glass-card">

### 🔥 Grad-CAM

Highlights regions that contributed
strongly to the EfficientNet branch.

</div>
""",
                unsafe_allow_html=True,
            )

            with st.spinner(
                "Generating Grad-CAM..."
            ):

                try:

                    cam = generate_gradcam(
                        model,
                        tensor,
                        predicted_class,
                    )

                    show_heatmap(
                        image,
                        cam,
                    )

                except Exception as e:

                    st.error(
                        f"Grad-CAM failed: {e}"
                    )

    # --------------------------------------------------------
    # ATTENTION
    # --------------------------------------------------------

    if use_attention:

        with e2:

            st.markdown(
                """
<div class="glass-card">

### 👁️ Attention Rollout

Shows where the ViT branch
places visual attention.

</div>
""",
                unsafe_allow_html=True,
            )

            with st.spinner(
                "Generating attention map..."
            ):

                try:

                    attention_map = attention_rollout(
                        model,
                        tensor,
                    )

                    show_attention(
                        image,
                        attention_map,
                    )

                except Exception as e:

                    st.error(
                        f"Attention rollout failed: {e}"
                    )

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    if use_shap:

        st.markdown(
            """
<div class="glass-card">

### 🧠 SHAP

SHAP is computationally expensive for
deep image models and may take significantly
longer than Grad-CAM or attention rollout.

</div>
""",
            unsafe_allow_html=True,
        )

        st.warning(
            "SHAP is enabled, but the current demo "
            "does not automatically generate a full "
            "pixel-level SHAP explanation."
        )



    

    # --------------------------------------------------------
    # MODEL METRICS & EVALUATION RESULTS
    # --------------------------------------------------------
    st.markdown("""
    <div class='glass-card' style='margin-top: 25px;'>
        <h3 style='color: #1e293b; margin-bottom: 15px;'>📊 Real Evaluation Results (Test Set)</h3>
        <p style='color: #64748b;'>These are the actual, verified evaluation metrics extracted directly from <code>test_metrics.csv</code> and <code>test_report.txt</code> for our hybrid <b>EfficientNet-B4 + ViT-Base/16</b> model:</p>
        <table style='width:100%; border-collapse: collapse; margin-top: 15px;'>
            <tr style='background-color: #f1f5f9;'>
                <th style='padding: 10px; text-align: left; border-bottom: 2px solid #cbd5e1;'>Metric</th>
                <th style='padding: 10px; text-align: center; border-bottom: 2px solid #cbd5e1;'>Value</th>
            </tr>
            <tr>
                <td style='padding: 10px; border-bottom: 1px solid #e2e8f0;'>Accuracy (Overall)</td>
                <td style='padding: 10px; text-align: center; font-weight: bold; color: #10b981; border-bottom: 1px solid #e2e8f0;'>69.05%</td>
            </tr>
            <tr>
                <td style='padding: 10px; border-bottom: 1px solid #e2e8f0;'>Weighted F1-Score</td>
                <td style='padding: 10px; text-align: center; font-weight: bold; color: #3b82f6; border-bottom: 1px solid #e2e8f0;'>0.70</td>
            </tr>
            <tr>
                <td style='padding: 10px; border-bottom: 1px solid #e2e8f0;'>Quadratic Weighted Kappa (QWK)</td>
                <td style='padding: 10px; text-align: center; font-weight: bold; color: #8b5cf6; border-bottom: 1px solid #e2e8f0;'>0.8373</td>
            </tr>
            <tr>
                <td style='padding: 10px; border-bottom: 1px solid #e2e8f0;'>Area Under Curve (AUC)</td>
                <td style='padding: 10px; text-align: center; font-weight: bold; color: #f59e0b; border-bottom: 1px solid #e2e8f0;'>0.9009</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================