"""
CAPTCHA GOTCHA! - Adversarial Attacks on AI-based CAPTCHA Systems

TO USE YOUR LOCAL IMAGE:
1. Find line ~450 in this file (search for "My Local Image")
2. Replace "path/to/your/local/image.jpg" with your actual image path
   Example: "dog.jpg" (if in same folder as app.py)
   Example: "/Users/yourname/Desktop/myimage.jpg" (absolute path)
3. Save and run: streamlit run app.py
"""

import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
import urllib.request
import io

# Page config
st.set_page_config(
    page_title="CAPTCHA GOTCHA!",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern, minimalistic design
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 2rem;
    }
    
    /* Title styling */
    .title-container {
        text-align: center;
        padding: 3rem 2rem;
        background: white;
        border: 2px solid #000000;
        border-radius: 8px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .main-title {
        font-size: 4rem;
        font-weight: 900;
        color: #000000;
        margin-bottom: 1rem;
        font-family: 'Helvetica Neue', Arial, sans-serif;
        letter-spacing: -2px;
        text-transform: uppercase;
    }
    
    .subtitle {
        font-size: 1.5rem;
        color: #000000;
        font-weight: 600;
        margin-top: 1rem;
        line-height: 1.4;
    }
    
    /* Description box */
    .description-box {
        background-color: #f8f9fa;
        border-left: 4px solid #1a1a1a;
        padding: 1.5rem;
        margin: 2rem 0;
        border-radius: 4px;
    }
    
    .description-text {
        color: #333;
        line-height: 1.6;
        font-size: 1rem;
    }
    
    /* Image cards */
    .image-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        height: 100%;
    }
    
    .image-card-title {
        font-size: 0.9rem;
        color: #666;
        text-align: center;
        margin-top: 1rem;
        font-weight: 500;
    }
    
    /* Upload section */
    .upload-section {
        border: 2px dashed #ccc;
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
        background: #fafafa;
        margin-bottom: 2rem;
    }
    
    /* Results section */
    .results-container {
        margin-top: 3rem;
        padding-top: 2rem;
        border-top: 1px solid #e0e0e0;
    }
    
    .result-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .result-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 0.5rem;
    }
    
    .result-text {
        color: #333;
        line-height: 1.6;
    }
    
    /* Success/failure badges */
    .success-badge {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        display: inline-block;
        font-weight: 600;
    }
    
    .failure-badge {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        display: inline-block;
        font-weight: 600;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Example buttons */
    .example-container {
        display: flex;
        gap: 0.5rem;
        justify-content: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'labels' not in st.session_state:
    st.session_state.labels = None
if 'cam' not in st.session_state:
    st.session_state.cam = None

@st.cache_resource
def load_model():
    """Load and cache the model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=True)
    model = model.to(device)
    model.eval()
    
    # Setup GradCAM
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # Load labels
    labels_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    with urllib.request.urlopen(labels_url) as url:
        labels = json.loads(url.read().decode())
    
    return model, cam, labels, device

def preprocess_image(image, return_numpy=False):
    """Preprocess image for model input"""
    device = st.session_state.device
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    if return_numpy:
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized).astype(np.float32) / 255.0
        return transform(image).unsqueeze(0).to(device), img_array
    
    return transform(image).unsqueeze(0).to(device)

def get_predictions(image_tensor, top_k=5):
    """Get model predictions"""
    model = st.session_state.model
    labels = st.session_state.labels
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    top_prob, top_idx = torch.topk(probabilities, top_k)
    
    predictions = []
    for i in range(top_k):
        predictions.append({
            'label': labels[top_idx[i].item()],
            'confidence': top_prob[i].item() * 100,
            'class_idx': top_idx[i].item()
        })
    
    return predictions

def generate_gradcam(image_tensor, img_array, target_class=None):
    """Generate GradCAM visualization"""
    cam = st.session_state.cam
    model = st.session_state.model
    
    if target_class is None:
        with torch.no_grad():
            output = model(image_tensor)
            target_class = output.argmax().item()
    
    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    visualization = show_cam_on_image(img_array, grayscale_cam, use_rgb=True)
    
    return visualization, grayscale_cam

def analyze_vulnerability(grayscale_cam, threshold=0.6):
    """Analyze vulnerability from GradCAM"""
    vulnerable_regions = grayscale_cam > threshold
    
    vulnerability_score = np.mean(grayscale_cam)
    concentration = np.sum(vulnerable_regions) / grayscale_cam.size
    
    return {
        'vulnerability_score': vulnerability_score,
        'concentration': concentration,
        'max_activation': np.max(grayscale_cam),
        'vulnerable_mask': vulnerable_regions
    }

def targeted_pgd_attack(image_tensor, target_class, epsilon=0.03, alpha=0.007, 
                       num_iter=40, vulnerability_mask=None):
    """PGD attack with vulnerability guidance"""
    device = st.session_state.device
    model = st.session_state.model
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    
    if vulnerability_mask is not None:
        mask = torch.from_numpy(vulnerability_mask).float().to(device)
        mask = mask.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
        mask = torch.nn.functional.interpolate(mask, size=(224, 224), mode='bilinear')
    else:
        mask = torch.ones(1, 3, 224, 224).to(device)
    
    perturbed = image_tensor.clone().detach()
    
    progress_bar = st.progress(0)
    for i in range(num_iter):
        perturbed.requires_grad = True
        
        output = model(perturbed)
        loss = nn.CrossEntropyLoss()(output, torch.tensor([target_class]).to(device))
        
        current_pred = output.argmax().item()
        current_conf = torch.nn.functional.softmax(output[0], dim=0)[target_class].item()
        
        if current_pred == target_class and current_conf > 0.5:
            progress_bar.progress(100)
            break
        
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            grad = perturbed.grad.data
            grad = grad * mask
            
            perturbed = perturbed - alpha * grad.sign()
            
            perturbation = torch.clamp(perturbed - image_tensor, -epsilon, epsilon)
            perturbed = image_tensor + perturbation
            
            perturbed = (perturbed * std + mean).clamp(0, 1)
            perturbed = (perturbed - mean) / std
        
        progress_bar.progress((i + 1) / num_iter)
    
    progress_bar.empty()
    return perturbed.detach()

def tensor_to_image(tensor):
    """Convert tensor to PIL image"""
    device = st.session_state.device
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    
    img = tensor.squeeze(0).cpu().detach().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = (img * 255).astype(np.uint8)
    
    return Image.fromarray(img)

def process_image(image, epsilon, use_vulnerability_guide):
    """Complete processing pipeline"""
    
    # Preprocess
    img_tensor, img_array = preprocess_image(image, return_numpy=True)
    
    # Get original predictions
    original_preds = get_predictions(img_tensor)
    
    # Generate GradCAM
    gradcam_viz, gradcam_heatmap = generate_gradcam(img_tensor, img_array)
    
    # Vulnerability analysis
    vulnerability = analyze_vulnerability(gradcam_heatmap)
    
    # Choose target class
    target_candidates = [p['class_idx'] for p in original_preds[1:4]]
    target_class = np.random.choice(target_candidates)
    
    # Run attack
    vulnerability_mask = vulnerability['vulnerable_mask'] if use_vulnerability_guide else None
    adv_tensor = targeted_pgd_attack(
        img_tensor, 
        target_class, 
        epsilon=epsilon/100,
        vulnerability_mask=vulnerability_mask
    )
    
    # Get adversarial predictions
    adv_preds = get_predictions(adv_tensor)
    
    # Convert to images
    adv_image = tensor_to_image(adv_tensor)
    original_img = tensor_to_image(img_tensor)
    
    # Generate adversarial GradCAM
    adv_img_array = np.array(adv_image.resize((224, 224))).astype(np.float32) / 255.0
    adv_gradcam_viz, _ = generate_gradcam(adv_tensor, adv_img_array, target_class=adv_preds[0]['class_idx'])
    
    # Calculate perturbation visualization
    diff = np.array(adv_image).astype(float) - np.array(original_img).astype(float)
    diff_normalized = ((diff - diff.min()) / (diff.max() - diff.min() + 1e-8) * 255).astype(np.uint8)
    perturbation_viz = Image.fromarray(diff_normalized)
    
    # Check attack success
    attack_success = adv_preds[0]['label'] != original_preds[0]['label']
    
    return {
        'original_preds': original_preds,
        'gradcam_viz': Image.fromarray(gradcam_viz),
        'vulnerability': vulnerability,
        'adv_image': adv_image,
        'adv_preds': adv_preds,
        'adv_gradcam': Image.fromarray(adv_gradcam_viz),
        'perturbation_viz': perturbation_viz,
        'attack_success': attack_success
    }

# Main app
def main():
    # Title section
    st.markdown("""
    <div class="title-container">
        <div class="main-title">CAPTCHA GOTCHA!</div>
        <div class="subtitle">Can CAPTCHA pull a got ya on AI?</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Description box
    st.markdown("""
    <div class="description-box">
        <p class="description-text">
        <strong>All of us have spent a good amount of time "proving they are human" before accessing a website.</strong>
        <br><br>
        The irony is that with AI models becoming so good at object detection, solving CAPTCHA challenges is a cake walk.
        <br><br>
        <strong>Is it possible to make CAPTCHA's AI resistant?</strong>
        <br><br>
        In this project, I make use of <strong>GradCAM</strong> to identify which parts of the image the model uses to make predictions, 
        then I add noise and fool the model.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    if st.session_state.model is None:
        with st.spinner("Loading model..."):
            model, cam, labels, device = load_model()
            st.session_state.model = model
            st.session_state.cam = cam
            st.session_state.labels = labels
            st.session_state.device = device
    
    # Upload section
    st.markdown("### Upload your image or choose from our examples")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=['png', 'jpg', 'jpeg'],
            label_visibility="collapsed"
        )
        
        st.markdown("**Example images:**")
        example_choice = st.radio(
            "Select an example",
            ["Golden Retriever"],
            label_visibility="collapsed"
        )
    
    with col2:
        if uploaded_file is not None or example_choice != "None":
            # Load image
            image = None
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert('RGB')
            elif example_choice != "None":
                # Load example image - support local and URL
                example_images = {
                    "Golden Retriever": "golden_dog.jpeg"
                }
                
                if example_choice in example_images:
                    try:
                        # Try to load as local file first
                        try:
                            image = Image.open(example_images[example_choice]).convert('RGB')
                        except:
                            # If not local, try URL
                            import requests
                            response = requests.get(example_images[example_choice])
                            image = Image.open(io.BytesIO(response.content)).convert('RGB')
                    except Exception as e:
                        st.error(f"Could not load example image. Please upload your own. Error: {str(e)}")
                        image = None
            
            # Only display if image was successfully loaded
            if image is not None:
                # Display uploaded image
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Simple run button with spacing
                st.markdown("<br>", unsafe_allow_html=True)
                run_analysis = st.button("Run Analysis", type="primary")
                
                # Use strong attack by default (hidden from user)
                epsilon = 7.0  # Strong attack for best results
                use_vulnerability = True  # Always use vulnerability guidance
                
                # Run analysis
                if run_analysis:
                    with st.spinner("Analyzing image and running adversarial attack..."):
                        results = process_image(image, epsilon, use_vulnerability)
                    
                    # Results section
                    st.markdown('<div class="results-container">', unsafe_allow_html=True)
                    st.markdown("## Analysis Results")
                    
                    # Four column layout for results
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown('<div class="image-card">', unsafe_allow_html=True)
                        st.image(image, use_column_width=True)
                        st.markdown('<p class="image-card-title">Original Image</p>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown("**AI Prediction:**")
                        st.write(f"**{results['original_preds'][0]['label']}**")
                        st.write(f"Confidence: {results['original_preds'][0]['confidence']:.1f}%")
                    
                    with col2:
                        st.markdown('<div class="image-card">', unsafe_allow_html=True)
                        st.image(results['gradcam_viz'], use_column_width=True)
                        st.markdown('<p class="image-card-title">GradCAM heatmap + Initial AI Prediction</p>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown("**Vulnerability Analysis:**")
                        st.write(f"Score: {results['vulnerability']['vulnerability_score']:.3f}")
                        st.write(f"Concentration: {results['vulnerability']['concentration']*100:.1f}%")
                    
                    with col3:
                        st.markdown('<div class="image-card">', unsafe_allow_html=True)
                        st.image(results['adv_image'], use_column_width=True)
                        st.markdown('<p class="image-card-title">Adversarial Image + Fooled AI prediction</p>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown("**After Attack:**")
                        if results['attack_success']:
                            st.markdown('<span class="success-badge">ATTACK SUCCESSFUL</span>', unsafe_allow_html=True)
                            st.write(f"**{results['adv_preds'][0]['label']}**")
                            st.write(f"Confidence: {results['adv_preds'][0]['confidence']:.1f}%")
                        else:
                            st.markdown('<span class="failure-badge">Attack Failed</span>', unsafe_allow_html=True)
                            st.write("Try a different image")
                    
                    with col4:
                        st.markdown('<div class="image-card">', unsafe_allow_html=True)
                        st.image(results['perturbation_viz'], use_column_width=True)
                        st.markdown('<p class="image-card-title">Perturbation Visualisation (Amplified)</p>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown("**Perturbation:**")
                        st.write("Imperceptible to humans")
                        st.write("Devastating to AI")
                    
                    # Conclusion
                    st.markdown("---")
                    st.markdown("### Conclusion")
                    
                    if results['attack_success']:
                        st.success(f"The AI was successfully fooled from '{results['original_preds'][0]['label']}' to '{results['adv_preds'][0]['label']}'. This demonstrates that small, imperceptible changes can make CAPTCHAs AI-resistant.")
                    else:
                        st.warning("The attack did not succeed. This image may be naturally robust.")
                    
                    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()