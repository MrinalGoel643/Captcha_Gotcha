# CAPTCHA GOTCHA! - Streamlit Deployment Guide

Clean, modern, minimalistic UI for demonstrating adversarial attacks on CAPTCHA systems.

---

## Quick Start (Local)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## Deploy to Streamlit Cloud (FREE)

### Step 1: Prepare Your Repository

1. Create a GitHub repository
2. Upload these files:
   - `app.py`
   - `requirements.txt`
   - `README.md` (this file)

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Main file path: `app.py`
6. Click "Deploy"

**That's it!** Your app will be live in ~5 minutes.

---

## Deploy to Hugging Face Spaces (FREE)

### Step 1: Create Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Choose "Streamlit" as SDK
4. Name it "captcha-gotcha"

### Step 2: Upload Files

Upload:
- `app.py`
- `requirements.txt`

### Step 3: Wait for Build

The space will automatically build and deploy!

**URL will be:** `https://huggingface.co/spaces/YOUR_USERNAME/captcha-gotcha`

---

## Features

### Clean UI Design
- Minimalistic layout matching your wireframe
- No emojis or clutter
- Professional typography
- Responsive grid layout

### Four-Panel Results
1. **Original Image** - Initial upload with AI prediction
2. **GradCAM Heatmap** - Shows where AI focuses + vulnerability analysis
3. **Adversarial Image** - Attacked image with fooled prediction
4. **Perturbation Viz** - Amplified noise visualization

### Interactive Controls
- Upload custom images
- Choose from example images (dog, panda, bus, banana)
- Adjust attack strength (epsilon slider)
- Toggle vulnerability-guided attacks
- Real-time progress bar during attack

---

## Customization

### Change Colors

Edit the CSS in `app.py`:

```python
# Current: Black and white minimalist
.main-title {
    color: #1a1a1a;  # Change title color
}

.description-box {
    border-left: 4px solid #1a1a1a;  # Change accent color
}
```

### Add More Examples

In `app.py`, update the `example_urls` dictionary:

```python
example_urls = {
    "Golden Retriever": "YOUR_IMAGE_URL",
    "Giant Panda": "YOUR_IMAGE_URL",
    # Add more examples
}
```

### Adjust Default Parameters

```python
epsilon = st.slider(
    "Attack Strength (Epsilon)",
    min_value=1.0,
    max_value=10.0,
    value=3.0,  # Change default value
    step=0.5
)
```

---

## Performance Notes

### First Load
- Model downloads (~100MB)
- Takes 30-60 seconds
- Cached for subsequent uses

### Per Image Analysis
- Without GPU: ~20-30 seconds
- With GPU: ~5-10 seconds

### Optimization Tips

For faster deployment:

1. **Use smaller model** (optional):
```python
model = models.resnet34(pretrained=True)  # Instead of resnet50
```

2. **Reduce iterations** (optional):
```python
num_iter=20  # Instead of 40 (slightly less effective)
```

3. **Enable GPU on Hugging Face**:
   - Upgrade to Pro ($9/month)
   - Get T4 GPU access
   - 5x faster processing

---

## Troubleshooting

### Issue: "Model not loading"
**Solution:** Check internet connection. Model downloads on first run.

### Issue: "Out of memory"
**Solution:** 
- Reduce image size
- Use CPU instead of GPU
- Restart the app

### Issue: "Slow performance"
**Solution:**
- Deploy on Hugging Face with GPU
- Or use Streamlit Cloud (free but CPU only)

### Issue: "Attack always fails"
**Solution:**
- Increase epsilon to 5-7
- Enable vulnerability-guided attack
- Try different images (animals work best)

---

## Example Usage Flow

### For Presentations:

1. **Upload golden retriever image**
2. **Set epsilon to 3-5**
3. **Enable vulnerability guidance**
4. **Click "Run Analysis"**
5. **Wait 20-30 seconds**
6. **Show results:**
   - Original: "Golden Retriever (95%)"
   - GradCAM shows focus on face/fur
   - After attack: "Tennis Ball (87%)"
   - Perturbation is imperceptible

### Live Demo Script:

> "Let me show you how this works. I'll upload this picture of a golden retriever. 
> The AI correctly identifies it with 95% confidence.
> 
> Now, using GradCAM, we can see where the AI focuses - mainly on the face and fur texture.
> 
> I'll run our vulnerability-guided attack... and watch what happens.
> 
> The image looks identical to us, but the AI now thinks it's a tennis ball with 87% confidence!
> 
> This tiny noise - which you can only see when amplified - completely fooled the AI.
> This proves CAPTCHAs CAN pull a 'got ya' on AI!"

---

## File Structure

```
captcha-gotcha/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── .streamlit/        # Optional config (create if needed)
    └── config.toml    # Streamlit configuration
```

### Optional: Create `.streamlit/config.toml`

```toml
[theme]
primaryColor = "#1a1a1a"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f8f9fa"
textColor = "#1a1a1a"
font = "sans serif"

[server]
maxUploadSize = 10
```

---

## Cost Analysis

### Free Options:

**Streamlit Cloud:**
- ✅ Free forever
- ✅ Easy deployment
- ❌ CPU only (slower)
- ❌ Public only

**Hugging Face Spaces (Free Tier):**
- ✅ Free forever
- ✅ Easy deployment
- ❌ CPU only (slower)
- ✅ Can be private

### Paid Options:

**Hugging Face Pro ($9/month):**
- ✅ T4 GPU access (5x faster)
- ✅ Private spaces
- ✅ Better performance

**Google Cloud Run:**
- Pay per use (~$5-20/month)
- GPU optional
- More complex setup

---

## Recommended Deployment

**For presentations/demos:** 
→ **Streamlit Cloud** (free, easy)

**For portfolio/public use:** 
→ **Hugging Face Spaces Free** (free, shareable)

**For production/fast performance:** 
→ **Hugging Face Pro** ($9/month, GPU)

---

## Sharing Your App

Once deployed, share via:

**Streamlit Cloud:**
`https://YOUR_USERNAME-captcha-gotcha-app-abc123.streamlit.app`

**Hugging Face:**
`https://huggingface.co/spaces/YOUR_USERNAME/captcha-gotcha`

---

## Next Steps

### Enhancements You Can Add:

1. **Download Results Button**
```python
# Add after results display
st.download_button(
    "Download Results",
    data=results_as_zip,
    file_name="captcha_gotcha_results.zip"
)
```

2. **Comparison Mode**
```python
# Compare multiple attacks side-by-side
col1, col2, col3 = st.columns(3)
# Show FGSM vs PGD vs Guided PGD
```

3. **Attack History**
```python
# Show past attacks in sidebar
if 'history' not in st.session_state:
    st.session_state.history = []
```

4. **Export GradCAM as Overlay**
```python
# Save GradCAM + original as single image
overlay = create_overlay(image, gradcam)
```

---

## Support

**Issues?** 
- Check troubleshooting section above
- Ensure all dependencies are installed
- Try with different images

**Questions?**
- Streamlit docs: [docs.streamlit.io](https://docs.streamlit.io)
- Hugging Face docs: [huggingface.co/docs](https://huggingface.co/docs)

---

## License

MIT License - Free to use for academic and personal projects.

---

## Credits

**Project:** CAPTCHA GOTCHA!
**Tech Stack:** PyTorch, ResNet50, GradCAM, Streamlit
**Method:** Vulnerability-Guided PGD Attack

---

**Ready to deploy? Let's go! 🚀**

```bash
streamlit run app.py
```
