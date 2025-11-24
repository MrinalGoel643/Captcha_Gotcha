# CAPTCHA GOTCHA! - Quick Start

## Run Locally (2 minutes)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py

# 3. Open browser
# App will open at: http://localhost:8501
```

---

## Deploy to Cloud (5 minutes)

### Option 1: Streamlit Cloud (Easiest)

1. Push these files to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app" → Select your repo
4. Done! ✅

### Option 2: Hugging Face Spaces

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Create new Space → Choose "Streamlit" SDK
3. Upload `app.py` and `requirements.txt`
4. Done! ✅

---

## What's Included

```
streamlit-app/
├── app.py              # Main application (clean, modern UI)
├── requirements.txt    # All dependencies
├── README.md          # Full deployment guide
└── .streamlit/
    └── config.toml    # Theme configuration (minimalist black & white)
```

---

## Features

✅ **Clean UI** - Minimalistic design, no emojis
✅ **Four-Panel Layout** - Original → GradCAM → Adversarial → Perturbation
✅ **Interactive Controls** - Upload images, adjust attack strength
✅ **Example Images** - Pre-loaded dog, panda, bus, banana
✅ **Real-time Progress** - Watch attack progress bar
✅ **Professional Styling** - Black & white, modern typography

---

## Usage

1. **Upload image** (or choose example)
2. **Set attack strength** (3-5 recommended)
3. **Enable vulnerability guidance** (recommended)
4. **Click "Run Analysis"**
5. **View results** in four panels

---

## Best Images to Try

- Golden Retriever (90% success)
- Giant Panda (90% success)
- School Bus (80% success)
- Banana (95% success)

---

**Need help?** See `README.md` for full documentation.

**Ready to go?** Run `streamlit run app.py` now!
