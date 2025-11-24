# Testing & Troubleshooting Checklist

## Pre-Deployment Testing

### Local Testing

- [ ] Install all dependencies: `pip install -r requirements.txt`
- [ ] Run app: `streamlit run app.py`
- [ ] App opens at `localhost:8501`
- [ ] No errors in terminal
- [ ] Model loads successfully (wait 30-60 seconds first run)

### UI Testing

- [ ] Title displays correctly: "CAPTCHA GOTCHA!"
- [ ] Subtitle shows: "Can CAPTCHA pull a got ya on AI?"
- [ ] Description box is visible and readable
- [ ] Upload button works
- [ ] Example radio buttons display
- [ ] Sliders work smoothly
- [ ] Checkbox toggles properly
- [ ] Run Analysis button is clickable

### Functionality Testing

#### Test 1: Upload Custom Image
- [ ] Upload a clear dog/cat image
- [ ] Image displays in preview
- [ ] Set epsilon to 3
- [ ] Enable vulnerability guidance
- [ ] Click "Run Analysis"
- [ ] Progress bar appears and completes
- [ ] Four panels display results
- [ ] Images are visible in all panels
- [ ] Text predictions are correct
- [ ] Success/failure badge shows

#### Test 2: Example Images
- [ ] Select "Golden Retriever"
- [ ] Image loads automatically
- [ ] Run analysis works
- [ ] Repeat for other examples

#### Test 3: Parameter Variations
- [ ] Test epsilon = 1 (should fail more often)
- [ ] Test epsilon = 5 (should succeed more)
- [ ] Test epsilon = 10 (should always succeed)
- [ ] Test with vulnerability OFF
- [ ] Test with vulnerability ON
- [ ] Compare results

### Performance Testing

- [ ] First run: 30-60 seconds (model download)
- [ ] Subsequent runs: 20-30 seconds
- [ ] No memory errors
- [ ] No crashes
- [ ] Progress bar is smooth

---

## Common Issues & Solutions

### Issue: "Module not found"

**Error:**
```
ModuleNotFoundError: No module named 'pytorch_grad_cam'
```

**Solution:**
```bash
pip install grad-cam
```

**Or install all:**
```bash
pip install -r requirements.txt
```

---

### Issue: "Model takes forever to load"

**Cause:** First-time model download (~100MB)

**Solution:**
- Wait 1-2 minutes on first run
- Check internet connection
- Model is cached after first download

**Check if model is cached:**
```bash
# Look for cached model
ls ~/.cache/torch/hub/checkpoints/
```

---

### Issue: "CUDA out of memory"

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution 1: Use CPU**
In `app.py`, force CPU:
```python
device = torch.device("cpu")  # Instead of checking for cuda
```

**Solution 2: Reduce batch size**
Already optimized (batch size = 1)

**Solution 3: Use smaller images**
Images are auto-resized to 224x224

---

### Issue: "Attack always fails"

**Symptoms:**
- Attack fails even with high epsilon
- Success badge never shows
- Same prediction before/after

**Solutions:**

1. **Increase epsilon:**
   - Try 5-7 instead of 3
   - Some images need stronger attacks

2. **Enable vulnerability guidance:**
   - Check the vulnerability guidance box
   - Much more effective

3. **Try different images:**
   - Animals work best (dogs, cats, pandas)
   - Avoid traffic lights (naturally robust)
   - Use clear, centered objects

4. **Check model predictions:**
   - If original prediction is wrong, attack will fail
   - Use images model can correctly classify

---

### Issue: "Streamlit port already in use"

**Error:**
```
Address already in use
```

**Solution:**
```bash
# Kill existing Streamlit processes
pkill -f streamlit

# Or use different port
streamlit run app.py --server.port 8502
```

---

### Issue: "Images not displaying"

**Symptoms:**
- Blank panels
- "Failed to load image" errors

**Solutions:**

1. **Check image format:**
   - Only PNG, JPG, JPEG supported
   - Convert if needed

2. **Check file size:**
   - Max 10MB (set in config)
   - Resize large images

3. **Check permissions:**
   - Ensure read permissions on uploaded files

---

### Issue: "Slow performance"

**Symptoms:**
- Takes >60 seconds per image
- UI freezes
- Browser becomes unresponsive

**Solutions:**

1. **Enable GPU (if available):**
   ```python
   # Check if GPU is being used
   print(torch.cuda.is_available())
   ```

2. **Reduce iterations:**
   In `app.py`, change:
   ```python
   num_iter=20  # Instead of 40
   ```

3. **Close other apps:**
   - Free up RAM
   - Close browser tabs

4. **Deploy to cloud with GPU:**
   - Hugging Face Pro ($9/month)
   - Much faster

---

### Issue: "Example images not loading"

**Error:**
```
Failed to load example image
```

**Solutions:**

1. **Check internet connection:**
   - Example URLs need internet access

2. **Use local examples instead:**
   - Download images to `/examples/` folder
   - Update code to load from local

3. **Try different example:**
   - Some URLs may be temporarily down

---

## Deployment Issues

### Streamlit Cloud

**Issue: "App won't start"**

**Check:**
- [ ] `requirements.txt` exists
- [ ] All packages have versions
- [ ] Python version compatible (3.8-3.11)

**Fix:**
Add to `requirements.txt`:
```
python-version>=3.8,<3.12
```

---

**Issue: "Build fails"**

**Common causes:**
- Incompatible package versions
- Missing dependencies
- Large model downloads timing out

**Fix:**
Use specific versions:
```
torch==2.0.1
torchvision==0.15.2
```

---

### Hugging Face Spaces

**Issue: "Out of memory"**

**Solution:**
- Upgrade to Pro for more RAM
- Or reduce model size (use ResNet34)

---

**Issue: "Build timeout"**

**Solution:**
- Hugging Face has build time limits
- Ensure model downloads quickly
- Consider pre-downloading model

---

## Performance Benchmarks

### Expected Times (CPU)

| Task | Duration |
|------|----------|
| First model load | 30-60s |
| Subsequent loads | <1s (cached) |
| Image preprocessing | <1s |
| GradCAM generation | 2-3s |
| Attack (40 iterations) | 20-30s |
| Total per image | 25-35s |

### Expected Times (GPU - T4)

| Task | Duration |
|------|----------|
| First model load | 20-30s |
| Subsequent loads | <1s (cached) |
| Image preprocessing | <1s |
| GradCAM generation | <1s |
| Attack (40 iterations) | 5-8s |
| Total per image | 7-10s |

---

## Pre-Launch Checklist

### Before Demo/Presentation

- [ ] Test with 3-4 sample images
- [ ] Verify all examples work
- [ ] Check attack success rate >80%
- [ ] Screenshots ready
- [ ] Backup plan (video recording)

### Before Deployment

- [ ] Remove debug print statements
- [ ] Test on clean Python environment
- [ ] Check all dependencies in requirements.txt
- [ ] Test deployment on target platform
- [ ] Verify example URLs work
- [ ] Update README with live demo link

### Before Going Public

- [ ] Add LICENSE file
- [ ] Update contact information
- [ ] Add screenshots to repo
- [ ] Test from different browsers
- [ ] Mobile responsiveness check
- [ ] Accessibility check

---

## Emergency Fixes

### Quick Fix: App crashes immediately

```bash
# 1. Update all packages
pip install --upgrade -r requirements.txt

# 2. Clear cache
rm -rf ~/.streamlit/cache/

# 3. Restart
streamlit run app.py
```

---

### Quick Fix: Model won't load

```python
# In app.py, add error handling
try:
    model = models.resnet50(pretrained=True)
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.info("Try: pip install --upgrade torch torchvision")
```

---

### Quick Fix: UI broken

```bash
# Clear Streamlit cache
streamlit cache clear

# Force reload
streamlit run app.py --server.runOnSave true
```

---

## Success Criteria

App is ready when:

✅ Loads without errors
✅ Model downloads and caches
✅ Upload works smoothly
✅ Examples load correctly
✅ Attack succeeds >80% of time
✅ Results display in 4 panels
✅ UI is clean and responsive
✅ No console errors
✅ Performance acceptable (<40s per image on CPU)

---

## Getting Help

**Still having issues?**

1. Check [Streamlit docs](https://docs.streamlit.io)
2. Check [PyTorch docs](https://pytorch.org/docs)
3. Search [Streamlit forum](https://discuss.streamlit.io)
4. Open GitHub issue

**For immediate help:**
- Check error messages carefully
- Google the exact error
- Check package versions
- Try on clean Python environment

---

**Remember:** Most issues are package version conflicts or missing dependencies. When in doubt, reinstall everything!

```bash
pip install --force-reinstall -r requirements.txt
```
