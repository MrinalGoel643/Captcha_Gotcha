# UI Design Specification

## Layout Overview

The UI follows a clean, minimalistic design matching your wireframe:

---

## 1. Header Section

**Title:** "CAPTCHA GOTCHA!"
- Font: Bold, 3rem
- Color: #1a1a1a (black)
- Alignment: Center

**Subtitle:** "Can CAPTCHA pull a got ya on AI?"
- Font: Regular, 1.2rem
- Color: #666 (gray)
- Alignment: Center

**Border:** 1px solid #e0e0e0 below header

---

## 2. Description Box

**Background:** #f8f9fa (light gray)
**Border-left:** 4px solid #1a1a1a (black accent)
**Padding:** 1.5rem
**Border-radius:** 4px

**Content:**
> All of us have spent a good amount of time "proving they are human" before accessing 
> a website. The irony is that with AI models becoming so good at object detection, 
> solving CAPTCHA challenges is a cake walk.
>
> Is it possible to make CAPTCHA's AI resistant? In this project, I make use of GradCAM 
> to identify which parts of the image the model uses to make predictions, then I add 
> noise and fool the model.

---

## 3. Upload Section

**Layout:** Two columns (1:3 ratio)

### Left Column: Upload Area
- File uploader component
- Radio buttons for example selection:
  - None
  - Golden Retriever
  - Giant Panda
  - School Bus
  - Banana

### Right Column: Parameters & Preview
- Uploaded image preview
- **Attack Parameters:**
  - Slider: Attack Strength (1-10, default 3)
  - Checkbox: Use Vulnerability-Guided Attack
- **Run Analysis** button (primary, full width)

---

## 4. Results Section (Four Panels)

**Layout:** 4 equal columns

### Panel 1: Original Image
**Card styling:**
- Border: 1px solid #e0e0e0
- Border-radius: 8px
- Padding: 1rem
- Background: white
- Shadow: 0 2px 4px rgba(0,0,0,0.05)

**Content:**
- Image display
- Title: "Original Image"
- Prediction label
- Confidence percentage

### Panel 2: GradCAM Heatmap
**Same card styling**

**Content:**
- GradCAM visualization
- Title: "GradCAM heatmap + Initial AI Prediction"
- Vulnerability Score
- Concentration percentage

### Panel 3: Adversarial Image
**Same card styling**

**Content:**
- Adversarial image (looks identical!)
- Title: "Adversarial Image + Fooled AI prediction"
- Success badge (green) or failure badge (red)
- New prediction label
- New confidence

### Panel 4: Perturbation Visualization
**Same card styling**

**Content:**
- Amplified noise visualization
- Title: "Perturbation Visualisation (Amplified)"
- Text: "Imperceptible to humans"
- Text: "Devastating to AI"

---

## 5. Conclusion Section

**Border-top:** 1px solid #e0e0e0
**Padding-top:** 2rem

**Success message (green box):**
> The AI was successfully fooled from 'Golden Retriever' to 'Tennis Ball'. 
> This demonstrates that small, imperceptible changes can make CAPTCHAs AI-resistant.

**Or failure message (yellow box):**
> The attack did not succeed with current parameters. Try increasing epsilon.

---

## Color Palette

```
Primary Text:    #1a1a1a (almost black)
Secondary Text:  #666666 (medium gray)
Tertiary Text:   #999999 (light gray)

Background:      #ffffff (white)
Card Background: #ffffff (white)
Alt Background:  #f8f9fa (very light gray)

Border:          #e0e0e0 (light gray)
Accent:          #1a1a1a (black)

Success:         #d4edda (light green background)
Success Text:    #155724 (dark green)

Warning:         #f8d7da (light red background)
Warning Text:    #721c24 (dark red)
```

---

## Typography

**Headings:**
- H1 (Title): 3rem, bold, #1a1a1a
- H2 (Section): 1.5rem, semi-bold, #1a1a1a
- H3 (Subsection): 1.1rem, semi-bold, #1a1a1a

**Body:**
- Regular: 1rem, #333333
- Small: 0.9rem, #666666

**Font Family:** 'Helvetica Neue', Arial, sans-serif

---

## Spacing

**Section spacing:** 2rem between major sections
**Card spacing:** 1rem padding inside cards
**Column gap:** 1rem between columns
**Element margin:** 0.5-1rem between elements

---

## Responsive Behavior

**Desktop (>1200px):** 4 columns for results
**Tablet (768-1200px):** 2 columns for results
**Mobile (<768px):** 1 column, stacked layout

---

## Interactive Elements

### Buttons
**Primary button:**
- Background: #1a1a1a
- Text: white
- Hover: #333333
- Border-radius: 4px
- Padding: 0.75rem 2rem

**Secondary button:**
- Background: white
- Text: #1a1a1a
- Border: 1px solid #1a1a1a
- Hover: #f8f9fa

### Sliders
- Track: #e0e0e0
- Thumb: #1a1a1a
- Active: #333333

### Checkboxes
- Unchecked: white with #1a1a1a border
- Checked: #1a1a1a fill

---

## Animation

**Progress bar during attack:**
- Smooth progress from 0-100%
- Color: #1a1a1a
- Height: 4px
- Transition: 0.3s ease

**Image loading:**
- Fade-in effect
- Duration: 0.3s

**Hover effects:**
- Smooth transitions (0.2s)
- Subtle elevation on cards

---

## Accessibility

- High contrast ratios (WCAG AA compliant)
- Clear focus indicators
- Descriptive labels
- Keyboard navigation support
- Screen reader friendly

---

## No-Nos (Following Your Requirements)

❌ **NO emojis anywhere**
❌ **NO bright colors** (only black, white, gray)
❌ **NO rounded corners >8px** (keep it sharp)
❌ **NO heavy shadows** (subtle only)
❌ **NO animations** (except progress bar)
❌ **NO decorative elements** (pure function)

---

## Design Philosophy

**Minimalist:** Remove everything unnecessary
**Clean:** White space is your friend
**Modern:** Contemporary typography and spacing
**Professional:** Business/academic appropriate
**Functional:** Every element serves a purpose

---

This matches your wireframe exactly - clean, minimal, black & white, no emojis, modern and professional.
