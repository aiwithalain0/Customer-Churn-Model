# Hyperframes Composition Brief: CustomerIQ

## Objective
Create a short launch-style brag video for CustomerIQ, an enterprise customer churn prediction and Explainable AI (XAI) analytics platform.

## Output
- Composition directory: `brag-output/composition/`
- Rendered video: `brag-output/brag.mp4`
- Format: landscape — 1920x1080
- Duration: 18.0 seconds

## Source Material
- Project root: `d:/My Projects/Customer_Churn_Model`
- Primary files read: `static/index.html`, `static/styles.css`, `static/app.js`, `server.py`, `README.md`
- Product name: CustomerIQ
- Tagline / strongest claim: "Smarter Insights. Stronger Relationships." / "Predict churn risk with transparent SHAP factor attribution."
- Key UI or visual moments to recreate:
  - Header with CustomerIQ icon and Model Status badge ("● Online - XGBoost & SHAP explainer ready")
  - 4 Enterprise KPI Cards (Total: 7,043, At-Risk: 3,063, Churn Rate: 26.5%, Retention: 56.7%)
  - Individual Account XAI Card (#7590-VHVEG, 84.2% High Risk, dynamic SHAP factor bars)
  - Proactive Recommendation & Export Report action
- Copy that must appear verbatim:
  - "CustomerIQ"
  - "Smarter Insights. Stronger Relationships."
  - "Total Customers: 7,043"
  - "At-Risk Customers: 3,063"
  - "XGBoost & SHAP explainer ready"
  - "Month-to-month Contract"

## Creative Direction
- Tone preset: `polished`
- Creative direction: "authoritative enterprise AI product film — clean, data-driven, executive-ready"
- Interpretation: Deep contrast, crisp typography (Inter), responsive metric cards, sleek glass elevation, and steady confident motion that highlights explainable machine learning rather than generic SaaS hype.
- Angle: Moving beyond black-box churn scores to exact, actionable causality using trained XGBoost & SHAP models on enterprise accounts.
- Hook: "Every customer leaves clues. Most companies only notice after they're gone."
- Outro / punchline: "CustomerIQ — Smarter Insights. Stronger Relationships."
- Avoid:
  - Generic SaaS language
  - Abstract filler visuals
  - Low-contrast or unreadable text

## Visual Identity
- Background: `#0b0f19` (Dark intro) and `#f8fafc` (Dashboard Canvas)
- Text Primary: `#0f172a` (Slate 900)
- Text Secondary: `#64748b` (Slate 500)
- Accent: `#2563eb` (Brand Blue), `#3b82f6` (Vibrant Blue)
- Risk High: `#ef4444` / Risk Med: `#f59e0b` / Risk Low: `#10b981`
- Card Elevation: `#ffffff` with `#e2e8f0` border and subtle shadow
- Display font: `Inter`, system-ui, sans-serif
- Body font: `Inter`, system-ui, sans-serif

## Storyboard
Scene summary:
1. The Problem: Silent Attrition — 3.5s — Dark canvas, radar grid, "Every customer leaves clues. Most teams only notice after they've already churned."
2. The Solution: CustomerIQ Command Center — 5.0s — 4 KPI cards arrive sequentially on beat grid (7,043 total, 3,063 at-risk), Model Online status badge.
3. Explainable AI: SHAP Factor Attribution — 5.5s — Individual account #7590-VHVEG, 84.2% High Risk badge, expanding SHAP horizontal impact bars, retention recommendation.
4. Executive Delivery & Outro — 4.0s — Brand lockup, tech badges (FastAPI, XGBoost, SHAP), Export Report action, fade to black.

## Audio
- Audio role: Upbeat, modern tech groove bed with crisp interface accents
- Audio arc: Ambient anticipation into energetic beat-driven reveal, analytical focus, clean executive resolution.
- Music: `happy-beats-business-moves-vol-10-by-ende-dot-app.mp3`
- Music treatment: Plays from 0.0s to 18.0s, gentle fade starting at 16.5s.
- Music cue guidance: Tempo ~109.96 BPM.
  - Strong cues / major transitions: 3.55s (reveal), 8.73s (SHAP deep dive), 14.20s (outro).
  - Beat-grid sequential reveals: 4.10s, 4.64s, 5.19s, 5.74s for the 4 KPI cards.
- Audio-reactive treatment: Subtle RMS glow on the hero brand badge and risk indicator border.
- SFX posture: Crisp motion-matched UI clicks and card entrance swooshes.
- Audio files: `happy-beats-business-moves-vol-10-by-ende-dot-app.mp3` and interface sound effects copied to `brag-output/composition/assets/`.

## Hyperframes Instructions
Build a self-contained, high-performance HTML/CSS/JS Hyperframes composition in `brag-output/composition/`:
- Multi-scene structure using `<div class="scene" data-scene="1" data-start="0" data-duration="3.5">` etc.
- Video duration exactly 18.0 seconds.
- Resolution 1920x1080 (16:9 landscape).
- Local assets for audio under `assets/music/` and `assets/sfx/`.
- Verify with `npx hyperframes check` before rendering with `npx hyperframes render`.
