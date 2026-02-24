"""
# Risk-Aware Diabetes Prediction with Probability Calibration

## Overview
Built a clinical decision-support pipeline that improves trust in diabetes predictions 
by calibrating model probabilities for risk-aware triage. The system enhances reliability 
without sacrificing classification accuracy.

## What I Built
- Gradient Boosting baseline classifier
- Isotonic Regression probability calibration
- 60/20/20 train–calibration–test split to prevent leakage
- Evaluation using Accuracy, F1, Brier Score, and Expected Calibration Error (ECE)
- Audit logging system for traceable risk decisions

## Key Results
- Accuracy: 92%
- Improved probability reliability (ECE reduced from 0.0042 → 0.0041)
- Generated structured HIGH_RISK / UNCERTAIN_TRIAGE flags for safer workflows

## Tech Stack
Python • Scikit-learn • Pandas • NumPy • Probability Calibration

## Impact
- Produces trustworthy risk scores instead of raw predictions
- Suitable for clinical triage and regulated AI environments
- Demonstrates model governance and accountability design

---
Academic ML project focused on reliability, calibration, and clinical AI safety.
"""
