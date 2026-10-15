# Diabetes Risk Prediction

Predicting diabetes and prediabetes risk from CDC health survey data (BRFSS 2015, ~254,000 respondents).

**Question:** Can basic health and lifestyle survey answers flag people at high risk of diabetes, so they can be referred for screening?

## Status
Exploration complete. Modeling in progress.

## Key findings so far
- Only 13.9% of respondents have diabetes or prediabetes, so accuracy is misleading; recall, F1 and ROC-AUC are used instead.
- Self-rated health is the strongest single signal: diabetes rate rises from 2.5% (excellent) to 37.9% (poor).
- High blood pressure goes with 4x the diabetes rate (24% vs 6%).
- Median BMI is 31 with diabetes vs 27 without, but the groups overlap heavily.

## Data
[CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators) from the UCI Machine Learning Repository. The notebook downloads it with the `ucimlrepo` package.

## How to run
pip install -r requirements.txt
jupyter notebook

*For learning purposes only, not medical advice.*
