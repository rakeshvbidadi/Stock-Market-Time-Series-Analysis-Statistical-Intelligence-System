# 📈 Stock Market Time Series Analysis & Statistical Intelligence System

## Overview

This project is a comprehensive statistical and time-series analysis framework developed using Python. It combines descriptive statistics, probability theory, inferential statistics, linear algebra, Principal Component Analysis (PCA), and time-series decomposition to analyze stock market behavior and generate actionable insights.

The project was developed as part of a university Data Science and Statistics assignment and demonstrates the practical application of:

* Descriptive Statistics
* Probability Theory
* Bayes Theorem
* Hypothesis Testing
* Confidence Intervals
* Linear Algebra
* Principal Component Analysis (PCA)
* Time Series Decomposition
* Data Visualization

---

## Dataset Information

| Attribute      | Value                        |
| -------------- | ---------------------------- |
| Dataset Name   | Stock Market Dataset         |
| Domain         | Financial Markets            |
| Period         | January 2021 – December 2022 |
| Observations   | 5,200                        |
| Variables      | 29                           |
| Companies      | 10                           |
| Sectors        | 6                            |
| Missing Values | 0                            |

### Companies Included

* AAPL
* MSFT
* GOOGL
* AMZN
* JPM
* BAC
* JNJ
* PFE
* WMT
* XOM

### Sectors Included

* Technology
* Financials
* Healthcare
* Energy
* Consumer Staples
* Consumer Discretionary

---

# Project Workflow

```text
Load Dataset
      ↓
Data Understanding
      ↓
Descriptive Statistics
      ↓
Probability Analysis
      ↓
Inferential Statistics
      ↓
Linear Algebra
      ↓
PCA Analysis
      ↓
Time Series Decomposition
      ↓
Visualization
      ↓
Insight Generation
      ↓
Final Report
```

---

# Features Implemented

## Dataset Understanding

* Dataset Preview
* Variable Classification
* Target Variable Identification
* Data Type Analysis

---

## Descriptive Statistics

For all major numerical variables:

* Mean
* Median
* Variance
* Standard Deviation
* Quartiles
* Interquartile Range (IQR)
* Outlier Detection

### Outlier Detection Method

```text
Lower Bound = Q1 − 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

---

## Probability Analysis

Calculated:

* P(Price Up)
* P(Bull Regime)
* P(Price Up | Bull Regime)
* P(Bull Regime | Price Up)

### Bayes Theorem

```math
P(Bull|Up)=\frac{P(Up|Bull)P(Bull)}{P(Up)}
```

### Result

A single-day stock increase provides very little evidence regarding market regime.

---

## Inferential Statistics

### Two-Sample T-Test

Objective:

Compare daily returns during Bull and Bear markets.

Result:

```text
t = 0.061
p = 0.9514
```

Conclusion:

No statistically significant difference.

---

### One-Way ANOVA

Objective:

Compare stock prices across sectors.

Result:

```text
F = 599.67
p < 0.0001
```

Conclusion:

Sector membership significantly affects stock price levels.

---

### Chi-Square Test

Objective:

Evaluate relationship between Trading Signal and Price Movement.

Result:

```text
χ² = 4224.85
p < 0.0001
```

Conclusion:

Strong association exists.

---

### Confidence Intervals

#### Mean Daily Return

```text
(-0.00614%, 0.06760%)
```

#### Probability of Price Increase

```text
(0.4941, 0.5213)
```

---

# Linear Algebra Concepts

The project implements:

* Matrix Representation
* Mean Centering
* Covariance Matrix
* Correlation Matrix
* Eigenvalue Decomposition
* Eigenvector Analysis

### Eigen Decomposition

```math
A = V \Lambda V^{-1}
```

Where:

* A = Covariance Matrix
* V = Eigenvectors
* Λ = Eigenvalues

---

# Principal Component Analysis (PCA)

### Purpose

Reduce dimensionality while retaining maximum information.

### PCA Results

| Component | Variance Explained |
| --------- | ------------------ |
| PC1       | 32.13%             |
| PC2       | 20.30%             |
| PC3       | 12.53%             |
| PC4       | 12.43%             |
| PC5       | 11.43%             |

### Key Finding

```text
Five principal components explain over 80% of total variance.
```

### PCA Interpretation

| Component | Meaning   |
| --------- | --------- |
| PC1       | Size      |
| PC2       | Momentum  |
| PC3       | Valuation |

---

# Time Series Decomposition

Two decomposition approaches were implemented:

## Moving Average Decomposition

Components:

* Trend
* Seasonal
* Residual

## STL Decomposition

STL (Seasonal-Trend decomposition using Loess)

Advantages:

* Robust
* Handles non-linear trends
* Better seasonal extraction

---

# Visualizations Generated

## Figure 1

Time Series Decomposition

```text
fig1_decomposition.png <img width="2684" height="2079" alt="image" src="https://github.com/user-attachments/assets/6fdabcd4-5e81-4754-940e-51f163495592" />

```

Shows:

* Original Series
* Trend
* Seasonal Component
* Residual Component

---

## Figure 2

Descriptive Statistics

```text
fig2_descriptive.png
```

Includes:

* Histograms
* Boxplots
* Bar Charts

---

## Figure 3

Relationship Analysis

```text
fig3_relationships.png
```

Includes:

* Scatter Plots
* Trend Analysis
* Grouped Bar Charts
* Stacked Bar Charts

---

## Figure 4

Multivariate Analysis

```text
fig4_multivariate.png
```

Includes:

* Correlation Heatmap
* Covariance Heatmap
* Scree Plot
* PCA Projection

---

# Key Insights

1. Daily returns do not significantly differ between Bull and Bear markets.
2. Sector membership strongly influences stock price levels.
3. Trading signals show a strong relationship with price movement.
4. Average daily return is statistically close to zero.
5. Market capitalization and stock price dominate the first PCA component.
6. Five principal components explain more than 80% of total variance.
7. Large-cap technology companies generate most market-cap outliers.
8. Single-day price increases provide little information about overall market regime.

---

# Technologies Used

* Python
* Pandas
* NumPy
* SciPy
* Scikit-Learn
* Statsmodels
* Matplotlib
* Seaborn

---

# Installation

```bash
pip install pandas numpy scipy matplotlib seaborn statsmodels scikit-learn
```

---

# Running the Project

```bash
python dsaproject.py
```

---

# Output Files

```text
fig1_decomposition.png
fig2_descriptive.png
fig3_relationships.png
fig4_multivariate.png
```

---

# Academic Concepts Covered

✅ Descriptive Statistics

✅ Probability Theory

✅ Bayes Theorem

✅ Confidence Intervals

✅ Hypothesis Testing

✅ T-Test

✅ ANOVA

✅ Chi-Square Test

✅ Linear Algebra

✅ Covariance & Correlation

✅ Eigenvalues & Eigenvectors

✅ Principal Component Analysis

✅ Time Series Decomposition

✅ Data Visualization

---

# Author

Rakesh V

Graduate Engineer – ADAS & Vehicle Applications

Data Science • Statistics • Machine Learning • Automotive Systems
