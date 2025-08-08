#!/usr/bin/env python3
"""
Simple example testing all adjustment methods in PyRegAdj.
"""

import sys
import os
# Add parent directory to path to import 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pyregadj import RegAdjustRCT

# Generate dummy data
np.random.seed(42)
n = 500

# Create covariates
age = np.random.normal(35, 10, n)
income = np.random.normal(50000, 20000, n)
education = np.random.normal(14, 3, n)

# Generate treatment assignment (50% treated)
treatment = np.random.binomial(1, 0.5, n)

# Generate outcome with treatment effect
baseline = 100 + 0.5 * age + 0.001 * income + 2 * education + np.random.normal(0, 10, n)
outcome = baseline + 15 * treatment  # True treatment effect = 15

# Create DataFrame
data = pd.DataFrame({
    'y': outcome,
    'd': treatment,
    'age': age,
    'income': income,
    'education': education
})

print(f"Data shape: {data.shape}")
print(f"Treatment group: {data['d'].sum()}")
print(f"Control group: {len(data) - data['d'].sum()}")
print(f"True treatment effect: 15.0")

# Initialize calculator
te = RegAdjustRCT()

# Test all adjustment methods
print("\n" + "="*60)
print("TESTING ALL ADJUSTMENT METHODS")
print("="*60)

# 1. Difference-in-means (no adjustment)
result = te.calculate(data, outcome='y', treatment='d', adjustment='none')
print(f"\n1. Difference-in-means:")
print(f"   Treatment Effect: {result['treatment_effect']:.2f}")
print(f"   Standard Error: {result['standard_error']:.2f}")
print(f"   P-value: {result['p_value']:.4f}")

# 2. Pooled regression adjustment
result = te.calculate(data, outcome='y', treatment='d', adjustment='pooled', 
                     covariates=['age', 'income', 'education'])
print(f"\n2. Pooled regression adjustment:")
print(f"   Treatment Effect: {result['treatment_effect']:.2f}")
print(f"   Standard Error: {result['standard_error']:.2f}")
print(f"   P-value: {result['p_value']:.4f}")

# 3. Groupwise regression adjustment
result = te.calculate(data, outcome='y', treatment='d', adjustment='groupwise', 
                     covariates=['age', 'income', 'education'])
print(f"\n3. Groupwise regression adjustment:")
print(f"   Treatment Effect: {result['treatment_effect']:.2f}")
print(f"   Standard Error: {result['standard_error']:.2f}")
print(f"   P-value: {result['p_value']:.4f}")

# 4. Centered covariates (pooled)
result = te.calculate(data, outcome='y', treatment='d', adjustment='pooled', 
                     covariates=['age', 'income', 'education'], centered=True)
print(f"\n4. Pooled regression (centered covariates):")
print(f"   Treatment Effect: {result['treatment_effect']:.2f}")
print(f"   Standard Error: {result['standard_error']:.2f}")
print(f"   P-value: {result['p_value']:.4f}")

# 5. Centered covariates (groupwise)
result = te.calculate(data, outcome='y', treatment='d', adjustment='groupwise', 
                     covariates=['age', 'income', 'education'], centered=True)
print(f"\n5. Groupwise regression (centered covariates):")
print(f"   Treatment Effect: {result['treatment_effect']:.2f}")
print(f"   Standard Error: {result['standard_error']:.2f}")
print(f"   P-value: {result['p_value']:.4f}")

# Test all ML methods
print("\n" + "="*60)
print("MACHINE LEARNING METHODS")
print("="*60)

ml_methods = ['rf', 'gbm', 'lasso', 'ridge', 'elastic-net']

# Without cross-fitting
print("\nWithout cross-fitting:")
for method in ml_methods:
    result = te.calculate(data, outcome='y', treatment='d', adjustment='ml', 
                         ml_method=method, covariates=['age', 'income', 'education'])
    print(f"\n{method.upper()}:")
    print(f"   Treatment Effect: {result['treatment_effect']:.2f}")
    print(f"   Standard Error: {result['standard_error']:.2f}")
    print(f"   P-value: {result['p_value']:.4f}")

# With cross-fitting
print("\nWith cross-fitting:")
for method in ml_methods:
    result = te.calculate(data, outcome='y', treatment='d', adjustment='ml', 
                         ml_method=method, covariates=['age', 'income', 'education'],
                         cross_fit=True, n_folds=5)
    print(f"\n{method.upper()} (cross-fit):")
    print(f"   Treatment Effect: {result['treatment_effect']:.2f}")
    print(f"   Standard Error: {result['standard_error']:.2f}")
    print(f"   P-value: {result['p_value']:.4f}")

print("\n" + "="*60)
print("ALL TESTS COMPLETED!")
print("="*60) 