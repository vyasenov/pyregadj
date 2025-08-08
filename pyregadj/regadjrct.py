"""
Treatment effects calculation for randomized experiments.

This module provides methods for estimating average treatment effects (ATE) in 
randomized controlled trials (RCTs) using various adjustment methods including regression
adjustment and machine learning approaches.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Union, Optional, List, Any
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.model_selection import KFold

from .constants import (
    VALID_ADJUSTMENTS, VALID_ML_METHODS, MIN_FOLDS, MAX_FOLDS, DEFAULT_FOLDS,
    DEFAULT_CROSS_FIT, DEFAULT_CENTERED, DEFAULT_ADJUSTMENT, RANDOM_SEED,
    ML_MODEL_CONFIGS, CONFIDENCE_LEVEL
)

class RegAdjustRCT:
    """
    Treatment effects calculator for randomized controlled trials.
    
    This class implements various methods for estimating average treatment effects (ATE)
    in randomized experiments, including unadjusted difference-in-means, regression
    adjustment, and machine learning-based approaches.
    
    Methods include:
    - Difference-in-means (unadjusted)
    - Pooled regression adjustment
    - Groupwise regression adjustment  
    - Machine learning adjustment with and without cross-fitting
    """
    
    def __init__(self):
        """Initialize the treatment effects calculator."""
        pass
    
    # === VALIDATION METHODS ===
    
    def _validate_data(self, data: pd.DataFrame, outcome: str, treatment: str, 
                      covariates: Optional[List[str]] = None) -> None:
        """
        Validate input data for treatment effects calculation.
        
        Checks data structure, required columns, data types, and treatment variable
        properties. Ensures data is suitable for treatment effects analysis.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data containing outcome and treatment variables
        outcome : str
            Name of the outcome variable (y; must be numeric)
        treatment : str
            Name of the treatment variable (d; must be binary 0/1)
        covariates : List[str], optional
            List of covariate variable names for regression adjustment
            
        Raises
        ------
        ValueError
            If data validation fails (missing columns, wrong types, etc.)
        """
        # Validate data structure
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
        
        if data.empty:
            raise ValueError("Data cannot be empty")
        
        # Validate required columns
        if outcome not in data.columns:
            raise ValueError(f"Outcome variable '{outcome}' not found in data")
        
        if treatment not in data.columns:
            raise ValueError(f"Treatment variable '{treatment}' not found in data")
        
        # Validate data content
        if data[outcome].isnull().any():
            raise ValueError(f"Outcome variable '{outcome}' contains missing values")
        
        if data[treatment].isnull().any():
            raise ValueError(f"Treatment variable '{treatment}' contains missing values")
        
        # Check treatment variable properties
        unique_treatment_values = data[treatment].unique()
        if not set(unique_treatment_values).issubset({0, 1}):
            raise ValueError(f"Treatment variable '{treatment}' must be binary (0/1)")
        
        if len(unique_treatment_values) < 2:
            raise ValueError("Treatment variable must have both 0 and 1 values")
        
        # Check data types
        if not pd.api.types.is_numeric_dtype(data[outcome]):
            raise ValueError(f"Outcome variable '{outcome}' must be numeric")
        
        # Validate covariates if provided
        if covariates is not None:
            for covariate in covariates:
                if covariate not in data.columns:
                    raise ValueError(f"Covariate '{covariate}' not found in data")
                if data[covariate].isnull().any():
                    raise ValueError(f"Covariate '{covariate}' contains missing values")
                if not pd.api.types.is_numeric_dtype(data[covariate]):
                    raise ValueError(f"Covariate '{covariate}' must be numeric")
    
    def _validate_parameters(self, adjustment: str, ml_method: Optional[str] = None,
                           covariates: Optional[List[str]] = None, centered: Optional[bool] = None,
                           cross_fit: Optional[bool] = None, n_folds: Optional[int] = None) -> None:
        """
        Validate parameter combinations for treatment effects calculation.
        
        Ensures parameter combinations are valid and consistent. Validates that
        required parameters are provided for each adjustment method.
        
        Parameters
        ----------
        adjustment : str
            Adjustment method ('none', 'pooled', 'groupwise', 'ml')
        ml_method : str, optional
            Machine learning method (required for adjustment='ml')
        covariates : List[str], optional
            List of covariate variable names (required for non-'none' adjustments)
        centered : bool, optional
            Whether to de-mean covariates (only for regression adjustments)
        cross_fit : bool, optional
            Whether to use cross-fitting for ML adjustment
        n_folds : int, optional
            Number of folds for cross-fitting (2-10, default 5)
            
        Raises
        ------
        ValueError
            If parameter combinations are invalid or inconsistent
        """
        # Validate adjustment method
        if adjustment not in VALID_ADJUSTMENTS:
            raise ValueError(f"adjustment must be one of {VALID_ADJUSTMENTS}, got '{adjustment}'")
        
        # Require covariates for non-'none' adjustments
        if adjustment != 'none':
            if covariates is None or len(covariates) == 0:
                raise ValueError(f"covariates must be provided when adjustment='{adjustment}'. Adjustment methods require at least one covariate.")
        
        # Validate based on adjustment type
        if adjustment == 'ml':
            # ML-specific validations
            if ml_method is None:
                raise ValueError("ml_method must be specified when adjustment='ml'")
            
            if ml_method not in VALID_ML_METHODS:
                raise ValueError(f"ml_method must be one of {VALID_ML_METHODS}, got '{ml_method}'")
            
            # Only validate cross_fit and n_folds if they are explicitly provided
            if cross_fit is not None and not cross_fit and n_folds is not None and n_folds != DEFAULT_FOLDS:
                raise ValueError("n_folds should only be specified when cross_fit=True")
            
            if cross_fit and n_folds is not None and (n_folds < MIN_FOLDS or n_folds > MAX_FOLDS):
                raise ValueError(f"n_folds must be between {MIN_FOLDS} and {MAX_FOLDS}")
            
            if centered is not None and centered != DEFAULT_CENTERED:
                raise ValueError(f"centered parameter is not available for adjustment='ml', got adjustment='{adjustment}'")
            
        else:
            # Regression-specific validations
            if ml_method is not None:
                raise ValueError(f"ml_method should only be specified when adjustment='ml', got adjustment='{adjustment}'")
            
            # Only validate cross_fit and n_folds if they are explicitly provided
            # For non-ML adjustments, these should only be validated if explicitly set to non-default values
            if cross_fit is not None and cross_fit != DEFAULT_CROSS_FIT:
                raise ValueError(f"cross_fit parameter is only available for adjustment='ml', got adjustment='{adjustment}'")
            
            if n_folds is not None and n_folds != DEFAULT_FOLDS:
                raise ValueError(f"n_folds parameter is only available for adjustment='ml', got adjustment='{adjustment}'")
    
    # === PUBLIC INTERFACE ===
    
    def calculate(self, data: pd.DataFrame, outcome: str, treatment: str, 
                 adjustment: str = DEFAULT_ADJUSTMENT, ml_method: Optional[str] = None, 
                 covariates: Optional[List[str]] = None, centered: bool = DEFAULT_CENTERED,
                 cross_fit: bool = DEFAULT_CROSS_FIT, n_folds: int = DEFAULT_FOLDS) -> Dict[str, Union[float, tuple, str]]:
        """
        Calculate treatment effects using various adjustment methods.
        
        Estimates the average treatment effect (ATE) using different approaches
        appropriate for randomized controlled trials (RCTs). All methods provide valid
        inference under random assignment. Prints results to console by default.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data containing outcome and treatment variables
        outcome : str
            Name of the outcome variable (y; must be numeric)
        treatment : str
            Name of the treatment variable (d; must be binary 0/1)
        adjustment : str, optional
            Adjustment method: 
            - 'none' for difference-in-means (default), 
            - 'pooled' for pooled linear regression adjustment, 
            - 'groupwise' for separate linear regressions by treatment group, 
            - 'ml' for machine learning adjustment
        ml_method : str, optional
            Machine learning method: (only used when adjustment='ml')
            - 'rf' for random forest, 
            - 'gbm' for gradient boosting, 
            - 'lasso' for lasso regression, 
            - 'ridge' for ridge regression,  
            - 'elastic-net' for elastic net regression
        covariates : List[str], optional
            List of covariate variable names for adjustment
        centered : bool, optional
            Whether to de-mean covariates (only for regression adjustments)
        cross_fit : bool, optional
            Whether to use cross-fitting for ML adjustment (only available for adjustment='ml'). Defaults to False
        n_folds : int, optional
            Number of folds for cross-fitting (only available for adjustment='ml' and cross_fit=True). Defaults to 5
            
        Returns
        -------
        dict
            Dictionary containing treatment effect estimates and inference:
            - treatment_effect: Estimated ATE
            - standard_error: Standard error of the estimate
            - p_value: P-value for testing H0: ATE = 0
            - confidence_interval: 95% confidence interval
            - treatment_mean: Mean outcome in treatment group
            - control_mean: Mean outcome in control group
            - n_treatment: Number of treated observations
            - n_control: Number of control observations
            - method: Method used for estimation
        """
        # Validate data
        self._validate_data(data, outcome, treatment, covariates)
        
        # Validate parameters (only check explicitly provided ones)
        self._validate_parameters(adjustment, ml_method, covariates, centered, cross_fit, n_folds)
        
        # Route to appropriate calculation method
        if adjustment == 'none':
            result = self._calculate_difference_in_means(data, outcome, treatment)
        elif adjustment == 'pooled':
            result = self._calculate_pooled_adjustment(data, outcome, treatment, covariates, centered)
        elif adjustment == 'groupwise':
            result = self._calculate_groupwise_adjustment(data, outcome, treatment, covariates, centered)
        elif adjustment == 'ml':
            result = self._calculate_ml_adjustment(data, outcome, treatment, ml_method, covariates, cross_fit, n_folds)
        else:
            # This should never happen due to validation, but keeping for safety
            raise ValueError(f"Unknown adjustment method: {adjustment}. Use 'none', 'pooled', 'groupwise', or 'ml'.")
        
        # Print results if requested
        self._print_results(result)
        
        return result
    
    # === UTILITY METHODS ===
    
    def _create_result_dict(self, treatment_effect: float, standard_error: float, 
                           treatment_mean: float, control_mean: float,
                           n_treatment: int, n_control: int, method: str,
                           **kwargs) -> Dict[str, Union[float, tuple, str]]:
        """
        Create standardized result dictionary with inference.
        
        Computes p-values and confidence intervals using t-distribution with
        heteroskedasticity-robust standard errors.
        
        Parameters
        ----------
        treatment_effect : float
            Estimated average treatment effect (ATE)
        standard_error : float
            Standard error of the ATE estimate
        treatment_mean : float
            Mean outcome in treatment group
        control_mean : float
            Mean outcome in control group
        n_treatment : int
            Number of treated observations
        n_control : int
            Number of control observations
        method : str
            Method used for estimation
        **kwargs : additional key-value pairs to include
            
        Returns
        -------
        dict
            Standardized result dictionary with treatment effect estimates and inference
        """
        # Calculate inference
        df = n_treatment + n_control - 2
        
        # T-statistic and p-value
        t_stat = treatment_effect / standard_error
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=df))
        
        # Confidence interval
        alpha = 1 - CONFIDENCE_LEVEL
        t_critical = stats.t.ppf(1 - alpha/2, df=df)
        margin_of_error = t_critical * standard_error
        confidence_interval = (treatment_effect - margin_of_error, treatment_effect + margin_of_error)
        
        # Create base result
        result = {
            'treatment_effect': treatment_effect,
            'standard_error': standard_error,
            'p_value': p_value,
            'confidence_interval': confidence_interval,
            'treatment_mean': treatment_mean,
            'control_mean': control_mean,
            'n_treatment': n_treatment,
            'n_control': n_control,
            'method': method
        }
        
        # Add any additional key-value pairs
        result.update(kwargs)
        
        return result
    
    def _print_results(self, result: Dict[str, Union[float, tuple, str]]) -> None:
        """
        Print formatted treatment effect results.
        
        Parameters
        ----------
        result : dict
            Result dictionary from treatment effects calculation
        """
        print(f"\n{'='*60}")
        print(f"TREATMENT EFFECT RESULTS: {result['method'].upper()}")
        print(f"{'='*60}")
        print(f"Treatment Effect: {result['treatment_effect']:.4f}")
        print(f"Standard Error:   {result['standard_error']:.4f}")
        print(f"P-value:          {result['p_value']:.4f}")
        print(f"95% CI:           ({result['confidence_interval'][0]:.4f}, {result['confidence_interval'][1]:.4f})")
        print(f"Treatment Mean:   {result['treatment_mean']:.4f}")
        print(f"Control Mean:     {result['control_mean']:.4f}")
        print(f"Sample Sizes:     T={result['n_treatment']}, C={result['n_control']}")
        print(f"Adjustment Method:  {result['method'].replace('_', ' ').title()}")
        
        # Print method-specific diagnostics
        if 'r_squared' in result:
            print(f"R-squared:        {result['r_squared']:.4f}")
        if 'adj_r_squared' in result:
            print(f"Adjusted R²:      {result['adj_r_squared']:.4f}")
        if 'r_squared_treatment' in result:
            print(f"R² (Treatment):   {result['r_squared_treatment']:.4f}")
            print(f"R² (Control):     {result['r_squared_control']:.4f}")
        if 'ml_method' in result:
            print(f"ML Method:        {result['ml_method'].upper()}")
        if 'cross_fit' in result and result['cross_fit']:
            print(f"Cross-fitting:     {result['n_folds']} folds")
        print(f"{'='*60}\n")
    
    # === CORE CALCULATION METHODS ===
    
    def _calculate_difference_in_means(self, data: pd.DataFrame, outcome: str, treatment: str) -> Dict[str, Union[float, tuple, str]]:
        """
        Calculate treatment effects using difference-in-means estimator.
        
        Estimates ATE as the difference between treatment and control group means.
        Uses heteroskedasticity-robust standard errors with pooled variance.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data containing outcome and treatment variables
        outcome : str
            Name of the outcome variable (y)
        treatment : str
            Name of the treatment variable (d)
            
        Returns
        -------
        dict
            Dictionary containing treatment effect estimates and inference
        """
        # Extract treatment and control groups
        treatment_group = data[data[treatment] == 1][outcome]
        control_group = data[data[treatment] == 0][outcome]
        
        # Calculate means
        treatment_mean = treatment_group.mean()
        control_mean = control_group.mean()
        
        # Calculate treatment effect (difference-in-means)
        treatment_effect = treatment_mean - control_mean
        
        # Calculate standard error (heteroskedasticity-robust)
        n_treatment = len(treatment_group)
        n_control = len(control_group)
        var_treatment = treatment_group.var()
        var_control = control_group.var()
        pooled_var = ((n_treatment - 1) * var_treatment + (n_control - 1) * var_control) / (n_treatment + n_control - 2)
        standard_error = np.sqrt(pooled_var * (1/n_treatment + 1/n_control))
        
        return self._create_result_dict(
            treatment_effect=treatment_effect,
            standard_error=standard_error,
            treatment_mean=treatment_mean,
            control_mean=control_mean,
            n_treatment=len(treatment_group),
            n_control=len(control_group),
            method='difference_in_means'
        )
    
    def _calculate_pooled_adjustment(self, data: pd.DataFrame, outcome: str, treatment: str, 
                                   covariates: List[str], centered: bool = False) -> Dict[str, Union[float, tuple, str]]:
        """
        Calculate treatment effects using pooled regression adjustment.
        
        Fits a single linear regression of outcome on treatment and covariates.
        The coefficient on treatment estimates the ATE. Uses heteroskedasticity-
        robust standard errors (HC1).
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data containing outcome and treatment variables
        outcome : str
            Name of the outcome variable (y; required)
        treatment : str
            Name of the treatment variable (d; required)
        covariates : List[str]
            List of covariate variable names for regression adjustment (required)
        centered : bool, optional
            Whether to de-mean covariates (default: False)
            
        Returns
        -------
        dict
            Dictionary containing treatment effect estimates and inference
        """
        # Prepare regression data
        y = data[outcome]
        d = data[treatment]
        
        # Create design matrix
        X = pd.DataFrame({'treatment': d})
        
        # Add covariates if provided
        for covariate in covariates:
            X[covariate] = data[covariate]
        
        if centered:
            # De-mean all covariates
            for covariate in covariates:
                X[covariate] = data[covariate] - data[covariate].mean()
        
        # Add constant term
        X = sm.add_constant(X)
        
        # Fit regression with heteroskedasticity-robust standard errors
        model = sm.OLS(y, X)
        results = model.fit(cov_type='HC1')
        
        # Extract treatment effect (coefficient on treatment variable)
        treatment_effect = results.params['treatment']
        standard_error = results.bse['treatment']

        # Calculate group means for comparison
        treatment_group = data[data[treatment] == 1][outcome]
        control_group = data[data[treatment] == 0][outcome]
        
        # Calculate means
        treatment_mean = treatment_group.mean()
        control_mean = control_group.mean()
        
        return self._create_result_dict(
            treatment_effect=treatment_effect,
            standard_error=standard_error,
            treatment_mean=treatment_mean,
            control_mean=control_mean,
            n_treatment=len(treatment_group),
            n_control=len(control_group),
            method='pooled_regression',
            r_squared=results.rsquared,
            adj_r_squared=results.rsquared_adj
        )
    
    def _calculate_groupwise_adjustment(self, data: pd.DataFrame, outcome: str, treatment: str, 
                                      covariates: List[str], centered: bool = False) -> Dict[str, Union[float, tuple, str]]:
        """
        Calculate treatment effects using groupwise regression adjustment.
        
        Fits separate linear regressions for each treatment group, then estimates
        ATE as the difference in average predicted outcomes. This allows covariate
        effects to differ by treatment group.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data containing outcome and treatment variables
        outcome : str
            Name of the outcome variable (y; required)
        treatment : str
            Name of the treatment variable (d; required)
        covariates : List[str]
            List of covariate variable names for regression adjustment (required)
        centered : bool, optional
            Whether to de-mean covariates (default: False)
            
        Returns
        -------
        dict
            Dictionary containing treatment effect estimates and inference
        """
        # Fit regression for treatment group
        treatment_pred, results_treatment = self._fit_single_group_regression(
            data, outcome, treatment, covariates, centered, treatment_value=1
        )
        
        # Fit regression for control group
        control_pred, results_control = self._fit_single_group_regression(
            data, outcome, treatment, covariates, centered, treatment_value=0
        )
        
        # Calculate treatment effect
        treatment_effect = treatment_pred - control_pred
        
        # Calculate standard error using residuals
        treatment_resid = results_treatment.resid
        control_resid = results_control.resid
        
        # Calculate standard error using pooled variance of residuals
        treatment_resid_series = pd.Series(treatment_resid)
        control_resid_series = pd.Series(control_resid)
        n_treatment = len(treatment_resid_series)
        n_control = len(control_resid_series)
        var_treatment = treatment_resid_series.var()
        var_control = control_resid_series.var()
        pooled_var = ((n_treatment - 1) * var_treatment + (n_control - 1) * var_control) / (n_treatment + n_control - 2)
        standard_error = np.sqrt(pooled_var * (1/n_treatment + 1/n_control))
        
        # Get sample sizes
        treatment_data = data[data[treatment] == 1].copy()
        control_data = data[data[treatment] == 0].copy()
        
        return self._create_result_dict(
            treatment_effect=treatment_effect,
            standard_error=standard_error,
            treatment_mean=treatment_pred,
            control_mean=control_pred,
            n_treatment=len(treatment_data),
            n_control=len(control_data),
            method='groupwise_regression',
            r_squared_treatment=results_treatment.rsquared,
            r_squared_control=results_control.rsquared,
            adj_r_squared_treatment=results_treatment.rsquared_adj,
            adj_r_squared_control=results_control.rsquared_adj
        )
    
    def _calculate_ml_adjustment(self, data: pd.DataFrame, outcome: str, treatment: str, 
                                ml_method: str, covariates: List[str], 
                                cross_fit: bool = False, n_folds: int = 5) -> Dict[str, Union[float, tuple, str]]:
        """
        Calculate treatment effects using machine learning adjustment with cross-fitting.
        
        Implements flexible regression adjustment using machine learning methods
        to estimate conditional expectations. Can use cross-fitting to prevent overfitting.
        Variance is estimated using influence functions.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data containing outcome and treatment variables
        outcome : str
            Name of the outcome variable (y; required)
        treatment : str
            Name of the treatment variable (d; required)
        ml_method : str
            Machine learning method: 'rf', 'gbm', 'lasso', 'ridge', 'elastic-net'
        covariates : List[str]
            List of covariate variable names for ML adjustment (required)
        cross_fit : bool, optional
            Whether to use cross-fitting (default: False)
        n_folds : int, optional
            Number of folds for cross-fitting (default: 5)
            
        Returns
        -------
        dict
            Dictionary containing treatment effect estimates and inference
        """
        # Create a copy of data for processing
        data_copy = data.copy()
        
        if cross_fit:
            # Use cross-fitting utility method
            data_copy = self._cross_fit_ml_models(data_copy, outcome, treatment, covariates, ml_method, n_folds)
        else:
            # Use non-cross-fitting utility method
            data_copy = self._fit_ml_models(data_copy, outcome, treatment, covariates, ml_method)
        
        # Calculate influence functions and treatment effects
        prop_treat_0 = (data_copy[treatment] == 0).mean()
        prop_treat_1 = (data_copy[treatment] == 1).mean()
        
        # Influence function for treatment = 0
        data_copy[f'u_{outcome}_0'] = np.where(
            data_copy[treatment] == 0,
            (1/prop_treat_0) * (data_copy[outcome] - data_copy[f'm_{outcome}_0']),
            0
        ) + data_copy[f'm_{outcome}_0']
        
        # Influence function for treatment = 1
        data_copy[f'u_{outcome}_1'] = np.where(
            data_copy[treatment] == 1,
            (1/prop_treat_1) * (data_copy[outcome] - data_copy[f'm_{outcome}_1']),
            0
        ) + data_copy[f'm_{outcome}_1']
        
        # Calculate treatment effect and inference
        treatment_effect = data_copy[f'u_{outcome}_1'].mean() - data_copy[f'u_{outcome}_0'].mean()
        
        # Calculate standard error using influence functions
        n = len(data_copy)
        
        # variance
        diff = data_copy[f'u_{outcome}_1'] - data_copy[f'u_{outcome}_0']
        var_treatment_effect = diff.var() / n
        standard_error = np.sqrt(var_treatment_effect)
        
        return self._create_result_dict(
            treatment_effect=treatment_effect,
            standard_error=standard_error,
            treatment_mean=data_copy[f'u_{outcome}_1'].mean(),
            control_mean=data_copy[f'u_{outcome}_0'].mean(),
            n_treatment=(data_copy[treatment] == 1).sum(),
            n_control=(data_copy[treatment] == 0).sum(),
            method=f'ml_adjustment_{ml_method}',
            ml_method=ml_method,
            cross_fit=cross_fit,
            n_folds=n_folds if cross_fit else None
        )
    
    # === SUPPORTING METHODS ===

    def _fit_single_group_regression(self, data: pd.DataFrame, outcome: str, treatment: str, 
                                    covariates: List[str], centered: bool, treatment_value: int) -> tuple:
        """
        Fit regression for a single treatment group.
        
        Helper method for groupwise adjustment. Fits linear regression on data
        from a specific treatment group and returns predicted mean outcome.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data containing outcome and treatment variables
        outcome : str
            Name of the outcome variable (y; required)
        treatment : str
            Name of the treatment variable (d; required)
        covariates : List[str]
            List of covariate variable names for regression adjustment
        centered : bool
            Whether to de-mean covariates
        treatment_value : int
            Treatment value to filter for (0 or 1)
            
        Returns
        -------
        tuple
            (predicted_value, regression_results)
        """
        # Filter data for the specific treatment group
        group_data = data[data[treatment] == treatment_value].copy()
        
        # Prepare design matrix
        X = group_data[covariates].copy()
        if centered:
            for covariate in covariates:
                X[covariate] = group_data[covariate] - group_data[covariate].mean()
        X = sm.add_constant(X)
        y = group_data[outcome]
        
        # Fit regression
        model = sm.OLS(y, X)
        results = model.fit()
        
        # Get predicted value
        predicted_value = results.fittedvalues.mean()
        
        return predicted_value, results
    
    def _fit_ml_models(self, data: pd.DataFrame, outcome: str, treatment: str, 
                       covariates: List[str], ml_method: str) -> pd.DataFrame:
        """
        Fit machine learning models for each treatment group without cross-fitting.

        Parameters
        ----------
        data : pd.DataFrame
            Input data containing outcome and treatment variables
        outcome : str
            Name of the outcome variable (y; required)
        treatment : str
            Name of the treatment variable (d; required)
        covariates : List[str]
            List of covariate variable names for ML adjustment
        ml_method : str
            Machine learning method: 'rf', 'gbm', 'lasso', 'ridge', 'elastic-net'
            
        Returns
        -------
        pd.DataFrame
            Data with prediction columns added for each treatment group
        """
        # Create a copy of data for processing
        data_copy = data.copy()
        
        # Initialize prediction columns
        data_copy[f'm_{outcome}_0'] = 0.0
        data_copy[f'm_{outcome}_1'] = 0.0
        
        # Fit models for each treatment group on all data
        for treat in [0, 1]:
            # Get data for current treatment group
            treat_mask = data_copy[treatment] == treat
            treat_data = data_copy[treat_mask]
            
            # Prepare features and target
            X_treat = treat_data[covariates]
            y_treat = treat_data[outcome]
            
            # Fit ML model
            model = self._get_ml_model(ml_method)
            
            model.fit(X_treat, y_treat)
            
            # Make predictions on all data
            X_all = data_copy[covariates]
            predictions = model.predict(X_all)
            data_copy[f'm_{outcome}_{treat}'] = predictions
        
        return data_copy
    
    def _get_ml_model(self, ml_method: str):
        """
        Get ML model instance for the specified method.
        
        Parameters
        ----------
        ml_method : str
            Machine learning method: 'rf', 'gbm', 'lasso', 'ridge', 'elastic-net'
            
        Returns
        -------
        sklearn estimator
            Configured ML model instance
        """
        if ml_method not in ML_MODEL_CONFIGS:
            raise ValueError(f"Unknown ML method: {ml_method}")
        
        config = ML_MODEL_CONFIGS[ml_method]
        
        if ml_method == 'rf':
            return RandomForestRegressor(**config)
        elif ml_method == 'gbm':
            return GradientBoostingRegressor(**config)
        elif ml_method == 'lasso':
            return Lasso(**config)
        elif ml_method == 'ridge':
            return Ridge(**config)
        elif ml_method == 'elastic-net':
            return ElasticNet(**config)
        else:
            raise ValueError(f"Unknown ML method: {ml_method}")
    
    def _cross_fit_ml_models(self, data: pd.DataFrame, outcome: str, treatment: str, 
                             covariates: Optional[List[str]] = None, 
                             ml_method: str = 'rf', n_folds: int = 5) -> pd.DataFrame:
        """
        Cross-fit machine learning models for each treatment group.
        
        Implements K-fold cross-fitting using sklearn's KFold splitter to prevent
        overfitting. Fits ML models on training folds and predicts on test folds
        for each treatment group.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data containing outcome and treatment variables
        outcome : str
            Name of the outcome variable (y; required)
        treatment : str
            Name of the treatment variable (d; required)
        covariates : List[str], optional
            List of covariate variable names for ML adjustment
        ml_method : str, optional
            Machine learning method: 'rf', 'gbm', 'lasso', 'ridge', 'elastic-net'
        n_folds : int, optional
            Number of folds for cross-fitting (default: 5)
            
        Returns
        -------
        pd.DataFrame
            DataFrame with prediction columns added for each treatment group
        """
        # Create a copy of data for processing
        data_copy = data.copy()
        
        # Initialize prediction columns
        data_copy[f'm_{outcome}_0'] = 0.0
        data_copy[f'm_{outcome}_1'] = 0.0
        
        # Create cross-validation splitter
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
        
        # Perform cross-fitting for each treatment group
        for treat in [0, 1]:
            # Get data for current treatment group
            treat_mask = data_copy[treatment] == treat
            treat_indices = data_copy[treat_mask].index
            
            if len(treat_indices) == 0:
                continue
            
            # Create cross-validation splits for this treatment group
            for train_idx, test_idx in cv.split(treat_indices):
                # Map back to original indices
                train_indices = treat_indices[train_idx]
                test_indices = treat_indices[test_idx]
                
                # Get training and test data
                train_data = data_copy.loc[train_indices]
                test_data = data_copy.loc[test_indices]
                
                # Prepare features and target
                X_train = train_data[covariates]
                y_train = train_data[outcome]
                X_test = test_data[covariates]
                
                # Fit ML model
                model = self._get_ml_model(ml_method)
                
                model.fit(X_train, y_train)
                
                # Make predictions
                predictions = model.predict(X_test)
                data_copy.loc[test_indices, f'm_{outcome}_{treat}'] = predictions
        
        return data_copy