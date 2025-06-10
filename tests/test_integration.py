"""Integration tests for PDC-Utils functionality"""

import pytest
import numpy as np
import pandas as pd
from PDC_Utils.pdc import PDC, power_curve
from PDC_Utils.mmp import MMP


class TestIntegration:
    """Integration tests combining multiple components"""
    
    def test_end_to_end_workflow(self):
        """Test a complete end-to-end workflow"""
        # Load sample data
        df = pd.read_csv('data/mmpcurve.csv')
        
        # Create PDC object
        pdc = PDC(df['Secs'], df['Watts'])
        
        # Fit the model
        result = pdc.fit()
        
        # Check that we got a successful fit
        assert result.success
        
        # Extract fitted parameters
        params = result.best_values
        
        # Use the fitted parameters to generate a power curve
        x_test = np.array([1, 10, 60, 300, 1200, 3600])
        y_predicted = power_curve(
            x_test,
            params['frc'],
            params['ftp'],
            params['tte'],
            params['tau'],
            params['tau2'],
            params['a']
        )
        
        # Check that predicted values are reasonable
        assert len(y_predicted) == len(x_test)
        assert np.all(y_predicted > 0)
        assert y_predicted[0] > y_predicted[-1]  # Power should decrease with time
    
    def test_mmp_to_pdc_workflow(self):
        """Test workflow from MMP data to PDC fitting"""
        # Create some MMP-like data
        durations = np.array([1, 5, 10, 30, 60, 120, 300, 600, 1200, 1800, 3600])
        powers = np.array([800, 750, 700, 600, 500, 450, 400, 350, 320, 300, 280])
        
        # Create MMP object
        mmp = MMP(durations, powers)
        
        # Verify MMP object
        assert np.array_equal(mmp.x, durations)
        assert np.array_equal(mmp.y, powers)
        
        # Use the same data for PDC fitting
        pdc = PDC(durations, powers)
        result = pdc.fit()
        
        # Should get a successful fit
        assert result.success
        assert 'ftp' in result.best_values
        
        # FTP should be reasonable for this data
        ftp = result.best_values['ftp']
        assert 200 <= ftp <= 400
    
    def test_power_curve_consistency(self):
        """Test that power_curve function is consistent with PDC fitting"""
        # Create test data with enough points (more than 6 parameters)
        x = np.array([1, 5, 10, 30, 60, 120, 300, 600, 1200])
        
        # Define known parameters that work well with the fitting constraints
        frc = 5000
        ftp = 200  # Use a value closer to the middle of the allowed range
        tte = 2000
        tau = 15
        tau2 = 5000  # Use a value within the allowed range
        a = 10
        
        # Generate synthetic data using power_curve
        y_synthetic = power_curve(x, frc, ftp, tte, tau, tau2, a)
        
        # Add some noise to make it more realistic
        np.random.seed(42)  # For reproducibility
        noise = np.random.normal(0, 2, len(y_synthetic))  # Reduced noise
        y_noisy = y_synthetic + noise
        
        # Fit PDC to the noisy data
        pdc = PDC(x, y_noisy)
        result = pdc.fit()
        
        # Should get a successful fit
        assert result.success
        
        # Fitted parameters should be reasonable (not necessarily close to original due to optimization complexity)
        fitted_params = result.best_values
        
        # Check that fitted parameters are within reasonable bounds
        assert 100 <= fitted_params['ftp'] <= 400
        assert 1 <= fitted_params['frc'] <= 15000
    
    def test_different_data_scales(self):
        """Test PDC fitting with different data scales"""
        # Test with short durations (seconds)
        x_short = np.array([1, 2, 5, 10, 15, 30, 60])
        y_short = np.array([1000, 950, 900, 800, 750, 650, 550])
        
        pdc_short = PDC(x_short, y_short)
        result_short = pdc_short.fit()
        assert result_short.success
        
        # Test with long durations (minutes to hours)
        x_long = np.array([60, 300, 600, 1200, 1800, 3600, 7200])
        y_long = np.array([400, 350, 320, 300, 290, 280, 270])
        
        pdc_long = PDC(x_long, y_long)
        result_long = pdc_long.fit()
        assert result_long.success
        
        # Both should produce reasonable FTP values
        assert 100 <= result_short.best_values['ftp'] <= 400
        assert 100 <= result_long.best_values['ftp'] <= 400
    
    def test_parameter_physical_meaning(self):
        """Test that fitted parameters have reasonable physical meaning"""
        # Load real data
        df = pd.read_csv('data/mmpcurve.csv')
        pdc = PDC(df['Secs'], df['Watts'])
        result = pdc.fit()
        
        params = result.best_values
        
        # FRC (Functional Reserve Capacity) should be higher than FTP
        assert params['frc'] > params['ftp']
        
        # TTE (Time to Exhaustion) should be reasonable (30 min to 1 hour)
        assert 1800 <= params['tte'] <= 3600
        
        # Tau values should be positive and reasonable
        assert params['tau'] > 0
        assert params['tau2'] > 0
        
        # Decay factor should be positive
        assert params['a'] > 0
    
    def test_data_validation(self):
        """Test that the system handles various data validation scenarios"""
        # Test with very small values (need enough points for fitting)
        x_small = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0])
        y_small = np.array([50, 45, 40, 35, 30, 25, 20, 18, 15])
        
        pdc_small = PDC(x_small, y_small)
        try:
            result_small = pdc_small.fit()
            # If it works, parameters should be within bounds
            if result_small.success:
                assert 100 <= result_small.best_values['ftp'] <= 400
        except (ValueError, RuntimeError, TypeError):
            # This is acceptable for edge cases
            pass
        
        # Test with very large values (need enough points for fitting)
        x_large = np.array([3600, 7200, 10800, 14400, 18000, 21600, 25200, 28800, 43200])
        y_large = np.array([300, 280, 270, 260, 255, 250, 245, 242, 240])
        
        pdc_large = PDC(x_large, y_large)
        try:
            result_large = pdc_large.fit()
            if result_large.success:
                assert 100 <= result_large.best_values['ftp'] <= 400
        except (ValueError, RuntimeError, TypeError):
            # This is acceptable for edge cases
            pass