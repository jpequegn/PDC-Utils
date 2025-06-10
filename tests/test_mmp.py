"""Tests for MMP (Mean Max Power) functionality"""

import pytest
import numpy as np
from PDC_Utils.mmp import MMP


class TestMMP:
    """Test the MMP class"""
    
    def setup_method(self):
        """Set up test data"""
        # Create sample time and power data
        self.x = np.array([1, 5, 10, 30, 60, 120, 300, 600, 1200, 1800, 3600])
        self.y = np.array([700, 650, 600, 500, 400, 350, 300, 280, 260, 250, 240])
    
    def test_mmp_initialization(self):
        """Test MMP class initialization"""
        mmp = MMP(self.x, self.y)
        
        assert np.array_equal(mmp.x, self.x)
        assert np.array_equal(mmp.y, self.y)
    
    def test_mmp_initialization_with_lists(self):
        """Test MMP initialization with Python lists"""
        x_list = [1, 10, 60, 300, 1200]
        y_list = [500, 400, 300, 250, 200]
        
        mmp = MMP(x_list, y_list)
        
        assert np.array_equal(mmp.x, x_list)
        assert np.array_equal(mmp.y, y_list)
    
    def test_mmp_fit_method_exists(self):
        """Test that MMP has a fit method"""
        mmp = MMP(self.x, self.y)
        
        # The fit method currently just passes, but it should exist
        assert hasattr(mmp, 'fit')
        assert callable(mmp.fit)
        
        # Should not raise an error when called
        result = mmp.fit()
        # Currently returns None since it's just a pass statement
        assert result is None
    
    def test_mmp_empty_data(self):
        """Test MMP behavior with empty data"""
        mmp = MMP([], [])
        
        # Should be able to create the object
        assert len(mmp.x) == 0
        assert len(mmp.y) == 0
        
        # fit() should still work (currently just passes)
        result = mmp.fit()
        assert result is None
    
    def test_mmp_single_point(self):
        """Test MMP with single data point"""
        mmp = MMP([60], [300])
        
        assert len(mmp.x) == 1
        assert len(mmp.y) == 1
        assert mmp.x[0] == 60
        assert mmp.y[0] == 300
        
        # fit() should work
        result = mmp.fit()
        assert result is None
    
    def test_mmp_data_types(self):
        """Test MMP with different data types"""
        # Test with integers
        mmp_int = MMP([1, 2, 3], [100, 200, 300])
        assert mmp_int.x[0] == 1
        assert mmp_int.y[0] == 100
        
        # Test with floats
        mmp_float = MMP([1.5, 2.5, 3.5], [100.5, 200.5, 300.5])
        assert mmp_float.x[0] == 1.5
        assert mmp_float.y[0] == 100.5
        
        # Test with numpy arrays
        x_np = np.array([1, 2, 3])
        y_np = np.array([100, 200, 300])
        mmp_np = MMP(x_np, y_np)
        assert np.array_equal(mmp_np.x, x_np)
        assert np.array_equal(mmp_np.y, y_np)
    
    def test_mmp_mismatched_data_length(self):
        """Test MMP behavior with mismatched x and y data lengths"""
        x = [1, 10, 60]
        y = [500, 400]  # One less value
        
        # Should be able to create the object
        mmp = MMP(x, y)
        
        # The arrays will have different lengths
        assert len(mmp.x) == 3
        assert len(mmp.y) == 2
        
        # fit() should still work (currently just passes)
        result = mmp.fit()
        assert result is None
    
    def test_mmp_negative_values(self):
        """Test MMP behavior with negative values"""
        # Test with negative time values
        x_neg = [-10, 10, 60]
        y_pos = [500, 400, 300]
        
        mmp = MMP(x_neg, y_pos)
        assert mmp.x[0] == -10
        
        # Test with negative power values
        x_pos = [10, 60, 300]
        y_neg = [-100, 200, 150]
        
        mmp = MMP(x_pos, y_neg)
        assert mmp.y[0] == -100
        
        # Both should work for initialization
        result = mmp.fit()
        assert result is None
    
    def test_mmp_large_datasets(self):
        """Test MMP with larger datasets"""
        # Create a larger dataset
        x_large = np.arange(1, 1001)  # 1 to 1000 seconds
        y_large = 1000 / np.sqrt(x_large)  # Realistic power decay
        
        mmp = MMP(x_large, y_large)
        
        assert len(mmp.x) == 1000
        assert len(mmp.y) == 1000
        
        # fit() should work
        result = mmp.fit()
        assert result is None
    
    def test_mmp_zero_values(self):
        """Test MMP behavior with zero values"""
        # Test with zero time (edge case)
        x_zero = [0, 10, 60]
        y_pos = [500, 400, 300]
        
        mmp = MMP(x_zero, y_pos)
        assert mmp.x[0] == 0
        
        # Test with zero power
        x_pos = [10, 60, 300]
        y_zero = [0, 200, 150]
        
        mmp = MMP(x_pos, y_zero)
        assert mmp.y[0] == 0
        
        # Both should work
        result = mmp.fit()
        assert result is None