"""Tests for FIT file processing functionality"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PDC_Utils.fit import FitLoader, load_fit_file, mmp_from_fit, pdc_from_fit


class TestFitLoader:
    """Test the FitLoader class"""
    
    def test_fitloader_nonexistent_file(self):
        """Test FitLoader with non-existent file"""
        with pytest.raises(FileNotFoundError):
            FitLoader("nonexistent_file.fit")
    
    def test_fitloader_wrong_extension_warning(self):
        """Test FitLoader warns about wrong file extension"""
        # Create a temporary file with wrong extension
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            with pytest.warns(UserWarning, match="File extension is not .fit"):
                loader = FitLoader(tmp_path)
                assert loader.filepath == Path(tmp_path)
        finally:
            os.unlink(tmp_path)
    
    def test_fitloader_correct_extension(self):
        """Test FitLoader with correct .fit extension"""
        # Create a temporary .fit file
        with tempfile.NamedTemporaryFile(suffix=".fit", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Should not raise warning
            loader = FitLoader(tmp_path)
            assert loader.filepath == Path(tmp_path)
            assert loader.filepath.suffix.lower() == '.fit'
        finally:
            os.unlink(tmp_path)
    
    def test_extract_power_data_interface(self):
        """Test that extract_power_data method exists and has correct interface"""
        # Create a temporary .fit file
        with tempfile.NamedTemporaryFile(suffix=".fit", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            loader = FitLoader(tmp_path)
            
            # Method should exist
            assert hasattr(loader, 'extract_power_data')
            assert callable(loader.extract_power_data)
            
            # Should raise ValueError for empty/invalid FIT file
            with pytest.raises(ValueError):
                loader.extract_power_data()
                
        finally:
            os.unlink(tmp_path)
    
    def test_get_power_duration_data_interface(self):
        """Test that get_power_duration_data method exists and has correct interface"""
        with tempfile.NamedTemporaryFile(suffix=".fit", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            loader = FitLoader(tmp_path)
            
            # Method should exist
            assert hasattr(loader, 'get_power_duration_data')
            assert callable(loader.get_power_duration_data)
            
            # Should raise ValueError for empty/invalid FIT file
            with pytest.raises(ValueError):
                loader.get_power_duration_data()
                
        finally:
            os.unlink(tmp_path)
    
    def test_compute_mmp_curve_interface(self):
        """Test that compute_mmp_curve method exists and has correct interface"""
        with tempfile.NamedTemporaryFile(suffix=".fit", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            loader = FitLoader(tmp_path)
            
            # Method should exist
            assert hasattr(loader, 'compute_mmp_curve')
            assert callable(loader.compute_mmp_curve)
            
            # Should raise ValueError for empty/invalid FIT file
            with pytest.raises(ValueError):
                loader.compute_mmp_curve([1, 10, 60])
                
        finally:
            os.unlink(tmp_path)


class TestFitUtilityFunctions:
    """Test utility functions for FIT file processing"""
    
    def test_load_fit_file(self):
        """Test load_fit_file function"""
        with tempfile.NamedTemporaryFile(suffix=".fit", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            loader = load_fit_file(tmp_path)
            assert isinstance(loader, FitLoader)
            assert loader.filepath == Path(tmp_path)
        finally:
            os.unlink(tmp_path)
    
    @patch('PDC_Utils.fit.FitLoader')
    def test_mmp_from_fit(self, mock_fit_loader):
        """Test mmp_from_fit function"""
        # Mock the FitLoader
        mock_loader_instance = Mock()
        mock_loader_instance.compute_mmp_curve.return_value = (
            np.array([1, 10, 60]),
            np.array([500, 400, 300])
        )
        mock_fit_loader.return_value = mock_loader_instance
        
        # Test the function
        mmp = mmp_from_fit("test.fit")
        
        # Check that FitLoader was called correctly
        mock_fit_loader.assert_called_once_with("test.fit")
        mock_loader_instance.compute_mmp_curve.assert_called_once_with(None)
        
        # Check that MMP object was created
        from PDC_Utils.mmp import MMP
        assert isinstance(mmp, MMP)
        assert np.array_equal(mmp.x, [1, 10, 60])
        assert np.array_equal(mmp.y, [500, 400, 300])
    
    @patch('PDC_Utils.fit.FitLoader')
    def test_pdc_from_fit(self, mock_fit_loader):
        """Test pdc_from_fit function"""
        # Mock the FitLoader
        mock_loader_instance = Mock()
        mock_loader_instance.compute_mmp_curve.return_value = (
            np.array([1, 10, 60]),
            np.array([500, 400, 300])
        )
        mock_fit_loader.return_value = mock_loader_instance
        
        # Test the function
        pdc = pdc_from_fit("test.fit", [1, 10, 60])
        
        # Check that FitLoader was called correctly
        mock_fit_loader.assert_called_once_with("test.fit")
        mock_loader_instance.compute_mmp_curve.assert_called_once_with([1, 10, 60])
        
        # Check that PDC object was created
        from PDC_Utils.pdc import PDC
        assert isinstance(pdc, PDC)
        assert np.array_equal(pdc.x, [1, 10, 60])
        assert np.array_equal(pdc.y, [500, 400, 300])
    
    @patch('PDC_Utils.fit.FitLoader')
    def test_mmp_from_fit_with_durations(self, mock_fit_loader):
        """Test mmp_from_fit function with custom durations"""
        # Mock the FitLoader
        mock_loader_instance = Mock()
        mock_loader_instance.compute_mmp_curve.return_value = (
            np.array([5, 30, 120]),
            np.array([450, 350, 250])
        )
        mock_fit_loader.return_value = mock_loader_instance
        
        # Test the function with custom durations
        custom_durations = [5, 30, 120]
        mmp = mmp_from_fit("test.fit", custom_durations)
        
        # Check that FitLoader was called correctly
        mock_fit_loader.assert_called_once_with("test.fit")
        mock_loader_instance.compute_mmp_curve.assert_called_once_with(custom_durations)
        
        # Check that MMP object was created with correct data
        from PDC_Utils.mmp import MMP
        assert isinstance(mmp, MMP)
        assert np.array_equal(mmp.x, [5, 30, 120])
        assert np.array_equal(mmp.y, [450, 350, 250])