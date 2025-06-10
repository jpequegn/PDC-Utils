"""Tests for core functionality"""

import pytest
from PDC_Utils.core import foo


class TestCore:
    """Test the core module"""
    
    def test_foo_function_exists(self):
        """Test that foo function exists and is callable"""
        assert callable(foo)
    
    def test_foo_function_returns_none(self):
        """Test that foo function returns None (since it just passes)"""
        result = foo()
        assert result is None
    
    def test_foo_function_no_parameters(self):
        """Test that foo function can be called without parameters"""
        # Should not raise any exception
        foo()
    
    def test_foo_function_multiple_calls(self):
        """Test that foo function can be called multiple times"""
        # Should not raise any exception
        for _ in range(5):
            result = foo()
            assert result is None