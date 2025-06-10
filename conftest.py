"""Pytest configuration and fixtures for PDC-Utils tests"""

import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def sample_power_data():
    """Fixture providing sample power duration data"""
    durations = np.array([1, 5, 10, 30, 60, 120, 300, 600, 1200, 1800, 3600])
    powers = np.array([700, 650, 600, 500, 400, 350, 300, 280, 260, 250, 240])
    return durations, powers


@pytest.fixture
def real_mmp_data():
    """Fixture providing real MMP data from CSV file"""
    df = pd.read_csv('data/mmpcurve.csv')
    return df['Secs'].values, df['Watts'].values


@pytest.fixture
def power_curve_params():
    """Fixture providing typical power curve parameters"""
    return {
        'frc': 5000,
        'ftp': 250,
        'tte': 2000,
        'tau': 15,
        'tau2': 5000,
        'a': 10
    }