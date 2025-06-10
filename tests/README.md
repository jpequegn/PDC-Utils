# PDC-Utils Test Suite

This directory contains comprehensive unit tests for the PDC-Utils library, which provides utilities for analyzing Power Duration Curves (PDC) and Mean Maximal Power (MMP) data from cycling and running activities.

## Test Coverage

The test suite achieves **63% overall code coverage** and includes tests for all major components:

### Core Modules Tested

1. **`test_core.py`** - Tests for the core module (100% coverage)
   - Basic functionality tests for the `foo()` function

2. **`test_pdc.py`** - Tests for Power Duration Curve functionality (100% coverage)
   - `power_curve()` function tests with various parameter combinations
   - `PDC` class initialization and fitting
   - Parameter bounds validation
   - Edge cases (empty data, single points, negative values)
   - Real data fitting with sample CSV data

3. **`test_mmp.py`** - Tests for Mean Max Power functionality (100% coverage)
   - `MMP` class initialization
   - Data type handling (lists, arrays, different numeric types)
   - Edge cases and validation

4. **`test_fit.py`** - Tests for FIT file processing functionality (49% coverage)
   - `FitLoader` class interface testing
   - File validation (extension checking, file existence)
   - Utility functions (`load_fit_file`, `mmp_from_fit`, `pdc_from_fit`)

5. **`test_integration.py`** - Integration tests combining multiple components
   - End-to-end workflow testing
   - Data validation across different scales
   - Parameter physical meaning validation
   - Cross-component consistency

## Key Test Features

### Comprehensive Edge Case Testing
- Empty datasets
- Single data points
- Mismatched array lengths
- Negative values
- Very small and very large values
- Insufficient data for fitting (mathematical constraints)

### Real Data Validation
- Uses actual sample data from `data/mmpcurve.csv`
- Validates fitted parameters are within physically meaningful ranges
- Tests reproducibility of fitting results

### Mathematical Constraints
- Ensures fitting algorithms respect the constraint that the number of data points must exceed the number of parameters (6)
- Validates parameter bounds match the physical constraints in the PDC model

### Integration Testing
- Tests complete workflows from data loading to parameter fitting
- Validates consistency between different components
- Tests parameter physical meaning and relationships

## Running the Tests

### Run all tests:
```bash
pytest tests/
```

### Run with coverage report:
```bash
pytest tests/ --cov=PDC_Utils --cov-report=term-missing
```

### Run specific test files:
```bash
pytest tests/test_pdc.py -v
pytest tests/test_integration.py -v
```

### Run with verbose output:
```bash
pytest tests/ -v
```

## Test Configuration

- **pytest.ini**: Configuration file with test discovery settings
- **conftest.py**: Shared fixtures for test data
- Tests are configured to suppress warnings for cleaner output
- Uses strict marker checking to prevent typos in test markers

## Coverage Details

| Module | Coverage | Notes |
|--------|----------|-------|
| `core.py` | 100% | Simple module, fully tested |
| `pdc.py` | 100% | Core functionality, comprehensive tests |
| `mmp.py` | 100% | Interface testing, all methods covered |
| `fit.py` | 49% | Interface testing only, FIT file processing not fully mocked |

The lower coverage in `fit.py` is due to the complexity of mocking the `fitdecode` library for FIT file processing. The tests focus on interface validation and error handling rather than full integration testing with actual FIT files.

## Test Data

The tests use:
- **Sample CSV data**: `data/mmpcurve.csv` with realistic power curve data
- **Synthetic data**: Generated using the `power_curve` function for controlled testing
- **Fixtures**: Predefined test data sets in `conftest.py`

## Mathematical Background

The tests validate the PDC model which uses 6 parameters:
- **FRC**: Functional Reserve Capacity (1-15000W)
- **FTP**: Functional Threshold Power (100-400W)  
- **TTE**: Time to Exhaustion (1800-3600s)
- **tau**: Short-term time constant (10-25s)
- **tau2**: Long-term time constant (10-25s)
- **a**: Decay factor (1-200)

The power curve equation: `P = FRC/t * (1 - exp(-t/tau)) + FTP * (1 - exp(-t/tau2)) - max(0, a * log(t/TTE))`

## Future Improvements

1. **Enhanced FIT file testing**: Add more comprehensive mocking for `fitdecode` library
2. **Performance testing**: Add tests for large datasets and performance benchmarks
3. **Visualization testing**: Add tests for any plotting functionality
4. **Error message validation**: More specific error message testing
5. **Property-based testing**: Use hypothesis for generating test cases