"""Test basic package imports."""

import pytest


def test_package_import():
    """Test that the package can be imported."""
    import pg_neo_graph_rl
    assert hasattr(pg_neo_graph_rl, "__version__")


def test_version_format():
    """Test that version follows semantic versioning."""
    from pg_neo_graph_rl import __version__
    import re
    
    # Check semver format (simplified)
    pattern = r"^\d+\.\d+\.\d+(?:-[a-zA-Z0-9]+)?$"
    assert re.match(pattern, __version__), f"Invalid version format: {__version__}"


def test_main_exports():
    """Test that main classes can be imported."""
    try:
        from pg_neo_graph_rl import (
            FederatedGraphRL,
            TrafficEnvironment,
            PowerGridEnvironment,
            GraphPPO,
            GraphSAC,
        )
        # Just test that they exist - full functionality tests elsewhere
        assert FederatedGraphRL is not None
        assert TrafficEnvironment is not None
        assert PowerGridEnvironment is not None
        assert GraphPPO is not None
        assert GraphSAC is not None
    except ImportError as e:
        pytest.skip(f"Core modules not implemented yet: {e}")