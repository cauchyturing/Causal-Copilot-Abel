"""Tests for algorithm adapters — verify ABC conformance and adapter pattern."""

import pytest

from causal_copilot.algorithms.adapters import (
    STABLE_ALGORITHMS,
    CALMAdapter,
    CDNODAdapter,
    CORLAdapter,
    DirectLiNGAMAdapter,
    DYNOTEARSAdapter,
    FCIAdapter,
    GESAdapter,
    GOLEMAdapter,
    GrangerCausalityAdapter,
    GRaSPAdapter,
    HybridAdapter,
    ICALiNGAMAdapter,
    NOTEARSLinearAdapter,
    NOTEARSNonlinearAdapter,
    PCAdapter,
    PCMCIAdapter,
    PCParallelAdapter,
    VARLiNGAMAdapter,
    XGESAdapter,
)
from causal_copilot.core.base import CausalDiscoveryBase

_ALL_ADAPTERS = [
    PCAdapter,
    GESAdapter,
    NOTEARSLinearAdapter,
    DirectLiNGAMAdapter,
    PCMCIAdapter,
    ICALiNGAMAdapter,
    GrangerCausalityAdapter,
    FCIAdapter,
    GRaSPAdapter,
    CDNODAdapter,
    VARLiNGAMAdapter,
    CALMAdapter,
    GOLEMAdapter,
    NOTEARSNonlinearAdapter,
    CORLAdapter,
    PCParallelAdapter,
    XGESAdapter,
    DYNOTEARSAdapter,
    HybridAdapter,
]


class TestAdapterConformance:
    """Verify all adapters conform to CausalDiscoveryBase ABC."""

    @pytest.mark.parametrize("cls", _ALL_ADAPTERS)
    def test_is_subclass(self, cls):
        assert issubclass(cls, CausalDiscoveryBase)

    @pytest.mark.parametrize("cls", _ALL_ADAPTERS)
    def test_can_instantiate(self, cls):
        adapter = cls()
        assert isinstance(adapter, CausalDiscoveryBase)
        assert isinstance(adapter.name, str) and len(adapter.name) > 0

    @pytest.mark.parametrize("cls", _ALL_ADAPTERS)
    def test_default_params(self, cls):
        adapter = cls()
        params = adapter.default_params()
        assert isinstance(params, dict)
        assert len(params) > 0

    @pytest.mark.parametrize("cls", _ALL_ADAPTERS)
    def test_params_override(self, cls):
        adapter = cls(params={"custom_key": 42})
        assert adapter.get_params()["custom_key"] == 42


class TestRegistry:
    def test_registry_has_19_algorithms(self):
        assert len(STABLE_ALGORITHMS) == 19

    def test_registry_names(self):
        expected = {
            "PC",
            "GES",
            "NOTEARSLinear",
            "DirectLiNGAM",
            "PCMCI",
            "ICALiNGAM",
            "GrangerCausality",
            "FCI",
            "GRaSP",
            "CDNOD",
            "VARLiNGAM",
            "CALM",
            "GOLEM",
            "NOTEARSNonlinear",
            "CORL",
            "PCParallel",
            "XGES",
            "DYNOTEARS",
            "Hybrid",
        }
        assert set(STABLE_ALGORITHMS.keys()) == expected

    def test_registry_values_are_classes(self):
        for name, cls in STABLE_ALGORITHMS.items():
            assert issubclass(cls, CausalDiscoveryBase), f"{name} is not a CausalDiscoveryBase subclass"
