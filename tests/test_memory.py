import pytest
from core.memory import get_device, get_dynamic_limits, get_gpu_info
import torch


class TestGetDevice:
    def test_force_cpu(self):
        device = get_device(force_cpu=True)
        assert device == torch.device('cpu')

    def test_default_device(self):
        device = get_device(force_cpu=False)
        # Should return cuda or cpu depending on environment
        assert device.type in ('cpu', 'cuda')


class TestGetDynamicLimits:
    def test_returns_expected_keys(self):
        limits = get_dynamic_limits()
        assert 'phrase_conflict_max' in limits
        assert 'embedding_batch_size' in limits
        assert 'phrase_chunk_size' in limits
        assert 'available_gb' in limits
        assert 'total_gb' in limits
        assert 'memory_tier' in limits

    def test_values_are_positive(self):
        limits = get_dynamic_limits()
        assert limits['phrase_conflict_max'] > 0
        assert limits['embedding_batch_size'] > 0
        assert limits['phrase_chunk_size'] > 0


class TestGetGpuInfo:
    def test_returns_dict(self):
        info = get_gpu_info()
        assert 'available' in info
        assert isinstance(info['available'], bool)
