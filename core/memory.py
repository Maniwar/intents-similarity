import torch
import psutil
import logging

logger = logging.getLogger(__name__)


def get_device(force_cpu=False):
    if force_cpu:
        return torch.device('cpu')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_gpu_info():
    if torch.cuda.is_available():
        return {
            'available': True,
            'name': torch.cuda.get_device_name(0),
            'memory_total': torch.cuda.get_device_properties(0).total_mem / (1024 ** 3),
        }
    return {'available': False, 'name': None, 'memory_total': 0}


def get_dynamic_limits():
    try:
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024 ** 3)
        total_gb = memory.total / (1024 ** 3)

        if available_gb < 4:
            tier = 'Low'
            phrase_conflict_max = 5_000
            embedding_batch_size = 16
            phrase_chunk_size = 128
        elif available_gb < 8:
            tier = 'Medium'
            phrase_conflict_max = 25_000
            embedding_batch_size = 32
            phrase_chunk_size = 256
        elif available_gb < 16:
            tier = 'Good'
            phrase_conflict_max = 100_000
            embedding_batch_size = 64
            phrase_chunk_size = 512
        else:
            tier = 'High'
            phrase_conflict_max = 500_000
            embedding_batch_size = 128
            phrase_chunk_size = 1024

        return {
            'phrase_conflict_max': phrase_conflict_max,
            'embedding_batch_size': embedding_batch_size,
            'phrase_chunk_size': phrase_chunk_size,
            'available_gb': round(available_gb, 2),
            'total_gb': round(total_gb, 2),
            'memory_tier': tier,
        }
    except Exception:
        logger.exception("Failed to read system memory, using conservative defaults")
        return {
            'phrase_conflict_max': 10_000,
            'embedding_batch_size': 32,
            'phrase_chunk_size': 256,
            'available_gb': 0,
            'total_gb': 0,
            'memory_tier': 'Unknown',
        }
