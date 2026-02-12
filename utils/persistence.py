import pickle
import json
import os
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

PROJECTS_DIR = os.path.join(os.path.expanduser('~'), '.intent_analyzer_projects')


def _ensure_dir():
    os.makedirs(PROJECTS_DIR, exist_ok=True)


def save_project(name, data_df, analysis_results, model_name, model_type,
                 baseline_results=None, search_history=None, generated_phrases=None):
    """Persist the current session to a local file.  Nothing leaves the machine."""
    _ensure_dir()
    filepath = os.path.join(PROJECTS_DIR, f"{name}.pkl")

    project = {
        'name': name,
        'data': data_df,
        'analysis_results': _serialise_results(analysis_results),
        'model_name': model_name,
        'model_type': model_type,
        'baseline_results': baseline_results,
        'search_history': search_history or [],
        'generated_phrases': generated_phrases or {},
        'saved_at': pd.Timestamp.now().isoformat(),
    }

    with open(filepath, 'wb') as f:
        pickle.dump(project, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("Project saved: %s", filepath)
    return filepath


def load_project(name):
    filepath = os.path.join(PROJECTS_DIR, f"{name}.pkl")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Project file not found: {filepath}")

    with open(filepath, 'rb') as f:
        project = pickle.load(f)

    logger.info("Project loaded: %s", filepath)
    return project


def list_projects():
    _ensure_dir()
    projects = []
    for f in sorted(os.listdir(PROJECTS_DIR)):
        if f.endswith('.pkl'):
            path = os.path.join(PROJECTS_DIR, f)
            stat = os.stat(path)
            projects.append({
                'name': f.replace('.pkl', ''),
                'size_mb': round(stat.st_size / (1024 * 1024), 2),
                'modified': pd.Timestamp.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M'),
            })
    return projects


def delete_project(name):
    filepath = os.path.join(PROJECTS_DIR, f"{name}.pkl")
    if os.path.exists(filepath):
        os.remove(filepath)
        logger.info("Project deleted: %s", filepath)


def _serialise_results(results):
    """Make analysis_results pickle-safe (numpy arrays -> lists)."""
    if results is None:
        return None
    safe = {}
    for k, v in results.items():
        if isinstance(v, np.ndarray):
            safe[k] = v.tolist()
        else:
            safe[k] = v
    return safe
