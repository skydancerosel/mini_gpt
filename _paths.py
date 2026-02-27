"""Path setup for mini_gpt project.

Import this module at the top of any script to configure Python paths
and backward-compatible module aliases for the reorganized directory structure.

Usage — add as the FIRST line of any script that imports local modules:

    Depth 1 (training/*.py):
        import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); import _paths  # noqa

    Depth 2 (analysis/**/*.py, experiments/**/*.py):
        import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); import _paths  # noqa
"""
import sys
import os
import importlib

ROOT = os.path.dirname(os.path.abspath(__file__))

# Add all source directories to Python path
_dirs = [
    'training',
    os.path.join('analysis', 'backbone'),
    os.path.join('analysis', 'basin'),
    os.path.join('analysis', 'switching'),
    os.path.join('analysis', 'fisher'),
    os.path.join('analysis', 'beta_sweep'),
    os.path.join('experiments', 'sgd_controls'),
    'figures',
]

for _d in _dirs:
    _p = os.path.join(ROOT, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Module aliases: old name -> new name (for renamed files)
_ALIASES = {
    'attractor_analysis': 'B1_basin_test',
    'basin_geometry': 'B6_basin_depth',
    'estimate_eval_noise': 'eval_noise',
    'directional_probing': 'switching_alignment',
    'backbone_fisher_analysis': 'rayleigh_quotients',
    'trajectory_pca_uncentered': 'trajectory_pca',
    'backbone_update_analysis': 'update_alignment',
    'backbone_gradient_analysis': 'residual_decomposition',
    'beta2_analysis': 'beta_summary',
}


class _AliasImporter:
    """Redirect imports of old module names to their renamed counterparts."""

    def find_module(self, fullname, path=None):
        return self if fullname in _ALIASES else None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]
        actual = importlib.import_module(_ALIASES[fullname])
        sys.modules[fullname] = actual
        return actual


sys.meta_path.insert(0, _AliasImporter())
