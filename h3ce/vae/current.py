"""Current research entry point, selected by the user on 2026-09-10.

Historical scripts retain their explicit locked FP16 configuration. New head
training/inference uses this entry; W8A8 is an explicit frozen experiment.
"""
from pathlib import Path
from h3ce.config import load_config
from .factory import load_backend
from .bridge import H3VAEBridge


def load_current_bridge(project_root, *, device='cuda', int8_gemm=False):
    root=Path(project_root).resolve()
    filename='project.int8.inference.yaml' if int8_gemm else 'project.int8.yaml'
    return H3VAEBridge(load_backend(load_config(root/'configs'/filename),root,device=device))
