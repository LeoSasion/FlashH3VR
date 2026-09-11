"""Explicit configuration dispatch; historical FP16 remains reproducible."""
def load_backend(config, root, *, device='cuda'):
    kwargs=dict(project_root=root, weights=config.vae.weights,
                components_lock=config.paths.components_lock, device=device)
    if config.vae.implementation=='aitoolkit_h3_pinned':
        from .aitoolkit_h3_backend import AIToolkitH3Backend
        return AIToolkitH3Backend.from_locked(**kwargs)
    from .int8_convrot_backend import Int8ConvRotH3Backend
    mode={'aitoolkit_h3_int8_dequant_fp16':'dequant_fp16','aitoolkit_h3_int8_mm':'int8_mm'}[config.vae.implementation]
    return Int8ConvRotH3Backend.from_locked(**kwargs,compute_mode=mode)
