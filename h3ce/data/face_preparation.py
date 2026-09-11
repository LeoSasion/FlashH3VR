"""Exact CPU preparation for owned face workers and source identity records."""
import hashlib
import threading

import numpy as np


BYTE_PREPARATIONS = ('legacy', 'inplace_rgb', 'direct_bgr')


class FaceBytePreparer:
    """Private per-worker float scratch; every returned byte image owns storage.

    Call only from its owning worker. The source and previously returned byte
    images are never modified, including when the scratch is reused next batch.
    """
    def __init__(self, mode):
        if mode not in BYTE_PREPARATIONS:
            raise ValueError('Explicit supported face byte preparation required')
        self.mode, self.scratch = mode, None
        self.owner = threading.get_ident()

    def prepare_bgr(self, frames):
        if threading.get_ident() != self.owner:
            raise RuntimeError('Face preparation scratch belongs to another thread')
        if len(frames) < 1 or not isinstance(frames[0], np.ndarray):
            raise ValueError('At least one FP32 HWC frame required')
        shape = frames[0].shape
        for frame in frames:
            if (not isinstance(frame, np.ndarray) or frame.dtype != np.float32 or frame.ndim != 3
                    or frame.shape[-1] != 3 or min(frame.shape) < 1 or frame.shape != shape):
                raise ValueError('Matching nonempty FP32 HWC frames required')
        if self.mode == 'legacy':
            # Match the existing executor: quantize the complete source batch
            # first, then create its list of BGR copies inside prediction.
            byte = [np.uint8(frame.clip(0,1)*255+.5) for frame in frames]
            return [np.ascontiguousarray(frame[:,:,::-1]) for frame in byte]
        output = []
        for frame in frames:
            if self.scratch is None or self.scratch.shape != shape:
                self.scratch = np.empty(shape, dtype=np.float32)
            # Separate float32 ufuncs retain the original rounding after each
            # operation. Do not fuse, change rounding or quantize through FP16.
            np.clip(frame, 0, 1, out=self.scratch)
            np.multiply(self.scratch, 255, out=self.scratch)
            np.add(self.scratch, .5, out=self.scratch)
            if self.mode == 'inplace_rgb':
                byte = self.scratch.astype(np.uint8)
                output.append(np.ascontiguousarray(byte[:,:,::-1]))
            else:
                byte = np.empty(shape, dtype=np.uint8)
                np.copyto(byte, self.scratch[:,:,::-1], casting='unsafe')
                output.append(byte)
        return output


def frame_sha256(frame):
    """Hash the same logical C-order bytes without copying contiguous frames."""
    data = memoryview(frame) if frame.flags.c_contiguous else frame.tobytes(order='C')
    return hashlib.sha256(data).hexdigest()
