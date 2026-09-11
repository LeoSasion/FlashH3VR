import struct
import av
import numpy as np
import pytest
from scripts.research_prores_container_color import raw_container_declaration,resolve_frame_color,frame_to_srgb


def atom(kind,payload):return struct.pack('>I',8+len(payload))+kind+payload


def movie(path,*,primaries=1,duplicate=False):
    colr=atom(b'colr',b'nclc'+struct.pack('>HHH',primaries,1,1))
    entry=atom(b'apcn',bytes(78)+colr+atom(b'pasp',struct.pack('>II',1,1)))
    track=atom(b'stsd',bytes(4)+struct.pack('>I',1)+entry)
    for label in (b'stbl',b'minf',b'mdia',b'trak'):track=atom(label,track)
    path.write_bytes(atom(b'mdat',b'colrnclc'+bytes(32))+atom(b'moov',track*(2 if duplicate else 1)))
    return path


def frame():
    f=av.VideoFrame(16,16,'yuv422p10le')
    for p,value in zip(f.planes,(512,480,570)):
        p.update(np.full((p.height,p.line_size//2),value,dtype='<u2').tobytes())
    f.color_range=f.colorspace=1;f.color_trc=f.color_primaries=2
    return f


def test_reads_declared_atom_ignoring_decoy_in_media_payload(tmp_path):
    path=movie(tmp_path/'sample.mov');d=raw_container_declaration(path)
    assert [d[k] for k in ('color_primaries','color_trc','colorspace')]==[1]*3
    with path.open('rb') as file:
        file.seek(d['colr_offset']);assert file.read(18)==atom(b'colr',b'nclc'+struct.pack('>HHH',1,1,1))


def test_rejects_unknown_container_or_multiple_video_declarations(tmp_path):
    for name,options in (('unknown',dict(primaries=2)),('ambiguous',dict(duplicate=True))):
        with pytest.raises(ValueError):raw_container_declaration(movie(tmp_path/(name+'.mov'),**options))


def test_rejects_truncated_container_boundaries(tmp_path):
    p=movie(tmp_path/'bad.mov');p.write_bytes(p.read_bytes()[:-1])
    with pytest.raises(ValueError):raw_container_declaration(p)


def test_resolves_only_unspecified_fields_and_preserves_original_metadata(tmp_path):
    d=raw_container_declaration(movie(tmp_path/'valid.mov'));f=frame();r=resolve_frame_color(f,d)
    assert set(r['effective_tags'].values())=={1} and set(r['inherited_fields'])=={'color_trc','color_primaries'}
    yy=(512-64)/876;cb=(480-512)/896;cr=(570-512)/896
    code=np.array([yy+1.5748*cr,yy-.1873242729306488*cb-.46812427293064884*cr,yy+1.8556*cb])
    linear=np.where(code<.081,code/4.5,((code+.099)/1.099)**(1/.45))
    expected=np.where(linear<=.0031308,12.92*linear,1.055*linear**(1/2.4)-.055)
    actual=frame_to_srgb(f,8,8,d)
    np.testing.assert_allclose(actual,np.broadcast_to(expected,actual.shape),atol=2e-7,rtol=0)
    assert int(f.color_trc)==2 and int(f.color_primaries)==2


def test_known_hdr_or_matrix_conflicts_are_rejected_without_overwrite(tmp_path):
    d=raw_container_declaration(movie(tmp_path/'valid.mov'))
    for field,value in (('color_trc',16),('color_primaries',9),('colorspace',6),('color_range',2)):
        f=frame();setattr(f,field,value)
        with pytest.raises(ValueError):frame_to_srgb(f,8,8,d)
        assert int(getattr(f,field))==value
