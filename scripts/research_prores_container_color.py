"""Explicit source-bound nclc resolution for unspecified ProRes frame fields."""
from pathlib import Path
import hashlib
import struct
from types import SimpleNamespace
import numpy as np
from h3ce.data.video import _color_values
from scripts.research_prores_rgb import sample_axis,resize_rgb,bt709_code_to_srgb

CONVERSION_ID='prores422p10_raw_nclc_bt709_unspecified_frame_resolution_srgb_v1'


def raw_container_declaration(path):
    """Walk MOV atoms, ignoring mdat bytes; accept exactly one apcn sample entry."""
    path=Path(path);entries=[]
    with path.open('rb') as file:
        def atoms(start,end,chain=()):
            pos=start
            while pos<end:
                if end-pos<8:raise ValueError('Truncated atom header')
                file.seek(pos);raw=file.read(16)
                size=int.from_bytes(raw[:4],'big');kind=raw[4:8];header=8
                if size==1:
                    if len(raw)<16:raise ValueError('Truncated extended atom')
                    size=int.from_bytes(raw[8:16],'big');header=16
                if size==0:size=end-pos
                if size<header or pos+size>end:raise ValueError('Atom outside parent bounds')
                current=chain+(kind.decode('latin1'),)
                if kind in (b'moov',b'trak',b'mdia',b'minf',b'stbl'):
                    atoms(pos+header,pos+size,current)
                elif kind==b'stsd':
                    file.seek(pos+header);prefix=file.read(8)
                    count=int.from_bytes(prefix[4:8],'big');cursor=pos+header+8
                    for number in range(count):
                        file.seek(cursor);entry_head=file.read(8)
                        length=int.from_bytes(entry_head[:4],'big');codec=entry_head[4:8]
                        if length<8 or cursor+length>pos+size:raise ValueError('Invalid sample-description bounds')
                        if codec==b'apcn':
                            if count!=1 or length<86:raise ValueError('Ambiguous ProRes sample description')
                            children=[];child=cursor+86
                            while child<cursor+length:
                                file.seek(child);head=file.read(8);n=int.from_bytes(head[:4],'big')
                                if n<8 or child+n>cursor+length:raise ValueError('Invalid visual sample-entry extension')
                                payload=file.read(n-8)
                                children.append(dict(type=head[4:8].decode('latin1'),offset=child,size=n,payload_hex=payload.hex()))
                                child+=n
                            colors=[c for c in children if c['type']=='colr']
                            aspects=[c for c in children if c['type']=='pasp']
                            if len(colors)!=1:raise ValueError('Exactly one actual colr declaration required')
                            color=colors[0];data=bytes.fromhex(color['payload_hex'])
                            if color['size']!=18 or data[:4]!=b'nclc':raise ValueError('Only explicit nclc source supported')
                            pri,trc,matrix=struct.unpack('>HHH',data[4:])
                            if (pri,trc,matrix)!=(1,1,1):raise ValueError('Container declaration is not supported BT709')
                            if len(aspects)!=1 or bytes.fromhex(aspects[0]['payload_hex'])!=struct.pack('>II',1,1):
                                raise ValueError('Explicit square-pixel pasp required')
                            entries.append(dict(atom_path='/'.join(current)+f'/apcn[{number}]/colr',entry_offset=cursor,
                                 colr_offset=color['offset'],colr_size=color['size'],colr_payload_hex=color['payload_hex'],
                                 color_primaries=pri,color_trc=trc,colorspace=matrix,extensions=children))
                        cursor+=length
                    if cursor!=pos+size:raise ValueError('Unparsed sample-description bytes')
                pos+=size
            if pos!=end:raise ValueError('Atom traversal did not close')
        atoms(0,path.stat().st_size)
    if len(entries)!=1:raise ValueError('Exactly one ProRes video sample entry required')
    digest=hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda:file.read(4*1024**2),b''):digest.update(block)
    return dict(status='raw_apcn_nclc_verified',source_path=str(path),source_sha256=digest.hexdigest(),
         source_bytes=path.stat().st_size,**entries[0])


def resolve_frame_color(frame,declaration):
    if declaration.get('status')!='raw_apcn_nclc_verified' or not declaration.get('source_sha256'):
        raise ValueError('A verified, source-bound container declaration is required')
    if [declaration[k] for k in ('colorspace','color_trc','color_primaries')]!=[1,1,1]:
        raise ValueError('Unsupported container color')
    raw={k:int(getattr(frame,k)) for k in ('color_range','colorspace','color_trc','color_primaries')}
    if raw['color_range']!=1 or raw['colorspace']!=1:
        raise ValueError('This source profile requires explicit per-frame limited range and BT709 matrix')
    if raw['color_trc'] not in (1,2) or raw['color_primaries'] not in (1,2):
        raise ValueError('Known conflicting frame color must not be overwritten')
    declared=SimpleNamespace(color_range=1,colorspace=1,color_trc=1,color_primaries=1)
    resolved=_color_values(frame,declared)
    if set(resolved.values())!={1}:raise ValueError('Unsupported effective color')
    return dict(raw_frame_tags=raw,effective_tags=resolved,
         inherited_fields=[k for k in ('color_trc','color_primaries') if raw[k]==2],
         inheritance_basis='Explicit nclc from this source file; unspecified frame field only',
         source_sha256=declaration['source_sha256'],colr_offset=declaration['colr_offset'])


def frame_to_srgb(frame,width,height,declaration):
    resolve_frame_color(frame,declaration)
    if frame.format.name!='yuv422p10le' or frame.interlaced_frame:
        raise ValueError('Only progressive planar10bit422 is supported')
    if min(width,height)<=0 or width>frame.width or height>frame.height or width*frame.height!=height*frame.width:
        raise ValueError('Aspect-preserving downscale required')
    if getattr(frame,'rotation',0) or any('DISPLAYMATRIX' in str(item.type).upper() for item in frame.side_data):
        raise ValueError('Rotation requires a separate transform')
    if any(any(marker in str(item.type).upper() for marker in ('MASTERING_DISPLAY','CONTENT_LIGHT','DYNAMIC_HDR','DOVI')) for item in frame.side_data):
        raise ValueError('HDR side data conflicts with the source profile')
    planes=[np.frombuffer(p,dtype='<u2').reshape(p.height,p.line_size//2)[:,:p.width].astype(np.float64) for p in frame.planes]
    if len(planes)!=3 or any(p.min()<0 or p.max()>1023 for p in planes):raise ValueError('Invalid10bit planes')
    y=(planes[0]-64)/876;centers=np.arange(frame.width,dtype=np.float64)/2
    cb=sample_axis((planes[1]-512)/896,centers,.5,-1);cr=sample_axis((planes[2]-512)/896,centers,.5,-1)
    kr=.2126;kb=.0722;kg=1-kr-kb
    r=y+2*(1-kr)*cr;b=y+2*(1-kb)*cb;g=y-2*kb*(1-kb)/kg*cb-2*kr*(1-kr)/kg*cr
    code=np.stack((r,g,b),axis=-1).clip(0,1).astype(np.float32)
    return bt709_code_to_srgb(resize_rgb(code,(height,width)))
