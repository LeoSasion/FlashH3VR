"""Nonlearned chunk planning and matching-frame overlap weights."""
import numpy as np

def chunk_plan(shot_ids, size=22, overlap=5):
    if not shot_ids or not 0 <= overlap < size:
        raise ValueError('Invalid video chunk configuration')
    ranges=[];shot_start=0
    for end in range(1,len(shot_ids)+1):
        if end<len(shot_ids) and shot_ids[end]==shot_ids[shot_start]:continue
        start=shot_start
        while True:
            stop=min(start+size,end)
            ranges.append({'start':start,'stop':stop,'shot':shot_ids[start],
                           'left_overlap':0 if start==shot_start else overlap,
                           'right_overlap':0 if stop==end else overlap})
            if stop==end:break
            start=stop-overlap
        shot_start=end
    return ranges

def blend_weights(chunk):
    n=chunk['stop']-chunk['start'];w=np.ones(n,dtype=np.float32)
    left,right=chunk['left_overlap'],chunk['right_overlap']
    if left:w[:left]=np.arange(1,left+1,dtype=np.float32)/(left+1)
    if right:w[-right:]=np.arange(right,0,-1,dtype=np.float32)/(right+1)
    return w
