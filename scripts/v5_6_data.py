#!/usr/bin/env python3
"""V5.6 registered object-sign and axis interventions; labels are never encoder inputs.

Production generation is released only by v5_6_extract. Software fixtures use
the software_test namespace or extend an old software_test block.
"""
from dataclasses import dataclass
import hashlib
import numpy as np
import v5_4_data as old
from v5_temporal_confirmation import render
from v5_5_counterfactual import cue_geometry, render_endpoint, composite_images, PAIR_IDS, array_identity

SCHEMA = 'v5.6-axis-object-1'
PATHS = old.PATHS.copy()
# Preserve all old unique slots, then add the two mixed-object poses.
POSE_NAMES = ('minus_minus', 'center', 'plus_plus', 'minus_plus', 'plus_minus')
UNIQUE_ORDER = tuple(p+':'+y for p in POSE_NAMES for y in ('a','b'))
X_ORDERS = np.array(((0,1),(2,1),(1,2),(1,0),(3,1),(4,1),(1,3),(1,4)), dtype=np.int64)
X_VIEW_INDICES = np.array([[2*a+y0,2*b+y1] for a,b in X_ORDERS for y0,y1 in PATHS], dtype=np.int64)
J_VIEW_INDICES = np.tile(old.VIEW_INDICES, (2,1))
AXIS_NAMES = ('none','Ax','Ay','Axy','Bx','By','Bxy')
AXIS_NEW_INDICES = np.array((1,2,4,5), dtype=np.int64)
AXIS_OLD_INDICES = np.array((0,3,6), dtype=np.int64)
AXIS_OLD_MASKS = np.array((0,4,2), dtype=np.int64)
OBJECT_CONDITION_INDICES = np.array(((0,1,2,3),(0,4,5,6)), dtype=np.int64)


@dataclass
class Block:
    unique_rgb: np.ndarray
    positions: np.ndarray
    x_targets: np.ndarray
    j_targets: np.ndarray
    metadata: dict

    def actor_inputs(self):
        return dict(rgb=self.unique_rgb[X_VIEW_INDICES],
                    times=np.tile(np.array([6.,7.], dtype=np.float32),(32,1)),
                    shutters=np.ones((32,2,2), dtype=np.float32))


def extend_block(base, role, block):
    """Add four images to an old six-image block without changing old pixels."""
    p=np.asarray(base.positions, dtype=np.float64)
    if p.shape != (3,4) or not np.array_equal(p[0]+p[2],2*p[1]):
        raise ValueError('Expected exact dyadic symmetric original poses')
    mixed=np.stack((np.concatenate((p[0,:2],p[2,2:])),np.concatenate((p[2,:2],p[0,2:]))))
    positions=np.concatenate((p,mixed))
    colors=[tuple(c) for c in base.metadata['colors']]
    nuisance=base.metadata['nuisance_x']
    new=np.stack([render(pose,colors,x) for pose in mixed for x in nuisance])
    images=np.concatenate((base.unique_rgb,new))
    start,end=positions[X_ORDERS[:,0]],positions[X_ORDERS[:,1]]
    x_targets=np.repeat(np.concatenate((end,end-start),axis=1).astype(np.float32),4,axis=0)
    j_targets=np.tile(base.targets,(2,1)).astype(np.float32)
    if not np.array_equal(x_targets[:16],base.targets): raise ValueError('Original view labels changed')
    # Each object's two-frame position and motion distributions are identical
    # between J and X, including the nuisance schedule. Joint signs differ.
    for obj in (0,1):
        cols=[2*obj,2*obj+1,4+2*obj,5+2*obj]
        j=np.concatenate((j_targets[:,cols],np.tile(PATHS,(8,1))),axis=1)
        x=np.concatenate((x_targets[:,cols],np.tile(PATHS,(8,1))),axis=1)
        if sorted(map(tuple,j.tolist())) != sorted(map(tuple,x.tolist())):
            raise ValueError('Per-object margins are not exactly matched')
    metadata=dict(schema=SCHEMA,role=role,block=block,group_id=f'{SCHEMA}/{role}/{block:06d}',
        old_metadata=base.metadata,unique_order=list(UNIQUE_ORDER),x_view_indices=X_VIEW_INDICES.tolist(),
        j_view_indices=J_VIEW_INDICES.tolist(),old_unique_count=6,new_unique_count=4,
        per_object_marginals_exact=True,independent_object_signs=True,independent_axis_interventions=False)
    return Block(images,positions,x_targets,j_targets,metadata)


def make_block(role, block):
    if role not in ('fit','confirm','software_test') or type(block) is not int:
        raise ValueError('Only fit/confirm/software_test; no original Test namespace')
    if not 0 <= block < (256 if role=='fit' else 64): raise ValueError('Registered block range exceeded')
    if role=='fit': return extend_block(old.make_block('fit',block),role,block)
    seed=int.from_bytes(hashlib.sha256(f'{SCHEMA}/{role}/{block}'.encode()).digest()[:8],'little')
    rng=np.random.default_rng(seed)
    center=rng.uniform(old.CENTER_LOWER,old.CENTER_UPPER)
    displacement=rng.uniform(.008,.018,4)*rng.choice([-1,1],4)
    center=np.round(center*2**20)/2**20;displacement=np.round(displacement*2**20)/2**20
    positions=np.stack((center-displacement,center,center+displacement))
    colors=[tuple(int(x) for x in c) for c in rng.integers(60,240,(2,3))]
    colors[0]=(235,colors[0][1]//2,60);colors[1]=(60,colors[1][1]//2,235)
    nuisance=rng.uniform(.1,.9,2)
    images=np.stack([render(p,colors,x) for p in positions for x in nuisance])
    start,end=positions[old.ORDERS[:,0]],positions[old.ORDERS[:,1]]
    targets=np.repeat(np.concatenate((end,end-start),axis=1).astype(np.float32),4,axis=0)
    base=old.Block(images,positions,targets,np.tile(np.concatenate((center,np.zeros(4))).astype(np.float32),(2,1)),
        dict(schema=SCHEMA,role=role,block=block,colors=[list(c) for c in colors],
             displacement=displacement.tolist(),nuisance_x=nuisance.tolist(),seed_namespace=f'{SCHEMA}/{role}'))
    return extend_block(base,role,block)


@dataclass
class AxisBlock:
    unique_rgb: np.ndarray
    targets: np.ndarray
    positions: np.ndarray
    metadata: dict

    def actor_inputs(self):
        return dict(rgb=self.unique_rgb,times=np.array([7.]*7,dtype=np.float32),
                    shutters=np.ones((7,2),dtype=np.float32))


def make_axes(split,pair,variant):
    """Other target and yellow stay at their original t6 positions."""
    if split not in ('dev','software_test') or (split=='dev' and pair not in PAIR_IDS):
        raise ValueError('Only the already observed V5.5 Dev range or software fixtures')
    original,colors,yellow=cue_geometry(split,pair,variant)
    bits=np.array(((0,0,0,0),(1,0,0,0),(0,1,0,0),(1,1,0,0),
                   (0,0,1,0),(0,0,0,1),(0,0,1,1)),dtype=np.int64)
    start=original[0].reshape(4);delta=(original[1]-original[0]).reshape(4)
    positions=np.stack([np.where(row,original[1].reshape(4),start) for row in bits])
    rendered=[render_endpoint(p.reshape(2,2),colors,yellow[0]) for p in positions]
    images=np.stack([x[0] for x in rendered]);masks=np.stack([x[1] for x in rendered])
    reference=composite_images(split,pair,variant)
    if not np.array_equal(images[AXIS_OLD_INDICES],reference[0][AXIS_OLD_MASKS]):
        raise ValueError('None/Axy/Bxy endpoint RGB no longer matches V5.5')
    edges=[]
    for obj,idx in enumerate(OBJECT_CONDITION_INDICES):
        for lo,hi,axis in ((0,1,'x'),(0,2,'y'),(1,3,'y'),(2,3,'x')):
            a,b=int(idx[lo]),int(idx[hi]);allowed=masks[a,obj]|masks[b,obj]
            if not np.array_equal(images[a][~allowed],images[b][~allowed]):
                raise ValueError('Axis change outside the intervened target support union')
            if not np.array_equal(images[a][masks[a,1-obj]],images[b][masks[b,1-obj]]):
                raise ValueError('Other target changed')
            if not np.array_equal(images[a][masks[a,2]],images[b][masks[b,2]]):
                raise ValueError('Yellow changed')
            edges.append(dict(start=a,end=b,object=obj,axis=axis,allowed_mask=array_identity(allowed),outside_support_unchanged=True,
                              changed_pixels=int(np.any(images[a]!=images[b],axis=-1).sum())))
    targets=np.concatenate((positions,bits*delta),axis=1).astype(np.float32)
    return AxisBlock(images,targets,positions.astype(np.float32),dict(schema=SCHEMA,split=split,pair=pair,
        variant=variant,conditions=list(AXIS_NAMES),coordinate_bits=bits.tolist(),new_indices=AXIS_NEW_INDICES.tolist(),
        old_indices=AXIS_OLD_INDICES.tolist(),old_masks=AXIS_OLD_MASKS.tolist(),
        object_condition_indices=OBJECT_CONDITION_INDICES.tolist(),old_v55_inputs=reference[3],
        colors=[list(c) for c in colors],nuisance_center=yellow[0],object_masks=[array_identity(m) for m in masks],
        all_eight_axis_edges_verified=True,edges=edges))
