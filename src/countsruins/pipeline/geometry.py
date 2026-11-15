"""Shared bounding-box matching, suppression and table ownership."""
import numpy as np
from scipy.optimize import linear_sum_assignment

BOX = ["x1", "y1", "x2", "y2"]


def iou(a, b):
    inter = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2])).prod(-1)
    area = lambda x: (x[:, 2:]-x[:, :2]).prod(1)
    return inter/np.maximum(area(a)[:, None]+area(b)[None, :]-inter, 1e-9)


def nms(d, threshold=.5):
    """Greedy NMS inside one frame and group: neighbouring table crops see the same object."""
    d = d.sort_values('score', ascending=False)
    boxes = d[BOX].to_numpy(float)
    keep, removed = [], np.zeros(len(d), bool)
    for i in range(len(d)):
        if removed[i]:
            continue
        keep.append(i)
        removed |= iou(boxes[i:i+1], boxes)[0] >= threshold
    return d.iloc[keep]


def box_iou(a, b):
    x, y, u, v = a
    p, q, r, s = b
    inter = max(0, min(u,r)-max(x,p))*max(0, min(v,s)-max(y,q))
    return inter / max(1, (u-x)*(v-y)+(r-p)*(s-q)-inter)


def arrays(rows):
    return np.asarray(rows,dtype=float).reshape(-1,4)


def pairs(a,b, threshold=.3):
    overlap=iou(a,b)
    good=overlap>=threshold
    r,c=linear_sum_assignment(-good.astype(float)-.01*overlap)
    return {int(x):int(y) for x,y in zip(r,c) if good[x,y]}


def geometry(boxes, frame_tables):
    inter=(np.minimum(boxes[:,None,2:],frame_tables[None,:,2:])-
           np.maximum(boxes[:,None,:2],frame_tables[None,:,:2])).clip(0).prod(2)
    share=inter/np.maximum((boxes[:,2:]-boxes[:,:2]).prod(1),1)[:,None]
    return share.argmax(1),share.max(1)


def assign(boxes, table_rows):
    owner, share = geometry(boxes[BOX].to_numpy(float), table_rows[BOX].to_numpy(float))
    return boxes.assign(owner=np.where(share >= .50, table_rows.row_id.to_numpy()[owner], None))
