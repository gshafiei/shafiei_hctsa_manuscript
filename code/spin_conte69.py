import numpy as np
import nibabel as nib
from scipy.spatial.distance import cdist


def get_gifti_centroids(surfaces, lhannot, rhannot):
    lhsurface, rhsurface = [nib.load(s) for s in surfaces]

    centroids, hemiid = [], []
    for n, (annot, surf) in enumerate(zip([lhannot, rhannot],
                                          [lhsurface, rhsurface])):
        vert, face = [d.data for d in surf.darrays]
        labels = np.squeeze(nib.load(annot).darrays[0].data)

        for lab in np.unique(labels):
            if lab == 0:
                continue
            coords = np.atleast_2d(vert[labels == lab].mean(axis=0))
            roi = vert[np.argmin(cdist(vert, coords), axis=0)[0]]
            centroids.append(roi)
            hemiid.append(n)

    centroids = np.row_stack(centroids)
    hemiid = np.asarray(hemiid)

    return centroids, hemiid
