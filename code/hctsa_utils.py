import scipy
import sklearn
import numpy as np
import spin_conte69
import seaborn as sns
import sklearn.metrics
import sklearn.decomposition
from mapalign.embed import compute_diffusion_map
from netneurotools import stats as netneurostats


def get_gradients(connectmat, ncomp):
    threshMat = connectmat.copy()
    np.fill_diagonal(threshMat, 0)

    # Generate percentile thresholds for 90th percentile
    perc = np.array([np.percentile(x, 90) for x in threshMat])

    # Threshold each row of the matrix by setting values below 90th perc to 0
    for i in range(threshMat.shape[0]):
        threshMat[i, threshMat[i, :] < perc[i]] = 0

    # Count negative values per row
    neg_values = np.array([sum(threshMat[i, :] < 0)
                          for i in range(threshMat.shape[0])])
    print('Negative values occur in %d rows' % sum(neg_values > 0))

    # remove negative ones
    threshMat[threshMat < 0] = 0

    cosSimilarity = sklearn.metrics.pairwise.cosine_similarity(threshMat)
    # de = DiffusionMapEmbedding().fit_transform(cosSimilarity)
    dme = compute_diffusion_map(cosSimilarity, n_components=ncomp,
                                return_result=True)

    # lambdas
    lambdas = dme[1]['lambdas']

    # gradients
    grads = dme[0]

    return grads, lambdas


def scatterregplot(x, y, title, xlab, ylab, pointsize):
    myplot = sns.scatterplot(x, y, facecolors='darkslategrey', s=pointsize,
                             legend=False)
    sns.regplot(x, y, scatter=False, ax=myplot,
                line_kws=dict(color='darkgray'))
    sns.despine(ax=myplot, trim=False)
    myplot.axes.set_title(title)
    myplot.axes.set_xlabel(xlab)
    myplot.axes.set_ylabel(ylab)
    myplot.figure.set_figwidth(5)
    myplot.figure.set_figheight(5)
    return myplot


def get_spinp(x, y, corrval, nspin, lhannot, rhannot, corrtype):
    surf_path = ('/home/gshafiei/data2/Projects/HCP_MEG/' +
                 'parcellationData/common/')
    surfaces = [surf_path + 'L.sphere.32k_fs_LR.surf.gii',
                surf_path + 'R.sphere.32k_fs_LR.surf.gii']
    lhannot = lhannot
    rhannot = rhannot

    centroids, hemiid = spin_conte69.get_gifti_centroids(surfaces, lhannot,
                                                         rhannot)
    spins, cost = netneurostats.gen_spinsamples(centroids, hemiid,
                                                n_rotate=nspin, seed=272)

    permuted_r = np.zeros((nspin, 1))
    for spin in range(nspin):
        if corrtype == 'spearman':
            permuted_r[spin] = scipy.stats.spearmanr(x[spins[:, spin]], y)[0]
        elif corrtype == 'pearson':
            permuted_r[spin] = scipy.stats.pearsonr(x[spins[:, spin]], y)[0]

    permmean = np.mean(permuted_r)
    pvalspin = (len(np.where(abs(permuted_r - permmean) >=
                             abs(corrval - permmean))[0])+1)/(nspin+1)
    return pvalspin


def get_spinidx(nspin, lhannot, rhannot):
    surf_path = ('/home/gshafiei/data2/Projects/HCP_MEG/' +
                 'parcellationData/common/')
    surfaces = [surf_path + 'L.sphere.32k_fs_LR.surf.gii',
                surf_path + 'R.sphere.32k_fs_LR.surf.gii']
    lhannot = lhannot
    rhannot = rhannot

    centroids, hemiid = spin_conte69.get_gifti_centroids(surfaces, lhannot,
                                                         rhannot)
    spins, cost = netneurostats.gen_spinsamples(centroids, hemiid,
                                                n_rotate=nspin, seed=272)
    return spins
