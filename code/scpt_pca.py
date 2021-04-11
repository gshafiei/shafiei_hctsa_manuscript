####################################
# PCA analysis
####################################

import bct
import scipy
import sklearn
import abagen
import numpy as np
import hctsa_utils
import spin_conte69
import pandas as pd
import seaborn as sns
import sklearn.metrics
import scipy.io as sio
import scipy.stats as stats
import sklearn.decomposition
import matplotlib.pyplot as plt
from netneurotools import plotting
from statsmodels.stats import multitest
from nibabel.freesurfer.io import read_annot
from netneurotools import stats as netneurostats


plt.rcParams['svg.fonttype'] = 'none'


# load all data
tsn = np.load('../data/discov_avgtsn_Schaefer400.npy')
avgFeatMat = np.load('../data/discov_avgFeatMat_Schaefer400.npy')
fc_average_discov = np.load('../data/discov_FC_average_Schaefer400.npy')
ctData_discov = np.load('../data/discov_corticalthickness_Schaefer400.npy')
myelinData_discov = np.load('../data/discov_myelination_Schaefer400.npy')
tsID_discov = np.loadtxt('../data/sharedIDdiscov_Schaefer400.txt')
tsID_discov = np.array(tsID_discov).astype(int)


# load parcellation-related data
lhlabels = ('../data/schaefer/HCP/fslr32k/gifti/' +
            'Schaefer2018_400Parcels_7Networks_order_lh.label.gii')
rhlabels = ('../data/schaefer/HCP/fslr32k/gifti/' +
            'Schaefer2018_400Parcels_7Networks_order_rh.label.gii')
labelinfo = np.loadtxt('../data/schaefer/HCP/fslr32k/gifti/' +
                       'Schaefer2018_400Parcels_7Networks_order_info.txt',
                       dtype='str', delimiter='tab')
coor = np.loadtxt('../data/schaefer/Schaefer_400_centres.txt', dtype=str)
coor = coor[:, 1:].astype(float)

rsnlabels = []
for row in range(0, len(labelinfo), 2):
    rsnlabels.append(labelinfo[row].split('_')[2])


# load required surfaces
surfaces = ['../data/surfaces/L.sphere.32k_fs_LR.surf.gii',
            '../data/surfaces/R.sphere.32k_fs_LR.surf.gii']


# load gene expression
schaefer_mni = ('../data/schaefer/MNI/' +
                'Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm.nii.gz')
expression = abagen.get_expression_data(schaefer_mni, exact=False,
                                        lr_mirror=True)

####################################
# PCA analysis on feature matrix
####################################
featMat = avgFeatMat

dataMat = stats.zscore(featMat)
pca = sklearn.decomposition.PCA(n_components=np.min(avgFeatMat.shape)-1,
                                svd_solver='full')
pca.fit(dataMat)
node_score = pca.transform(dataMat)
pc_wei = pca.components_

# var explained in PCA
varexpall = pca.explained_variance_ratio_
myplot = sns.scatterplot(np.arange(len(varexpall[:10])), varexpall[:10],
                         facecolors='darkslategrey', s=50)
sns.despine(ax=myplot, trim=False)
myplot.axes.set_title('PCA - hctsa feat')
myplot.axes.set_xlabel('components')
myplot.axes.set_ylabel('variance explained (%)')
myplot.figure.set_figwidth(5)
myplot.figure.set_figheight(5)

# plot on brain surface
ncomp = 0
toplot = node_score[:, ncomp]
brains = plotting.plot_conte69(toplot, lhlabels, rhlabels,
                               vmin=np.percentile(toplot, 2.5),
                               vmax=np.percentile(toplot, 97.5),
                               colormap='viridis',
                               colorbartitle=('hctsa node score - PC %s - ' +
                                              'VarExp = %0.3f')
                               % (ncomp+1, varexpall[ncomp]),
                               surf='inflated')


####################################
# compare PCs with fc gradients
####################################
grads, lambdas = hctsa_utils.get_gradients(fc_average_discov, ncomp=10)

# variance explained
varexpfc = lambdas/np.sum(lambdas)
myplot = sns.scatterplot(np.arange(len(varexpfc)), varexpfc,
                         facecolors='darkslategrey', s=75)
sns.despine(ax=myplot, trim=False)
myplot.axes.set_title('diffusion map embedding - fc')
myplot.axes.set_xlabel('components')
myplot.axes.set_ylabel('variance explained (%)')
myplot.figure.set_figwidth(5)
myplot.figure.set_figheight(5)

# plot on surface
ncomp = 0
toplot = grads[:, ncomp]
brains = plotting.plot_conte69(toplot, lhlabels, rhlabels,
                               vmin=np.percentile(toplot, 2.5),
                               vmax=np.percentile(toplot, 97.5),
                               colormap='viridis',
                               colorbartitle='fc gradient %s - VarExp = %0.3f'
                               % (ncomp+1, varexpfc[ncomp]), surf='inflated')

# correlate and plot
ncomp1 = 0
ncomp2 = 0
x = node_score[:, ncomp1]
y = grads[:, ncomp2]

# # y can also be average ct or myelin
# avgmyelin = np.mean(myelinData_discov, axis=0)
# avgct = np.mean(ctData_discov, axis=0)
# y = avgct

corr = scipy.stats.spearmanr(x, y)
pvalspin = hctsa_utils.get_spinp(x, y, corrval=corr[0], nspin=10000,
                                 corrtype='spearman',
                                 lhannot=lhlabels, rhannot=rhlabels)

title = 'spearman r = %1.3f - p (spin) = %1.4f' % (corr[0], pvalspin)
xlab = 'partial fc gradient'
ylab = 'hctsa node score - PC%s' % (ncomp1+1)
hctsa_utils.scatterregplot(y, x, title, xlab, ylab, pointsize=50)


####################################
# PCA analysis on gene expression
####################################
# use abagen's gene list to get brain-elated genes
celltype = abagen.fetch_gene_group('Brain')

ind = expression.columns.isin(celltype)

dataMat = np.array(expression.iloc[:, [i for i, x in enumerate(ind) if x]]).T
pcaGene = sklearn.decomposition.PCA(n_components=10)
pcaGene.fit(dataMat)
gene_score = pcaGene.transform(dataMat)
pc_wei_gene = pcaGene.components_

# var explained in PCA
varexpall = pcaGene.explained_variance_ratio_
myplot = sns.scatterplot(np.arange(len(varexpall)), varexpall,
                         facecolors='darkslategrey', s=75)
sns.despine(ax=myplot, trim=False)
myplot.axes.set_title('PCA - gene expression (brain specific)')
myplot.axes.set_xlabel('components')
myplot.axes.set_ylabel('variance explained (%)')

# plot on surface
ncomp = 0
toplot = pc_wei_gene[ncomp, :]
brains = plotting.plot_conte69(toplot, lhlabels, rhlabels,
                               vmin=np.percentile(toplot, 2.5),
                               vmax=np.percentile(toplot, 97.5),
                               colormap='magma',
                               colorbartitle=('gene exp - PC %s - ' +
                                              'VarExp = %0.3f')
                               % (ncomp+1, varexpall[ncomp]), surf='inflated')

# correlate and plot
ncomp1 = 0
ncomp2 = 0
x = node_score[:, ncomp1]
y = pc_wei_gene[ncomp2, :]

corr = scipy.stats.spearmanr(x, y)
pvalspin = hctsa_utils.get_spinp(x, y, corrval=corr[0], nspin=10000,
                                 corrtype='spearman',
                                 lhannot=lhlabels, rhannot=rhlabels)

title = 'spearman r = %1.3f - p (spin) = %1.4f' % (corr[0], pvalspin)
xlab = 'hctsa node score - PC %s' % (ncomp1+1)
ylab = 'gene expression (brain specific) - PC 1'
hctsa_utils.scatterregplot(x, y, title, xlab, ylab, 50)

####################################
# mass univariate comparison
####################################
# Pearson r
X = node_score[:, :2]
Y = stats.zscore(featMat)

rho = np.zeros((X.shape[1], Y.shape[1]))
for i in range(X.shape[1]):
    for j in range(Y.shape[1]):
        tmpcorr = scipy.stats.pearsonr(X[:, i], Y[:, j])
        rho[i, j] = tmpcorr[0]

# spin tests
centroids, hemiid = spin_conte69.get_gifti_centroids(surfaces, lhlabels,
                                                     lhlabels)
spins, cost = netneurostats.gen_spinsamples(centroids, hemiid,
                                            n_rotate=10000, seed=272)

n_spins = spins.shape[1]
rhoPerm = np.zeros((X.shape[1], Y.shape[1], n_spins))
for spin in range(n_spins):
    for i in range(X.shape[1]):
        for j in range(Y.shape[1]):
            rhoPerm[i, j, spin] = scipy.stats.pearsonr(
                                    X[spins[:, spin], i],
                                    Y[:, j])[0]

pvalPerm = np.zeros((X.shape[1], Y.shape[1]))
corrected_pval = np.zeros((X.shape[1], Y.shape[1]))
sigidx = []
nonsigidx = []
for comp in range(X.shape[1]):
    for feat in range(Y.shape[1]):
        permmean = np.mean(rhoPerm[comp, feat, :])
        pvalPerm[comp, feat] = (len(np.where(abs(rhoPerm[comp, feat, :]
                                                 - permmean) >=
                                             abs(rho[comp, feat]
                                                 - permmean))[0])+1)/(n_spins +
                                                                      1)
    multicomp = multitest.multipletests(pvalPerm[comp, :], alpha=0.001,
                                        method='fdr_bh')
    corrected_pval[comp, :] = multicomp[1]
    sigidx.append(np.where(multicomp[1] < 0.001)[0])
    nonsigidx.append(np.where(multicomp[1] >= 0.001)[0])

# load all operation IDs
hctsaOper = sio.loadmat('../data/hctsaOperations.mat')
Operations = hctsaOper['hctsaOperations']
operCodeString = np.array(list(Operations[1:, 0]))
operName = np.array(list(Operations[1:, 1]))
operKeywords = np.array(list(Operations[1:, 2]))
operID = np.array(list(Operations[1:, 3]))
operMasterID = np.array(list(Operations[1:, 4]))


# find the indices of operation IDs
sharedts = np.intersect1d(tsID_discov, operID.astype(int), return_indices=True)
sharedCodeString = operCodeString[sharedts[2], :]
sharedName = operName[sharedts[2], :]
sharedKeywords = operKeywords[sharedts[2], :]
sharedID = operID[sharedts[2], :]
sharedMasterID = operMasterID[sharedts[2], :]

allCodeString = []
allName = []
allKeywords = []
allID = []
allMasterID = []

for ii in range(len(sharedCodeString)):
    allCodeString.append(sharedCodeString[ii][0])
    allName.append(sharedName[ii][0])
    allKeywords.append(sharedKeywords[ii][0])
    allID.append(sharedID[ii][0])
    allMasterID.append(sharedMasterID[ii][0])

# generate csv files with all feature correlations (Tables S3 and S4)
var = 0
univarCorr = pd.DataFrame(data=allCodeString, columns=['CodeString'])
univarCorr['Name'] = allName
univarCorr['Keywords'] = allKeywords
univarCorr['ID'] = allID
univarCorr['MasterID'] = allMasterID

univarCorr['loading'] = rho[var, :]
# univarCorr['correctedPval'] = corrected_pval[var, :]
univarCorr['absoluteLoading'] = np.abs(rho[var, :])

univarCorr.sort_values('absoluteLoading', ascending=False, inplace=True)


# plot significant vs nonsignificant correlations
var = 0

sortedcorr = np.sort(rho[var, :])
sortingidx = np.argsort(rho[var, :])
sortedpval = pvalPerm[var, sortingidx]

rhoCopy = rho.copy()
rhoCopy[0, nonsigidx[0]] = 0
rhoCopy[1, nonsigidx[1]] = 0

sortedrhoCopy = np.sort(rhoCopy[var, :])
sigthresh1 = np.max(np.where(sortedrhoCopy < 0)[0])
sigthresh2 = np.min(np.where(sortedrhoCopy > 0)[0])

myfig = plt.figure()
plt.plot(sortedcorr)
plt.vlines([sigthresh1, sigthresh2], ymin=-1, ymax=1)
plt.xlabel('# of features')
plt.ylabel('pearson r')
plt.title('PC%s, sigthresh1 = %s, sigthresh2 = %s' % (str(var+1),
                                                      str(sigthresh1),
                                                      str(sigthresh2)))
myfig.set_figwidth(7)
myfig.set_figheight(7)

####################################
# Brain scores vs neurosynth data
####################################
X = node_score[:, :2]

neurosynth = pd.read_csv('../data/neurosynth_schaefer400.csv')

modif_neurosynth = neurosynth.drop([0, 201])
labels = modif_neurosynth.columns[1:]

Y = np.array(modif_neurosynth.iloc[:, 1:])
Y = stats.zscore(Y)

# univariate analysis
rho = np.zeros((X.shape[1], Y.shape[1]))
for i in range(X.shape[1]):
    for j in range(Y.shape[1]):
        tmpcorr = scipy.stats.pearsonr(X[:, i], Y[:, j])  # or pearsonr
        rho[i, j] = tmpcorr[0]


# calculate p-values from spin tests
centroids, hemiid = spin_conte69.get_gifti_centroids(surfaces, lhlabels,
                                                     lhlabels)
spins, cost = netneurostats.gen_spinsamples(centroids, hemiid,
                                            n_rotate=10000, seed=272)

n_spins = spins.shape[1]
rhoPerm = np.zeros((X.shape[1], Y.shape[1], n_spins))
for spin in range(n_spins):
    for i in range(X.shape[1]):
        for j in range(Y.shape[1]):
            rhoPerm[i, j, spin] = scipy.stats.pearsonr(
                                    X[spins[:, spin], i],
                                    Y[:, j])[0]


# correct p-values for multiple comparisons
pvalPerm = np.zeros((X.shape[1], Y.shape[1]))
corrected_pval = np.zeros((X.shape[1], Y.shape[1]))
sigidx = []
nonsigidx = []
for comp in range(X.shape[1]):
    for feat in range(Y.shape[1]):
        permmean = np.mean(rhoPerm[comp, feat, :])
        pvalPerm[comp, feat] = (len(np.where(abs(rhoPerm[comp, feat, :]
                                                 - permmean) >=
                                             abs(rho[comp, feat]
                                                 - permmean))[0])+1)/(n_spins +
                                                                      1)
    multicomp = multitest.multipletests(pvalPerm[comp, :], alpha=0.05,
                                        method='bonferroni')
    corrected_pval[comp, :] = multicomp[1]
    sigidx.append(np.where(multicomp[1] < 0.05)[0])
    nonsigidx.append(np.where(multicomp[1] >= 0.05)[0])


# plot results
ncomp = 1
sortedcorr = np.sort(rho[ncomp, :])

rhoCopy = rho.copy()
rhoCopy[ncomp, nonsigidx[ncomp]] = 0

sortedrhoCopy = np.sort(rhoCopy[ncomp, :])
sigthresh1 = np.max(np.where(sortedrhoCopy < 0)[0])
sigthresh2 = np.min(np.where(sortedrhoCopy > 0)[0])

myfig = plt.figure()
plt.plot(sortedcorr)
plt.vlines([sigthresh1, sigthresh2], ymin=np.min(sortedcorr),
           ymax=np.max(sortedcorr))
plt.xlabel('Neurosynth terms')
plt.ylabel('pearson correlation of terms with PC %s weights' % str(ncomp+1))
plt.title('PC%s, sigthresh1 = %s, sigthresh2 = %s' % (str(ncomp+1),
                                                      str(sigthresh1),
                                                      str(sigthresh2)))
myfig.set_figwidth(7)
myfig.set_figheight(7)


# bar plots
sigcorr = rho[ncomp, sigidx[ncomp]]
siglabels = labels[sigidx[ncomp]]

sortedsigcorr = np.sort(sigcorr)
sortedsiglabels = siglabels[np.argsort(sigcorr)]

fig, ax = plt.subplots()
ax.barh(np.arange(len(sortedsigcorr)), sortedsigcorr)
ax.set_xlim(-0.6, 0.6)
plt.yticks(np.arange(len(sortedsigcorr)), sortedsiglabels, rotation=0)
fig.set_figwidth(7)
fig.set_figheight(len(sigidx[ncomp])*0.6)
plt.title('PC%s - significant Neurosynth' % (str(ncomp+1)))

####################################
# PCs in networks
####################################
# von economo
ve_schaefer = sio.loadmat('../data/economo_Schaefer400')
veidx = ve_schaefer['pdata']
velabelsuniq = ['PM', 'AC1', 'AC2', 'PSS', 'PS', 'LB', 'IC']

velabelsfull = np.empty_like(rsnlabels)
for n, ve in enumerate(velabelsuniq):
    tempidx = np.where(veidx == n+1)[0]
    velabelsfull[tempidx] = ve

# laminar differentiation
laminar_diff = sio.loadmat('../data/mesulam_Schaefer400')
mesulamidx = laminar_diff['pdata']
mesulamlabelsuniq = []
for label in range(len(laminar_diff['mesulamLabels'][0])):
    mesulamlabelsuniq.append(laminar_diff['mesulamLabels'][0][label][0])

mesulamlabelsfull = np.empty_like(rsnlabels)
for n, label in enumerate(mesulamlabelsuniq):
    tempidx = np.where(mesulamidx == n+1)[0]
    mesulamlabelsfull[tempidx] = label

# intrinsic networks
rsnlabelsfull = rsnlabels
rsnlabelsuniq = np.unique(rsnlabels)

# plot
ncomp = 1
classname = 'Mesulam'

if classname == 'von Economo':
    network_labels = velabelsfull
    network_labels_uniq = velabelsuniq
elif classname == 'Yeo':
    network_labels = rsnlabelsfull
    network_labels_uniq = rsnlabelsuniq
elif classname == 'Mesulam':
    network_labels = mesulamlabelsfull
    network_labels_uniq = mesulamlabelsuniq

# plot
network_score = pd.DataFrame(data={'hctsa node score': node_score[:, ncomp],
                             'network assignment': network_labels})
medianVal = np.vstack([network_score.loc[network_score[
                      'network assignment'].eq(netw),
                      'hctsa node score'].median() for netw in
                      network_labels_uniq])
idx = np.argsort(medianVal.squeeze())
plot_order = [network_labels_uniq[k] for k in idx]
sns.set(style='ticks', palette='pastel')
myplot = sns.boxplot(x='network assignment', y='hctsa node score',
                     data=network_score,
                     width=0.4, fliersize=3, showcaps=False,
                     order=plot_order, showfliers=False)

sns.despine(ax=myplot, offset=5, trim=True)
myplot.axes.set_title('PC %s' % (ncomp+1))
myplot.figure.set_figwidth(10)
myplot.figure.set_figheight(6)

####################################
# Evolutionary expansion
####################################
# load data and parcellate
schaefer_fsaverage6 = read_annot('../data/schaefer/FreeSurfer5.3/' +
                                 'fsaverage6/label/' +
                                 'rh.Schaefer2018_400Parcels_7Networks' +
                                 '_order.annot')
evolutionExpData = np.loadtxt('../data/evolutionaryExpansion/' +
                              'Hill2010_evo_fsaverage6.txt')

parcelIDs = np.unique(schaefer_fsaverage6[0])
parcelIDs = np.delete(parcelIDs, 0)

parcellatedData = np.zeros((len(parcelIDs), 1))

for IDnum in parcelIDs:
    idx = np.where(schaefer_fsaverage6[0] == IDnum)[0]
    parcellatedData[IDnum-1, 0] = np.nanmean(evolutionExpData[idx])


# plot on brain surface
toplot = np.vstack((parcellatedData, parcellatedData))
brains = plotting.plot_conte69(toplot, rhlabels, rhlabels,
                               vmin=np.percentile(toplot, 2.5),
                               vmax=np.percentile(toplot, 97.5),
                               colormap='viridis',
                               colorbartitle=('evolutionary expansion'),
                               surf='inflated')


# correlate and plot
# only right hemisphere
ncomp = 0
x = node_score[200:, ncomp]
y = parcellatedData.flatten()

corr = scipy.stats.spearmanr(x, y)

nspin = 10000
spinidx = hctsa_utils.get_spinidx(nspin=nspin, lhannot=lhlabels,
                                  rhannot=rhlabels)
rh_spinidx = spinidx[200:, :] - 200

permuted_r = np.zeros((nspin, 1))
for spin in range(nspin):
    permuted_r[spin] = scipy.stats.spearmanr(x[rh_spinidx[:, spin]], y)[0]

permmean = np.mean(permuted_r)
pvalspin = (len(np.where(abs(permuted_r - permmean) >=
                         abs(corr[0] - permmean))[0])+1)/(nspin+1)

title = 'spearman r = %1.3f - p (spin) = %1.4f' % (corr[0], pvalspin)
xlab = 'node score PC%s' % (ncomp+1)
ylab = 'evolutionary expansion'
hctsa_utils.scatterregplot(x, y, title, xlab, ylab, pointsize=50)


# mirror left and right hemispheres
ncomp = 0
x = node_score[:, ncomp]
y = np.vstack((parcellatedData, parcellatedData)).flatten()

corr = scipy.stats.spearmanr(x, y)

pvalspin = hctsa_utils.get_spinp(x, y, corrval=corr[0], nspin=10000,
                                 corrtype='spearman',
                                 lhannot=lhlabels, rhannot=rhlabels)

title = 'spearman r = %1.3f - p (spin) = %1.4f' % (corr[0], pvalspin)
ylab = 'node score PC%s' % (ncomp+1)
xlab = 'evolutionary expansion'
hctsa_utils.scatterregplot(y, x, title, xlab, ylab, pointsize=50)

####################################
# Participation coefficient
####################################
# tsn and participation coefficient
uniqlabels, uniqidx = np.unique(rsnlabels, return_index=True)
uniqlabels = uniqlabels[np.argsort(uniqidx)]
rsnidx = np.zeros((400, 1))
for n, rsn in enumerate(uniqlabels):
    idx = np.where(np.array(rsnlabels) == rsn)[0]
    rsnidx[idx] = n

participCoef = bct.participation_coef(fc_average_discov, rsnidx)

toplot = participCoef
brains = plotting.plot_conte69(toplot, lhlabels, rhlabels,
                               vmin=np.percentile(toplot, 2.5),
                               vmax=np.percentile(toplot, 97.5),
                               colormap='viridis',
                               colorbartitle=('fc participation coefficient'),
                               surf='inflated')

# plot and correlate
ncomp = 0
x = participCoef
y = node_score[:, ncomp]
corr = scipy.stats.spearmanr(x, y)
pvalspin = hctsa_utils.get_spinp(x, y, corrval=corr[0], nspin=10000,
                                 corrtype='spearman',
                                 lhannot=lhlabels, rhannot=rhlabels)

title = 'spearman r = %1.3f - p (spin) = %1.4f' % (corr[0], pvalspin)
xlab = 'participation coefficient'
ylab = 'hctsa node score - PC%s' % (ncomp+1)
hctsa_utils.scatterregplot(x, y, title, xlab, ylab, pointsize=50)
