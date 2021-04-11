####################################
# Temporal profile similarity vs structural and functional networks
####################################

import scipy
import sklearn
import numpy as np
import hctsa_utils
import pandas as pd
import seaborn as sns
import sklearn.metrics
import scipy.io as sio
import matplotlib.colors
import scipy.stats as stats
import sklearn.decomposition
import matplotlib.pyplot as plt
from netneurotools import plotting
from scipy.optimize import curve_fit
from scipy.interpolate import interpn
from dominance_analysis import Dominance


plt.rcParams['svg.fonttype'] = 'none'


# load all data
tsn = np.load('../data/discov_avgtsn_Schaefer400.npy')
fc_average_discov = np.load('../data/discov_FC_average_Schaefer400.npy')
sc_consensus_discov = np.load('../data/discov_consensusSC_Schaefer400.npy')
wei_sc_consensus = np.load('../data/discov_wei_consensusSC_Schaefer400.npy')
ctData_discov = np.load('../data/discov_corticalthickness_Schaefer400.npy')
myelinData_discov = np.load('../data/discov_myelination_Schaefer400.npy')


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


# exponential fit
def func(x, a, b, c):
    return a * np.exp(-b * x) + c


####################################
# tsn vs distance
####################################
# underlying structure
mask = np.mask_indices(400, np.triu, 1)
masked_tsn = tsn[mask]

# plot tsn distance scatter plot
distance = sklearn.metrics.pairwise_distances(coor)

x = distance[mask]
y = tsn[mask]

data, x_e, y_e = np.histogram2d(x, y, bins=30, density=True)
z = interpn((0.5*(x_e[1:]+x_e[:-1]), 0.5*(y_e[1:]+y_e[:-1])), data,
            np.vstack([x, y]).T, method='splinef2d', bounds_error=False)


popt, pcov = curve_fit(func, x, y, bounds=(-1, 2))
modeled_y = func(x, *popt)

figure = plt.scatter(x, y, c=z, cmap='RdBu_r', marker='.', label='data',
                     rasterized=True)
plt.plot(np.array(sorted(x)), np.array(sorted(modeled_y, reverse=True)),
         'k-', lw=4, label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

plt.xlabel('distance - upper triangle')
plt.ylabel('mean tsn - upper triangle')
sns.despine(trim=True)
plt.title('exponential fit: y = a * exp(b*x) + c')
plt.legend()
figure.figure.set_figwidth(10)
figure.figure.set_figheight(10)

####################################
# tsn: sc vs no-sc
####################################
# underlying structure
mask = np.mask_indices(400, np.triu, 1)
masked_tsn = tsn[mask]

masked_sc = sc_consensus_discov[mask]
scidx = np.where(masked_sc == 1)[0]
noscidx = np.where(masked_sc == 0)[0]

tsn_withsc = masked_tsn[scidx]
tsn_nosc = masked_tsn[noscidx]

[tvals, pvals] = scipy.stats.ttest_ind(np.array(tsn_withsc),
                                       np.array(tsn_nosc))
label = ['connected'] * len(tsn_withsc) + ['not connected'] * len(tsn_nosc)
struct_tsn = pd.DataFrame(data={'temporal similarity':
                          np.concatenate((tsn_withsc, tsn_nosc)),
                          'structural connectivity': label})
sns.set(style='ticks', palette='pastel')
flierprops = dict(marker='.')
myplot = sns.boxplot(x='structural connectivity', y='temporal similarity',
                     data=struct_tsn, palette=['m', 'g'],
                     width=0.4, fliersize=3, showcaps=False,
                     flierprops=flierprops, showfliers=False)
sns.despine(ax=myplot, offset=5, trim=True)
myplot.axes.set_title('t-value = %1.3f, - p = %1.3f' % (tvals, pvals))
myplot.figure.set_figwidth(7)
myplot.figure.set_figheight(6)


# rewired null
# Note: "null_models_sc_bin_conslength.mat" is not shared, but can be
# generated using "scpt_make_rewiredNullSC.m" code
rewirednull = sio.loadmat('../data/null_models_sc_bin_conslength.mat')
randomnet_sc = rewirednull['randomnetw_sc']

mask = np.mask_indices(400, np.triu, 1)

masked_sc = sc_consensus_discov[mask]
scidx = np.where(masked_sc == 1)[0]
noscidx = np.where(masked_sc == 0)[0]

masked_tsn = tsn[mask]
tsn_withsc = masked_tsn[scidx]
tsn_nosc = masked_tsn[noscidx]

empirical_meanDiff = np.mean(tsn_withsc) - np.mean(tsn_nosc)

rewired_meanValDiff = np.zeros((1, 10000))

for nrand in range(10000):
    rewired_sc = randomnet_sc[nrand, 0]
    rewired_masked_sc = rewired_sc[mask]
    rewired_scidx = np.where(rewired_masked_sc == 1)[0]
    rewired_noscidx = np.where(rewired_masked_sc == 0)[0]

    rewired_tsn_withsc = masked_tsn[rewired_scidx]
    rewired_tsn_nosc = masked_tsn[rewired_noscidx]

    rewired_meanValDiff[:, nrand] = np.mean(rewired_tsn_withsc) - np.mean(
                                    rewired_tsn_nosc)
    print('\nRewired network %s' % nrand)

# pval
dist_mean = np.mean(rewired_meanValDiff)
nn = np.where((np.abs(rewired_meanValDiff-dist_mean) >=
               np.abs(empirical_meanDiff-dist_mean)) |
              (np.abs(np.abs(empirical_meanDiff) -
               np.abs(rewired_meanValDiff)) < 10**-6))
rewired_pval = (len(nn[0])+1)/10001

myfig = plt.figure()
ax = sns.distplot(rewired_meanValDiff, hist=False)
ax.set_ylim([0, 180])
plt.vlines(empirical_meanDiff, ymin=0,
           ymax=170)
plt.xlabel('difference of the means')
plt.ylabel('kernel density estimate')
plt.title('sc nosc tsn, pvalRewired = %s' % rewired_pval)
myfig.set_figwidth(7)
myfig.set_figheight(7)


####################################
# tsn: within-between rsn
####################################
# within between fc networks
edge_status = np.zeros(np.shape(fc_average_discov))
for i in range(fc_average_discov.shape[0]):
    for j in range(fc_average_discov.shape[1]):
        if rsnlabels[i] == rsnlabels[j]:
            edge_status[i, j] = 1
        else:
            edge_status[i, j] = 0

# within-between tsn
mask = np.mask_indices(400, np.triu, 1)
masked_edgestatus = edge_status[mask]
masked_tsn = tsn[mask]
tsn_withinidx = np.where(masked_edgestatus == 1)
tsn_within = masked_tsn[tsn_withinidx[0]][:, np.newaxis]
tsn_betweenidx = np.where(masked_edgestatus == 0)
tsn_between = masked_tsn[tsn_betweenidx[0]][:, np.newaxis]

empirical_meanDiff = np.mean(tsn_within) - np.mean(tsn_between)

[tvals, pvals] = scipy.stats.ttest_ind(tsn_within, tsn_between)
label = ['within'] * len(tsn_within) + ['between'] * len(tsn_between)
withinbetween_tsn = pd.DataFrame(data={'temporal similarity':
                                 np.concatenate((tsn_within.flatten(),
                                                 tsn_between.flatten())),
                                 'tsn connection status': label})
sns.set(style='ticks', palette='pastel')
flierprops = dict(marker='.')
myplot = sns.boxplot(x='tsn connection status', y='temporal similarity',
                     data=withinbetween_tsn, palette=['m', 'g'],
                     width=0.4, fliersize=3, showcaps=False,
                     flierprops=flierprops, showfliers=False)
sns.despine(ax=myplot, offset=5, trim=True)
myplot.axes.set_title('t-value = %1.3f, - p = %1.3f' % (tvals, pvals))
myplot.figure.set_figwidth(7)
myplot.figure.set_figheight(6)


# with spin
nspin = 10000
spinidx = hctsa_utils.get_spinidx(nspin=nspin, lhannot=lhlabels,
                                  rhannot=rhlabels)

# within-between tsn
mask = np.mask_indices(400, np.triu, 1)
masked_tsn = tsn[mask]
# masked_tsn = tsnresid
tsn_withinidx = np.where(masked_edgestatus == 1)
tsn_within = masked_tsn[tsn_withinidx[0]][:, np.newaxis]
tsn_betweenidx = np.where(masked_edgestatus == 0)
tsn_between = masked_tsn[tsn_betweenidx[0]][:, np.newaxis]

empirical_meanDiff = np.mean(tsn_within) - np.mean(tsn_between)

perm_meanValDiff = np.zeros((1, nspin))

for spin in range(nspin):
    perm_netw_labels = [rsnlabels[k] for k in spinidx[:, spin]]

    # within between fc networks
    perm_edge_status = np.zeros(np.shape(fc_average_discov))
    for i in range(fc_average_discov.shape[0]):
        for j in range(fc_average_discov.shape[1]):
            if perm_netw_labels[i] == perm_netw_labels[j]:
                perm_edge_status[i, j] = 1
            else:
                perm_edge_status[i, j] = 0
    perm_masked_edgestatus = perm_edge_status[mask]

    perm_tsn_withinidx = np.where(perm_masked_edgestatus == 1)
    perm_tsn_within = masked_tsn[perm_tsn_withinidx[0]][:, np.newaxis]
    perm_tsn_betweenidx = np.where(perm_masked_edgestatus == 0)
    perm_tsn_between = masked_tsn[perm_tsn_betweenidx[0]][:, np.newaxis]

    perm_meanValDiff[:, spin] = np.mean(perm_tsn_within) - np.mean(
                                    perm_tsn_between)
    print('\nSpin %s' % spin)

# pval
dist_mean = np.mean(perm_meanValDiff)
nn = np.where((np.abs(perm_meanValDiff-dist_mean) >=
               np.abs(empirical_meanDiff-dist_mean)) |
              (np.abs(np.abs(empirical_meanDiff) -
               np.abs(perm_meanValDiff)) < 10**-6))
spin_pval = (len(nn[0])+1)/(nspin+1)

myfig = plt.figure()
ax = sns.distplot(perm_meanValDiff, hist=False)
ax.set_ylim([0, 70])
plt.vlines(empirical_meanDiff, ymin=0,
           ymax=68)
plt.xlabel('difference of the means')
plt.ylabel('kernel density estimate')
plt.title('within between tsn, pvalSpin = %s' % spin_pval)
myfig.set_figwidth(7)
myfig.set_figheight(7)


####################################
# tsn vs fc
####################################
# tsn fc scatter plot
uniqlabels, uniqidx = np.unique(rsnlabels, return_index=True)
uniqlabels = uniqlabels[np.argsort(uniqidx)]
rsnidx = np.zeros((400, 1))
for n, rsn in enumerate(uniqlabels):
    idx = np.where(np.array(rsnlabels) == rsn)[0]
    rsnidx[idx] = n

rsnlabelsabb = ['VIS', 'SM', 'DA', 'VA', 'LIM', 'FP', 'DMN']

colors = sns.color_palette('hls', n_colors=len(np.unique(rsnidx)))
# add grey
colors.append(tuple(matplotlib.colors.to_rgb('gainsboro')))

# rsn assignment of edges
edge_assignments = np.zeros(np.shape(fc_average_discov))
edge_status = np.zeros(np.shape(fc_average_discov))
for i in range(fc_average_discov.shape[0]):
    for j in range(fc_average_discov.shape[1]):
        if rsnidx[i, :] == rsnidx[j, :]:
            edge_assignments[i, j] = rsnidx[i, :]
            edge_status[i, j] = 1
        else:
            edge_assignments[i, j] = 7
            edge_status[i, j] = 0
pointstyle = edge_status[mask].astype(str)
pointstyle[np.where(pointstyle == '1.0')] = 'within'
pointstyle[np.where(pointstyle == '0.0')] = 'between'
pointsize = edge_status[mask]
pointsize[np.where(pointsize == 1)] = 4
pointsize[np.where(pointsize == 0)] = 2
pointcolor = edge_assignments[mask].astype(str)
rsnlabelsabb.append('inter rsn')
for net in range(len(rsnlabelsabb)):
    pointcolor[np.where(pointcolor == str(net+0.0))] = rsnlabelsabb[net]
pointcolor_order = rsnlabelsabb


# plot tsn fc scatter plot
corr = scipy.stats.spearmanr(fc_average_discov[mask], tsn[mask])
figure = sns.scatterplot(fc_average_discov[mask], tsn[mask], hue=pointcolor,
                         hue_order=pointcolor_order, size=pointsize,
                         size_order=[2, 4], legend='full', palette=colors,
                         rasterized=True)
sns.regplot(fc_average_discov[mask], tsn[mask], scatter=False, ax=figure,
            line_kws=dict(color='darkgray'))
plt.xlabel('fc average - upper triangle')
plt.ylabel('mean tsn - upper triangle')
plt.title('distance regressed - spearman r = %1.3f, - p = %1.3f' % corr)
figure.figure.set_figwidth(10)
figure.figure.set_figheight(10)
sns.despine(trim=True)


# plot distance regressed version (exponential trend)
distance = sklearn.metrics.pairwise_distances(coor)

x = distance[mask]
y = tsn[mask]
y_fc = fc_average_discov[mask]


popt, pcov = curve_fit(func, x, y, bounds=(-1, 2))
modeled_y = func(x, *popt)

popt, pcov = curve_fit(func, x, y_fc, bounds=(-1, 2))
modeled_y_fc = func(x, *popt)


# residuals
tsnresid = y - modeled_y
fcresid = y_fc - modeled_y_fc

corr_distreg = scipy.stats.spearmanr(fcresid, tsnresid)
figure = sns.scatterplot(fcresid, tsnresid, hue=pointcolor,
                         hue_order=pointcolor_order, size=pointsize,
                         size_order=[2, 4], legend='full', palette=colors,
                         rasterized=True)
sns.regplot(fcresid, tsnresid, scatter=False, ax=figure,
            line_kws=dict(color='darkgray'))
plt.xlabel('fc average partial (residuals) - upper triangle')
plt.ylabel('mean tsn (residuals) - upper triangle')
plt.title('distance regressed - spearman r = %1.3f, - p = %1.3f'
          % corr_distreg)
figure.figure.set_figwidth(10)
figure.figure.set_figheight(10)
sns.despine(trim=True)

####################################
# tsn plot
####################################
# plot tsn with 7-network assignments
extrVal = np.max([abs(np.min(tsn)), abs(np.max(tsn))])
figure = plotting.plot_mod_heatmap(tsn, rsnidx.flatten().astype(int),
                                   figsize=(6.4, 4.8), cmap='RdBu_r',
                                   vmin=-extrVal, vmax=extrVal,
                                   xlabels=list(rsnlabelsabb),
                                   ylabels=list(rsnlabelsabb),
                                   rasterized=True)


# for 17 networks
lhlabels = ('../data/schaefer/HCP/fslr32k/gifti/' +
            'Schaefer2018_400Parcels_17Networks_order_lh.label.gii')
rhlabels = ('../data/schaefer/HCP/fslr32k/gifti/' +
            'Schaefer2018_400Parcels_17Networks_order_rh.label.gii')
labelinfo = np.loadtxt('../data/schaefer/HCP/fslr32k/gifti/' +
                       'Schaefer2018_400Parcels_17Networks_order_info.txt',
                       dtype='str', delimiter='tab')
rsnlabels17 = []
for row in range(0, len(labelinfo), 2):
    rsnlabels17.append(labelinfo[row].split('_')[2])

uniqlabels, uniqidx = np.unique(rsnlabels17, return_index=True)
uniqlabels = uniqlabels[np.argsort(uniqidx)]
rsnidx = np.zeros((400, 1))
for n, rsn in enumerate(uniqlabels):
    idx = np.where(np.array(rsnlabels17) == rsn)[0]
    rsnidx[idx] = n
    print(n)
    print(rsn)

figure = plotting.plot_mod_heatmap(tsn, rsnidx.flatten().astype(int),
                                   figsize=(6.4, 4.8), cmap='RdBu_r',
                                   vmin=-extrVal, vmax=extrVal,
                                   xlabels=uniqlabels,
                                   ylabels=uniqlabels,
                                   rasterized=True)

####################################
# Dominance analysis
####################################
# sc
wei_sc_consensus[wei_sc_consensus == 0] = np.nan
logSC = np.log(wei_sc_consensus)
logSC_fixed = logSC + 2*(-np.nanmin(logSC))
logSC_fixed = np.nan_to_num(logSC_fixed)

mask = np.mask_indices(400, np.triu, 1)
masked_sc = sc_consensus_discov[mask]
scidx = np.where(masked_sc == 1)[0]
noscidx = np.where(masked_sc == 0)[0]

masked_weighted_sc = logSC_fixed[mask]
weightedSC_withsc = masked_weighted_sc[scidx][:, np.newaxis]

# tsn
masked_tsn = tsn[mask]
tsn_withsc = masked_tsn[scidx][:, np.newaxis]
tsn_nosc = masked_tsn[noscidx][:, np.newaxis]

# fc
masked_fc = fc_average_discov[mask]
fc_withsc = masked_fc[scidx][:, np.newaxis]
fc_nosc = masked_fc[noscidx][:, np.newaxis]

# distance
distance = sklearn.metrics.pairwise_distances(coor)

masked_distance = distance[mask]
distance_withsc = masked_distance[scidx][:, np.newaxis]
distance_nosc = masked_distance[noscidx][:, np.newaxis]

# dominance analysis
stacked_data = np.hstack((weightedSC_withsc, fc_withsc, distance_withsc,
                          tsn_withsc))
data = pd.DataFrame(stats.zscore(stacked_data), columns=['SC', 'FC',
                                                         'distance', 'tsn'])
dominance_regression = Dominance(data=data, target='tsn', objective=1)
incr_variable_rsquare = dominance_regression.incremental_rsquare()
dominance_regression.dominance_stats()
