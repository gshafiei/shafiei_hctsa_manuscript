# construct 3D matrices as nodextsfeaturesxsubj
import h5py
import numpy as np
import time

tspath = '/path/to/hctsaOutputs/'

subjList = np.loadtxt('path/to/subjectLists/discovSubjList.txt', dtype=str)
tsID = np.loadtxt('../../data/sharedIDdiscov_Schaefer400.txt')
tsID = np.array(tsID).astype(int)

for dataset in range(2):
    if dataset == 0:
        inpath = tspath + 'hctsaHCP_outputs/discovTest/'
    elif dataset == 1:
        inpath = tspath + 'hctsaHCP_outputs/discovReTest/'
    sharedTS_LR = np.zeros((len(tsID), 400, len(subjList)))
    sharedTS_RL = np.zeros((len(tsID), 400, len(subjList)))

    print('\nDataset', dataset)
    print('---------------------------------------------------')

    for subj in range(len(subjList)):
        start = time.time()
        inMat = (inpath + '%s_LR_Schaefer400_N.mat' % subjList[subj])
        with h5py.File(inMat, 'r') as src:
            refs = src.get('Operations/ID')[()]
            refID = np.hstack([src[ref][()].squeeze() for ref in refs[0]])
            ts = np.array(src['TS_DataMat'])
        availidx = np.intersect1d(refID, tsID, return_indices=True)
        myTS = ts[availidx[1], :]
        sharedTS_LR[:, :, subj] = myTS

        inMat = (inpath + '%s_RL_Schaefer400_N.mat' % subjList[subj])
        with h5py.File(inMat, 'r') as src:
            refs = src.get('Operations/ID')[()]
            refID = np.hstack([src[ref][()].squeeze() for ref in refs[0]])
            ts = np.array(src['TS_DataMat'])
        availidx = np.intersect1d(refID, tsID, return_indices=True)
        myTS = ts[availidx[1], :]
        sharedTS_RL[:, :, subj] = myTS
        end = time.time()
        print('\nSubj', subj, 'of', len(subjList), 'done!',
              '\nRunning time = ', end-start, 'seconds!')
    if dataset == 0:
        outpath = tspath + 'hctsaHCP_outputs/discovTest/'
        np.save(outpath + 'sharedTS_LR_Schaefer400.npy', sharedTS_LR)
        np.save(outpath + 'sharedTS_RL_Schaefer400.npy', sharedTS_RL)
    elif dataset == 1:
        outpath = tspath + 'hctsaHCP_outputs/discovReTest/'
        np.save(outpath + 'sharedTS_LR_Schaefer400.npy', sharedTS_LR)
        np.save(outpath + 'sharedTS_RL_Schaefer400.npy', sharedTS_RL)
