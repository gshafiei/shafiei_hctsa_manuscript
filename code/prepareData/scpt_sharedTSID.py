
import h5py
import numpy as np
import time

tspath = '/path/to/hctsaOutputs/'

subjList = np.loadtxt('path/to/subjectLists/discovSubjList.txt', dtype=str)

IDList = []

for dataset in range(2):
    if dataset == 0:
        inpath = tspath + 'hctsaHCP_outputs/discovTest/'
    elif dataset == 1:
        inpath = tspath + 'hctsaHCP_outputs/discovReTest/'

    print('\nDataset', dataset)
    print('---------------------------------------------------')

    for subj in range(len(subjList)):
        start = time.time()
        inMat = (inpath + '%s_LR_Schaefer400_N.mat' % subjList[subj])
        with h5py.File(inMat, 'r') as src:
            refs = src.get('Operations/ID')[()]
            refID = np.hstack([src[ref][()].squeeze() for ref in refs[0]])
        IDList.append(set(refID))

        inMat = (inpath + '%s_RL_Schaefer400_N.mat' % subjList[subj])
        with h5py.File(inMat, 'r') as src:
            refs = src.get('Operations/ID')[()]
            refID = np.hstack([src[ref][()].squeeze() for ref in refs[0]])
        IDList.append(set(refID))
        end = time.time()
        print('\nSubj', subj, 'of', len(subjList), 'done!',
              '\nRunning time = ', end-start, 'seconds!')

sharedID = np.array(list(set.intersection(*IDList))).astype(int)
np.savetxt(tspath + 'hctsaHCP_outputs/sharedIDdiscov_Schaefer400.txt',
           sharedID)
