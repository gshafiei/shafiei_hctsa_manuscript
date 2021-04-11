% load binary consensus sc data
datapth = '../../data/'
sc_consensus = load(fullfile(datapath, 'discov_consensus_sc_Schaefer400_matlab.mat'));
distance = load(fullfile(datapath, 'discov_distance_Schaefer400_matlab.mat'));

sc = double(sc_consensus.sc_consensus);
D = double(distance.distance);
nrandnet = 10000;
nnodes = 400;

randomnetw_sc = cell(nrandnet,1);
for randnet = 1:nrandnet
    tic
    randomnetw_sc{randnet}= ...
        fcn_match_length_degree_distribution(sc,...
        D, 10, 20*nnodes);
    fprintf('\nRandomization %i done!\n', randnet)
    toc
end

save(fullfile(datapath,'null_models_sc_bin_conslength.mat'),'randomnetw_sc')
