% run hctsa
datapath = '/path/to/subjectLists';
tspath = ('/path/to/timeSeriesData/');

% load subject lists of discovery group
discovID = load(fullfile(datapath,'discovSubjList.mat'));
subjList = cellstr(discovID);

for dataset = 1:2
    tstart = tic;
    if dataset == 1
        inpath = strcat(tspath,'discovTest/');
        outpath = ('path/to/output/discovTest/');
    elseif dataset == 2
        inpath = strcat(tspath,'discovReTest/');
        outpath = ('path/to/output/discovReTest/');
    end

    parfor (subj = 1:length(subjList),36)
        tic
        inputFile = strcat(inpath,[subjList{subj},'_RL_Schaefer400.mat']);
        outputFile = strcat(outpath,[subjList{subj},'_RL_Schaefer400.mat']);
        if ~isfile(outputFile)
            TS_init(inputFile,[],[],0,outputFile);
            TS_compute(0,[],[],'missing',outputFile,0);
            TS_normalize([],[],outputFile,[]);
        end

        inputFile = strcat(inpath,[subjList{subj},'_LR_Schaefer400.mat']);
        outputFile = strcat(outpath,[subjList{subj},'_LR_Schaefer400.mat']);
        if ~isfile(outputFile)
            TS_init(inputFile,[],[],0,outputFile);
            TS_compute(0,[],[],'missing',outputFile,0);
            TS_normalize([],[],outputFile,[]);
        end

        fprintf('\nDataset%i - Subj%i - done!\n',dataset,subj)
        toc
    end
    fprintf('DATASET%i done!\n',dataset)
    toc(tstart)
    fprintf('\n----------------------------------------------------------\n')
end
