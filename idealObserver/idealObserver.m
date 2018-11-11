% script to generate ideal observer plots
% data is generated using generateTestData script

% loading data
chdir(fullfile(cs231nRootPath,'data'))
load('testSet_100Samp_IdealObserver.mat')
Images = testImages;
Images_NoNoise = testImages_NoNoise;
Labels = testLabels;
Contrasts = testContrasts;
Freqs = testFreqs;

% initialising variables
%scanFreq      = logspace(0.1, 1.5441, 5); 
%scanContrast  = logspace(-3.5, -1, 10);
%nImages = 10; % number of noisy and stimulus images generated 
scanFreq = unique(Freqs);
scanContrast = unique(Contrasts);

ioAccuracy = zeros(length(scanFreq),length(scanContrast));

for iterSp = 1:length(scanFreq)
    for iterC  = 1:length(scanContrast)
        
        freq = scanFreq(iterSp);
        contrast = scanContrast(iterC); 
        
        % finding data corresponding to freq and contrast        
        idx = (Contrasts == contrast & Freqs == freq);
        sample = Images(:,:,idx);
        sampleLabels = Labels(idx);
        
        % finding isorates
        idx0 = (Contrasts == contrast & Freqs == freq & Labels == 0);
        isoRate_c0 = reshape(mean(Images(:,:,idx0),3),[],1);
        idx1 = (Contrasts == contrast & Freqs == freq & Labels == 1);
        isoRate_c1 = reshape(mean(Images(:,:,idx1),3),[],1);
        
        % find accuracy 
        ioAccuracy(iterSp, iterC) = PoissonMLE(sample,sampleLabels, isoRate_c0, isoRate_c1);
           
    end
end

%% plotting data
% refer plotCSF
figure()
legend_name = [];
for iterSp = 1:length(scanFreq)
    semilogx(scanContrast,ioAccuracy(iterSp,:),'-o')
    hold on
    legend_name = [legend_name;sprintf("spatial frequency = %f cpd",scanFreq(iterSp))]; 
end
legend(legend_name')
xlabel('Contrast (log)')
ylabel('Mean Accuracy')


%% fitting data to find csf
% refer plotCSF

refAccuracy = 0.75; % reference accuracy for CSF plot
csfValue = zeros(length(scanFreq),1);

for iterSp = 1:length(scanFreq)
    % using interpolation to find the accuracy at 75%
    [x,iy,~] = unique(ioAccuracy(iterSp,:));
    y = log(scanContrast(iy));
    fitValue = interp1(x,y,refAccuracy);
    csfValue(iterSp) = 1 / (10^fitValue);
end

% plotting contrast sensitivity function
figure()
loglog(scanFreq, csfValue,'k-o')
ylabel('contrast sensitivity')
xlabel('spatial frequency')


%% saving data
% to save freqRange, contrastRange, ioAccuracy, csfAccuracy