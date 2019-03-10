contrastValues = logspace(-3.5, log(0.5)/log(10), 18);
outPath = '/share/wandell/data/reith/coneMosaik/temporal_case_freq1_var_contrasts/';
mkdir(outPath);
allStar = tic;
for i = 1:length(contrastValues)
    start = tic;
    contrast = contrastValues(i);
    createResnetTrainingData('outPath', outPath, 'contrast', contrast, 'frequency', 1, 'signalGridSize', 1, 'signalLocation', 1, ...
        'sceneSizeDegs', 2, 'gridZoom', 1, 'nTrialsNum', 1, 'nPathSteps', 10);
    sprintf("Time for for data generation with contrast of %d is %d seconds", contrast, round(toc(start)));
end
 sprintf("Time for for whole program is %d seconds",round(toc(allStar)));