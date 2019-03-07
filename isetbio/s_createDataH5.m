contrastValues = logspace(-6.5, 1, 18);
outPath = '/share/wandell/data/reith/coneMosaik/static_caser_freq1_var_contrasts/';
mkdir(outputFolder);

for i = 1:length(contrastValues)
    contrast = contrastValues(i);
    createResnetTrainingData('outPath', outPath, 'contrast', contrast, 'frequency', 1, 'signalGridSize', 1, 'signalLocation', 1, ...
        'sceneSizeDegs', 2, 'gridZoom', 2, 'nTrialsNum', 1, 'nPathSteps', 1);
end