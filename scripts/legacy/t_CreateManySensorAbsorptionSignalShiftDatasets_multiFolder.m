% t_CreateManyConeAbsorptionSignalNoiseDatasets
% Create multiple datasets consisting of the cone absorption of signals
% with and without added noise
%
% Description:
%    This tutorial creates a dataset for each of the contrast values in
%    "contrastValues". The resulting dataset consists of a signal, whose strength is 
%    defined by "contrast". The signal consists of stripes that can be seen
%    with a frequency of "frequency" on the generated image. In addition to
%    the signal, there is noise added as well. 
%    This image is then processed by a cone absorption function that 
%    simulates the stimulus generated, would this signal/noise image be 
%    absorbed by eye cones. The resulting image is then center cropped from
%    a size of 249x249 to a size of 227x227. This does not affect the 
%    created signal pattern itself, as it is pretty much only visible
%    within the cropped 227x227 center. 
%
%    "numSamples" images with noise are generated for each frequency, as
%    well as "numSamples" of images with noise only (-> no signal). For
%    each of these "numSamples" images, two addtitional images are created,
%    each consisting of a mean image (only signal, no noise).  {Why two
%    when one is enough? - Just for the user to verify that different runs
%    create the same no noise image. Also, to preserve the dimensional
%    structure.}
%
% See Also:
%    CreateConeAbsorptionSignalNoiseDataset_function

% Values to set
outDirBase = '/share/wandell/data/reith/frequencies_experiment/';
status = mkdir(outputFolder);
numSamples = 2;
frequencies = linspace(1,20,20);
% contrastValues = [0.0003, 0.0002, 0.0004];
contrastValues = 0.1;
contrastFreqPairs = [];
shiftValues = logspace(-4, -1, 12);

for i = 1:length(contrastValues)
    for j = 1:length(frequencies)       
        contrast = contrastValues(i);
        freq = frequencies(j);
        contrastFreqPairs = cat(1, contrastFreqPairs, [contrast, freq]);
    end
end

for f = 1:length(frequencies)
    freq = frequencies(f);
    % This creates the resulting datasets
    fprintf('starting at %s\n', datetime('now'))
    contrast = contrastValues;
    shiftValue = shiftValues;
    fileName = sprintf('freq_%s_contrast_%s_shift_%s_pi_per_3000',join(string(freq),'-'), strrep(sprintf("%.2f", contrast), '.', '_'), 'x');
    disp(fileName);
    outputFolder = [outDirBase fileName '/'];
    status = mkdir(outputFolder);
    CreateSensorAbsorptionSignalShiftDataset_function(freq, contrast, shiftValue, numSamples, fileName, outputFolder)
    fprintf('ending at %s\n', datetime('now'))
end