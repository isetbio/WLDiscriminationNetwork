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

imageDir = '/share/wandell/data/reith/matlabData/generated_circles/';
imageNames = dir([imageDir '*.bmp']);
imagePaths = fullfile(imageDir, {imageNames.name});
numSamples = 2;
frequencies = 4;
% contrastValues = [0.0003, 0.0002, 0.0004];
contrastValues = logspace(-4.5, 0, 20);
contrastFreqPairs = [];
shiftValues = 0;

for i = 1:length(contrastValues)
    for j = 1:length(frequencies)       
        contrast = contrastValues(i);
        freq = frequencies(j);
        contrastFreqPairs = cat(1, contrastFreqPairs, [contrast, freq]);
    end
end

for i = 1:length(imagePaths)
    imagePath = imagePaths(i);
    imagePath = imagePath{1};
    [~,fname,~] = fileparts(imagePath);
    outputFolder = ['/share/wandell/data/reith/circles_experiment_v4/' fname '/'];
    status = mkdir(outputFolder);
    % This creates the resulting datasets
    fprintf('starting at %s\n', datetime('now'))
    contrast = contrastValues;
    shiftValue = shiftValues;
    fileName = sprintf('%d_samplesPerClass_freq_%s_contrast_%s_image_%s',numSamples, join(string(frequencies),'-'), 'x', fname);
    disp(fileName);
    CreateContrastDatasetFromImage_function(frequencies, contrast, shiftValue, numSamples, fileName, outputFolder, imagePath)
    fprintf('ending at %s\n', datetime('now'))
end