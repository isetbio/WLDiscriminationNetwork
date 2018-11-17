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
outputFolder = '/black/localhome/reith/Desktop/projects/WLDiscriminationNetwork/deepLearning/data/mat_files';
numSamples = 200;
frequencies = [1 2 3 4 5 6 7 8 9];
contrastValues = [0.005 0.0075  0.01 0.0125 0.015 0.02 0.03 0.04 0.08 0.16 0.32];

% This creates the resulting datasets
parfor i = 1:length(contrastValues)
    fprintf('starting at %s\n', datetime('now'))
    contrast = contrastValues(i);
    fileName = sprintf('%d_samplesPerClass_freq_%s_contrast_%s',numSamples, join(string(frequencies),'-'), strrep(num2str(contrast), '.', '_'));
    disp(fileName);
    CreateConeAbsorptionSignalNoiseDataset_function(frequencies, contrast, numSamples, fileName, outputFolder)
    fprintf('ending at %s\n', datetime('now'))
end