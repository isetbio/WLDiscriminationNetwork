
% Create datasets for various variables
outputFolder = '/black/localhome/reith/Desktop/projects/WLDiscriminationNetwork/deepLearning/data/mat_files';
name = 'freq_8_contrast_00005';
numSamples = 5000;
values = [0.00125 0.0025 0.005 0.01 0.02 0.04 0.08 0.16];
parfor i = 1:length(values)
    val = values(i)
    fileName = sprintf('%d_samplesPerClass_freq_8_contrast_%s',numSamples, strrep(num2str(val), '.', '_'));
    disp(fileName);
    createDataFunction(8, i, numSamples, fileName, outputFolder)
end