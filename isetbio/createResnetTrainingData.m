function result = createResnetTrainingData(varargin)
%% Choose parameters for RESNET and SVM comparisons
%  
% Training in this case for harmonic detection thresholds
% We are examining only the rectangular cone mosaics
%
% NOTE TO BW:   Add to displayGet() an argument to return the XYZ for
%               a particular set of RGB values
%                  XYZ = displayGet(d,'xyz',rgbValues)
%         
% FR/BW ISETBio Team, 2019


% TODO
%   Add position of the target in the visual field.  That way we can
%   produce targets at different locations
%

% Examples:
%{
% nTrials, nRows, nCols, nTime
  defaultData = createResnetTrainingData;
  ieMovie(squeeze(defaultData.coneExcitationsSignal(1,:,:,:)));
%}

%% Create a presentation display
% We employ an existing display specification, here an Apple LCD display.

% Create presentation display and place it 57 cm in front of the eye
presentationDisplay = displayCreate('LCD-Apple', 'viewing distance', 0.57);

% Double the display resolution
presentationDisplay = displaySet(presentationDisplay,'dpi', 2*96);

%% Create a harmonic test stimulus
%

% Let's choose these parameters:    [4 8 16 32]
% 
p = inputParser;
addParameter(p, 'frequency',4);
addParameter(p, 'contrast', 0.6);
addParameter(p, 'sceneSizeDegs', 1);
addParameter(p, 'meanL',36);
addParameter(p, 'shift', 0);
addParameter(p, 'phaseD', 90);
addParameter(p, 'sigmaD',0.10);
addParameter(p, 'orientationDegs', 0);
addParameter(p, 'nPathSteps', 20);
addParameter(p, 'nTrialsNum',5);
addParameter(p, 'outPath', -1);
parse(p, varargin{:});

spatialFrequency = p.Results.frequency;
contrast         = p.Results.contrast;
sceneSizeDegs    = p.Results.sceneSizeDegs;
meanL            = p.Results.meanL;   % Mean display luminance
shift            = p.Results.shift;
phaseD           = p.Results.phaseD;
sigmaD           = p.Results.sigmaD;
orientationDegs  = p.Results.orientationDegs;
nPathSteps       = p.Results.nPathSteps;
nTrialsNum       = p.Results.nTrialsNum;
outPath          = p.Results.outPath;

% Filled in only when there is output argument in the calling
% function.
result = [];

name = sprintf('%d_samplesPerClass_frames_%d_freq_%s_contrast_%s',nTrialsNum, nPathSteps, join(string(spatialFrequency),'-'), strrep(sprintf("%.8f", contrast), '.', '_'));
saveName = fullfile(outPath, name);

% Parameter struct for a Gabor stimulus 
stimParams = struct(...
    'spatialFrequencyCyclesPerDeg', spatialFrequency, ... % cycles/deg
    'orientationDegs', orientationDegs, ...               % 0 degrees (rotation)
    'phaseDegs', phaseD + shift, ...                          % spatial phase in degrees (0 is cos phase and 90 is sin)
    'sizeDegs', sceneSizeDegs, ...                        % D x D degrees
    'sigmaDegs', sigmaD, ...                              % sigma of Gaussian envelope, in degrees
    'contrast', contrast,...                              % Michelson contrast
    'meanLuminanceCdPerM2', meanL, ...                    % mean luminance
    'pixelsAlongWidthDim', [], ...                        % pixels- width dimension
    'pixelsAlongHeightDim', [] ...                        % pixel- height dimension
    );

% Generate a scene representing the 10% Gabor stimulus as realized on the presentationDisplay
signalScene = generateGaborScene(...
    'stimParams', stimParams,...
    'presentationDisplay', presentationDisplay);

% sceneWindow(signalScene);

% Visualize the generated scene
%{
visualizeScene(testScene, ...
    'displayRadianceMaps', false);
%}

%% Create a cone mosaic object with a 5 msec integration window

% Generate a hexagonal cone mosaic with ecc-based cone quantal efficiency
theMosaic = coneMosaic;
% This is a way to make only red cones
%   The order is Missing, Red, Green and Blue
%   
% theMosaic.spatialDensity = [0,1,0,0];
theMosaic.setSizeToFOV(0.5*sceneSizeDegs);
theMosaic.integrationTime = 5/1000;      % Integration time in ms
%% *Step 3.* Create the null stimulus - a 0% contrast Gabor

% Zero contrast for the null stimulus 
stimParams.contrast = 0.0;

% Generate a scene representing the 10% Gabor stimulus as realized on the presentationDisplay
nullScene = generateGaborScene(...
    'stimParams', stimParams,...
    'presentationDisplay', presentationDisplay);

% sceneWindow(nullScene);

%% *Step 4.* Compute the optical images of the 2 scenes

% Generate wavefront-aberration derived human optics
theOI = oiCreate('wvf human');

% Compute the retinal image of the test stimulus
theSignalOI = oiCompute(theOI, signalScene);

% oiWindow(theSignalOI);

% Compute the retinal image of the null stimulus
theNullOI = oiCompute(theOI, nullScene);

%% Compute a time series of mosaic responses to signal and the null stimuli

% Generate instances of eye movement paths. This step will
% take a few minutes to complete.
% moved to top - nTrialsNum   = 2;

% This makes it 100 ms trial for a 5 ms integration 
startPath  =  6;  
% moved to top - nPathSteps = 20;
emPathLength = nPathSteps + startPath - 1;    

theMosaic.noiseFlag = 'none';

theMosaic.emGenSequence(emPathLength,'nTrials',nTrialsNum,'rSeed',randi(1e9,1));
emPositions = theMosaic.emPositions(:,(startPath:emPathLength),:);
coneExcitationsNull = theMosaic.compute(theNullOI,'emPath',emPositions);
theMosaic.name = 'Null';

theMosaic.emGenSequence(emPathLength,'nTrials',nTrialsNum,'rSeed',randi(1e9,1));
emPositions = theMosaic.emPositions;
coneExcitationsSignal = theMosaic.compute(theSignalOI,'emPath',emPositions);

disp("interesting");
contrasts = zeros(nTrialsNum,1);
contrasts(:) = contrast;
spatialFrequencyCyclesPerDeg = zeros(nTrialsNum, 1);
spatialFrequencyCyclesPerDeg(:) = spatialFrequency;
phaseDegs = zeros(nTrialsNum, 1);
phaseDegs(:) = phaseD;
shifts = zeros(nTrialsNum, 1);
shifts(:) = shift;
meanLuminance = zeros(nTrialsNum, 1);
meanLuminance(:) = meanL;
sigmaGaussianEnvelope = zeros(nTrialsNum, 1);
sigmaGaussianEnvelope(:) = sigmaD;
rotation = zeros(nTrialsNum, 1);
rotation(:) = orientationDegs;

% If the data are returned, don't write them out.
% If the data are not returned, write them out.
if nargout == 0
    outFile = sprintf('%s.h5',saveName);
    
    % currDate = datestr(now,'mm-dd-yy_HH_MM');
    % The new function is worse.
    hdf5write(outFile, ...
        'coneExcitationsSignal', coneExcitationsSignal, ...
        'coneExcitationsNull', coneExcitationsNull, ...
        'spatialFrequencyCyclesPerDeg', spatialFrequencyCyclesPerDeg, ...
        'phaseDegs', phaseDegs, ...
        'shift', shifts, ...
        'meanLuminance', meanLuminance, ...
        'sigmaGaussianEnvelope', sigmaGaussianEnvelope, ...
        'rotation', rotation, ...
        'contrast', contrasts);
else
    
    result.coneExcitationsSignal = coneExcitationsSignal;
    result.coneExcitationsNull   = coneExcitationsNull;
    %{
        'spatialFrequencyCyclesPerDeg', spatialFrequencyCyclesPerDeg, ...
        'phaseDegs', phaseDegs, ...
        'shift', shifts, ...
        'meanLuminance', meanLuminance, ...
        'sigmaGaussianEnvelope', sigmaGaussianEnvelope, ...
        'rotation', rotation, ...
        'contrast', contrasts);
    %}
end

%{
% theMosaic.window;
% theMosaic.plot('Eye Movement Path');
%}

%% END

end

