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
%presentationDisplay = displaySet(presentationDisplay,'dpi', 2*96);
presentationDisplay = displaySet(presentationDisplay,'dpi', 0.5*96);


%% Create a harmonic test stimulus
%

% Let's choose these parameters:    [4 8 16 32]
% 
p = inputParser;
addParameter(p, 'frequency',4);
addParameter(p, 'contrasts', 0.6);
addParameter(p, 'sceneSizeDegs', 1);
addParameter(p, 'meanL',36);
addParameter(p, 'shift', 0);
addParameter(p, 'phaseD', 90);
addParameter(p, 'sigmaD',0.10);
addParameter(p, 'orientationDegs', 0);
addParameter(p, 'nPathSteps', 20);
addParameter(p, 'nTrialsNum',5);
addParameter(p, 'outPath', -1);
addParameter(p, 'signalGridSize', 1);
addParameter(p, 'signalLocation', 1);
addParameter(p, 'gridZoom', 1); 
parse(p, varargin{:});

spatialFrequency = p.Results.frequency;
contrasts         = p.Results.contrasts;
sceneSizeDegs    = p.Results.sceneSizeDegs;
meanL            = p.Results.meanL;   % Mean display luminance
shift            = p.Results.shift;
phaseD           = p.Results.phaseD;
sigmaD           = p.Results.sigmaD;
orientationDegs  = p.Results.orientationDegs;
nPathSteps       = p.Results.nPathSteps;
nTrialsNum       = p.Results.nTrialsNum;
outPath          = p.Results.outPath;
signalGridSize   = p.Results.signalGridSize;
signalLocation   = p.Results.signalLocation;
gridZoom         = p.Results.gridZoom;

% Filled in only when there is output argument in the calling
% function.
result = [];

name = sprintf('%d_samplesPerClass_frames_%d_freq_%s_contrast_%s',nTrialsNum, nPathSteps, join(string(spatialFrequency),'-'), strrep(sprintf("%.8f", contrasts), '.', '_'));
saveName = fullfile(outPath, name);

% Parameter struct for a Gabor stimulus 
stimParams = struct(...
    'spatialFrequencyCyclesPerDeg', spatialFrequency, ... % cycles/deg
    'orientationDegs', orientationDegs, ...               % 0 degrees (rotation)
    'phaseDegs', phaseD + shift, ...                      % spatial phase in degrees (0 is cos phase and 90 is sin)
    'sizeDegs', sceneSizeDegs, ...                        % D x D degrees
    'sigmaDegs', sigmaD, ...                              % sigma of Gaussian envelope, in degrees
    'contrast', contrasts(1),...                          % Michelson contrast
    'meanLuminanceCdPerM2', meanL, ...                    % mean luminance
    'pixelsAlongWidthDim', [], ...                        % pixels- width dimension
    'pixelsAlongHeightDim', [], ...                       % pixel- height dimension
    'signalGridSize', signalGridSize, ...                 % grid size, where signal location can be specified (2x2 grid -> 2)
    'signalLocation', signalLocation, ...                 % location, where to put the signal. In a 2x2 grid: 1 -> upperLeft, 2 -> upperRight, 3 -> lowerRight etc. etc.
    'gridZoom', gridZoom ...                              % zoom the grid into the display. Useful, if some extra padding around is needed to include eye movement..
    );

contrastsResult = [];
excitationsData = [];
% sceneWindow(signalScene);

% Visualize the generated scene
%{
visualizeScene(testScene, ...
    'displayRadianceMaps', false);
%}

%% Create a cone mosaic object with a 5 msec integration window

% Generate a hexagonal cone mosaic with ecc-based cone quantal efficiency
% set seed, so that coneMosaic stays the same for each contrast etc..
rng(42);
theMosaic = coneMosaic;
% This is a way to make only red cones
%   The order is Missing, Red, Green and Blue
%   
% theMosaic.spatialDensity = [0,1,0,0];
theMosaic.setSizeToFOV(0.5*sceneSizeDegs);
theMosaic.integrationTime = 10/1000;      % Integration time in ms
%% *Step 3.* Create the null stimulus - a 0% contrast Gabor

% Zero contrast for the null stimulus 
stimParams.contrast = 0.0;
contrastsResult = cat(1, contrastsResult, stimParams.contrast);
% Generate a scene representing the 10% Gabor stimulus as realized on the presentationDisplay
nullScene = generateGaborScene(...
    'stimParams', stimParams,...
    'presentationDisplay', presentationDisplay);

% sceneWindow(nullScene);

%% *Step 3.1 create signal stimulus
% Generate a scene representing the 10% Gabor stimulus as realized on the presentationDisplay
signalScenes = [];
for i = 1:length(contrasts)
stimParams.contrast = contrasts(i);
contrastsResult = cat(1, contrastsResult, stimParams.contrast);
signalScene = generateGaborScene(...
    'stimParams', stimParams,...
    'presentationDisplay', presentationDisplay);
signalScenes = [signalScenes signalScene];
end
%% *Step 4.* Compute the optical images of the 2 scenes

% Generate wavefront-aberration derived human optics
theOI = oiCreate('wvf human');

% oiWindow(theSignalOI);

% Compute the retinal image of the null stimulus
theNullOI = oiCompute(theOI, nullScene);

%% Compute a time series of mosaic responses to signal and the null stimuli

% Generate instances of eye movement paths. This step will
% take a few minutes to complete.
% moved to top - nTrialsNum   = 2;

% This makes it 100 ms trial for a 5 ms integration 
startPath  =  10;  
% moved to top - nPathSteps = 20;
emPathLength = nPathSteps + startPath - 1;    

theMosaic.noiseFlag = 'none';

theMosaic.emGenSequence(emPathLength,'nTrials',nTrialsNum,'rSeed',randi(1e9,1));

if nPathSteps == 1
    % static  case:
    emPositions = theMosaic.emPositions(:,1,:);
else
    % non static case:
    emPositions = theMosaic.emPositions(:,(startPath:emPathLength),:);
end
coneExcitationsNull = theMosaic.compute(theNullOI,'emPath',emPositions);
excitationsData = cat(4, excitationsData, coneExcitationsNull);
theMosaic.name = 'Null';

% I think it's better if both, signal and no signal, have the same eye
% movement path.
% theMosaic.emGenSequence(emPathLength,'nTrials',nTrialsNum,'rSeed',randi(1e9,1));
% emPositions = theMosaic.emPositions;

for i = 1:length(signalScenes)
   sc = signalScenes(i);
   % Compute the retinal image of the test stimulus
   theSignalOI = oiCompute(theOI, sc);
   coneExcitationsSignal = theMosaic.compute(theSignalOI,'emPath',emPositions);
   excitationsData = cat(4, excitationsData, coneExcitationsSignal);
end


disp("interesting");
spatialFrequencyCyclesPerDeg = zeros(nTrialsNum*(length(contrasts)+1), 1);
spatialFrequencyCyclesPerDeg(:) = spatialFrequency;
phaseDegs = zeros(nTrialsNum*(length(contrasts)+1), 1);
phaseDegs(:) = phaseD;
shifts = zeros(nTrialsNum*(length(contrasts)+1), 1);
shifts(:) = shift;
meanLuminance = zeros(nTrialsNum*(length(contrasts)+1), 1);
meanLuminance(:) = meanL;
sigmaGaussianEnvelope = zeros(nTrialsNum*(length(contrasts)+1), 1);
sigmaGaussianEnvelope(:) = sigmaD;
rotation = zeros(nTrialsNum*(length(contrasts)+1), 1);
rotation(:) = orientationDegs;

% If the data are returned, don't write them out.
% If the data are not returned, write them out.
if nargout == 0
    outFile = sprintf('%s.h5',saveName);
    
    % currDate = datestr(now,'mm-dd-yy_HH_MM');
    % The new function is worse.
    hdf5write(outFile, ...
        'mosaicPattern', theMosaic.pattern, ...
        'excitationsData', excitationsData, ...
        'coneExcitationsSignal', coneExcitationsSignal, ...
        'coneExcitationsNull', coneExcitationsNull, ...
        'spatialFrequencyCyclesPerDeg', spatialFrequencyCyclesPerDeg, ...
        'phaseDegs', phaseDegs, ...
        'shift', shifts, ...
        'meanLuminance', meanLuminance, ...
        'sigmaGaussianEnvelope', sigmaGaussianEnvelope, ...
        'rotation', rotation, ...
        'contrast', contrastsResult);
else
    
    result.coneExcitationsSignal        = coneExcitationsSignal;
    result.coneExcitationsNull          = coneExcitationsNull;
    result.spatialFrequencyCyclesPerDeg = spatialFrequencyCyclesPerDeg;
    result.phaseDegs                    = phaseDegs;
    result.shift                        = shifts;
    result.meanLuminance                = meanLuminance;
    result.sigmaGaussianEnvelope        = sigmaGaussianEnvelope;
    result.rotation                     = rotation;
    result.contrast                     = contrastsResult;

end

%{
% theMosaic.window;
% theMosaic.plot('Eye Movement Path');
%}

%% END

end

