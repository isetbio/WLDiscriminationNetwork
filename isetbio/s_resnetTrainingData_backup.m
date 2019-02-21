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

%% Create a presentation display
% We employ an existing display specification, here an Apple LCD display.

% Create presentation display and place it 57 cm in front of the eye
presentationDisplay = displayCreate('LCD-Apple', 'viewing distance', 0.57);

% Double the display resolution
presentationDisplay = displaySet(presentationDisplay,'dpi', 2*96);

%% Create a cone mosaic object with a 5 msec integration window

% Generate a hexagonal cone mosaic with ecc-based cone quantal efficiency
theMosaic = coneMosaic;
% This is a way to make only red cones
%   The order is Missing, Red, Green and Blue
%   
% theMosaic.spatialDensity = [0,1,0,0];
theMosaic.setSizeToFOV(0.5*sceneSizeDegs);
theMosaic.integrationTime = 5/1000;      % Integration time in ms

%% Create a harmonic test stimulus
%

% Let's choose these parameters:    [4 8 16 32]
% 

spatialFrequency = 4;
contrast         = 0.6;
sceneSizeDegs    = 1;
meanL            = 75;   % Mean display luminance

% Parameter struct for a Gabor stimulus 
stimParams = struct(...
    'spatialFrequencyCyclesPerDeg', spatialFrequency, ... % cycles/deg
    'orientationDegs', 0, ...               % 0 degrees (rotation)
    'phaseDegs', 90, ...                    % spatial phase in degrees (0 is cos phase and 90 is sin)
    'sizeDegs', sceneSizeDegs, ...          % D x D degrees
    'sigmaDegs', 0.10, ...                   % sigma of Gaussian envelope, in degrees
    'contrast', contrast,...                % Michelson contrast
    'meanLuminanceCdPerM2', meanL, ...      % mean luminance
    'pixelsAlongWidthDim', [], ...          % pixels- width dimension
    'pixelsAlongHeightDim', [] ...          % pixel- height dimension
    );

% Generate a scene representing the 10% Gabor stimulus as realized on the presentationDisplay
signalScene = generateGaborScene(...
    'stimParams', stimParams,...
    'presentationDisplay', presentationDisplay);

sceneWindow(signalScene);

% Visualize the generated scene
%{
visualizeScene(testScene, ...
    'displayRadianceMaps', false);
%}
%% *Step 3.* Create the null stimulus - a 0% contrast Gabor

% Zero contrast for the null stimulus 
stimParams.contrast = 0.0;

% Generate a scene representing the 10% Gabor stimulus as realized on the presentationDisplay
nullScene = generateGaborScene(...
    'stimParams', stimParams,...
    'presentationDisplay', presentationDisplay);

sceneWindow(nullScene);

%% *Step 4.* Compute the optical images of the 2 scenes

% Generate wavefront-aberration derived human optics
theOI = oiCreate('wvf human');

% Compute the retinal image of the test stimulus
theSignalOI = oiCompute(theOI, signalScene);

oiWindow(theSignalOI);

% Compute the retinal image of the null stimulus
theNullOI = oiCompute(theOI, nullScene);

%% Compute a time series of mosaic responses to signal and the null stimuli

% Generate instances of eye movement paths. This step will
% take a few minutes to complete.
nTrialsNum   = 2;

% This makes it 100 ms trial for a 5 ms integration 
startPath  =  6;  
nPathSteps = 20;
emPathLength = nPathSteps + startPath - 1;    

theMosaic.noiseFlag = 'none';

theMosaic.emGenSequence(emPathLength,'nTrials',nTrialsNum,'rSeed',randi(1e9,1));
emPositions = theMosaic.emPositions(:,(startPath:emPathLength),:);
coneExcitationsNull = theMosaic.compute(theNullOI,'emPath',emPositions);
theMosaic.name = 'Null';

theMosaic.emGenSequence(emPathLength,'nTrials',nTrialsNum,'rSeed',randi(1e9,1));
emPositions = theMosaic.emPositions;
coneExcitationsSignal = theMosaic.compute(theSignalOI,'emPath',emPositions);

%{
% theMosaic.window;
% theMosaic.plot('Eye Movement Path');
%}

%% END
