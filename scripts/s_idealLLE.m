% Illustrate the log likelihood calculation
%
% This is the start of choosing the most likely of several different
% patterns.  We might set the orientation, phase, frequency of the
% harmonic, for example, and calculate the log likelihood of the data with
% respect to these different stimuli.
%
%


%% Calculate the ideal
fov = 5;
p = harmonicP;
p.freq = 5;
scene = sceneCreate('harmonic',p);
scene = sceneSet(scene,'fov',fov);

oi = oiCreate;
oi = oiCompute(oi,scene);

% This is the part where we calculate the noisy samples.
sensor = sensorCreate('monochrome');
sensor = sensorSetSizeToFOV(sensor,fov);
sensorIdeal = sensorSet(sensor,'noise flag',0);

eTime = 1e-3;
sensor = sensorSet(sensor,'exp time',eTime);
sensorIdeal = sensorSet(sensorIdeal,'exp time',eTime);

% This is the part where we calculate the true mean.
sensorIdeal = sensorCompute(sensorIdeal,oi);
sensorIdeal = sensorSet(sensorIdeal,'name','ideal');
ieAddObject(sensorIdeal); sensorWindow;

param.mean = sensorGet(sensorIdeal,'electrons');
param.mean = param.mean(:);

%% Now calculate something just a little different in frequency
fov = 5;
p = harmonicP;
p.freq = 5.001;   % Could loop on this to show log likelihood decline
scene = sceneCreate('harmonic',p);
scene = sceneSet(scene,'fov',fov);

oi = oiCreate;
oi = oiCompute(oi,scene);

sensor = sensorCompute(sensor,oi);
ieAddObject(sensor); sensorWindow;

%%
samples = sensorGet(sensor,'electrons');
samples = samples(:);

ieLogLikelihood(samples,param)

%%

% Examples:
%{
% This is about as good as it gets.  The samples each hit the mean
clear param
param.mean = 1:10;
samples    = 1:10;
ieLogLikelihood(samples,param)
%}
%{
% Notice the LogLikelihood is much smaller if we flip the means
clear param
param.mean = 1:10;
samples    = fliplr(1:10);
ieLogLikelihood(samples,param)
%}
%{
% The normal is pretty much like the Gaussian
clear param
samples    = 1:10;
param.mean = 1:10;
param.sd   = sqrt(param.mean);
ieLogLikelihood(samples,param,'distribution','normal')
%}
%{
clear param
samples    = 1:10;
param.mean = 1:10;
param.sd   = 3*(ones(size(param.mean)));
ieLogLikelihood(samples,param,'distribution','normal')
%}
%{
% For large values, the normal and poisson are pretty much identical
clear param
samples    = (1:10)*1e3;
param.mean = samples;
param.sd   = sqrt(samples);
ieLogLikelihood(samples,param,'distribution','normal')
%}
%{
clear param
samples    = (1:10)*1e3;
param.mean = samples;
ieLogLikelihood(samples,param,'distribution','poisson')
%}
