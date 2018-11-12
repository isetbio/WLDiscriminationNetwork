function logLikelihood =  ieLogLikelihood(samples, param, varargin)
% Calculate logLikelihood (LLE) given samples and distribution parameters
%
% Syntax
%  logLikelihood =  ieLogLikelihood(samples, param, varargin)
%
% Brief description
%   Given a probability density function we calculate a log likelihood for
%   a vector of samples values, x, and the parameters of the distribution. 
%   
%      sum(log(pdens(x|dparam)))
%
%   The likelihood is the product across an array of sample values each
%   with its own parameter.  The log likelihood is the sum of the logs of
%   the prob densities.
%
% Inputs:
%   samples  - Vector of sample values
%   param    - Vector of distribution parameters
%              For Poisson this is a vector of means.  For Normal this is
%              param.mean(:) and param.sd(:)
%
% Optional key/value pairs
%   pdistribution - {'Poisson'}
%
% Output:
%   likelihood
%
% Wandell
%
% See also

% Examples:
%{
% This is about as good as it gets.  The samples each hit the mean
clear param
param.mean = 1:10;   % Signal known exactly
samples    = 1:10;   % Sample measurements
ieLogLikelihood(samples,param)  % The likelihood of observing this sample
%}
%{
% Notice the LogLikelihood is much smaller if we flip the means
clear param
param.mean = 1:10;          % Signal known exactly
samples    = fliplr(1:10);  % The sample
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

%% Parse parameters
p = inputParser;
p.addRequired('samples',@isvector);
p.addRequired('param',@(x)(isvector(param) || isstruct(param)));

vFunc = @(x)(ismember(x,{'poisson','normal'}));
p.addParameter('distribution','poisson',vFunc)

p.parse(samples,param,varargin{:});

pdistribution = p.Results.distribution;

%%
switch lower(pdistribution)
    case 'poisson'
        % Poisson has only one parameter
        logLikelihood = sum(log(poisspdf(samples,param.mean)));
    case 'normal'
        logLikelihood = sum(log(normpdf(samples,param.mean,param.sd)));
    otherwise
        error('%s distribution not yet implemented\n',pdistribution);
end


end
