function [fractionCorrect,dPrime] = iePoissonIdealObserver(alphaMean,betaMean)
% Ideal observer fraction correct in a TAFC two stimuli with Poisson noise
%
% Syntax:
%   fractionCorrect = iePoissonIdealObserver(alphaMeanResponses,betaMeanResponses)
%
% Brief description:
%  The fraction correct in a TAFC task using the formula for a Poisson
%  ideal observer developed by Geisler, 1984, JOSA A, 1, pp. 775 ff.
%
%  The Geisler formula returns dPrime and this routine then converts to
%  TAFC percent correct by numerically integrating the area under the ROC
%  curve for that dPrime, assuming normal distributions with equal variance
%  in the conversion.
%
%  The two input arguments should be vectors of the same length. Each
%  represents the mean of the Poisson distributed responses for the two
%  stimulus types.
%
% Author: NC, ISETBIO Team
%
% See also:
%   ieCorrectFromDprime

%% When the mean responses are identical, we return chance.
if (all(alphaMean == betaMean))
    fractionCorrect = 0.5;
    dPrime = 0;
    return;
end

%% In general we need only consider locations means differ
%
index = find(alphaMean ~= betaMean);

% This comes from the appendix of the paper
numerator = sum( (betaMean(index)-alphaMean(index)).*log(betaMean(index)./alphaMean(index)) );
denominator = 0.5*sum( (betaMean(index)+alphaMean(index)).*(log(betaMean(index)./alphaMean(index)).^2) );
dPrime = numerator / sqrt(denominator);

% Call into our dPrime to TAFC conversion function
fractionCorrect = dPrimeToTAFCFractionCorrect(dPrime);

if (isnan(dPrime) || isnan(fractionCorrect))
    error('Should not get NaN here.  Probably there are some zero mean response cones');
end
