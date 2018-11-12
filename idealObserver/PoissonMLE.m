function accuracy =  PoissonMLE(sample,sampleLabels, isoRate_c0, isoRate_c1)
% Calculate Poisson maximum likelihood estimate (MLE)
%
% Brief dand find if stimulus is present or not.
%
% Inputs: 
% sample    - the coneMosaic absorption used to be classified
% isoRate_c0, isoRate_c1  - the mean isomerization rate (distribution) 
%                   for the whole cones (w/ and w/o stimulus)
% sampleLabels - true labels for the data

% Output:
% accuracy of classification

    N = length(sampleLabels);
    timeInterval = 1;
    
    likelihood_c0 = zeros(N,1);
    likelihood_c1 = zeros(N,1);
    difference = zeros(N,1);
    
    for niter = 1:N
        % rounding to integer - it is not, but should be?
        curr_image = double(reshape(round(sample(:,:,niter)),[],1));

        % Finding log likelihood of cone response given no stimulus
        temp_c0 = poisspdf(curr_image,isoRate_c0*timeInterval);
        likelihood_c0(niter) = sum(log(temp_c0));

        % Finding log likelihood of cone response given no stimulus
        temp_c1 = poisspdf(curr_image,isoRate_c1*timeInterval);
        likelihood_c1(niter) = sum(log(temp_c1));

        % comparing likelihoods
        % if difference > 0 => class = stimulus
        % if difference < 0 => class = no stimulus
        difference(niter) = likelihood_c1(niter) - likelihood_c0(niter);
    end

    % finding prediction
    pred = (difference>0)
    accuracy = sum(pred == sampleLabels)/N;


end