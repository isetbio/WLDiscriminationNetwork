% script to plot CSF function using the accuracy data from experiments
% uses data SVM accuracy.mat

%% loading data
chdir(fullfile(cs231nRootPath,'data'))
load('SVMaccuracy.mat')

% contrast range - scanContrast
% frequency range - scanFreq
svmAccuracy = accuracy;

% plotting data
figure()
legend_name = [];
for iterSp = 1:length(scanFreq)
    semilogx(scanContrast,svmAccuracy(iterSp,:),'-o')
    hold on
    legend_name = [legend_name;sprintf("spatial frequency = %f cpd",scanFreq(iterSp))]; 
end
legend(legend_name')
xlabel('Contrast (log)')
ylabel('Mean Accuracy')
%% find contrast corresponding to 75% accuracy 

refAccuracy = 0.75; % reference accuracy for CSF plot
csfValue = zeros(length(scanFreq),1);

for iterSp = 1:length(scanFreq)
    % using interpolation to find the accuracy at 75%
    [x,iy,~] = unique(svmAccuracy(iterSp,:));
    y = log(scanContrast(iy));
    fitValue = interp1(x,y,refAccuracy);
    csfValue(iterSp) = 1 / (10^fitValue);
    
    % plotting fit
    %{
    xq = 0.5:0.05:0.9;
    yq = interp1(x,y,xq);
    figure()
    plot(y,x,'-o')
    hold on
    plot(yq,xq,'-o')
    plot(fitValue,refAccuracy,'k*')
    legend('original data','fitted data','contrast at desired accuracy')
    %}  
end

% plotting contrast sensitivity function
figure()
loglog(scanFreq, csfValue,'k-o')
ylabel('contrast sensitivity')
xlabel('spatial frequency')

%% saving data
save('SVM_csf_curve','csfValue','scanFreq','refAccuracy')