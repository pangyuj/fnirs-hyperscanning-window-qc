% This function calculates the robust correlation of the NIRS Brain AnalyzIR Toolbox (robust_corrcoef; Shevlyakov and
% Smirnov, 2011) for time series with missing samples, over the samples valid in both channels of each pair
%
% Inputs
% Y: time series (sample x channel), NaN for missing samples
%
% Outputs
% R: correlation matrix (channel x channel)
% P: two-sided p-values
% Zw: standardized and weighted samples (sample x channel, 0 for missing samples)
%
% Written by Pangyu Joo 9/28/2026

function [R, P, Zw] = robust_corr_pairwise(Y)
tune = 4.685; % tuning constant of the toolbox weight function
warning('off', 'stats:statrobustfit:IterationLimit'); % bisquare IRLS reaching its iteration limit

%% robust standardization of each channel
valid = ~isnan(Y);
Z = (Y - median(Y, 1, 'omitnan'))./(1.4826*mad(Y, 1, 1));

%% sample weights: smallest weight across channels (weight function of the toolbox)
U = Z./(mad(Z, 0, 1)/0.6745)/tune;
Wc = (1 - U.^2).*(abs(U) < 1);
Wc(~valid) = 1; % missing samples do not affect the weight of the sample
Wt = min(Wc, [], 2);
Zw = Z.*Wt;
Zw(~valid) = 0;

%% correlation of each pair from the robust regressions in both directions (as the toolbox)
n_ch = size(Y, 2);
n = double(valid)'*double(valid); % common samples of each pair
R = NaN(n_ch);
for i = 1:n_ch
    for j = i+1:n_ch
        s = valid(:,i) & valid(:,j);
        x = Zw(s,i);
        y = Zw(s,j);
        if n(i,j) < 3 || all(x == x(1)) || all(y == y(1))
            continue % fewer than 3 common samples or a constant series
        end
        b1 = robustfit(x, y, 'bisquare', [], 'off'); % slope of channel j on channel i
        b2 = robustfit(y, x, 'bisquare', [], 'off'); % slope of channel i on channel j
        R(i,j) = sign(b1 + b2)*sqrt(abs(b1*b2));
        R(j,i) = R(i,j);
    end
end
R = min(max(R, -1, 'includenan'), 1, 'includenan'); % |r| > 1 set to +-1, as in the toolbox
R(logical(eye(n_ch)) & n >= 3) = 1;

%% p-values
Tstat = R.*sqrt((n - 2)./(1 - R.^2));
P = 2*tcdf(-abs(Tstat), n - 2);
P(logical(eye(n_ch))) = 1;
end
