% This function computes the activation (ar_irls_gap) and connectivity (robust_corr_pairwise) Z-scores of one simulated
% recording for several masks of retained samples (Z = 0 without a result)
%
% Inputs
% tp: template (sim_template)
% y: simulated HbO (time x channel)
% V: masks of the retained samples (cell of time x channel logical)
% highpass: high-pass cut-off (Hz)
% ar_order_sec: maximum AR model order (sec)
%
% Outputs
% z_act: Z of each condition x channel x mask
% z_con: Z of each pair x section x mask
%
% Written by Pangyu Joo 10/6/2026

function [z_act, z_con] = sim_zscores(tp, y, V, highpass, ar_order_sec)
h1 = tp.h;
h1.data = y;
n_ch = size(y, 2);
n_cond = size(tp.X, 2);
trend_fun = @(t) nirs.design.trend.dctmtx(t, highpass);

%% activation
z_act = zeros(n_cond * n_ch, numel(V));
for s = 1:numel(V)
    hs = h1;
    hs.data(:, ~any(V{s}, 1)) = NaN; % removed channels
    S = ar_irls_gap(hs, trend_fun, ar_order_sec, V{s});
    Z = zeros(n_cond, n_ch);
    for k = 1:n_ch
        rows = find(S.variables.source == k);
        [~, o_c] = ismember(tp.conds, S.variables.cond(rows));
        z = -norminv(max(S.p(rows(o_c)), eps) / 2) .* sign(S.beta(rows(o_c)));
        z(isnan(z)) = 0;
        Z(:, k) = z;
    end
    z_act(:, s) = Z(:);
end

%% connectivity
hp = eeg.modules.BandPassFilter();
hp.do_downsample = 0;
hp.lowpass = [];
hp.highpass = highpass;
hc = hp.run(h1);
[yw, f] = nirs.math.innovations(hc.data, ceil(ar_order_sec * hc.Fs)); % prewhitened on the continuous series
yw(1:max(cellfun(@length, f)), :) = NaN; % start-up samples of the whitening filter
lin = find(triu(true(n_ch), 1));
n_p = numel(lin);
n_sec = size(tp.sec_ind, 2);
z_con = zeros(n_sec * n_p, numel(V));
for s = 1:numel(V)
    ys = yw;
    ys(~V{s}) = NaN;
    for k = 1:n_sec
        [Rk, Pk] = robust_corr_pairwise(ys(tp.sec_ind(:, k), :));
        z = -norminv(max(Pk(lin), eps) / 2) .* sign(Rk(lin));
        z(isnan(z)) = 0;
        z_con((k - 1) * n_p + (1:n_p), s) = z;
    end
end
end
