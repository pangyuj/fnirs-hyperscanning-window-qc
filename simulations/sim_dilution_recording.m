% This function simulates one recording with dilution noise spread across channels and returns the Z-scores and true
% effects at every noise scale and rejection threshold
%
% Inputs
% tp: template (sim_template)
% max_noise: largest SD of the dilution noise across channels (x true response)
% noise_scales: scales of the dilution noise
% thresholds: rejection thresholds (maximum allowed noise SD)
% highpass: high-pass cut-off (Hz)
% ar_order_sec: maximum AR model order (sec)
% keep_example: return the example traces
%
% Outputs
% o: Z-scores and true effects (z_act, t_act, z_con, t_con), retained channels (used), noise SD of each channel
%   (ch_noise_levels), example traces (keep_example)
%
% Written by Pangyu Joo 10/6/2026

function [o] = sim_dilution_recording(tp, max_noise, noise_scales, thresholds, highpass, ar_order_sec, keep_example)
[N1, n_ch] = size(tp.h.data);
%% true effects: active condition x channel, connected pairs and sections, responses and fluctuations
active = rand(size(tp.X, 2), n_ch) < 0.5;
match = reshape(randperm(n_ch, 2 * floor(n_ch / 2)), 2, [])'; % random channel pairs
conn_match = rand(size(match, 1), 1) < 0.5; % connected pairs
conn_sec = rand(size(tp.sec_ind, 2), 1) < 0.5; % connected sections
T_pair = false(n_ch);
T_pair(sub2ind([n_ch, n_ch], match(conn_match, 1), match(conn_match, 2))) = true;
T_pair = T_pair | T_pair';
t_con = reshape(T_pair(triu(true(n_ch), 1)) & conn_sec', [], 1); % true connectivity of each pair x section
brain = randn(N1, n_ch);
true_signal = tp.X * active;
in_conn = any(tp.sec_ind(:, conn_sec), 2); % samples of the connected sections
for p = find(conn_match)'
    common = randn(N1, 1) .* in_conn;
    true_signal(:, match(p, :)) = true_signal(:, match(p, :)) + common;
end

%% dilution noise SD spread uniformly across channels from 0 to max_noise
ch_noise_levels = linspace(0, max_noise, n_ch);
ch_noise_levels = ch_noise_levels(randperm(n_ch));
base_dil_noise = randn(N1, n_ch) .* ch_noise_levels;

%% Z-scores of every noise scale and threshold
n_level = numel(noise_scales);
o.z_act = zeros(size(tp.X, 2) * n_ch, numel(thresholds), n_level, 'single');
o.z_con = zeros(numel(t_con), numel(thresholds), n_level, 'single');
o.used = zeros(n_level, numel(thresholds));
if keep_example
    o.y = zeros(N1, n_ch, n_level);
end
for level_i = 1:n_level
    y = true_signal + brain + noise_scales(level_i) * base_dil_noise;
    keep_ch = arrayfun(@(th) noise_scales(level_i) * ch_noise_levels <= th, thresholds, 'UniformOutput', false); % retained channels
    V = cellfun(@(k) repmat(k, N1, 1), keep_ch, 'UniformOutput', false);
    [z_act, z_con] = sim_zscores(tp, y, V, highpass, ar_order_sec);
    o.z_act(:, :, level_i) = z_act;
    o.z_con(:, :, level_i) = z_con;
    o.used(level_i, :) = cellfun(@mean, keep_ch);
    if keep_example
        o.y(:, :, level_i) = y;
    end
end
o.t_act = active(:);
o.t_con = t_con;
o.ch_noise_levels = ch_noise_levels;
if keep_example
    o.t1 = tp.h.time;
    o.true_signal = true_signal;
end
end
