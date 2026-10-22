% This function simulates one recording at a false-effect level and returns the Z-scores and true effects of no
% removal, channel-level removal and window-level removal of the bad windows
%
% Inputs
% tp: template (sim_template)
% false_lv: fraction of channel-windows with false effects
% dil_sd: SD of the dilution noise (x true response)
% false_size: amplitude of the false effect (x true response)
% window_sec: window length (sec)
% qt_quality: clean-window fraction of the channel-level removal
% highpass: high-pass cut-off (Hz)
% ar_order_sec: maximum AR model order (sec)
% keep_example: return the example traces
%
% Outputs
% o: Z-scores and true effects of each strategy (z_act, t_act, z_con, t_con), retained data (used), example traces
%   (keep_example)
%
% Written by Pangyu Joo 10/6/2026

function [o] = sim_false_effect_recording(tp, false_lv, dil_sd, false_size, window_sec, qt_quality, highpass, ar_order_sec, keep_example)
t1 = tp.h.time;
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
resp = tp.X * active;
brain = randn(N1, n_ch);
shared = zeros(N1, n_ch);
in_conn = any(tp.sec_ind(:, conn_sec), 2); % samples of the connected sections
for p = find(conn_match)'
    common = randn(N1, 1) .* in_conn;
    shared(:, match(p, :)) = shared(:, match(p, :)) + common;
end

spw = round(window_sec * tp.h.Fs); % samples per window
win_i = floor((0:N1 - 1)' / spw) + 1; % window of each sample
n_w = win_i(end);

%% dilution: stationary background noise
dil_noise = dil_sd * randn(N1, n_ch);

%% false effect: canonical HRF waveform from the onset of a window of an optode
p_opt = 1 - sqrt(1 - false_lv); % optode probability for the channel fraction false_lv
fo = rand(n_w, tp.n_opt) < p_opt;
hrf = nirs.design.basis.Canonical(); % the canonical HRF of the true responses and the GLM
kernel = nirs.design.basis.Canonical.getImpulseResponse(hrf.peakTime, hrf.peakDisp, hrf.uShootTime, hrf.uShootDisp, ...
    hrf.ratio, (0:1 / tp.h.Fs:hrf.duration)');
kernel = kernel / max(abs(kernel));
false_opt = zeros(N1 + numel(kernel), tp.n_opt);
for w = 1:n_w
    for opt = 1:tp.n_opt
        if fo(w, opt)
            idx = (w - 1) * spw + (1:numel(kernel));
            false_opt(idx, opt) = false_opt(idx, opt) + false_size * sign(randn()) * kernel;
        end
    end
end
false_opt = false_opt(1:N1, :);
fal = fo(:, tp.opt(:, 1)) | fo(:, tp.opt(:, 2)); % bad windows (window x channel)

%% measured signal: true effects removed in the bad windows
false_signal = false_opt(:, tp.opt(:, 1)) + false_opt(:, tp.opt(:, 2)); % false effect of each channel
true_signal = resp + shared;
true_signal(fal(win_i, :)) = 0;
y = true_signal + brain + dil_noise + false_signal;

%% strategies: no removal, channel-level removal, window-level removal
good = ~fal;
V = {true(N1, n_ch), repmat(mean(good, 1) >= qt_quality, N1, 1), good(win_i, :)};
[z_act, z_con] = sim_zscores(tp, y, V, highpass, ar_order_sec);
o.z_act = single(z_act);
o.t_act = active(:);
o.z_con = single(z_con);
o.t_con = t_con;
o.used = cellfun(@(v) mean(v(:)), V);
if keep_example
    o.label = double(fal)';
    o.t_win = t1(1) + ((1:n_w) - 0.5) * window_sec;
    cand = find(any(active, 1));
    [~, i_k] = min(abs(mean(good(:, cand), 1) - 0.75)); % active channel with a clean fraction closest to 75%
    o.ex_k = cand(i_k);
    o.ex_true = resp(:, o.ex_k);
    o.ex_meas = y(:, o.ex_k);
    o.t1 = t1;
end
end
