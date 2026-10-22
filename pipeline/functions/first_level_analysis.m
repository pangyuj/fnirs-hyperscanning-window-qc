% This function estimates the task activation (AR-IRLS GLM) and the task-versus-rest connectivity (robust correlation,
% resting-state null matched in the valid windows) of one recording, with the rejected samples excluded
%
% Inputs
% task, task_rej: task hemoglobin (nirs.core.Data, conditions as stimulus) without and with rejected samples (NaN)
% rest, rest_rej: the same for the resting-state recording
% n_perm: number of resting-state draws
% highpass: high-pass cut-off (Hz, default 0.01)
% fs_new: sampling rate of the analyses (Hz, default 1)
% ar_order_sec: maximum AR model order (sec, default 4)
% min_event_dur: minimum event duration (sec, default 1)
% window_sec: window length of the quality assessment (sec, default 5)
%
% Outputs
% R: activation (act_z, act_p, act_n: channel x condition), connectivity (conn_r, conn_z, conn_p, conn_sig, pair_n:
%   channel x channel x condition), conds, link
%
% Written by Pangyu Joo 10/6/2026

function [R] = first_level_analysis(task, task_rej, rest, rest_rej, n_perm, highpass, fs_new, ar_order_sec, min_event_dur, window_sec)
alpha = 0.05; % significance level (FDR q)
%% parameters
if ~exist('highpass','var')
    highpass = 0.01; % high-pass cut-off (Hz)
end
if ~exist('fs_new','var')
    fs_new = 1; % sampling rate of the analyses (Hz)
end
if ~exist('ar_order_sec','var')
    ar_order_sec = 4; % maximum AR model order (sec)
end
if ~exist('min_event_dur','var')
    min_event_dur = 1; % minimum event duration (sec)
end
if ~exist('window_sec','var')
    window_sec = 5; % window length of the quality assessment (sec)
end

%% data
task_bad = isnan(task_rej.data); % rejected samples of the task recording
rest_bad = isnan(rest_rej.data); % rejected samples of the resting-state recording
conds = task.stimulus.keys; % conditions
n_cond = numel(conds);
n_ch = height(task.probe.link);
stim = Dictionary(); % conditions with the minimum event duration
for k = 1:n_cond
    st = task.stimulus(conds{k});
    st.dur = max(st.dur, min_event_dur);
    stim(conds{k}) = st;
end

hp = eeg.modules.BandPassFilter(); % high-pass filter
hp.do_downsample = 0;
hp.lowpass = [];
hp.highpass = highpass;
rs = nirs.modules.Resample(); % resampling
rs.Fs = fs_new;

%% activation
glm = rs.run(task); % without high-pass filtering (drift terms in the design matrix)
glm.stimulus = stim;
valid = ~map_mask(task.time, task_bad, glm.time); % samples entering the regression
S = ar_irls_gap(glm, @(t) nirs.design.trend.dctmtx(t, highpass), ar_order_sec, valid);

act_z = NaN(n_ch, n_cond);
act_p = NaN(n_ch, n_cond);
act_n = zeros(n_ch, n_cond);
for k = 1:n_cond
    % retained samples of each channel in the segments of the condition
    st = stim(conds{k});
    in_seg = false(size(task.time));
    for e = 1:numel(st.onset)
        in_seg = in_seg | (task.time >= st.onset(e) & task.time < st.onset(e) + st.dur(e));
    end
    n_ret = sum(~task_bad(in_seg, :), 1)';

    % Z of the condition against the implicit baseline
    T = S.ttest(conds{k});
    [~, row] = ismember(task.probe.link(:, {'source', 'detector', 'type'}), T.variables(:, {'source', 'detector', 'type'}));
    p = T.p(row);
    ok = n_ret > 0 & ~isnan(p); % channels with a result
    act_z(ok, k) = -norminv(max(p(ok), eps) / 2) .* sign(T.tstat(row(ok)));
    act_p(ok, k) = p(ok);
    act_n(ok, k) = n_ret(ok);
end

%% connectivity
W = [rs.run(hp.run(task)), rs.run(hp.run(rest))];
for i = 1:2
    Y = W(i).data;
    ok = ~any(isnan(Y), 1); % channels without missing data
    Yw = NaN(size(Y));
    [Yw(:, ok), filt] = nirs.math.innovations(Y(:, ok), ceil(ar_order_sec * W(i).Fs)); % prewhitened on the continuous data
    n0 = max(cellfun(@length, filt)); % start-up samples of the longest whitening filter
    W(i).data = Yw(n0 + 1:end, :);
    W(i).time = W(i).time(n0 + 1:end);
end
task_w = W(1);
rest_w = W(2);
task_w.data(map_mask(task.time, task_bad, task_w.time)) = NaN; % rejected samples on the new time base
rest_w.data(map_mask(rest.time, rest_bad, rest_w.time)) = NaN;

% resting-state windows: rows of the samples valid in both channels of each pair, in time order (the first L of them
% replace a task window with L valid samples); samples standardized and weighted over the recording (Zr)
[~, ~, chrom] = unique(task.probe.link.type); % chromophore of each channel
[ch1, ch2] = find(triu(chrom == chrom', 1)); % channel pairs of the same chromophore
n_p = numel(ch1);
lin = [sub2ind([n_ch, n_ch], ch1, ch2), sub2ind([n_ch, n_ch], ch2, ch1)]; % positions of the pairs in the matrices
[~, ~, Zr] = robust_corr_pairwise(rest_w.data);
[~, ~, rest_win] = unique(floor((rest_w.time - rest.time(1)) / window_sec)); % window of each sample
n_w = max(rest_win);
L_max = max(accumarray(rest_win, 1));
c_rest = zeros(n_w, n_p); % valid samples of each window
rest_rows = zeros(L_max, n_w, n_p); % rows of these samples
for w = 1:n_w
    rows_w = find(rest_win == w);
    v = ~isnan(rest_w.data(rows_w, :));
    v = v(:, ch1) & v(:, ch2);
    c_rest(w, :) = sum(v, 1);
    [i_r, i_p] = find(v);
    n_v = cumsum(v, 1);
    k = n_v(v); % rank of each valid sample in its window
    rest_rows(sub2ind([L_max, n_w, n_p], k(:), w * ones(numel(k), 1), i_p(:))) = rows_w(i_r(:));
end
n_elig = zeros(L_max, n_p); % windows with at least L valid samples
elig = cell(L_max, 1); % their indices, listed first in each column
for L = 1:L_max
    n_elig(L, :) = sum(c_rest >= L, 1);
    [~, elig{L}] = sort(c_rest < L, 1);
end

conn_r = NaN(n_ch, n_ch, n_cond);
conn_z = NaN(n_ch, n_ch, n_cond);
conn_p = NaN(n_ch, n_ch, n_cond);
conn_sig = false(n_ch, n_ch, n_cond);
pair_n = zeros(n_ch, n_ch, n_cond);
for k = 1:n_cond
    % concatenated segments of the condition and the window of the quality assessment of each sample
    st = stim(conds{k});
    seg = cell(numel(st.onset), 1);
    win = cell(numel(st.onset), 1);
    for e = 1:numel(st.onset)
        [~, i0] = min(abs(task_w.time - st.onset(e)));
        i1 = min(i0 + round(st.dur(e) * fs_new) - 1, size(task_w.data, 1));
        seg{e} = task_w.data(i0:i1, :);
        win{e} = floor((task_w.time(i0:i1) - task.time(1)) / window_sec);
    end
    Y = cat(1, seg{:});
    [~, ~, win] = unique(cat(1, win{:})); % window of each sample
    missing = isnan(Y);
    pair_n(:, :, k) = double(~missing)' * double(~missing); % samples valid in both channels
    r_obs = robust_corr_pairwise(Y);
    r_obs_p = r_obs(lin(:, 1))';
    z_obs = atanh(min(max(r_obs_p, -1 + eps, 'includenan'), 1 - eps, 'includenan')); % Fisher z
    z_ref = z_obs;
    z_ref(isnan(z_ref)) = 0; % reference of the null sums (numerical stability)

    % valid samples of each pair in each window of the task segments, number of windows with L valid samples, and pairs
    % matched by the resting state: at least as many windows with >= L valid samples as the task segments (every L),
    % and at least 1/alpha distinct draws of the resting-state windows
    C = zeros(max(win), n_p);
    for g = 1:max(win)
        v = ~missing(win == g, :);
        C(g, :) = sum(v(:, ch1) & v(:, ch2), 1);
    end
    q = zeros(L_max, n_p);
    for L = 1:L_max
        q(L, :) = sum(C == L, 1);
    end
    n_s = (1:L_max) * q; % valid samples of each pair
    log_draws = gammaln(n_elig + q) - gammaln(q + 1) - gammaln(n_elig); % distinct draws of the windows of each L (log)
    log_draws(q == 0) = 0;
    matched = all(n_elig >= cumsum(q, 1, 'reverse'), 1) & sum(log_draws, 1) >= log(1 / alpha) & all(C <= L_max, 1) & ~isnan(r_obs_p);

    % null distribution: each window of the task segments with L valid samples replaced by the first L valid samples of
    % a random resting-state window with at least L valid samples of the pair; robust correlation of the toolbox
    % (robust_corr_pairwise) on these samples
    n_def = zeros(1, n_p); % draws with a defined correlation
    s1 = zeros(1, n_p); % sum of the null z (relative to z_ref)
    s2 = zeros(1, n_p); % sum of squares
    parfor b = 1:n_perm
        warning('off', 'stats:statrobustfit:IterationLimit'); % bisquare IRLS reaching its iteration limit
        r_null = NaN(1, n_p);
        for j = find(matched)
            rows = zeros(n_s(j), 1); % resting-state samples of the draw
            pos = 0;
            for L = find(q(:, j))'
                w = elig{L}(ceil(rand(q(L, j), 1) * n_elig(L, j)), j); % windows drawn among the eligible ones
                rows(pos + (1:L * q(L, j))) = reshape(rest_rows(1:L, w, j), [], 1);
                pos = pos + L * q(L, j);
            end
            x = Zr(rows, ch1(j));
            y = Zr(rows, ch2(j));
            if all(x == x(1)) || all(y == y(1))
                continue % draws of equal samples
            end
            b1 = robustfit(x, y, 'bisquare', [], 'off');
            b2 = robustfit(y, x, 'bisquare', [], 'off');
            r_null(j) = sign(b1 + b2) * sqrt(abs(b1 * b2));
        end
        r_null = min(max(r_null, -1, 'includenan'), 1, 'includenan'); % |r| > 1 set to +-1, as in the toolbox
        defined = ~isnan(r_null);
        d = atanh(min(max(r_null, -1 + eps), 1 - eps)) - z_ref;
        d(~defined) = 0;
        n_def = n_def + defined;
        s1 = s1 + d;
        s2 = s2 + d.^2;
    end
    mu = z_ref + s1 ./ n_def; % null mean
    sd = sqrt(max(s2 - s1.^2 ./ n_def, 0) ./ (n_def - 1)); % null SD
    z = (z_obs - mu) ./ sd; % task-versus-rest Z
    z(~matched) = NaN;
    p = 2 * (1 - normcdf(abs(z))); % two-sided p-value of the Z

    % FDR across the channel pairs of each chromophore in the recording
    sig = false(1, n_p);
    for c = 1:max(chrom)
        m = ~isnan(p) & chrom(ch1)' == c; % pairs of chromophore c with a Z
        sig(m) = mafdr(p(m)', 'BHFDR', true) < alpha;
    end
    conn_r(:, :, k) = r_obs;
    conn_z(lin + n_ch^2 * (k - 1)) = [z; z]';
    conn_p(lin + n_ch^2 * (k - 1)) = [p; p]';
    conn_sig(lin + n_ch^2 * (k - 1)) = [sig; sig]';
end

%% results
R.conds = conds;
R.link = task.probe.link;
R.act_z = act_z;
R.act_p = act_p;
R.act_n = act_n;
R.conn_r = conn_r;
R.conn_z = conn_z;
R.conn_p = conn_p;
R.conn_sig = conn_sig;
R.pair_n = pair_n;
end
%% functions
% This function maps rejected samples to a new time base: each rejected sample marks the nearest new sample (as
% nirs.modules.Resample restores NaN positions); samples outside the new time range are ignored
function bad_new = map_mask(t_old, bad_old, t_new)
bad_new = false(numel(t_new), size(bad_old, 2));
half_step = median(diff(t_new)) / 2;
for c = 1:size(bad_old, 2)
    ib = find(bad_old(:, c));
    idx = dsearchn(t_new(:), t_old(ib));
    near = abs(t_new(idx) - t_old(ib)) <= half_step;
    bad_new(unique(idx(near)), c) = true;
end
end
