% This function generates dummy raw-intensity recordings (task and resting state) of one dyad with known activation,
% connectivity and false-effect windows (model of the validation simulations with a cardiac pulsation)
%
% Inputs
% layout: montage of one participant (srcPos, detPos in mm, link)
% dil_sd: SD of the dilution noise (scalar, or one value per channel of the dyad)
% false_frac: fraction of channel-windows with a false effect
% cardiac_amp: amplitude of the cardiac pulsation (x true response)
% truth_in: known states of another dyad to reuse (empty for new states)
%
% Outputs
% task, rest: nirs.core.Data of raw intensity (participant 1, then participant 2)
% truth: known states (conds, active, pairs, conn_conds, false_win)
%
% Written by Pangyu Joo 10/6/2026

function [task, rest, truth] = make_dummy_dyad(layout, dil_sd, false_frac, cardiac_amp, truth_in)
lambda = [760; 850]; % wavelengths of the dummy device (nm)
false_size = 2; % amplitude of the false effect (x true response)
window_sec = 5; % window of the quality assessment (sec)
Fs = 25; % sampling rate (Hz)
rest_dur = 300; % resting-state duration (sec)
%% task design
names = {'Examination', 'Diagnostic_reasoning', 'Treatment_planning', 'Resource_utilization', 'Additional_comments', ...
    'Highlight', 'Look_at_peer_giver', 'Look_at_peer_receiver'};
sec_dur = [90, 120, 85, 20, 15] .* (0.8 + 0.4 * rand(1, 5)); % section durations (sec)
sec_on = 20 + [0, cumsum(sec_dur(1:end - 1) + 15)]; % section onsets (sec)
span = [sec_on(1), sec_on(end) + sec_dur(end)]; % task span (sec)
n_ev = [3, 40, 20]; % events of the event conditions
onset = [num2cell(sec_on), arrayfun(@(n) sort(span(1) + diff(span) * rand(n, 1)), n_ev, 'UniformOutput', false)];
dur = [num2cell(sec_dur), {5 + 5 * rand(n_ev(1), 1), 0.3 + 2.7 * rand(n_ev(2), 1), 0.3 + 2.7 * rand(n_ev(3), 1)}];
stim = Dictionary();
for k = 1:numel(names)
    st = nirs.design.StimulusEvents();
    st.name = names{k};
    st.onset = onset{k}(:);
    st.dur = dur{k}(:);
    st.amp = ones(numel(onset{k}), 1);
    stim(names{k}) = st;
end

%% probe of the dyad
n_src = size(layout.srcPos, 1); % sources of a participant
n_det = size(layout.detPos, 1); % detectors of a participant
n_ch = 2 * height(layout.link); % channels of the dyad
src = [layout.link.source; layout.link.source + n_src]; % source of each channel
det = [layout.link.detector; layout.link.detector + n_det]; % detector of each channel
link = table(repelem(src, 2), repelem(det, 2), repmat(lambda, n_ch, 1), 'VariableNames', {'source', 'detector', 'type'});
probe = nirs.core.Probe([layout.srcPos; layout.srcPos], [layout.detPos; layout.detPos], link);
opt = [src, 2 * n_src + det]; % optodes (source, detector) of each channel
n_opt = 2 * (n_src + n_det); % optodes of the dyad
part = [ones(n_ch / 2, 1); 2 * ones(n_ch / 2, 1)]; % participant of each channel
dil_sd = dil_sd(:)' .* ones(1, n_ch); % dilution SD of each channel

% modified Beer-Lambert law of each channel: OD = [HbO, HbR] x EL'
E = nirs.media.getspectra(lambda); % extinction coefficients
L = probe.distances(1:2:end); % source-detector distance of each channel
EL = cell(n_ch, 1);
for c = 1:n_ch
    EL{c} = E(:, 1:2) * L(c) * nirs.modules.BeerLambertLaw().PPF / 1e6; % OD per uM of HbO and HbR
end

%% task recording
conds = stim.keys; % condition names
t_end = 0;
for k = 1:numel(conds)
    st = stim(conds{k});
    t_end = max([t_end; st.onset + st.dur]); % end of the last event
end
t = (0:1 / Fs:ceil(t_end) + 30)'; % time vector (30 s after the last event)
basis = Dictionary();
basis('default') = nirs.design.basis.Canonical(); % canonical HRF
[X, names] = nirs.design.createDesignMatrix(stim, t, basis);
[~, ord] = ismember(conds, names);
X = X(:, ord) ./ max(abs(X(:, ord)), [], 1); % response of each condition (peak 1)

% true effects
if isempty(truth_in)
    active = rand(numel(conds), n_ch) < 0.5; % active condition x channel
    match = reshape(randperm(n_ch), 2, [])'; % random channel pairs of the dyad
    pairs = match(rand(size(match, 1), 1) < 0.5, :); % connected pairs
    conn_cond = rand(numel(conds), 1) < 0.5; % connected conditions
else
    active = truth_in.active;
    pairs = truth_in.pairs;
    conn_cond = ismember(conds, truth_in.conn_conds);
end
in_conn = false(size(t)); % samples of the connected conditions
for k = find(conn_cond)'
    st = stim(conds{k});
    for e = 1:numel(st.onset)
        in_conn = in_conn | (t >= st.onset(e) & t < st.onset(e) + st.dur(e));
    end
end
t1 = (0:ceil(t(end)))'; % 1-Hz time base of the white fluctuations
shared = zeros(numel(t), n_ch); % common fluctuation of the connected pairs (white at 1 Hz, SD 1)
for p = 1:size(pairs, 1)
    common = interp1(t1, randn(numel(t1), 1), t, 'spline') .* in_conn;
    shared(:, pairs(p, :)) = shared(:, pairs(p, :)) + common;
end
truth.conds = conds;
truth.active = active;
truth.pairs = pairs;
truth.conn_conds = conds(conn_cond);

%% measured recordings: brain fluctuation, false effects, cardiac pulsation and dilution noise
hrf = nirs.design.basis.Canonical();
kernel = nirs.design.basis.Canonical.getImpulseResponse(hrf.peakTime, hrf.peakDisp, hrf.uShootTime, ...
    hrf.uShootDisp, hrf.ratio, (0:1 / Fs:hrf.duration)');
kernel = kernel / max(abs(kernel)); % false-effect waveform (peak 1)
t_rec = {t, (0:1 / Fs:rest_dur)'}; % time vectors of the task and resting-state recordings
true_rec = {X * active + shared, zeros(numel(t_rec{2}), n_ch)}; % true effects of HbO
spw = round(window_sec * Fs); % samples per window
for r = 1:2
    t = t_rec{r};
    N = numel(t);

    % false effects: canonical HRF waveform from the onset of a window of an optode
    win_i = floor((0:N - 1)' / spw) + 1; % window of each sample
    fo = rand(win_i(end), n_opt) < 1 - sqrt(1 - false_frac); % onsets (window x optode) for the channel fraction false_frac
    false_opt = zeros(N + numel(kernel), n_opt); % false effect of each optode
    [w_on, o_on] = find(fo);
    for e = 1:numel(w_on)
        idx = (w_on(e) - 1) * spw + (1:numel(kernel));
        false_opt(idx, o_on(e)) = false_opt(idx, o_on(e)) + false_size * sign(randn) * kernel;
    end
    false_win = fo(:, opt(:, 1)) | fo(:, opt(:, 2)); % false-effect windows of each channel
    decoupled = false_win(win_i, :); % samples of the decoupled windows

    % hemodynamic components of HbO: true effects lost in the decoupled windows, brain fluctuation (white at 1 Hz)
    hemo = true_rec{r};
    hemo(decoupled) = 0;
    t1 = (0:ceil(t(end)))';
    hemo = hemo + interp1(t1, randn(numel(t1), n_ch), t, 'spline') + false_opt(1:N, opt(:, 1)) + false_opt(1:N, opt(:, 2));

    % cardiac pulsation of each participant, lost in the decoupled windows
    cardiac = zeros(N, n_ch);
    for p = 1:2
        phase = cumsum(0.1 * randn(N, 1) * 2 * pi / Fs); % drifting phase
        cardiac(:, part == p) = repmat(sin(2 * pi * (1 + 0.1 * randn) * t + phase), 1, sum(part == p));
    end
    cardiac(decoupled) = 0;

    % optical density with the dilution noise of each wavelength, and raw intensity
    OD = zeros(N, 2 * n_ch);
    for c = 1:n_ch
        iEL = pinv(EL{c}); % OD to uM
        sd_od = dil_sd(c) * sqrt(Fs) / norm(iEL(1, :)); % OD noise giving the HbO SD dil_sd at 1 Hz
        OD(:, 2 * c - [1, 0]) = [hemo(:, c) + cardiac_amp * cardiac(:, c), -hemo(:, c) / 2] * EL{c}' + sd_od * randn(N, 2);
    end
    rec(r) = nirs.core.Data();
    rec(r).data = 100 * exp(-OD);
    rec(r).time = t;
    rec(r).probe = probe;
    if r == 1
        truth.false_win = false_win;
    end
end
task = rec(1);
task.stimulus = stim;
task.description = 'dummy task recording';
rest = rec(2);
rest.description = 'dummy resting-state recording';
end
