% This function preprocesses raw fNIRS intensity (interpolation, optical density, TDDR, wavelet filtering), rejects the
% windows of each channel with SCI or PSP below threshold (QT-NIRS) and converts the data to hemoglobin
%
% Inputs
% raw: nirs.core.Data array of raw intensity
% link_groups: links of each participant in raw (cell of link indices)
% sci_th: SCI threshold
% psp_th: PSP threshold
% window_sec: window length (sec, default 5)
% max_missing: maximum fraction of missing samples for interpolation (default 0.05)
%
% Outputs
% hb: hemoglobin (nirs.core.Data array)
% hb_rej: hemoglobin with the samples of the rejected windows set to NaN
% QC: SCI, PSP and retained windows of each recording and participant
%
% Written by Pangyu Joo 10/6/2026

function [hb, hb_rej, QC] = preprocess_window_qc(raw, link_groups, sci_th, psp_th, window_sec, max_missing)
fcut = [0.5, 2.5]; % cardiac band of SCI and PSP (Hz)
%% parameters
if ~exist('window_sec','var')
    window_sec = 5; % window length (sec)
end
if ~exist('max_missing','var')
    max_missing = 0.05; % maximum fraction of missing samples for interpolation
end

%% missing samples
for i = 1:numel(raw)
    missing = mean(isnan(raw(i).data), 1); % fraction of missing samples of each link
    fill = missing > 0 & missing < max_missing;
    raw(i).data(:, fill) = fillmissing(raw(i).data(:, fill), 'spline');
    raw(i).data(:, missing >= max_missing) = NaN;
end

%% optical density and artifact correction
od = nirs.modules.OpticalDensity().run(raw);
wavelet = nirs.modules.WaveletFilter();
wavelet.sthresh = 3; % threshold of the wavelet coefficients (robust SD)
for i = 1:numel(od)
    ok = ~any(isnan(od(i).data), 1); % links without missing data
    [U, S, V] = svd(od(i).data(:, ok), 'econ'); % temporal principal components
    od(i).data(:, ok) = nirs.math.tddr(U, od(i).Fs, false) * S * V'; % TDDR of the components
    tmp = od(i);
    tmp.data = od(i).data(:, ok);
    tmp = wavelet.run(tmp); % wavelet filtering
    od(i).data(:, ok) = tmp.data;
end

%% hemoglobin
hb = nirs.modules.BeerLambertLaw().run(od);

%% window quality
QC = struct();
hb_rej = hb;
for i = 1:numel(od)
    n_samples = size(hb(i).data, 1);
    for p = 1:numel(link_groups)
        od_p = od(i);
        od_p.data = od_p.data(:, link_groups{p});
        od_p.probe.link = od_p.probe.link(link_groups{p}, :);

        % SCI, PSP and their combined decision of every channel-window
        qm = qtnirs(od_p, 'freqCut', fcut, 'window', window_sec, 'overlap', 0, 'sciThreshold', sci_th, ...
            'pspThreshold', psp_th, 'conditionsMask', 'resting', 'dodFlag', 0, 'guiFlag', 0);
        Q = struct();
        Q.link = array2table(qm.good_combo_link(:, 1:2), 'VariableNames', {'source', 'detector'});
        Q.sci = qm.sci_array;
        Q.psp = qm.power_array;
        Q.good = qm.combo_array;
        Q.missing = false(height(Q.link), 1);
        Q.window_samples = qm.sampPerWindow;
        Q.n_windows = qm.n_windows;
        Q.sci_th = sci_th;
        Q.psp_th = psp_th;
        Q.window_sec = window_sec;

        % samples of the rejected windows set to NaN for both chromophores
        for c = 1:height(Q.link)
            ln = od_p.probe.link.source == Q.link.source(c) & od_p.probe.link.detector == Q.link.detector(c);
            Q.missing(c) = any(isnan(od_p.data(:, ln)), 'all');
            keep = false(n_samples, 1);
            keep(1:Q.n_windows * Q.window_samples) = repelem(Q.good(c, :)', Q.window_samples);
            ch = hb(i).probe.link.source == Q.link.source(c) & hb(i).probe.link.detector == Q.link.detector(c);
            hb_rej(i).data(~keep, ch) = NaN;
        end
        QC(i, 1).participant(p) = Q;
    end
end
end
