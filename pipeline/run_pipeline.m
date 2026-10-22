% This script runs the analysis pipeline of the paper on dummy hyperscanning recordings and draws Figures 1C, 2 and 3.
% To analyze a study, replace the dummy recordings and the montage by those of the study
clear; clc; close all;
addpath('functions')

% set save path and generate folder if not exist
savepath = 'results/';
if ~exist(savepath, "dir"); mkdir(savepath); end

% montage of one participant: 20-channel bilateral prefrontal montage of the paper, channels sorted by source and
% detector (channel order of the NIRS Brain AnalyzIR Toolbox), order: display order of the channels (AF8-F8 to AF7-F7)
layout.src_names = {'AF7', 'Fpz', 'AF8', 'AF3', 'AF4', 'Fz'}; % source labels
layout.det_names = {'F7', 'Fp1', 'Fp2', 'F8', 'F5', 'AFz', 'F6', 'F1', 'F2'}; % detector labels
layout.srcPos = [-46, 55, -1; 0, 69, -4; 46, 55, -1; -23, 65, 22; 23, 65, 22; 0, 46, 49]; % source positions (MNI, mm)
layout.detPos = [-54, 37, 10; -24, 68, -6; 24, 68, -6; 54, 37, 10; -46, 52, 18; 0, 66, 24; 46, 52, 18; ...
    -22, 46, 44; 22, 46, 44]; % detector positions (MNI, mm)
source = [3; 3; 3; 5; 5; 5; 2; 5; 6; 2; 6; 2; 4; 6; 4; 4; 1; 4; 1; 1]; % sources of the channels in the display order
detector = [4; 7; 3; 7; 3; 9; 3; 6; 9; 6; 6; 2; 6; 8; 2; 8; 2; 5; 5; 1]; % detectors of the channels in the display order
[layout.link, sort_idx] = sortrows(table(source, detector), {'source', 'detector'}); % channels in the toolbox order
layout.names = strcat(layout.src_names(layout.link.source), '-', layout.det_names(layout.link.detector))'; % channel labels
[~, layout.order] = sort(sort_idx); % display order of the channels
n_ch = height(layout.link); % channels of a participant

% dummy recordings
n_dyad = 4; % number of dyads
dil_max = 1.5; % dilution noise SD of the channels spread from 0 to dil_max
false_frac = 0.2; % fraction of channel-windows with false effects
cardiac_amp = 2; % amplitude of the cardiac pulsation

% analysis parameters
sci_th = 0.6; % SCI threshold
psp_th = 0.05; % PSP threshold
n_perm = 10000; % resting-state draws of the connectivity null distribution
chrom = 'hbo'; % chromophore of the group analyses
sections = {'Examination', 'Diagnostic_reasoning', 'Treatment_planning', 'Resource_utilization', 'Additional_comments'}; % section conditions

% generate the dummy recordings (true effects drawn for dyad 1 and shared by all dyads)
truth = [];
for d = 1:n_dyad
    rng(d)
    [task(d), rest(d), truth] = make_dummy_dyad(layout, dil_max * rand(1, 2 * n_ch), false_frac, cardiac_amp, truth);
end
lk = task(1).probe.link;
link_groups = {find(lk.source <= size(layout.srcPos, 1)), find(lk.source > size(layout.srcPos, 1))}; % links of each participant

% preprocessing and window-wise quality control
[hb_task, rej_task, QC] = preprocess_window_qc(task, link_groups, sci_th, psp_th);
[hb_rest, rej_rest] = preprocess_window_qc(rest, link_groups, sci_th, psp_th);

% first-level activation and connectivity
for d = 1:n_dyad
    fprintf('dyad %d \n', d)
    tic
    R(d) = first_level_analysis(hb_task(d), rej_task(d), hb_rest(d), rej_rest(d), n_perm);
    toc
end

% group statistics: weighted Z-method (Stouffer et al., 1949; Liptak, 1958; Whitlock, 2005) with each Z weighted by the
% square root of its retained samples, and Benjamini-Hochberg FDR of the group p-values; activation of each role
% (participant of the recording) and condition across the channels of the role, connectivity of each condition across
% the channel pairs of all roles. Network summary: pairs significant at the first level (FDR within the recording) with
% a positive or negative Z in each subnetwork (intra-brain of each role, inter-brain of each pair of roles), relative
% to the pairs with a Z and averaged across dyads
n_d = numel(R); % dyads
conds = R(1).conds;
n_cond = numel(conds);
n_role = numel(QC(1).participant); % roles (participants of a recording)
lk = R(1).link;
rows = cell(1, n_role); % channels of each role (rows of R.link)
for r = 1:n_role
    Q = QC(1).participant(r);
    rows{r} = find(ismember([lk.source, lk.detector], [Q.link.source, Q.link.detector], 'rows') & strcmpi(lk.type, chrom));
end
G.conds = conds;
G.rows = rows;
for r = 1:n_role
    Z = NaN(numel(rows{r}), n_d, n_cond);
    N = zeros(numel(rows{r}), n_d, n_cond);
    P = NaN(numel(rows{r}), n_d, n_cond);
    for d = 1:n_d
        Z(:, d, :) = R(d).act_z(rows{r}, :);
        N(:, d, :) = R(d).act_n(rows{r}, :);
        P(:, d, :) = R(d).act_p(rows{r}, :);
    end
    sig = false(size(P)); % first-level significance within the role
    for k = 1:n_cond
        % weighted Z-method (no weight without a result) and FDR across the channels with a result
        zg = sum(sqrt(N(:, :, k)) .* Z(:, :, k), 2, 'omitnan') ./ sqrt(sum(N(:, :, k), 2));
        zg(sum(N(:, :, k), 2) == 0) = NaN;
        p = 2 * (1 - normcdf(abs(zg)));
        q = NaN(size(p));
        q(~isnan(p)) = mafdr(p(~isnan(p)), 'BHFDR', true);
        G.act_z{r}(:, k) = zg;
        G.act_p{r}(:, k) = p;
        G.act_q{r}(:, k) = q;
        for d = 1:n_d
            ok = ~isnan(P(:, d, k));
            sig(ok, d, k) = mafdr(P(ok, d, k), 'BHFDR', true) < 0.05;
        end
    end
    G.act_z_dyad{r} = Z;
    G.act_n_dyad{r} = N;
    G.act_sig_dyad{r} = sig;
end
idx = vertcat(rows{:}); % channels of all roles
role = repelem((1:n_role)', cellfun(@numel, rows)); % role of each channel
n_e = numel(idx);
ut = triu(true(n_e), 1); % channel pairs
net = {}; % pairs of the subnetworks: intra-brain of each role, inter-brain of each pair of roles
G.pct_names = {};
for r1 = 1:n_role
    net{end + 1} = ut & (role == r1) & (role' == r1);
    G.pct_names{end + 1} = sprintf('intra %d', r1);
end
for r1 = 1:n_role
    for r2 = r1 + 1:n_role
        net{end + 1} = (role == r1) & (role' == r2);
        G.pct_names{end + 1} = sprintf('inter %d-%d', r1, r2);
    end
end
G.conn_z = NaN(n_e, n_e, n_cond);
G.conn_q = NaN(n_e, n_e, n_cond);
G.pct = zeros(n_cond, 2 * numel(net));
for k = 1:n_cond
    Ze = NaN(nnz(ut), n_d);
    Ne = zeros(nnz(ut), n_d);
    pct = zeros(n_d, 2 * numel(net));
    for d = 1:n_d
        Zc = R(d).conn_z(idx, idx, k);
        Nc = R(d).pair_n(idx, idx, k);
        Nc(isnan(Zc)) = 0; % no weight without a Z-score
        Ze(:, d) = Zc(ut);
        Ne(:, d) = Nc(ut);

        % first-level significant pairs of each subnetwork
        sig = R(d).conn_sig(idx, idx, k);
        valid = ~isnan(Zc); % pairs with a task-versus-rest Z
        for m = 1:numel(net)
            v = net{m} & valid;
            pct(d, 2 * m - 1) = 100 * sum(sig(v) & Zc(v) > 0) / max(sum(v(:)), 1);
            pct(d, 2 * m) = 100 * sum(sig(v) & Zc(v) < 0) / max(sum(v(:)), 1);
        end
    end
    % weighted Z-method and FDR across the pairs with a result
    zg = sum(sqrt(Ne) .* Ze, 2, 'omitnan') ./ sqrt(sum(Ne, 2));
    zg(sum(Ne, 2) == 0) = NaN;
    p = 2 * (1 - normcdf(abs(zg)));
    qg = NaN(size(p));
    qg(~isnan(p)) = mafdr(p(~isnan(p)), 'BHFDR', true);
    Zm = NaN(n_e);
    Zm(ut) = zg;
    Zm = Zm';
    Zm(ut) = zg; % symmetric
    Qm = NaN(n_e);
    Qm(ut) = qg;
    Qm = Qm';
    Qm(ut) = qg;
    G.conn_z(:, :, k) = Zm;
    G.conn_q(:, :, k) = Qm;
    G.pct(k, :) = mean(pct, 1);
end
save(sprintf('%sresults.mat', savepath), 'R', 'G', 'QC', '-v7.3');

%% figures
lab.names = layout.names; % channel labels
lab.roles = {'giver', 'receiver'}; % roles of the participants
lab.dyads = compose('D%d', 1:n_dyad); % dyad labels
lab.conds = {'Examination', 'Diagnostic reasoning', 'Treatment planning', 'Resource utilization', 'Additional comments', ...
    'Highlight', 'Look at peer (giver)', 'Look at peer (receiver)'}; % condition labels (order of G.conds)
ex = struct('rec', 2, 'role', 2, 'ch', find(strcmp(layout.names, 'AF8-Fp2')), 'cond', 'Examination'); % example cell of Figure 2
show_cond = 'Treatment_planning'; % condition of Figure 3A-C
min_event_dur = 1; % minimum event duration of the GLM (sec, as in first_level_analysis), for the display of short events
color_red = [0.85, 0.25, 0.20];
color_gray = [0.75, 0.75, 0.75];
color_green = [0.2, 0.6, 0.3];
color_dark = [0.15, 0.15, 0.15];
color_event = [0.3, 0.3, 0.3];
section_colors = [0.55, 0.71, 0.90; 0.99, 0.75, 0.44; 0.60, 0.85, 0.60; 0.85, 0.65, 0.85; 0.95, 0.90, 0.50];
if numel(sections) > size(section_colors, 1)
    section_colors = lines(numel(sections));
end
dyad_colors = [0.27, 0.47, 0.75; 0.87, 0.52, 0.20; 0.45, 0.65, 0.30; 0.60, 0.40, 0.70; lines(max(n_dyad - 4, 0))];
font_tick = 13; % tick labels
font_label = 14; % axis labels and panel titles
font_legend = 12; % legends

% Figure 1C: montage
fig = figure('Visible', 'off');
plot_fnirs_brain(layout, zeros(n_ch, 1), zeros(n_ch, 1), [180, 10], parula(256), true, layout.src_names, layout.det_names);
exportgraphics(fig, sprintf('%sFigure1C_montage.png', savepath), 'Resolution', 300)

% Figure 2: window-wise quality assessment and how the retained windows enter the statistics, for the example cell:
% (A) SCI and PSP of every window of the example channel in the representative recording, with the thresholds
% (B) SCI and PSP of every channel-window of all recordings and participants, with the windows of (A)
% (C) conditions and the 1-Hz HbO of the example channel: samples of the retained and the rejected windows
% (D) retained samples of the example channel, a second channel and the samples retained in both (channel pair)
% (E) participant Z of the example cell (bars), their weights (square root of the retained samples, normalized), the
%     weighted contributions stacked in the group bar with their sum (group Z), and the Z at the FDR threshold
Q = QC(ex.rec).participant(ex.role);
lk = hb_task(ex.rec).probe.link;
rows = G.rows{ex.role}; % HbO channels of the role
[~, q_row] = ismember([lk.source(rows), lk.detector(rows)], [Q.link.source, Q.link.detector], 'rows'); % window quality rows
frac = mean(Q.good(q_row, :), 2); % retained fraction of each channel
frac(Q.missing(q_row) | (1:numel(rows))' == ex.ch) = NaN;
[~, ch_pair] = min(abs(frac - 0.5)); % second channel: retained fraction closest to one half
cols = rows([ex.ch, ch_pair]);
retained = ~isnan(rej_task(ex.rec).data(:, cols));
t = hb_task(ex.rec).time;
good = Q.good(q_row(ex.ch), :);
vals = {Q.sci(q_row(ex.ch), :), Q.psp(q_row(ex.ch), :)};
thr = [Q.sci_th, Q.psp_th];
t_window = ((1:Q.n_windows) - 0.5) * Q.window_sec;
t_lim = [0, Q.n_windows * Q.window_sec];
k_ex = find(strcmp(G.conds, ex.cond));
rep_label = sprintf('%s %s, %s', lab.dyads{ex.rec}, lab.roles{ex.role}, lab.names{ex.ch});

% SCI and PSP of every non-missing channel-window of all recordings
all_sci = [];
all_psp = [];
all_good = [];
for i = 1:numel(QC)
    for p = 1:numel(QC(i).participant)
        Qi = QC(i).participant(p);
        keep = repmat(~Qi.missing, 1, Qi.n_windows);
        all_sci = [all_sci; Qi.sci(keep)];
        all_psp = [all_psp; Qi.psp(keep)];
        all_good = [all_good; Qi.good(keep)];
    end
end
all_good = logical(all_good);

% participant Z, weights and contributions of the example cell, Z at the FDR threshold of its role and condition
z_dyad = G.act_z_dyad{ex.role}(ex.ch, :, k_ex);
n_ret = G.act_n_dyad{ex.role}(ex.ch, :, k_ex); % retained samples of each dyad
w_dyad = sqrt(n_ret) / sqrt(sum(n_ret)); % normalized weights (sum of squares 1)
c_dyad = w_dyad .* z_dyad; % weighted contributions
z_group = sum(c_dyad, 'omitnan');
sec_dyad = n_ret / hb_task(ex.rec).Fs; % retained time (sec)
p_sorted = sort(G.act_p{ex.role}(~isnan(G.act_p{ex.role}(:, k_ex)), k_ex));
m = numel(p_sorted);
k_max = find(p_sorted <= (1:m)' * 0.05 / m, 1, 'last'); % Benjamini-Hochberg boundary
if isempty(k_max), k_max = 1; end % no rejection: boundary of the smallest p-value
z_crit = norminv(1 - k_max * 0.05 / m / 2);

% layout of Figure 2
fig = figure('Color', 'w', 'Position', [50, 25, 960, 1210], 'Visible', 'off');
block = @(pos) uipanel(fig, 'Units', 'normalized', 'Position', pos, 'BorderType', 'none', 'BackgroundColor', 'w');
tl_ab = tiledlayout(block([0, 0.74, 1, 0.26]), 1, 12, 'TileSpacing', 'compact', 'Padding', 'compact');
tl_c = tiledlayout(block([0, 0.37, 1, 0.37]), 6, 1, 'TileSpacing', 'none', 'Padding', 'compact');
tl_de = tiledlayout(block([0, 0, 1, 0.37]), 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% A: SCI and PSP of every window of the example channel
tl_a = tiledlayout(tl_ab, 2, 1, 'TileSpacing', 'tight', 'Padding', 'none');
tl_a.Layout.Tile = 1;
tl_a.Layout.TileSpan = [1, 8];
y_names = {'SCI', 'PSP'};
for v = 1:2
    ax = nexttile(tl_a, v);
    hold on; grid on; box on;
    plot(t_window, vals{v}, '-', 'Color', color_gray, 'LineWidth', 0.8);
    scatter(t_window(~good), vals{v}(~good), 14, color_gray, 'filled', 'MarkerEdgeColor', color_dark);
    scatter(t_window(good), vals{v}(good), 14, color_green, 'filled', 'MarkerEdgeColor', color_dark);
    yline(thr(v), '-', 'Color', color_red, 'LineWidth', 1.5);
    xlim(t_lim);
    ylabel(y_names{v}, 'FontSize', font_label);
    set(ax, 'FontSize', font_tick);
    if v == 1
        title(sprintf('\\bfA\\rm  Window-wise SCI and PSP (%s)', rep_label), 'FontSize', font_label, 'FontWeight', 'normal');
        ax.TitleHorizontalAlignment = 'left';
        ylim([-1, 1]);
        set(ax, 'XTickLabel', []);
    else
        xlabel('Time (s)', 'FontSize', font_label);
    end
end

% B: SCI and PSP of every channel-window of all recordings
ax = nexttile(tl_ab, 9, [1, 4]);
hold on; grid on; box on;
pos = all_psp > 0; % values shown on the logarithmic axis
scatter(all_sci(pos & ~all_good), all_psp(pos & ~all_good), 3, color_gray, 'filled', 'MarkerFaceAlpha', 0.3);
scatter(all_sci(pos & all_good), all_psp(pos & all_good), 3, color_green, 'filled', 'MarkerFaceAlpha', 0.3);
ex_pos = vals{2} > 0;
scatter(vals{1}(ex_pos & ~good), vals{2}(ex_pos & ~good), 14, color_gray, 'filled', 'MarkerEdgeColor', color_dark);
scatter(vals{1}(ex_pos & good), vals{2}(ex_pos & good), 14, color_green, 'filled', 'MarkerEdgeColor', color_dark);
set(ax, 'YScale', 'log');
y_lim = [min(all_psp(pos)) * 0.8, max(all_psp(pos)) * 1.25];
plot([thr(1), thr(1), 1], [y_lim(2), thr(2), thr(2)], '-', 'Color', color_red, 'LineWidth', 1.5);
xlim([-1, 1]);
ylim(y_lim);
xlabel('SCI', 'FontSize', font_label);
ylabel('PSP', 'FontSize', font_label);
title('\bfB\rm  All participants', 'FontSize', font_label, 'FontWeight', 'normal');
ax.TitleHorizontalAlignment = 'left';
set(ax, 'FontSize', font_tick);
h_ab = gobjects(1, 4);
h_ab(1) = scatter(ax, nan, nan, 20, color_gray, 'filled');
h_ab(2) = scatter(ax, nan, nan, 20, color_green, 'filled');
h_ab(3) = scatter(ax, nan, nan, 20, 'w', 'filled', 'MarkerEdgeColor', color_dark);
h_ab(4) = plot(ax, nan, nan, '-', 'Color', color_red, 'LineWidth', 1.5);
lgd = legend(h_ab, {'Rejected window', 'Retained window', 'Window of the channel in A', 'Threshold'}, 'NumColumns', 4, ...
    'FontSize', font_legend, 'box', 'off');
lgd.Layout.Tile = 'north';

% C: conditions and the samples entering the GLM
ax_c1 = nexttile(tl_c, 1, [2, 1]);
hold on; box on;
event_names = setdiff(G.conds, sections, 'stable'); % event conditions, drawn in their own row
for k = 1:numel(G.conds)
    st = hb_task(ex.rec).stimulus(G.conds{k});
    if ismember(G.conds{k}, sections)
        y_row = 1;
        color = section_colors(strcmp(sections, G.conds{k}), :);
        dur = st.dur;
    else
        y_row = 1 + find(strcmp(event_names, G.conds{k}));
        color = color_event;
        dur = max(st.dur, min_event_dur);
    end
    for e = 1:numel(st.onset)
        patch(ax_c1, st.onset(e) + [0, dur(e), dur(e), 0], y_row + [-0.35, -0.35, 0.35, 0.35], color, 'EdgeColor', 'none');
    end
end
xlim(ax_c1, t_lim);
ylim(ax_c1, [0.5, numel(event_names) + 1.5]);
set(ax_c1, 'YDir', 'reverse', 'YTick', 1:numel(event_names) + 1, 'YTickLabel', [{'Sections'}, lab.conds(ismember(G.conds, event_names))], ...
    'XTickLabel', [], 'TickLength', [0, 0], 'FontSize', font_tick);
title(ax_c1, sprintf('\\bfC\\rm  Samples entering the GLM (%s)', rep_label), 'FontSize', font_label, 'FontWeight', 'normal');
ax_c1.TitleHorizontalAlignment = 'left';

% HbO at 1 Hz as in the GLM, samples of the rejected windows on the 1-Hz time base
ax_c2 = nexttile(tl_c, 3, [4, 1]);
hold on; grid on; box on;
rs = nirs.modules.Resample();
rs.Fs = 1;
hb_1hz = rs.run(hb_task(ex.rec));
t1 = hb_1hz.time;
y = hb_1hz.data(:, cols(1));
y_ret = y;
y_ret(interp1(t, double(retained(:, 1)), t1, 'nearest', 0) ~= 1) = NaN;
h_c = gobjects(1, 2 + numel(sections));
h_c(2) = plot(ax_c2, t1, y, '-', 'Color', color_gray, 'LineWidth', 1.0);
h_c(1) = plot(ax_c2, t1, y_ret, '-', 'Color', color_dark, 'LineWidth', 1.2);
xlim(ax_c2, t_lim);
xlabel(ax_c2, 'Time (s)', 'FontSize', font_label);
ylabel(ax_c2, 'HbO (\muM)', 'FontSize', font_label);
set(ax_c2, 'FontSize', font_tick);
for k = 1:numel(sections)
    h_c(2 + k) = patch(ax_c2, nan, nan, section_colors(k, :), 'EdgeColor', 'none');
end
[~, i_sec] = ismember(sections, G.conds);
lgd = legend(h_c, [{'Retained samples', 'Rejected samples'}, lab.conds(i_sec)], 'NumColumns', 4, 'FontSize', font_legend, 'box', 'off');
lgd.Layout.Tile = 'south';
linkaxes([ax_c1, ax_c2], 'x');

% D: samples of a channel pair
ax = nexttile(tl_de, 1);
t_s = 0:floor(t(end)); % 1-s grid
state = zeros(3, numel(t_s));
for e = 1:2
    state(e, :) = interp1(t, double(retained(:, e)), t_s, 'nearest', 0);
end
state(3, :) = 2 * (state(1, :) & state(2, :));
imagesc(ax, t_s, 1:3, state);
colormap(ax, [color_gray; color_green; color_dark]);
clim(ax, [-0.5, 2.5]);
yticks(ax, 1:3);
yticklabels(ax, {lab.names{ex.ch}, lab.names{ch_pair}, 'Pair'});
xlabel('Time (s)', 'FontSize', font_label);
title('\bfD\rm  Samples of a channel pair', 'FontSize', font_label, 'FontWeight', 'normal');
ax.TitleHorizontalAlignment = 'left';
set(ax, 'FontSize', font_tick, 'TickLength', [0, 0]);
h_d = gobjects(1, 3);
h_d(1) = patch(ax, nan, nan, color_green, 'EdgeColor', 'none');
h_d(2) = patch(ax, nan, nan, color_dark, 'EdgeColor', 'none');
h_d(3) = patch(ax, nan, nan, color_gray, 'EdgeColor', 'none');
legend(h_d, {'Retained', 'Retained in both', 'Rejected'}, 'Location', 'southoutside', 'NumColumns', 3, 'FontSize', font_legend, 'box', 'off');

% E: combination of the participant Z-scores
ax = nexttile(tl_de, 2);
hold on; grid on; box on;
y_group = n_d + 1;
h_e = gobjects(1, 4);
for d = 1:n_d
    if isnan(z_dyad(d))
        text(0, d, 'no retained window', 'HorizontalAlignment', 'center', 'FontSize', font_legend);
        continue
    end
    h_e(1) = patch(ax, [0, z_dyad(d), z_dyad(d), 0], d + [-0.3, -0.3, 0.3, 0.3], dyad_colors(d, :), 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    h_e(2) = patch(ax, [0, c_dyad(d), c_dyad(d), 0], d + [-0.15, -0.15, 0.15, 0.15], dyad_colors(d, :), 'EdgeColor', 'none');
end

% group row: positive contributions stacked to the right of zero, negative ones to the left
pos_end = 0;
neg_end = 0;
for d = 1:n_d
    c = c_dyad(d);
    if isnan(c), continue, end
    if c >= 0
        x0 = pos_end; pos_end = pos_end + c;
    else
        x0 = neg_end; neg_end = neg_end + c;
    end
    patch(ax, [x0, x0 + c, x0 + c, x0], y_group + [-0.3, -0.3, 0.3, 0.3], dyad_colors(d, :), 'EdgeColor', 'w');
end
h_e(3) = plot([z_group, z_group], y_group + [-0.45, 0.45], '-', 'Color', color_dark, 'LineWidth', 2.5);
h_e(4) = xline(z_crit, '--', 'Color', color_red, 'LineWidth', 1.5);
xline(-z_crit, '--', 'Color', color_red, 'LineWidth', 1.5);
xline(0, 'k-');
xlim([min([z_dyad, -z_crit, neg_end], [], 'omitnan'), max([z_dyad, z_crit, pos_end], [], 'omitnan')] * 1.15);
ylim([0.4, y_group + 0.5]);
row_labels = lab.dyads(1:n_d);
for d = find(~isnan(z_dyad))
    row_labels{d} = sprintf('%s (%.0f s)', lab.dyads{d}, sec_dyad(d));
end
row_labels{end + 1} = sprintf('Group (Z = %.2f)', z_group);
set(ax, 'YDir', 'reverse', 'YTick', 1:y_group, 'YTickLabel', row_labels, 'FontSize', font_tick);
xlabel('Z', 'FontSize', font_label);
title(sprintf('\\bfE\\rm  Group Z (%s, %s, %s)', lab.roles{ex.role}, lab.names{ex.ch}, lab.conds{k_ex}), ...
    'FontSize', font_label, 'FontWeight', 'normal');
ax.TitleHorizontalAlignment = 'left';
legend(h_e, {'Participant Z', 'w \times Z', 'Group Z (sum of w \times Z)', 'FDR threshold (q = 0.05)'}, ...
    'Location', 'southoutside', 'NumColumns', 2, 'FontSize', font_legend, 'box', 'off');

% page of the figure size for printing
for ax = findall(fig, 'Type', 'axes')'
    ax.Toolbar.Visible = 'off'; % axes toolbar not exported
end
fig.PaperUnits = 'inches';
fig.PaperSize = fig.Position(3:4) / 96;
fig.PaperPosition = [0, 0, fig.PaperSize];
print(fig, sprintf('%sFigure2_window_quality.png', savepath), '-dpng', '-r300');

% Figure 3, condition show_cond: (A, B) group Z of the activation of each role on the cortical surface (plot_fnirs_brain),
% channels with FDR q < 0.05 in color; (C) circos plot of the participant Z of each channel (one ring per dyad,
% channels without first-level significance in gray, channels without retained data blank) and the group Z of the
% channel pairs with FDR q < 0.05 (curves inside the circle)
c_lims = [-6, 6]; % color limits of the Z-scores
n_c = 128;
cmap_conn = [linspace(0.0, 1, n_c)', linspace(0.3, 1, n_c)', linspace(0.5, 1, n_c)'; ...
    linspace(1, 0.7, n_c)', linspace(1, 0.1, n_c)', linspace(1, 0.0, n_c)']; % connectivity colormap (blue-white-red)
role_title = cellfun(@(s) [upper(s(1)), s(2:end)], lab.roles, 'UniformOutput', false);
order = layout.order(:); % display order of the channels
k = find(strcmp(G.conds, show_cond));
fig_cond = figure('Name', show_cond, 'Position', [100, 100, 800, 900], 'Visible', 'off');
tl = tiledlayout(fig_cond, 3, 2);
ax_r = gobjects(1, 2);
for r = 1:2
    ax_r(r) = nexttile(tl);
    z = G.act_z{r}(:, k);
    z(isnan(z)) = 0;
    plot_fnirs_brain(layout, z, double(G.act_q{r}(:, k) < 0.05), [180, 10], parula(256), false);
    title(sprintf('%s Activation (Group Z-map)', role_title{r}), 'FontSize', 12);
    clim(ax_r(r), c_lims);
end
cbr = colorbar(ax_r(2), 'position', [0.9, 0.7, 0.02, 0.2]);
cbr.Label.String = 'Activations (Z-score)';
colormap(ax_r(2), parula(256));

% circos plot: nodes of role 2 (right sector), then role 1 (left sector), clockwise from the top
ax_c = nexttile(tl, [2, 2]);
node = [G.act_z_dyad{2}(order, :, k); G.act_z_dyad{1}(order, :, k)]; % node x ring (dyad)
node_gray = ~[G.act_sig_dyad{2}(order, :, k); G.act_sig_dyad{1}(order, :, k)];
idx = [n_ch + order; order];
edge = G.conn_z(idx, idx, k);
edge(~(G.conn_q(idx, idx, k) < 0.05)) = NaN; % pairs with FDR q < 0.05
node_labels = [string(layout.names(order)); string(layout.names(order))];
sector_labels = [repmat(string(role_title{2}), n_ch, 1); repmat(string(role_title{1}), n_ch, 1)];
ring_labels = compose("D%d", 1:size(node, 2));
r_inner = 1.0; % radius of the edge ends
r_outer = 1.1; % radius of the first ring
ring_width = 0.08; % width of a ring
ring_gap = 0.02; % gap between rings
gap = 0.2; % gap between sectors (rad)
[n_node, n_ring] = size(node);
first = [true; sector_labels(2:end) ~= sector_labels(1:end - 1)]; % first node of each sector
sector_names = sector_labels(first);
sector_sizes = diff([find(first); n_node + 1])';
step = (2 * pi - numel(sector_sizes) * gap) / n_node; % angle per node
theta = zeros(n_node, 1);
sector_centers = zeros(numel(sector_sizes), 1);
ang = pi / 2 - gap / 2 - step / 2;
i = 1;
for s = 1:numel(sector_sizes)
    angs = ang - (0:sector_sizes(s) - 1) * step;
    theta(i:i + sector_sizes(s) - 1) = angs;
    sector_centers(s) = atan2(sum(sin(angs)), sum(cos(angs)));
    ang = ang - sector_sizes(s) * step - gap;
    i = i + sector_sizes(s);
end
arc_width = step * 0.9;
hold(ax_c, 'on');
axis(ax_c, 'equal');
axis(ax_c, 'off');
for r = 1:n_ring
    radius = r_outer + (r - 1) * (ring_width + ring_gap);
    for n = 1:n_node
        arc = linspace(theta(n) - arc_width / 2, theta(n) + arc_width / 2, 20);
        x_in = radius * cos(arc);
        y_in = radius * sin(arc);
        x_out = (radius + ring_width) * cos(fliplr(arc));
        y_out = (radius + ring_width) * sin(fliplr(arc));
        if ~isnan(node(n, r))
            rgb = parula(256);
            rgb = rgb(round(1 + 255 * (min(max(node(n, r), c_lims(1)), c_lims(2)) - c_lims(1)) / diff(c_lims)), :);
            if node_gray(n, r)
                rgb = repmat(0.2989 * rgb(1) + 0.5870 * rgb(2) + 0.1140 * rgb(3), 1, 3); % grayscale
            end
            patch(ax_c, 'XData', [x_in, x_out], 'YData', [y_in, y_out], 'FaceColor', rgb, 'EdgeColor', 'none');
        else
            patch(ax_c, 'XData', [x_in, x_out], 'YData', [y_in, y_out], 'FaceColor', [1, 1, 1], 'EdgeColor', [0, 0, 0]);
            line(ax_c, [x_in(1), x_out(1)], [y_in(1), y_out(1)], 'Color', [0, 0, 0]);
        end
    end
    text(ax_c, 0, radius + ring_width / 2, ring_labels(r), 'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'middle', 'FontSize', 10, 'FontWeight', 'bold', 'Color', [0.4, 0.4, 0.4], 'Interpreter', 'none');
end
label_radius = r_outer + (n_ring - 1) * (ring_width + ring_gap) + ring_width + 0.05;
for n = 1:n_node
    rot = rad2deg(theta(n));
    align = 'left';
    if cos(theta(n)) < 0
        rot = rot + 180;
        align = 'right';
    end
    text(ax_c, label_radius * cos(theta(n)), label_radius * sin(theta(n)), node_labels(n), ...
        'HorizontalAlignment', align, 'VerticalAlignment', 'middle', 'Rotation', rot, 'FontSize', 8, 'Interpreter', 'none');
end
[n1, n2] = find(triu(~isnan(edge), 1));
t = linspace(0, 1, 50);
for e = 1:numel(n1)
    rgb = cmap_conn(round(1 + (size(cmap_conn, 1) - 1) * (min(max(edge(n1(e), n2(e)), c_lims(1)), c_lims(2)) - c_lims(1)) / diff(c_lims)), :);
    x = (1 - t).^2 * r_inner * cos(theta(n1(e))) + t.^2 * r_inner * cos(theta(n2(e))); % quadratic Bezier through the center
    y = (1 - t).^2 * r_inner * sin(theta(n1(e))) + t.^2 * r_inner * sin(theta(n2(e)));
    plot(ax_c, x, y, '-', 'Color', [rgb, 0.8], 'LineWidth', 4);
end
for s = 1:numel(sector_names)
    rot = rad2deg(sector_centers(s)) - 90;
    if cos(sector_centers(s)) < 0
        rot = rot + 180;
    end
    text(ax_c, r_outer * 1.7 * cos(sector_centers(s)), r_outer * 1.7 * sin(sector_centers(s)), sector_names(s), ...
        'HorizontalAlignment', 'center', 'Rotation', rot, 'FontWeight', 'bold', 'FontSize', 12, 'Interpreter', 'none');
end
cb1 = colorbar(ax_r(1), 'location', 'southoutside', 'position', [0.175, 0.04, 0.3, 0.015]);
cb1.Label.String = 'Activations (Z-score)';
colormap(ax_r(1), parula(256));
clim(ax_c, c_lims);
colormap(ax_c, cmap_conn);
cb2 = colorbar(ax_c, 'location', 'southoutside', 'position', [0.555, 0.041, 0.3, 0.015]);
cb2.Label.String = '\DeltaConnectivity (Z-score)';
cb2.Ticks = -6:3:6;
exportgraphics(fig_cond, sprintf('%sFigure3_%s.png', savepath, show_cond), 'Resolution', 300)

% Figure 3, all conditions: (D, E) group Z of each channel x condition of each role (* q < 0.05, ** q < 0.01,
% *** q < 0.001); (F) significant pairs (%) of the intra-brain network of each role and of the inter-brain network
fig_sum = figure('Name', 'Summary Statistics Heatmaps', 'Position', [50, 50, 650, 900], 'Visible', 'off');
tl = tiledlayout(fig_sum, 3, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
for r = 1:2
    ax = nexttile(tl, [1, 3]);
    z = G.act_z{r}(order, :)';
    q = G.act_q{r}(order, :)';
    imagesc(ax, z, 'AlphaData', ~isnan(z));
    colormap(ax, parula(256));
    clim(ax, c_lims);
    cb = colorbar(ax);
    cb.Label.String = 'Activations (Z-score)';
    set(ax, 'YTick', 1:n_cond, 'YTickLabel', lab.conds, 'XTick', 1:n_ch, 'XTickLabel', layout.names(order), ...
        'XTickLabelRotation', 90, 'XDir', 'reverse', 'TickLabelInterpreter', 'none');
    title(ax, sprintf('%s Activation (Group Z-score)', role_title{r}), 'FontSize', 11);
    [ri, ci] = find(q < 0.05);
    for i = 1:numel(ri)
        stars = repmat('*', 1, 1 + (q(ri(i), ci(i)) < 0.01) + (q(ri(i), ci(i)) < 0.001));
        text(ax, ci(i), ri(i), stars, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Color', 'w', ...
            'FontSize', 10, 'FontWeight', 'bold');
    end
end

% significant pairs (%) of the intra-brain networks and the inter-brain network
blocks = {G.pct(:, 1:4), G.pct(:, 5:6)};
x_labels = {{sprintf('%s(+)', role_title{1}), sprintf('%s(-)', role_title{1}), sprintf('%s(+)', role_title{2}), ...
    sprintf('%s(-)', role_title{2})}, {'Inter(+)', 'Inter(-)'}};
titles = {'\DeltaIntra-connectivity', '\DeltaInter-connectivity'};
for b = 1:2
    ax = nexttile(tl, [1, 3 - b]);
    imagesc(ax, blocks{b});
    colormap(ax, sky(256));
    cb = colorbar(ax, 'location', 'southoutside');
    cb.Label.String = 'Significant Edges (%)';
    set(ax, 'YTick', 1:n_cond, 'XTick', 1:size(blocks{b}, 2), 'XTickLabel', x_labels{b});
    if b == 1
        set(ax, 'YTickLabel', lab.conds, 'TickLabelInterpreter', 'none');
    else
        set(ax, 'YTickLabel', {});
    end
    title(ax, titles{b}, 'FontSize', 11);
    for i = 1:n_cond
        for j = 1:size(blocks{b}, 2)
            text_color = 'w';
            if blocks{b}(i, j) < max(blocks{b}(:)) * 0.5
                text_color = 'k';
            end
            text(ax, j, i, sprintf('%.2f%%', blocks{b}(i, j)), 'HorizontalAlignment', 'center', 'Color', text_color, ...
                'FontWeight', 'bold', 'FontSize', 8);
        end
    end
end
exportgraphics(fig_sum, sprintf('%sFigure3_summary.png', savepath), 'Resolution', 300)
