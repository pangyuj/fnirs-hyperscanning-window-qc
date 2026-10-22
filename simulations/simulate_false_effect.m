% This script compares no removal, channel-level removal (75% clean windows) and window-level removal of the windows
% with false effects in simulated recordings (AUROC of activation and connectivity, Supplementary Figure 2 of the paper).
% The random generators and computational threads are fixed, so that the results of the paper are reproduced
clear; clc; close all;
addpath('../pipeline/functions') % ar_irls_gap, robust_corr_pairwise

% template: timing and channels of the giver of the representative dyad (no fNIRS signal)
load('paper_template.mat', 'template')
tp = sim_template(template);

% simulation parameters
n_rep = 1000; % simulated recordings at each false-effect level
false_level = [0:0.05:0.3, 0.4:0.1:0.9]; % fractions of windows with false effects
dil_sd = 2; % SD of the dilution noise (x true response)
false_size = 2; % amplitude of the false effect (x true response)
window_sec = 5; % window length (sec)
qt_quality = 0.75; % clean-window fraction of the channel-level removal
highpass = 0.01; % high-pass cut-off (Hz)
ar_order_sec = 4; % maximum AR model order (sec)
n_boot = 2000; % bootstrap samples of the AUROC
strategy_names = {'No removal', 'Channel-level removal (75%)', 'Window-level removal (this study)'};
f_res = 'False_Effect_Results.mat'; % results file
f_fig = 'Figure_False_Effect'; % figure file

% simulate the recordings of every false-effect level
[j1, j2] = ndgrid(1:numel(false_level), 1:n_rep);
jobs = [j1(:), j2(:)]; % false-effect level and repetition of each simulation
out = cell(size(jobs, 1), 1);
n_threads = maxNumCompThreads(1); % one computational thread as on parallel workers, for the same numerics without a pool
tic
parfor i = 1:size(jobs, 1)
    rng(i, 'threefry');
    out{i} = sim_false_effect_recording(tp, false_level(jobs(i, 1)), dil_sd, false_size, window_sec, qt_quality, highpass, ar_order_sec, false);
end
toc
maxNumCompThreads(n_threads);
save(f_res, 'jobs', 'out', 'false_level', 'strategy_names', 'n_rep', '-v7.3');

%% AUROC estimation
n_c = numel(false_level);
used = zeros(n_c, 3); % retained data of each strategy
auc = {zeros(n_c, 3, 3), zeros(n_c, 3, 3)}; % AUROC and 95% CI of the activation and the connectivity
fields = {'z_act', 't_act'; 'z_con', 't_con'};
edges = [-inf, -10:0.01:10, inf]; % histogram bins of the Z-scores
for c_i = 1:n_c
    o = out(jobs(:, 1) == c_i);
    n_r = numel(o);
    used(c_i, :) = mean(cell2mat(cellfun(@(x) x.used, o, 'UniformOutput', false)), 1);
    rng(c_i, 'twister');
    for m = 1:2
        Z = double(cell2mat(cellfun(@(x) x.(fields{m, 1}), o, 'UniformOutput', false)));
        T = cell2mat(cellfun(@(x) x.(fields{m, 2}), o, 'UniformOutput', false));
        rec = repelem((1:n_r)', numel(o{1}.(fields{m, 2})), 1); % simulated recording of each unit

        % bootstrap weights of the simulated recordings (resampled with replacement)
        W = ones(n_boot + 1, n_r);
        for b = 1:n_boot
            W(b + 1, :) = accumarray(randi(n_r, n_r, 1), 1, [n_r, 1])';
        end

        % AUROC (Mann-Whitney) of each strategy from the weighted histograms of the true and absent effects
        A = zeros(n_boot + 1, 3);
        for s = 1:3
            bin = discretize(Z(:, s), edges);
            Pc = W * accumarray([rec(T), bin(T)], 1, [n_r, numel(edges) - 1]);
            Nc = W * accumarray([rec(~T), bin(~T)], 1, [n_r, numel(edges) - 1]);
            A(:, s) = sum(Pc .* (cumsum(Nc, 2) - 0.5 * Nc), 2) ./ (sum(Pc, 2) .* sum(Nc, 2));
        end
        auc{m}(c_i, :, :) = reshape([A(1, :)', prctile(A(2:end, :), [2.5, 97.5])'], 1, 3, 3);
    end
end
auc_act = auc{1};
auc_con = auc{2};
save(f_res, 'auc_act', 'auc_con', 'used', '-append');

% example recording
ex_cond = false_level(ceil(numel(false_level) / 2)); % false-effect level of the example
rng(1, 'twister');
ex = sim_false_effect_recording(tp, ex_cond, dil_sd, false_size, window_sec, qt_quality, highpass, ar_order_sec, true);

% draw figure
font_tick = 12;
font_label = 14;
font_title = 14;
font_legend = 12;
false_color = [0.85, 0.20, 0.15];
colors = [0.60, 0.60, 0.60; 0.27, 0.47, 0.75; 0.20, 0.60, 0.30];
x = 100 * false_level;
fig = figure('Color', 'w', 'Position', [50, 50, 1600, 950], 'Visible', 'off');
tl = tiledlayout(fig, 2, 6, 'TileSpacing', 'compact', 'Padding', 'compact');

% A: false-effect windows across channels
ax = nexttile(tl, [1, 3]);
imagesc(ax, ex.t_win, 1:size(ex.label, 1), ex.label);
colormap(ax, [1, 1, 1; false_color]);
clim(ax, [0, 1]);
cb = colorbar(ax, 'Ticks', [0.25, 0.75], 'TickLabels', {'Clean', 'False effect'});
cb.FontSize = font_tick;
xlabel(ax, 'Time (s)', 'FontSize', font_label);
ylabel(ax, 'Channel', 'FontSize', font_label);
title(ax, sprintf('\\bfA\\rm  Example false-effect windows across channels (target %g%%)', 100 * ex_cond), 'FontSize', font_title);
ax.TitleHorizontalAlignment = 'left';
ax.FontSize = font_tick;

% B: example channel signal (drift terms of the GLM removed for display)
ax = nexttile(tl, [1, 3]);
hold(ax, 'on');
box(ax, 'on');
T = nirs.design.trend.dctmtx(ex.t1, highpass);
y = ex.ex_meas - T * (T \ ex.ex_meas);
y_lim = 1.1 * max(abs([prctile(y, [1; 99]); ex.ex_true]));
for w = find(ex.label(ex.ex_k, :))
    t0 = ex.t1(1) + (w - 1) * window_sec;
    patch(ax, [t0, t0 + window_sec, t0 + window_sec, t0], y_lim * [-1, -1, 1, 1], false_color, 'EdgeColor', 'none', 'FaceAlpha', 0.18);
end
h_b = gobjects(1, 3);
h_b(1) = patch(ax, nan, nan, false_color, 'EdgeColor', 'none', 'FaceAlpha', 0.18);
h_b(2) = plot(ax, ex.t1, y, 'Color', [0.45, 0.45, 0.45], 'LineWidth', 0.8);
h_b(3) = plot(ax, ex.t1, ex.ex_true, 'k', 'LineWidth', 1.8);
xlim(ax, [ex.t1(1), ex.t1(end)]);
ylim(ax, y_lim * [-1, 1]);
xlabel(ax, 'Time (s)', 'FontSize', font_label);
ylabel(ax, 'HbO (\times true effect)', 'FontSize', font_label);
title(ax, sprintf('\\bfB\\rm  Channel %d: measured HbO and true response', ex.ex_k), 'FontSize', font_title);
ax.TitleHorizontalAlignment = 'left';
ax.FontSize = font_tick;
legend(ax, h_b, {'False-effect window', 'Measured HbO', 'True response'}, 'Location', 'northeast', 'Orientation', 'horizontal', 'FontSize', font_legend);

% C: retained data
ax = nexttile(tl, [1, 2]);
hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
h_s = gobjects(1, 3);
for s = 1:3
    h_s(s) = plot(ax, x, 100 * used(:, s), 'o-', 'Color', colors(s, :), 'MarkerFaceColor', colors(s, :), 'LineWidth', 1.4, 'MarkerSize', 4);
end
xlim(ax, [min(x), max(x)]);
ylim(ax, [0, 105]);
ax.XTick = x;
xlabel(ax, 'Windows with false effects (%)', 'FontSize', font_label);
ylabel(ax, 'Data retained (%)', 'FontSize', font_label);
title(ax, '\bfC\rm  Data retention by strategy', 'FontSize', font_title);
ax.TitleHorizontalAlignment = 'left';
ax.FontSize = font_tick;
legend(ax, h_s, strategy_names, 'Location', 'southoutside', 'Orientation', 'vertical', 'FontSize', font_legend);

% D, E: AUROC of the activation and the connectivity with 95% CIs
titles = {'\bfD\rm  Activation recovery', '\bfE\rm  Connectivity recovery'};
ax_auc = gobjects(1, 2);
for m = 1:2
    ax = nexttile(tl, [1, 2]);
    hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
    for s = 1:3
        fill(ax, [x, fliplr(x)], [auc{m}(:, s, 3)', fliplr(auc{m}(:, s, 2)')], colors(s, :), 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
        plot(ax, x, auc{m}(:, s, 1), 'o-', 'Color', colors(s, :), 'MarkerFaceColor', [1, 1, 1], 'LineWidth', 1.5, 'MarkerSize', 5);
    end
    yline(ax, 0.5, 'k:', 'HandleVisibility', 'off');
    xlim(ax, [min(x), max(x)]);
    ax.XTick = x;
    xlabel(ax, 'Windows with false effects (%)', 'FontSize', font_label);
    ylabel(ax, 'AUROC', 'FontSize', font_label);
    title(ax, titles{m}, 'FontSize', font_title);
    ax.TitleHorizontalAlignment = 'left';
    ax.FontSize = font_tick;
    ax_auc(m) = ax;
end
y_min = min([0.5, min(auc_act(:, :, 2), [], 'all'), min(auc_con(:, :, 2), [], 'all')]) - 0.02;
y_max = max([max(auc_act(:, :, 3), [], 'all'), max(auc_con(:, :, 3), [], 'all')]) + 0.02;
linkaxes(ax_auc, 'y');
ylim(ax_auc(1), [y_min, y_max]);

print(fig, [f_fig, '.png'], '-dpng', '-r600');
print(fig, [f_fig, '.pdf'], '-dpdf', '-bestfit');
