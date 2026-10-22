% This script tests how retaining or excluding channels with stationary background (dilution) noise changes the
% recovery of true activation and connectivity in simulated recordings (AUROC, Supplementary Figure 1 of the paper).
% The random generators and computational threads are fixed, so that the results of the paper are reproduced
clear; clc; close all;
addpath('../pipeline/functions') % ar_irls_gap, robust_corr_pairwise

% template: timing and channels of the giver of the representative dyad (no fNIRS signal)
load('paper_template.mat', 'template')
tp = sim_template(template);

% simulation parameters
n_rep = 1000; % simulated recordings
max_noise = 10; % largest SD of the dilution noise across channels (x true response)
noise_scales = linspace(0.2, 1, 5); % scales of the dilution noise
thresholds = [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, Inf]; % rejection thresholds (maximum allowed noise SD)
highpass = 0.01; % high-pass cut-off (Hz)
ar_order_sec = 4; % maximum AR model order (sec)
n_boot = 2000; % bootstrap samples of the AUROC
f_res = 'Dilution_Threshold_Results.mat'; % results file
f_fig = 'Figure_Dilution_Threshold'; % figure file

% simulate the recordings
out = cell(n_rep, 1);
n_threads = maxNumCompThreads(1); % one computational thread as on parallel workers, for the same numerics without a pool
tic
parfor i = 1:n_rep
    rng(i, 'threefry');
    out{i} = sim_dilution_recording(tp, max_noise, noise_scales, thresholds, highpass, ar_order_sec, false);
end
toc
maxNumCompThreads(n_threads);
save(f_res, 'out', 'max_noise', 'noise_scales', 'thresholds', 'n_rep', '-v7.3');

%% AUROC estimation
n_level = numel(noise_scales);
n_th = numel(thresholds);
used = zeros(n_level, n_th); % retained channels
auc = {zeros(n_th, 3, n_level), zeros(n_th, 3, n_level)}; % AUROC, AUROC - SE and AUROC + SE of the activation and the connectivity
fields = {'z_act', 't_act'; 'z_con', 't_con'};
edges = [-inf, -10:0.01:10, inf]; % histogram bins of the Z-scores
for level_i = 1:n_level
    used(level_i, :) = mean(cell2mat(cellfun(@(x) x.used(level_i, :), out, 'UniformOutput', false)), 1);
    rng(1, 'twister');
    for m = 1:2
        Z = double(cell2mat(cellfun(@(x) x.(fields{m, 1})(:, :, level_i), out, 'UniformOutput', false)));
        T = cell2mat(cellfun(@(x) x.(fields{m, 2}), out, 'UniformOutput', false));
        rec = repelem((1:n_rep)', numel(out{1}.(fields{m, 2})), 1); % simulated recording of each unit

        % bootstrap weights of the simulated recordings (resampled with replacement)
        W = ones(n_boot + 1, n_rep);
        for b = 1:n_boot
            W(b + 1, :) = accumarray(randi(n_rep, n_rep, 1), 1, [n_rep, 1])';
        end

        % AUROC (Mann-Whitney) of each threshold from the weighted histograms of the true and absent effects
        A = zeros(n_boot + 1, n_th);
        for s = 1:n_th
            bin = discretize(Z(:, s), edges);
            Pc = W * accumarray([rec(T), bin(T)], 1, [n_rep, numel(edges) - 1]);
            Nc = W * accumarray([rec(~T), bin(~T)], 1, [n_rep, numel(edges) - 1]);
            A(:, s) = sum(Pc .* (cumsum(Nc, 2) - 0.5 * Nc), 2) ./ (sum(Pc, 2) .* sum(Nc, 2));
        end
        auc{m}(:, :, level_i) = A(1, :)' + [0, -1, 1] .* std(A(2:end, :), 0, 1)';
    end
end
auc_act = auc{1};
auc_con = auc{2};
save(f_res, 'auc_act', 'auc_con', 'used', '-append');

% example recording
rng(1, 'twister');
ex = sim_dilution_recording(tp, max_noise, noise_scales, thresholds, highpass, ar_order_sec, true);

% draw figure
font_tick = 12;
font_label = 14;
font_title = 14;
font_legend = 12;
colors = lines(n_level);
noise_labels = arrayfun(@(v) sprintf('Noise scale = %.1f', v), noise_scales, 'UniformOutput', false);
fig = figure('Color', 'w', 'Position', [50, 50, 1500, 1000], 'Visible', 'off');
tl_traces = tiledlayout(uipanel(fig, 'Position', [0.05, 0.58, 0.90, 0.38], 'BorderType', 'none'), 3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
tl_metrics = tiledlayout(uipanel(fig, 'Position', [0.05, 0.06, 0.90, 0.46], 'BorderType', 'none'), 1, 3, 'TileSpacing', 'loose', 'Padding', 'compact');

% A: example channels (clean, medium and heavy dilution)
[~, clean_k] = min(ex.ch_noise_levels);
[~, noisy_k] = max(ex.ch_noise_levels);
[~, mid_k] = min(abs(ex.ch_noise_levels - max_noise / 2));
ex_chans = [clean_k, mid_k, noisy_k];
ex_titles = {'Clean Channel', 'Medium Dilution Channel', 'Heavily Diluted Channel'};
for c = 1:3
    ax = nexttile(tl_traces);
    hold(ax, 'on'); box(ax, 'on');
    for level_i = 1:n_level
        plot(ax, ex.t1, ex.y(:, ex_chans(c), level_i), 'Color', colors(level_i, :), 'LineWidth', 0.8);
    end
    plot(ax, ex.t1, ex.true_signal(:, ex_chans(c)), 'k', 'LineWidth', 1.5);
    xlim(ax, [ex.t1(1), ex.t1(end)]);
    ylabel(ax, 'HbO');
    title(ax, sprintf('\\bfA%d\\rm  %s (Base SD = %.2f)', c, ex_titles{c}, ex.ch_noise_levels(ex_chans(c))), 'FontSize', font_title);
    ax.TitleHorizontalAlignment = 'left';
    ax.FontSize = font_tick;
    if c == 3
        xlabel(ax, 'Time (s)');
        legend(ax, [noise_labels, {'True Signal'}], 'Location', 'southoutside', 'Orientation', 'horizontal', 'FontSize', font_legend);
    end
end

% B: retained channels, C, D: AUROC of the activation and the connectivity
x_labels = arrayfun(@(v) sprintf('%g', v), thresholds, 'UniformOutput', false);
x_labels{end} = 'All (\infty)';
x_idx = 1:n_th;
metrics = {100 * used', auc_act, auc_con};
y_names = {'Retained Channels (%)', 'AUROC (Mean \pm SE)', 'AUROC (Mean \pm SE)'};
titles = {'\bfB\rm  Data Retained vs Threshold', '\bfC\rm  Activation AUROC vs Threshold', '\bfD\rm  Connectivity AUROC vs Threshold'};
for m = 1:3
    ax = nexttile(tl_metrics);
    hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
    for level_i = 1:n_level
        if m == 1
            plot(ax, x_idx, metrics{m}(:, level_i), 'o-', 'Color', colors(level_i, :), 'LineWidth', 1.5, 'MarkerFaceColor', [1, 1, 1], 'MarkerSize', 4);
        else
            fill(ax, [x_idx, fliplr(x_idx)], [metrics{m}(:, 3, level_i); flipud(metrics{m}(:, 2, level_i))]', colors(level_i, :), ...
                'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
            plot(ax, x_idx, metrics{m}(:, 1, level_i), 'o-', 'Color', colors(level_i, :), 'LineWidth', 1.5, 'MarkerFaceColor', [1, 1, 1], 'MarkerSize', 5);
        end
    end
    xlim(ax, [0.6, n_th + 0.4]);
    if m == 1
        ylim(ax, [0, 105]);
        legend(ax, noise_labels, 'Location', 'best', 'FontSize', font_legend);
    else
        yline(ax, 0.5, 'k:', 'HandleVisibility', 'off');
        ylim(ax, [0.45, 1.0]);
    end
    ax.XTick = x_idx;
    ax.XTickLabel = x_labels;
    ax.XTickLabelRotation = 45;
    xlabel(ax, 'Noise Rejection Threshold (Max Allowed SD)', 'FontSize', font_label);
    ylabel(ax, y_names{m}, 'FontSize', font_label);
    title(ax, titles{m}, 'FontSize', font_title);
    ax.FontSize = font_tick;
end

print(fig, [f_fig, '.png'], '-dpng', '-r300');
print(fig, [f_fig, '.pdf'], '-dpdf', '-bestfit');
