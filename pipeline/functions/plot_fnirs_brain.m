% This function plots channel values on the cortical surface of the FieldTrip template, interpolated from the channel
% midpoints within 25 mm of a channel and blended to gray by the significance of each channel
%
% Inputs
% probe: optode positions (srcPos, detPos, mm) and channels (link)
% val: value of each channel
% sig: significance of each channel (0-1, 0 = gray)
% view_angle: view of the axes
% cmap: colormap
% show_optodes: draw the optodes and the channel lines
% src_labels, det_labels: labels of the sources and detectors
%
% Written by Pangyu Joo 10/6/2026

function plot_fnirs_brain(probe, val, sig, view_angle, cmap, show_optodes, src_labels, det_labels)
dist_threshold = 25; % distance from a channel within which the surface is colored (mm)
gray = [0.7, 0.7, 0.7]; % color of the surface without channels
ft_defaults

%% values on the template surface
src = probe.srcPos(probe.link.source, :);
det = probe.detPos(probe.link.detector, :);
pos_ch = (src + det) / 2; % channel midpoints
[~, ft_path] = ft_version;
mesh = ft_read_headshape(fullfile(ft_path, 'template', 'sourcemodel', 'cortex_8196.surf.gii'));
cfg = [];
cfg.parameter = 'pow';
source = struct('pos', pos_ch, 'pow', val(:), 'inside', 1:size(pos_ch, 1), 'outside', []);
interp = ft_sourceinterpolate(cfg, source, mesh);
interp_val = interp.pow;
source.pow = double(sig(:));
interp = ft_sourceinterpolate(cfg, source, mesh);
interp_sig = interp.pow;

% vertex colors: colormap within the distance threshold, blended to gray by the significance
min_dist = min(pdist2(mesh.pos, pos_ch), [], 2); % distance of each vertex to the nearest channel
norm_val = max(0, min(1, (interp_val - min(val)) / (max(val) - min(val)))); % scaled to the range of the values
rgb = cmap(max(1, min(size(cmap, 1), round(norm_val * (size(cmap, 1) - 1)) + 1)), :);
s = max(0, min(1, interp_sig));
colors = s .* rgb + (1 - s) .* gray;
colors(min_dist > dist_threshold, :) = repmat(gray, sum(min_dist > dist_threshold), 1);

%% surface
patch('Faces', mesh.tri, 'Vertices', mesh.pos, 'FaceVertexCData', colors, 'FaceColor', 'interp', 'EdgeColor', 'none');
material dull;
lighting gouraud;
camlight;
axis off;
axis image;
hold on;

%% optodes and channel lines
if show_optodes
    for c = 1:size(pos_ch, 1)
        plot3([src(c, 1), det(c, 1)], [src(c, 2), det(c, 2)], [src(c, 3), det(c, 3)], 'Color', [0.85, 0.85, 0.85], 'LineWidth', 1.0);
    end
    all_pos = [probe.srcPos; probe.detPos];
    radius = 0.015 * max(max(all_pos, [], 1) - min(all_pos, [], 1)); % optode radius
    [Xs, Ys, Zs] = sphere(32);
    pos = {probe.srcPos, probe.detPos};
    color = {[0.9, 0.1, 0.1], [0.1, 0.1, 0.9]}; % sources red, detectors blue
    labels = {src_labels, det_labels};
    for o = 1:2
        for i = 1:size(pos{o}, 1)
            surf(radius * Xs + pos{o}(i, 1), radius * Ys + pos{o}(i, 2), radius * Zs + pos{o}(i, 3), 'FaceColor', color{o}, 'LineStyle', 'none');
        end
        for i = 1:numel(labels{o})
            text(pos{o}(i, 1), pos{o}(i, 2), pos{o}(i, 3) + radius, labels{o}{i}, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
        end
    end
    h_src = plot3(nan, nan, nan, 'o', 'MarkerFaceColor', color{1} * 0.8, 'MarkerEdgeColor', 'none', 'MarkerSize', 7, 'LineStyle', 'none');
    h_det = plot3(nan, nan, nan, 'o', 'MarkerFaceColor', color{2} * 0.8, 'MarkerEdgeColor', 'none', 'MarkerSize', 7, 'LineStyle', 'none');
    legend([h_src, h_det], {'Sources', 'Detectors'}, 'Location', 'northwest', 'box', 'off', 'AutoUpdate', 'off');
end
view(view_angle);
end
