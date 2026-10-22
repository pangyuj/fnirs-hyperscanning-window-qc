% This function builds the template of the validation simulations (1-Hz time base, conditions, responses, sections and
% optodes) from the timing and channels of a task recording (paper_template.mat)
%
% Inputs
% template: time, conds, onset, dur, source, detector and sections of the task recording
%
% Outputs
% tp: template (h, X, conds, sec_ind, opt, n_opt)
%
% Written by Pangyu Joo 10/6/2026

function [tp] = sim_template(template)
t1 = template.time(:);
n_ch = numel(template.source);
%% conditions
stim = Dictionary();
for k = 1:numel(template.conds)
    st = nirs.design.StimulusEvents();
    st.name = template.conds{k};
    st.onset = template.onset{k};
    st.dur = template.dur{k};
    st.amp = ones(numel(st.onset), 1);
    stim(template.conds{k}) = st;
end

%% responses (canonical HRF, peak 1)
basis = Dictionary();
basis('default') = nirs.design.basis.Canonical();
[X, names] = nirs.design.createDesignMatrix(stim, t1, basis);
[~, ord] = ismember(template.conds, names);
tp.X = X(:, ord) ./ max(abs(X(:, ord)), [], 1);
tp.conds = template.conds;

%% sections
tp.sec_ind = false(numel(t1), numel(template.sections));
for k = 1:numel(template.sections)
    st = stim(template.sections{k});
    for e = 1:numel(st.onset)
        tp.sec_ind(:, k) = tp.sec_ind(:, k) | (t1 >= st.onset(e) & t1 < st.onset(e) + st.dur(e));
    end
end

%% channels and optodes
[~, ~, i_src] = unique(template.source);
[~, ~, i_det] = unique(template.detector);
tp.opt = [i_src, max(i_src) + i_det]; % optodes (source, detector) of each channel
tp.n_opt = max(i_src) + max(i_det);
link = table((1:n_ch)', ones(n_ch, 1), repmat({'hbo'}, n_ch, 1), 'VariableNames', {'source', 'detector', 'type'}); % one source per channel
h = nirs.core.Data();
h.time = t1;
h.data = zeros(numel(t1), n_ch);
h.probe = nirs.core.Probe(zeros(n_ch, 3), zeros(1, 3), link);
h.stimulus = stim;
tp.h = h;
end
