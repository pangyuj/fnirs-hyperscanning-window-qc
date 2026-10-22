% This function fits a first-level GLM to fNIRS data with excluded samples by AR-IRLS (Barker et al., 2013): the AR
% model is fitted on the continuous data and the excluded samples are removed from the regression only
%
% Inputs
% data: nirs.core.Data (time x channel), NaN for missing samples
% trend_func: function of time returning trend regressors
% Pmax_sec: maximum AR model order (sec)
% valid_mask: samples entering the regression (time x channel, logical)
%
% Outputs
% S: nirs.core.ChannelStats of the task regressors (NaN for channels that cannot be fitted)
%
% Written by Pangyu Joo 9/28/2026

function [S] = ar_irls_gap(data, trend_func, Pmax_sec, valid_mask)
warning('off', 'stats:statrobustfit:IterationLimit'); % bisquare IRLS reaching its iteration limit
tune = 4.685; % bisquare tuning constant
max_iter = 10; % maximum number of IRLS iterations
tol = 1e-2; % relative change of the coefficients for convergence
glm = nirs.modules.AR_IRLS(); % toolbox GLM module, for its design check nirs.modules.AbstractGLM/checkRank
basis = Dictionary();
basis('default') = nirs.design.basis.Canonical(); % canonical HRF

%% design matrix
t = data.time;
Fs = data.Fs;
Pmax = ceil(Pmax_sec*Fs);
[X, names] = nirs.design.createDesignMatrix(data.stimulus, t, basis);
X(isnan(X)) = 0;
D = [X trend_func(t)];
ncond = size(X,2);

%% channel-wise AR-IRLS
Y = data.data - mean(data.data, 1, 'omitnan');
nchan = size(Y,2);
beta = NaN(ncond, nchan);
covb = NaN(ncond, ncond, nchan);
dfe = NaN(1, nchan);
for c = 1:nchan
    y = Y(:,c);
    valid = valid_mask(:,c) & ~isnan(y); % samples entering the regression
    ar_rows = movsum(double(~isnan(y)), [Pmax 0], 'Endpoints', 'fill') == Pmax+1;
    if sum(ar_rows) <= Pmax
        continue % AR model not estimable
    end

    % regressors with nonzero values on the valid samples, as nirs.modules.AR_IRLS removes conditions without events;
    % a design that the toolbox check refuses (rank-deficient) is not fitted
    keep = find(any(D(valid,:) ~= 0, 1));
    try
        glm.checkRank(D(valid,keep));
    catch err
        if ~strcmp(err.message, 'Design matrix is rank deficient.')
            rethrow(err)
        end
        continue % rank-deficient design
    end
    if sum(valid) <= numel(keep)
        continue % no residual degree of freedom
    end

    % missing samples filled for the whitening filter only
    y_fill = fillmissing(y, 'spline', 'EndValues', 'nearest');

    % ordinary least squares as the initial estimate; the AR model is fitted on the residual of all
    % non-missing samples, including samples excluded from the regression by valid_mask
    B = D(valid,keep)\y(valid);
    fitted = true;
    for iter = 1:max_iter
        % AR model of the residual: nested least squares fits of orders 1...Pmax from one QR factorization, on the
        % samples whose lags (forward prediction) or leads (backward prediction) are all valid, order chosen by BIC
        res = y - D(:,keep)*B;
        lagged = NaN(length(res), Pmax); % res(t-k)
        led = NaN(length(res), Pmax); % res(t+k)
        for k = 1:Pmax
            lagged(k+1:end, k) = res(1:end-k);
            led(1:end-k, k) = res(k+1:end);
        end
        fwd = ~isnan(res) & all(~isnan(lagged), 2); % the same samples for every order
        bwd = ~isnan(res) & all(~isnan(led), 2);
        target = [res(fwd); res(bwd)];
        lagged = [lagged(fwd,:); led(bwd,:)];
        m = length(target);
        [Q, R] = qr(lagged, 0);
        z = Q'*target;
        rss = sum(target.^2) - cumsum(z.^2);
        bic = m*log(rss/m) + (1:Pmax)'*log(m);
        [~, p_best] = min(bic);
        phi = R(1:p_best,1:p_best)\z(1:p_best); % e(t) = res(t) - phi(1)*res(t-1) - ... - phi(p)*res(t-p)

        % AR whitening filter applied to the data and the design; each column is referenced to its first sample so
        % that the filter starts from rest, and the filtered reference level is added back
        filt = [1; -phi];
        yf = filter(filt, 1, y_fill - y_fill(1)) + sum(filt)*y_fill(1);
        Df = filter(filt, 1, D(:,keep) - D(1,keep)) + sum(filt)*D(1,keep);

        % regressors estimable after whitening: linearly independent columns on the valid samples (QR with column
        % pivoting) in their original order, with the rank tolerance of robustfit (statrobustfit) so that robustfit
        % drops no further column
        Dw = Df(valid,:);
        [~, Rq, piv] = qr(Dw, 0);
        sel = sort(piv(1:sum(abs(diag(Rq)) > abs(Rq(1,1))*max(size(Dw))*eps(class(Rq)))));
        keep = keep(sel);
        Dw = Dw(:,sel);
        B_old = B(sel);
        if isempty(sel) || size(Dw,1) <= size(Dw,2)
            fitted = false; % no estimable regressor or no residual degree of freedom
            break
        end
        [B, st] = robustfit(Dw, yf(valid), 'bisquare', tune, 'off');
        if norm(B - B_old)/norm(B_old) < tol
            break
        end
    end
    if ~fitted
        continue
    end
    % covariance from the robust residual scale and the robustly weighted whitened design
    % (Barker et al., 2013), degrees of freedom by the Satterthwaite approximation of the
    % weighted residual projection
    task_cols = find(keep <= ncond); % kept columns that are task regressors
    Xw = st.w.*Dw;
    V = st.robust_s^2*pinv(Xw'*Xw);
    beta(keep(task_cols), c) = B(task_cols);
    covb(keep(task_cols), keep(task_cols), c) = V(task_cols, task_cols);

    % Satterthwaite degrees of freedom tr(A)^2/tr(A^2) of A = H'*H with H = diag(w) - P, P the orthogonal projection
    % onto the columns of Xw, from traces with an orthonormal basis of Xw instead of the n x n matrices
    w = st.w;
    [Qx, Rx] = qr(Xw, 0);
    k = sum(abs(diag(Rx)) > max(size(Xw))*eps(abs(Rx(1,1))));
    Qx = Qx(:,1:k);
    M1 = Qx'*(w.*Qx);
    M2 = Qx'*(w.^2.*Qx);
    M3 = Qx'*(w.^3.*Qx);
    trA = sum(w.^2) - 2*trace(M1) + k;
    trA2 = sum(w.^4) - 4*trace(M3) + 4*trace(M2) + 2*sum(M1.^2, 'all') - 4*trace(M1) + k;
    dfe(c) = trA^2/trA2;
end

%% channel statistics (nirs.core.ChannelStats)
link = data.probe.link;
cond = repmat(names(:)', [nchan 1]);
variables = [repmat(link, [ncond 1]) table(cond(:), 'VariableNames', {'cond'})];
beta_t = beta';
beta_all = beta_t(:);
covb_all = zeros(nchan*ncond);
dfe_all = zeros(nchan*ncond, 1);
dfe(isnan(dfe)) = 0; % channels that were not fitted
for c = 1:nchan
    idx = (0:ncond-1)*nchan + c; % positions of channel c in the condition-major order
    covb_all(idx, idx) = covb(:,:,c);
    dfe_all(idx) = dfe(c);
end

% sort as nirs.core.ChannelStats/sorted does, so that the variable-wise dfe stays aligned
[variables, order] = nirs.util.sortrows(variables, {'source', 'detector', 'type', 'cond'});
S = nirs.core.ChannelStats();
S.variables = variables;
S.beta = beta_all(order);
S.covb = covb_all(order, order);
S.dfe = dfe_all(order);
S.description = data.description;
S.demographics = data.demographics;
S.probe = data.probe;
S.probe.link = nirs.util.sortrows(link, {'source', 'detector', 'type'});
stim = Dictionary();
for j = 1:data.stimulus.count
    ev = data.stimulus.values{j};
    s = nirs.design.StimulusEvents();
    s.name = ev.name;
    s.dur = mean(ev.dur);
    stim(data.stimulus.keys{j}) = s;
end
S.basis.base = basis;
S.basis.Fs = Fs;
S.basis.stim = stim;
end
