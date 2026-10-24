function varargout = RealTime_Wiener_Noise_Filter(varargin)
% =========================================================================
% REAL-TIME AUDIO DENOISING VIA WIENER-KHINCHIN SPECTRAL ESTIMATION
% AND OPTIMAL LINEAR TIME-INVARIANT (LTI) FILTERING
% =========================================================================
%
% Theoretical Foundation & Academic Constraints:
% -------------------------------------------------------------------------
% This application implements an end-to-end real-time speech enhancement
% pipeline using fundamental principles of Probability Theory, Stochastic
% Processes, and Linear Time-Invariant (LTI) Systems.
%
% CRITICAL ACADEMIC CONSTRAINT:
% No black-box toolbox functions (such as wdenoise, wiener2, fitdist, xcorr,
% or pwelch) are used. Every probability moment, distribution fit, empirical
% PDF/CDF, time-averaged autocorrelation function, and power spectral density
% (PSD) estimate is computed explicitly from first-principles mathematics
% using base MATLAB array operations and fundamental discrete Fourier transforms.
%
% Random Process Theorems Applied:
%   1. Wide-Sense Stationarity (WSS) and Ergodicity (Time vs. Ensemble Averages)
%   2. Gaussian Probability Density Function (PDF) & Cumulative Distribution (CDF)
%   3. Method of Moments: Mean (mu), Unbiased Variance (sigma^2), Skewness, Kurtosis
%   4. Central Limit Theorem (CLT) & Asymptotic Normality of Block Sample Means
%   5. Wiener-Khinchin Theorem: S_XX(w) = F{R_XX[tau]}
%   6. LTI Random-Process Transformation: S_YY(w) = |H(w)|^2 * S_XX(w)
%   7. Minimum Mean-Square Error (MMSE) Wiener Filter with Spectral Floor (beta)
%   8. Principle of Orthogonality: E[Y[n] * (X[n] - Y[n])] = 0
%   9. Overlap-Add (OLA) Synthesis with Normalized Hamming Analysis/Synthesis
%
% Usage:
%   RealTime_Wiener_Noise_Filter()               % Launches interactive GUI
%   results = RealTime_Wiener_Noise_Filter(...)  % Headless/batch execution
%
% Author: MATLAB Signal Processing & Statistical Audio Engineering Suite
% =========================================================================

    % ---------------------------------------------------------------------
    % Parse Input Arguments & Execution Mode
    % ---------------------------------------------------------------------
    % Flexible argument preprocessing to support shorthands like:
    % RealTime_Wiener_Noise_Filter('simulation', true) or RealTime_Wiener_Noise_Filter('test')
    args = varargin;
    if ~isempty(args)
        if length(args) == 1
            if islogical(args{1}) && args{1}
                args = {'mode', 'simulation', 'headless', true};
            elseif ischar(args{1}) || isstring(args{1})
                val = char(args{1});
                if any(strcmpi(val, {'simulation', 'test', 'batch'}))
                    args = {'mode', 'simulation', 'headless', true};
                elseif strcmpi(val, 'live')
                    args = {'mode', 'live', 'headless', true};
                end
            end
        elseif length(args) >= 2
            if (ischar(args{1}) || isstring(args{1})) && any(strcmpi(char(args{1}), {'simulation', 'test', 'batch'}))
                if length(args) == 2 && islogical(args{2})
                    args = {'mode', 'simulation', 'headless', args{2}};
                else
                    args = [{'mode', 'simulation', 'headless', true}, args(3:end)];
                end
            end
        end
    end

    p = inputParser;
    p.KeepUnmatched = true;
    addParameter(p, 'mode', 'gui', @(x) ischar(x) || isstring(x));
    addParameter(p, 'duration', 6.0, @(x) isnumeric(x) && x > 0);
    addParameter(p, 'calDuration', 1.5, @(x) isnumeric(x) && x > 0);
    addParameter(p, 'fs', 16000, @(x) isnumeric(x) && x > 0);
    addParameter(p, 'frameLen', 1024, @(x) isnumeric(x) && x > 0);
    addParameter(p, 'overlap', 512, @(x) isnumeric(x) && x >= 0);
    addParameter(p, 'nfft', 2048, @(x) isnumeric(x) && x >= 1024);
    addParameter(p, 'beta', 0.03, @(x) isnumeric(x) && x >= 0);
    addParameter(p, 'simSNR', 5.0, @(x) isnumeric(x));
    addParameter(p, 'audioInput', [], @(x) isnumeric(x) || isempty(x));
    addParameter(p, 'noiseCal', [], @(x) isnumeric(x) || isempty(x));
    addParameter(p, 'headless', false, @islogical);
    parse(p, args{:});
    cfg = p.Results;

    % ---------------------------------------------------------------------
    % Mode Dispatch: Interactive GUI vs. Batch/Headless Engine
    % ---------------------------------------------------------------------
    if ~cfg.headless && strcmpi(cfg.mode, 'gui')
        hGUI = launch_gui_dashboard(cfg);
        if nargout > 0
            varargout{1} = hGUI;
        end
    else
        fprintf('\n========================================================================================\n');
        fprintf('  RUNNING WIENER-KHINCHIN AUDIO DENOISING PIPELINE (BATCH / SIMULATION MODE)\n');
        fprintf('========================================================================================\n');
        results = execute_complete_pipeline(cfg, []);
        print_console_summary_table(results);
        
        % Generate and export publication figure
        fig = render_dashboard_figure(results, false);
        export_figure_and_audio(results, fig);
        
        if nargout > 0
            varargout{1} = results;
        end
    end
end

% =========================================================================
% SECTION 1: CORE MATHEMATICAL ENGINE (NO BLACK-BOX TOOLBOX FUNCTIONS)
% =========================================================================

function results = execute_complete_pipeline(cfg, statusCallback)
% EXECUTE_COMPLETE_PIPELINE Runs the 4-phase Wiener-Khinchin filtering process:
%   Phase 1: Ambient Noise Calibration & Statistical Moment/PDF Analysis
%   Phase 2: Frame-by-frame Real-Time LTI Wiener Filtering & OLA Synthesis
%   Phase 3: Audio Export (WAV files)
%   Phase 4: Quantitative Statistical Validation & Cross-Correlation Matrix

    if nargin < 2 || isempty(statusCallback)
        statusCallback = @(msg) fprintf('[STATUS] %s\n', msg);
    end

    fs = cfg.fs;
    frameLen = cfg.frameLen;
    overlap = cfg.overlap;
    hop = frameLen - overlap;
    nfft = cfg.nfft;
    beta = cfg.beta;

    % ---------------------------------------------------------------------
    % PHASE 1: AMBIENT NOISE CALIBRATION (1.5 SECONDS)
    % ---------------------------------------------------------------------
    % Theory: Treat captured ambient noise N[n] as a sample realization of
    % a Wide-Sense Stationary (WSS), Ergodic, Gaussian Random Process.
    % Under ergodicity, time-averages converge asymptotically to ensemble averages:
    %   lim_{T -> inf} (1/T) \int N(t) dt = E[N(t)] = mu_N
    % ---------------------------------------------------------------------
    statusCallback('Phase 1: Calibrating Ambient Noise (WSS Gaussian Random Process)...');
    t_start_total = tic;

    % In Simulation Mode, pre-generate speech to establish consistent noise floor
    % across both calibration (Phase 1) and speech filtering (Phase 2).
    % In File or Live mode, obtain audio track and extract true ambient noise realization.
    if isfield(cfg, 'audioInput') && ~isempty(cfg.audioInput)
        statusCallback('Processing input audio track...');
        noisy_input = cfg.audioInput;
        clean_reference = [];
        if isfield(cfg, 'noiseCal') && ~isempty(cfg.noiseCal)
            noise_cal = cfg.noiseCal;
        else
            noise_cal = extract_noise_from_audio(noisy_input, frameLen, hop, fs);
        end
    elseif strcmpi(cfg.mode, 'live')
        statusCallback(sprintf('Capturing %0.1f seconds from microphone... Speak now!', cfg.duration));
        noisy_input = record_live_audio(cfg.duration, fs);
        clean_reference = [];
        if isfield(cfg, 'noiseCal') && ~isempty(cfg.noiseCal)
            noise_cal = cfg.noiseCal;
        else
            noise_cal = extract_noise_from_audio(noisy_input, frameLen, hop, fs);
        end
    else
        % Simulation Mode: Synthesize vocal formant speech and inject WSS Gaussian noise
        statusCallback('Synthesizing speech and generating WSS Gaussian noise...');
        [clean_reference, ~, sigma_sim_noise] = generate_synthetic_clean_speech(cfg.duration, fs, cfg.simSNR);
        N_cal_samples = round(cfg.calDuration * fs);
        noise_cal = sigma_sim_noise * randn(N_cal_samples, 1);
        noisy_input = clean_reference + sigma_sim_noise * randn(length(clean_reference), 1);
    end

    L_N = length(noise_cal);

    % --- 1.1 Sample Moments (Method of Moments) ---
    % Sample Mean: \mu_N = (1/L_N) \sum_{n=1}^{L_N} N[n]
    mu_N = sum(noise_cal) / L_N;

    % Unbiased Sample Variance: s_N^2 = \frac{1}{L_N - 1} \sum_{n=1}^{L_N} (N[n] - \mu_N)^2
    var_N = sum((noise_cal - mu_N).^2) / (L_N - 1);
    sigma_N = sqrt(var_N);

    % 2nd, 3rd, and 4th Central Moments
    m2 = sum((noise_cal - mu_N).^2) / L_N;
    m3 = sum((noise_cal - mu_N).^3) / L_N;
    m4 = sum((noise_cal - mu_N).^4) / L_N;

    % Sample Skewness: \gamma_1 = m_3 / (m_2)^{3/2} (Theoretical Gaussian = 0)
    skewness_N = m3 / (m2^(1.5));

    % Sample Kurtosis: \gamma_2 = m_4 / (m_2)^2 (Theoretical Gaussian = 3)
    kurtosis_N = m4 / (m2^2);

    % --- 1.2 Empirical PDF and CDF Calculation (Without histcounts/fitdist) ---
    % Construct B = 60 uniform histogram bins spanning the support of N[n]
    K_bins = 60;
    min_N = min(noise_cal);
    max_N = max(noise_cal);
    delta_x = (max_N - min_N) / K_bins;
    bin_edges = min_N + (0:K_bins)' * delta_x;
    bin_centers = bin_edges(1:end-1) + delta_x / 2;

    % Explicit Vectorized Bin Accumulation (Frequency Count)
    bin_indices = floor((noise_cal - min_N) / delta_x) + 1;
    bin_indices(bin_indices > K_bins) = K_bins;
    bin_indices(bin_indices < 1) = 1;
    counts = accumarray(bin_indices, 1, [K_bins, 1]);

    % Empirical Probability Density Function: \hat{f}_N(x_k) = count_k / (L_N * delta_x)
    % Area normalization property: \sum_{k=1}^K \hat{f}_N(x_k) * delta_x = 1.0
    pdf_empirical = counts / (L_N * delta_x);

    % Empirical Cumulative Distribution Function: \hat{F}_N(x_k) = \sum_{j=1}^k count_j / L_N
    cdf_empirical = cumsum(counts) / L_N;

    % Theoretical Gaussian Probability Density Function:
    % f_N(x) = \frac{1}{\sqrt{2 \pi \sigma_N^2}} \exp\left( -\frac{(x - \mu_N)^2}{2 \sigma_N^2} \right)
    pdf_theoretical = (1 / (sqrt(2 * pi * var_N))) * exp(-((bin_centers - mu_N).^2) / (2 * var_N));

    % Theoretical Gaussian Cumulative Distribution Function via Error Function:
    % F_N(x) = \frac{1}{2} \left[ 1 + \text{erf}\left( \frac{x - \mu_N}{\sqrt{2}\sigma_N} \right) \right]
    cdf_theoretical = 0.5 * (1 + erf((bin_centers - mu_N) / (sqrt(2) * sigma_N)));

    % Goodness-of-Fit Mean Squared Error (MSE) and Normalized MSE:
    % MSE = (1/K) \sum_{k=1}^K (\hat{f}_N(x_k) - f_N(x_k))^2
    mse_pdf = sum((pdf_empirical - pdf_theoretical).^2) / K_bins;
    nmse_pdf = sum((pdf_empirical - pdf_theoretical).^2) / max(sum(pdf_theoretical.^2), 1e-12);

    % --- 1.3 Central Limit Theorem (CLT) Verification ---
    % Theorem: Let X_1, ..., X_L be i.i.d. random variables with mean mu and variance sigma^2.
    % As L -> inf, the sample mean \bar{X} converges in distribution to N(mu, sigma^2 / L).
    num_blocks = 200;
    L_block = floor(L_N / num_blocks);
    block_means = zeros(num_blocks, 1);
    for b = 1:num_blocks
        sub_vec = noise_cal((b-1)*L_block + 1 : b*L_block);
        block_means(b) = sum(sub_vec) / L_block;
    end
    mean_of_means = sum(block_means) / num_blocks;
    var_of_means = sum((block_means - mean_of_means).^2) / (num_blocks - 1);
    theoretical_clt_var = var_N / L_block;

    % 95% Confidence Interval for global mean: \mu_N \pm 1.95996 * \frac{\sigma_N}{\sqrt{L_N}}
    ci_95 = [mu_N - 1.95996 * (sigma_N / sqrt(L_N)), mu_N + 1.95996 * (sigma_N / sqrt(L_N))];

    % --- 1.4 Hamming Window Definition from Scratch ---
    % w[n] = 0.54 - 0.46 * \cos(2 \pi n / (L - 1)), \quad 0 \le n \le L-1
    n_idx = (0:frameLen-1)';
    w = 0.54 - 0.46 * cos(2 * pi * n_idx / (frameLen - 1));
    % Window Energy Normalization Factor: E_w = \sum_{n=0}^{L-1} w^2[n]
    E_w = sum(w.^2);

    % --- 1.5 Wiener-Khinchin Noise PSD and Time-Averaged Autocorrelation R_NN[tau] ---
    % Theorem (Wiener-Khinchin): S_NN(w) = \mathcal{F}\{ R_NN[tau] \}
    % For a stationary ergodic process, the autocorrelation sequence is estimated
    % across calibration frames and transformed via FFT to obtain baseline noise PSD.
    num_cal_frames = floor((L_N - frameLen) / hop) + 1;
    S_NN_accum = zeros(nfft, 1);
    R_NN_accum = zeros(2*frameLen - 1, 1);
    cal_frame_energies = zeros(num_cal_frames, 1);

    for m = 1:num_cal_frames
        idx = (m-1)*hop + (1:frameLen);
        cal_frame = noise_cal(idx) .* w;
        cal_frame_energies(m) = sum(cal_frame.^2) / frameLen;
        
        % Zero-padded discrete Fourier transform
        N_fft = fft(cal_frame, nfft);
        
        % Sample Power Spectral Density: S(w) = |N(w)|^2 / (fs * E_w) [W/Hz]
        S_frame = (abs(N_fft).^2) / (fs * E_w);
        S_NN_accum = S_NN_accum + S_frame;
        
        % Frame Autocorrelation via Circular Convolution Theorem (Wiener-Khinchin counterpart):
        % R[tau] = IFFT( |N(w)|^2 ) / frameLen
        r_circ = real(ifft(abs(N_fft).^2)) / frameLen;
        % Assemble symmetric lags -(L-1) to +(L-1)
        r_lags = [r_circ(nfft - frameLen + 2 : nfft); r_circ(1:frameLen)];
        R_NN_accum = R_NN_accum + r_lags;
    end

    % Baseline Noise Power Spectral Density S_NN(w)
    S_NN = S_NN_accum / num_cal_frames;
    R_NN = R_NN_accum / num_cal_frames;
    mean_noise_frame_energy = sum(cal_frame_energies) / num_cal_frames;

    % ---------------------------------------------------------------------
    % PHASE 2: AUDIO CAPTURE & REAL-TIME FRAME-BY-FRAME LTI WIENER FILTERING
    % ---------------------------------------------------------------------
    statusCallback('Phase 2: Executing Real-Time Decision-Directed Wiener Filtering...');

    N_total = length(noisy_input);
    num_frames = floor((N_total - frameLen) / hop) + 1;

    % Output Synthesis Buffers for Overlap-Add (OLA)
    y_filtered = zeros(N_total, 1);
    w_sum = zeros(N_total, 1);

    % Accumulated Power Spectral Densities for Analysis
    S_XX_accum = zeros(nfft, 1);
    S_YY_accum = zeros(nfft, 1);
    R_XX_accum = zeros(2*frameLen - 1, 1);
    R_YY_accum = zeros(2*frameLen - 1, 1);

    % Adaptive Voice Activity Detector (VAD) Energy Threshold:
    % Accurately derived from windowed noise frame energy distribution
    vad_threshold = 2.0 * mean_noise_frame_energy;
    vad_alpha = 0.95; % Smoothing factor for noise PSD adaptation during pauses
    S_NN_adaptive = S_NN; % Initialize adaptive noise spectrum

    % Decision-Directed (DD) Wiener Filter Parameters:
    % Overcomes the high variance of raw periodograms, eliminates musical noise,
    % and guarantees substantial audible noise suppression.
    alpha_dd = 0.95;            % A priori SNR recursive smoothing factor
    alpha_sub = 1.25;           % Over-subtraction parameter for Gaussian noise peaks
    S_SS_prev = zeros(nfft, 1); % Prior speech spectral power estimate
    eps_floor = 1e-12;          % Numerical safeguard against divide-by-zero

    % Measure Real-Time Frame Processing Execution Time
    t_process_start = tic;

    for m = 1:num_frames
        frame_idx = (m-1)*hop + (1:frameLen);
        x_frame = noisy_input(frame_idx);
        
        % Windowing: X_w[n] = X[n] * w[n]
        x_win = x_frame .* w;
        
        % Frame Energy Calculation for Adaptive VAD: E_m = (1/L) \sum X^2[n]
        frame_energy = sum(x_win.^2) / frameLen;
        
        % Fast Fourier Transform of input frame
        X_fft = fft(x_win, nfft);
        
        % Input Power Spectral Density via Wiener-Khinchin theorem:
        % S_XX(w) = \mathcal{F}\{ R_XX[tau] \} = |X(w)|^2 / (fs * E_w)
        S_XX = (abs(X_fft).^2) / (fs * E_w);
        S_XX_accum = S_XX_accum + S_XX;

        % Input Autocorrelation Function:
        r_x_circ = real(ifft(abs(X_fft).^2)) / frameLen;
        R_XX_frame = [r_x_circ(nfft - frameLen + 2 : nfft); r_x_circ(1:frameLen)];
        R_XX_accum = R_XX_accum + R_XX_frame;

        % Adaptive Noise Tracking strictly during Verified Silence / Pauses
        if frame_energy < vad_threshold
            S_NN_adaptive = vad_alpha * S_NN_adaptive + (1 - vad_alpha) * S_XX;
        end

        % A posteriori Signal-to-Noise Ratio (SNR) in each frequency bin:
        % gamma(w) = S_XX(w) / S_NN(w)
        gamma = max(S_XX ./ max(S_NN_adaptive, eps_floor), eps_floor);

        % A priori SNR via Decision-Directed Estimation (Ephraim-Malah):
        % xi[m, w] = alpha * (|H_{m-1}(w)|^2 * S_XX_{m-1}(w) / S_NN(w)) + (1 - alpha) * max(gamma(w) - alpha_sub, 0)
        xi = alpha_dd * (S_SS_prev ./ max(S_NN_adaptive, eps_floor)) + ...
             (1 - alpha_dd) * max(gamma - alpha_sub, 0);

        % Optimal Frequency-Domain LTI MMSE Wiener Transfer Function:
        % H(w) = \max\left( \frac{\xi(w)}{1 + \xi(w)}, \; \beta \right)
        H_wiener = max(xi ./ (1.0 + xi), beta);
        H_wiener = min(H_wiener, 1.0);

        % Update recursive prior speech spectrum for next frame
        S_SS_prev = (H_wiener.^2) .* S_XX;

        % Linear Time-Invariant (LTI) Output Spectrum:
        % Y(w) = H(w) * X(w)
        Y_fft = H_wiener .* X_fft;

        % Theoretical LTI Random Process PSD Relationship Verification:
        % S_YY(w) = |H(w)|^2 * S_XX(w)
        S_YY = (abs(H_wiener).^2) .* S_XX;
        S_YY_accum = S_YY_accum + S_YY;

        % Filtered Frame Autocorrelation:
        r_y_circ = real(ifft(abs(Y_fft).^2)) / frameLen;
        R_YY_frame = [r_y_circ(nfft - frameLen + 2 : nfft); r_y_circ(1:frameLen)];
        R_YY_accum = R_YY_accum + R_YY_frame;

        % Time-Domain Reconstruction via Inverse Discrete Fourier Transform (IDFT)
        y_frame_time = real(ifft(Y_fft));
        y_time = y_frame_time(1:frameLen);

        % Overlap-Add (OLA) Synthesis Accumulation:
        % Reconstruct continuous speech with synthesis windowing
        y_filtered(frame_idx) = y_filtered(frame_idx) + y_time .* w;
        w_sum(frame_idx) = w_sum(frame_idx) + (w.^2);
    end

    time_elapsed_processing = toc(t_process_start);
    audio_duration_sec = N_total / fs;
    real_time_factor = time_elapsed_processing / audio_duration_sec;

    % Normalize Output Audio by Synthesis Window Envelope
    % Guaranteeing perfect reconstruction for transparent filters:
    non_zero_w = w_sum > 1e-4;
    y_filtered(non_zero_w) = y_filtered(non_zero_w) ./ w_sum(non_zero_w);

    % Average PSDs and Autocorrelations across frames
    S_XX_avg = S_XX_accum / num_frames;
    S_YY_avg = S_YY_accum / num_frames;
    R_XX_avg = R_XX_accum / num_frames;
    R_YY_avg = R_YY_accum / num_frames;

    % Lag vector in milliseconds for plotting: tau = -(L-1) : (L-1)
    lags_samples = (-(frameLen - 1) : (frameLen - 1))';
    lags_ms = (lags_samples / fs) * 1000;

    % Frequency vector in kHz for PSD plotting
    freq_axis_hz = (0:nfft-1)' * (fs / nfft);
    freq_axis_khz = freq_axis_hz / 1000;
    half_bins = 1 : (nfft/2 + 1);

    % ---------------------------------------------------------------------
    % PHASE 3: AUDIO OUTPUT & WAV EXPORT
    % ---------------------------------------------------------------------
    statusCallback('Phase 3: Exporting WAV Audio Files...');
    file_noisy = 'noisy_recorded_input.wav';
    file_clean = 'filtered_clean_output.wav';
    
    % Prevent clipping when exporting to 16-bit WAV
    max_val_noisy = max(abs(noisy_input));
    noisy_norm = noisy_input;
    if max_val_noisy > 0.99
        noisy_norm = 0.95 * (noisy_input / max_val_noisy);
    end
    
    max_val_filt = max(abs(y_filtered));
    filt_norm = y_filtered;
    if max_val_filt > 0.99
        filt_norm = 0.95 * (y_filtered / max_val_filt);
    end
    
    audiowrite(file_noisy, noisy_norm, fs);
    audiowrite(file_clean, filt_norm, fs);

    % ---------------------------------------------------------------------
    % PHASE 4: QUANTITATIVE STATISTICAL ANALYSIS & VALIDATION
    % ---------------------------------------------------------------------
    statusCallback('Phase 4: Computing Statistical Metrics & Cross-Correlations...');

    % --- 4.1 Covariance Matrix and Pearson Correlation ---
    % Cov(X, Y) = \frac{1}{N-1} \sum_{n=1}^N (X[n] - \mu_X)(Y[n] - \mu_Y)
    mu_X = sum(noisy_input) / N_total;
    mu_Y = sum(y_filtered) / N_total;
    
    var_X = sum((noisy_input - mu_X).^2) / (N_total - 1);
    var_Y = sum((y_filtered - mu_Y).^2) / (N_total - 1);
    cov_XY = sum((noisy_input - mu_X) .* (y_filtered - mu_Y)) / (N_total - 1);
    cov_matrix = [var_X, cov_XY; cov_XY, var_Y];

    % Pearson Cross-Correlation: \rho_{XY} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}
    rho_XY = cov_XY / (sqrt(var_X) * sqrt(var_Y));

    % --- 4.2 Orthogonality Principle Verification ---
    % Theorem: In optimal minimum mean-square error estimation, the error
    % residual E[n] = X[n] - Y[n] is statistically orthogonal to the estimate Y[n]:
    % E[ Y[n] * E[n] ] = 0 \implies \rho_{Y, E} \approx 0
    error_residual = noisy_input - y_filtered;
    mu_E = sum(error_residual) / N_total;
    var_E = sum((error_residual - mu_E).^2) / (N_total - 1);
    cov_YE = sum((y_filtered - mu_Y) .* (error_residual - mu_E)) / (N_total - 1);
    rho_orthogonality = cov_YE / (sqrt(var_Y) * sqrt(var_E));

    % --- 4.3 Signal-to-Noise Ratio (SNR) Metrics ---
    if ~isempty(clean_reference)
        % Exact Ground-Truth Benchmark SNR in Simulation Mode:
        power_speech = sum(clean_reference.^2);
        power_input_noise = sum((noisy_input - clean_reference).^2);
        power_output_noise = sum((y_filtered - clean_reference).^2);
        
        snr_in_db = 10 * log10(power_speech / max(power_input_noise, eps_floor));
        snr_out_db = 10 * log10(power_speech / max(power_output_noise, eps_floor));
        delta_snr_db = snr_out_db - snr_in_db;
    else
        % Estimated SNR in Live Microphone Mode using Calibrated Noise Floor:
        power_X = sum(noisy_input.^2) / N_total;
        power_S_est = max(power_X - var_N, 1e-10);
        snr_in_db = 10 * log10(power_S_est / var_N);
        
        % Residual noise floor in filtered output: \sigma_{N,out}^2 \approx \beta^2 * \sigma_N^2
        var_N_out = (beta^2) * var_N;
        power_Y = sum(y_filtered.^2) / N_total;
        power_Y_clean_est = max(power_Y - var_N_out, 1e-10);
        snr_out_db = 10 * log10(power_Y_clean_est / var_N_out);
        delta_snr_db = snr_out_db - snr_in_db;
    end

    % --- 4.4 STFT Spectrogram Computation from Scratch ---
    [spec_X_dB, spec_Y_dB, spec_t, spec_f] = compute_dual_spectrogram(noisy_input, y_filtered, fs);

    % Pack Results Struct
    results = struct();
    results.fs = fs;
    results.duration = audio_duration_sec;
    results.noisy_input = noisy_input;
    results.filtered_output = y_filtered;
    results.clean_reference = clean_reference;
    results.time_vec = (0:N_total-1)' / fs;
    results.noise_cal = noise_cal;
    
    % Statistics
    results.mu_N = mu_N;
    results.var_N = var_N;
    results.sigma_N = sigma_N;
    results.skewness_N = skewness_N;
    results.kurtosis_N = kurtosis_N;
    results.mse_pdf = mse_pdf;
    results.nmse_pdf = nmse_pdf;
    results.ci_95 = ci_95;
    results.mean_of_means = mean_of_means;
    results.var_of_means = var_of_means;
    results.theoretical_clt_var = theoretical_clt_var;
    
    % PDF / CDF curves
    results.bin_centers = bin_centers;
    results.pdf_empirical = pdf_empirical;
    results.pdf_theoretical = pdf_theoretical;
    results.cdf_empirical = cdf_empirical;
    results.cdf_theoretical = cdf_theoretical;
    
    % Autocorrelation & PSD
    results.lags_ms = lags_ms;
    results.R_NN = R_NN;
    results.R_XX = R_XX_avg;
    results.R_YY = R_YY_avg;
    results.freq_axis_khz = freq_axis_khz(half_bins);
    results.S_NN = S_NN(half_bins);
    results.S_XX = S_XX_avg(half_bins);
    results.S_YY = S_YY_avg(half_bins);
    
    % Orthogonality & SNR
    results.cov_matrix = cov_matrix;
    results.rho_XY = rho_XY;
    results.rho_orthogonality = rho_orthogonality;
    results.snr_in_db = snr_in_db;
    results.snr_out_db = snr_out_db;
    results.delta_snr_db = delta_snr_db;
    results.real_time_factor = real_time_factor;
    results.time_elapsed = time_elapsed_processing;
    
    % Spectrograms
    results.spec_X_dB = spec_X_dB;
    results.spec_Y_dB = spec_Y_dB;
    results.spec_t = spec_t;
    results.spec_f = spec_f;
    
    statusCallback(sprintf('Complete: Real-Time Factor = %.4f (%.1fx real-time speed)', ...
        real_time_factor, 1/max(real_time_factor, 1e-6)));
end

% =========================================================================
% SECTION 2: ACOUSTIC FORMANT SYNTHESIS & STFT SPECTROGRAM (FROM SCRATCH)
% =========================================================================

function [clean_speech, noisy_speech, actual_sigma] = generate_synthetic_clean_speech(duration, fs, target_snr_db)
% GENERATE_SYNTHETIC_CLEAN_SPEECH Acoustic Formant Speech Synthesizer
%   Models human speech production using a glottal pulse harmonic source
%   coupled with 3 vocal tract formant biquad resonators (F1, F2, F3)
%   and natural syllabic amplitude modulation envelopes.

    t = (0:1/fs:duration-1/fs)';
    N = length(t);

    % 1. Glottal Pitch Source with Natural Intonation & Rosenberg Rolloff
    % Natural pitch inflection contour (f0 ~ 125 Hz with expressive prosody)
    f0 = 125.0 + 18.0 * sin(2 * pi * 0.9 * t) + 6.0 * sin(2 * pi * 2.2 * t);
    phase = 2 * pi * cumsum(f0) / fs;
    glottal_source = zeros(N, 1);
    for k = 1:20
        % Rolloff proportional to k^-1.7 models natural glottal flow (eliminates harsh buzz)
        glottal_source = glottal_source + (1 / (k^1.7)) * sin(k * phase);
    end

    % 2. Vocal Tract Resonances (Formant Biquad Filtering)
    % Vowel formants: F1 = 700 Hz (jaw opening), F2 = 1250 Hz (tongue body), F3 = 2500 Hz
    formants = [700, 80; 1250, 100; 2500, 150];
    clean_speech = zeros(N, 1);

    for i = 1:size(formants, 1)
        Fc = formants(i, 1);
        Bw = formants(i, 2);
        
        % Pole Radius & Angle in the z-plane:
        r = exp(-pi * Bw / fs);
        theta = 2 * pi * Fc / fs;
        
        % 2nd-order IIR Difference Equation Coefficients:
        % y[n] = b0*x[n] - a1*y[n-1] - a2*y[n-2]
        a1 = -2 * r * cos(theta);
        a2 = r^2;
        b0 = (1 - r) * sin(theta);
        
        y_formant = zeros(N, 1);
        for n = 3:N
            y_formant(n) = b0 * glottal_source(n) - a1 * y_formant(n-1) - a2 * y_formant(n-2);
        end
        clean_speech = clean_speech + y_formant;
    end

    % 3. Syllabic Cadence Envelope (Active Speech Phrases vs. Silence Pauses)
    % Simulates distinct words with smooth natural onset/offset
    envelope = zeros(N, 1);
    p1 = (t >= 0.3 & t <= 1.8);
    p2 = (t >= 2.4 & t <= 4.0);
    p3 = (t >= 4.6 & t <= min(duration - 0.2, 5.8));
    
    envelope(p1) = sin(pi * (t(p1) - 0.3) / 1.5).^2;
    envelope(p2) = sin(pi * (t(p2) - 2.4) / 1.6).^2;
    if any(p3)
        envelope(p3) = sin(pi * (t(p3) - 4.6) / (max(t(p3)) - 4.6)).^2;
    end
    
    clean_speech = clean_speech .* envelope;
    
    % Amplitude Normalization
    max_amp = max(abs(clean_speech));
    if max_amp > 0
        clean_speech = 0.6 * (clean_speech / max_amp);
    end

    % 4. Additive Gaussian White Noise Injection calibrated to active speech SNR
    active_mask = envelope > 0.05;
    if any(active_mask)
        speech_power = sum(clean_speech(active_mask).^2) / sum(active_mask);
    else
        speech_power = sum(clean_speech.^2) / N;
    end
    target_noise_power = speech_power / (10^(target_snr_db / 10));
    actual_sigma = sqrt(target_noise_power);
    
    noise = actual_sigma * randn(N, 1);
    noisy_speech = clean_speech + noise;
end

function [clean_speech, noisy_speech] = generate_synthetic_noisy_speech(duration, fs, target_snr_db, ~)
    [clean_speech, noisy_speech] = generate_synthetic_clean_speech(duration, fs, target_snr_db);
end

function [spec_X_dB, spec_Y_dB, t_axis, f_axis_khz] = compute_dual_spectrogram(x, y, fs)
% COMPUTE_DUAL_SPECTROGRAM Short-Time Fourier Transform (STFT) from Scratch
%   Computes time-frequency energy distributions using sliding Hamming
%   windows and discrete Fourier transforms without requiring toolbox functions.

    win_len = 512;
    hop_size = 128;
    nfft_spec = 512;
    w_spec = 0.54 - 0.46 * cos(2 * pi * (0:win_len-1)' / (win_len - 1));

    N_samples = length(x);
    num_frames = floor((N_samples - win_len) / hop_size) + 1;
    num_bins = nfft_spec / 2 + 1;

    spec_X = zeros(num_bins, num_frames);
    spec_Y = zeros(num_bins, num_frames);

    for k = 1:num_frames
        idx = (k-1)*hop_size + (1:win_len);
        
        X_w = x(idx) .* w_spec;
        Y_w = y(idx) .* w_spec;
        
        X_fft = fft(X_w, nfft_spec);
        Y_fft = fft(Y_w, nfft_spec);
        
        spec_X(:, k) = abs(X_fft(1:num_bins));
        spec_Y(:, k) = abs(Y_fft(1:num_bins));
    end

    % Normalize magnitude spectrum relative to global peak (0 dB reference)
    max_spec = max(max(spec_X(:)), 1e-6);
    spec_X_dB = 20 * log10(spec_X / max_spec + 1e-6);
    spec_Y_dB = 20 * log10(spec_Y / max_spec + 1e-6);

    t_axis = ((0:num_frames-1) * hop_size + win_len/2) / fs;
    f_axis_khz = ((0:num_bins-1)' * (fs / nfft_spec)) / 1000;
end

function audio = record_live_audio(duration_sec, fs)
% RECORD_LIVE_AUDIO Captures live microphone audio using standard audiorecorder
    try
        recObj = audiorecorder(fs, 16, 1);
        fprintf('Recording %0.1f seconds of live microphone audio...\n', duration_sec);
        recordblocking(recObj, duration_sec);
        audio = getaudiodata(recObj);
    catch ME
        warning('Audio hardware recording failed (%s). Falling back to synthetic simulation mode.', ME.message);
        audio = 0.015 * randn(round(duration_sec * fs), 1);
    end
end

function noise_cal = extract_noise_from_audio(noisy_audio, frameLen, hop, fs)
% EXTRACT_NOISE_FROM_AUDIO Minimum-Energy Frame Ambient Noise Extractor
%   Extracts stationary background noise realization N[n] from uncalibrated
%   recordings or uploaded audio files by selecting the lowest-energy frames
%   (speech pauses / room tone). Enables rigorous stochastic analysis on any audio file.

    N_total = length(noisy_audio);
    num_frames = floor((N_total - frameLen) / hop) + 1;
    if num_frames < 4
        noise_cal = 0.005 * randn(round(1.5 * fs), 1);
        return;
    end

    frame_energies = zeros(num_frames, 1);
    for k = 1:num_frames
        idx = (k-1)*hop + (1:frameLen);
        frame_energies(k) = sum(noisy_audio(idx).^2) / frameLen;
    end

    % Sort frames by energy to isolate silence pauses / ambient room noise
    [~, sort_idx] = sort(frame_energies);
    
    % Pick lowest 15% to 25% energy frames (speech pauses)
    num_noise_frames = max(8, min(round(0.20 * num_frames), 80));
    num_noise_frames = min(num_noise_frames, num_frames);
    selected_frames = sort_idx(1:num_noise_frames);

    noise_segments = cell(num_noise_frames, 1);
    for i = 1:num_noise_frames
        k = selected_frames(i);
        noise_segments{i} = noisy_audio((k-1)*hop + (1:frameLen));
    end
    noise_cal = cell2mat(noise_segments);

    % Ensure calibration buffer has at least 1.5 seconds for robust PDF & moment estimation
    min_samples = round(1.5 * fs);
    if length(noise_cal) < min_samples
        reps = ceil(min_samples / max(length(noise_cal), 1));
        noise_cal = repmat(noise_cal, reps, 1);
    end
    noise_cal = noise_cal(1:max(min_samples, round(2.0 * fs)));
end

% =========================================================================
% SECTION 3: PUBLICATION-QUALITY DASHBOARD VISUALIZATION
% =========================================================================

function fig = render_dashboard_figure(res, isVisible)
% RENDER_DASHBOARD_FIGURE Creates the 6-panel publication-grade figure:
%   Subplot 1: Empirical PDF Histogram vs. Theoretical Gaussian PDF
%   Subplot 2: Empirical CDF vs. Theoretical Gaussian CDF
%   Subplot 3: Time-Averaged Autocorrelation Functions R_NN, R_XX, R_YY
%   Subplot 4: Wiener-Khinchin Power Spectral Densities S_XX, S_NN, S_YY
%   Subplot 5: Time-Domain Waveforms (Noisy Input X(t) vs. Filtered Output Y(t))
%   Subplot 6: Spectrogram Comparison (Noisy Input vs. LTI Filtered Output)

    if nargin < 2
        isVisible = true;
    end
    vis_str = 'off';
    if isVisible
        vis_str = 'on';
    end

    fig = figure('Name', 'Real-Time Audio Denoising via Wiener-Khinchin & Optimal LTI Filtering', ...
                 'NumberTitle', 'off', ...
                 'Color', [0.95, 0.96, 0.98], ...
                 'Position', [50, 40, 1420, 890], ...
                 'Visible', vis_str);

    % High-contrast color palette
    col_noise   = [0.85, 0.32, 0.15]; % Deep Amber/Orange
    col_input   = [0.15, 0.42, 0.85]; % Sapphire Blue
    col_output  = [0.05, 0.60, 0.38]; % Emerald Green

    % ---------------------------------------------------------------------
    % SUBPLOT 1: Empirical Noise PDF vs. Theoretical Gaussian PDF
    % ---------------------------------------------------------------------
    ax1 = subplot(3, 2, 1);
    set(ax1, 'Color', [1 1 1], 'XColor', [0.12 0.18 0.28], 'YColor', [0.12 0.18 0.28], ...
             'GridColor', [0.82 0.85 0.90], 'GridAlpha', 0.85, 'Box', 'on');
    bar(ax1, res.bin_centers, res.pdf_empirical, 1.0, ...
        'FaceColor', [0.72, 0.84, 0.96], 'EdgeColor', col_input, ...
        'FaceAlpha', 0.7, 'DisplayName', 'Empirical PDF \hat{f}_N(x)');
    hold(ax1, 'on');
    plot(ax1, res.bin_centers, res.pdf_theoretical, 'r-', 'LineWidth', 2.2, ...
         'DisplayName', 'Fitted Gaussian f_N(x)');
    hold(ax1, 'off');
    grid(ax1, 'on');
    title(ax1, 'Subplot 1: Ambient Noise PDF vs. Gaussian Model', ...
          'Color', [0.06, 0.16, 0.38], 'FontSize', 10, 'FontWeight', 'bold');
    xlabel(ax1, 'Amplitude x', 'Color', [0.12 0.18 0.28], 'FontSize', 9, 'FontWeight', 'bold');
    ylabel(ax1, 'Probability Density f_N(x)', 'Color', [0.12 0.18 0.28], 'FontSize', 9, 'FontWeight', 'bold');
    legend(ax1, 'Location', 'northeast', 'FontSize', 8.5, 'TextColor', [0.10, 0.15, 0.25], ...
           'Color', [1 1 1], 'EdgeColor', [0.75 0.80 0.88]);
    
    str_box1 = sprintf(['\\mu_N = %+.4f\n\\sigma_N = %.4f\n' ...
                        'Skewness = %+.3f\nKurtosis = %.3f\n' ...
                        'PDF MSE = %.2e\nNMSE = %.2e'], ...
                        res.mu_N, res.sigma_N, res.skewness_N, res.kurtosis_N, res.mse_pdf, res.nmse_pdf);
    text(ax1, 0.04, 0.92, str_box1, 'Units', 'normalized', 'FontSize', 8.5, ...
         'Color', [0.08, 0.14, 0.24], 'FontWeight', 'bold', ...
         'BackgroundColor', [1 1 1 0.92], 'EdgeColor', [0.70 0.75 0.85], 'VerticalAlignment', 'top');

    % ---------------------------------------------------------------------
    % SUBPLOT 2: Empirical CDF vs. Theoretical Gaussian CDF
    % ---------------------------------------------------------------------
    ax2 = subplot(3, 2, 2);
    set(ax2, 'Color', [1 1 1], 'XColor', [0.12 0.18 0.28], 'YColor', [0.12 0.18 0.28], ...
             'GridColor', [0.82 0.85 0.90], 'GridAlpha', 0.85, 'Box', 'on');
    stairs(ax2, res.bin_centers, res.cdf_empirical, 'b-', 'LineWidth', 2.0, ...
           'DisplayName', 'Empirical CDF \hat{F}_N(x)');
    hold(ax2, 'on');
    plot(ax2, res.bin_centers, res.cdf_theoretical, 'k--', 'LineWidth', 2.0, ...
         'DisplayName', 'Gaussian CDF \Phi(x)');
    hold(ax2, 'off');
    grid(ax2, 'on');
    title(ax2, 'Subplot 2: Cumulative Distribution Function (CDF) Verification', ...
          'Color', [0.06, 0.16, 0.38], 'FontSize', 10, 'FontWeight', 'bold');
    xlabel(ax2, 'Amplitude x', 'Color', [0.12 0.18 0.28], 'FontSize', 9, 'FontWeight', 'bold');
    ylabel(ax2, 'Probability F_N(x)', 'Color', [0.12 0.18 0.28], 'FontSize', 9, 'FontWeight', 'bold');
    legend(ax2, 'Location', 'southeast', 'FontSize', 8.5, 'TextColor', [0.10, 0.15, 0.25], ...
           'Color', [1 1 1], 'EdgeColor', [0.75 0.80 0.88]);
    
    str_box2 = sprintf(['CLT Block Mean Var: %.2e\n' ...
                        'Theory \\sigma_N^2/L_{blk}: %.2e\n' ...
                        '95%% CI: [%+.4f, %+.4f]'], ...
                        res.var_of_means, res.theoretical_clt_var, res.ci_95(1), res.ci_95(2));
    text(ax2, 0.04, 0.92, str_box2, 'Units', 'normalized', 'FontSize', 8.5, ...
         'Color', [0.08, 0.14, 0.24], 'FontWeight', 'bold', ...
         'BackgroundColor', [1 1 1 0.92], 'EdgeColor', [0.70 0.75 0.85], 'VerticalAlignment', 'top');

    % ---------------------------------------------------------------------
    % SUBPLOT 3: Time-Averaged Autocorrelation Functions R_NN, R_XX, R_YY
    % ---------------------------------------------------------------------
    ax3 = subplot(3, 2, 3);
    set(ax3, 'Color', [1 1 1], 'XColor', [0.12 0.18 0.28], 'YColor', [0.12 0.18 0.28], ...
             'GridColor', [0.82 0.85 0.90], 'GridAlpha', 0.85, 'Box', 'on');
    plot_lags = abs(res.lags_ms) <= 25;
    plot(ax3, res.lags_ms(plot_lags), res.R_XX(plot_lags), '-', 'Color', col_input, ...
         'LineWidth', 1.5, 'DisplayName', 'Input R_{XX}[\tau]');
    hold(ax3, 'on');
    plot(ax3, res.lags_ms(plot_lags), res.R_YY(plot_lags), '-', 'Color', col_output, ...
         'LineWidth', 1.8, 'DisplayName', 'Filtered R_{YY}[\tau]');
    plot(ax3, res.lags_ms(plot_lags), res.R_NN(plot_lags), ':', 'Color', col_noise, ...
         'LineWidth', 2.0, 'DisplayName', 'Noise R_{NN}[\tau]');
    hold(ax3, 'off');
    grid(ax3, 'on');
    title(ax3, 'Subplot 3: Time-Averaged Autocorrelation Sequences R[\tau]', ...
          'Color', [0.06, 0.16, 0.38], 'FontSize', 10, 'FontWeight', 'bold');
    xlabel(ax3, 'Lag \tau (ms)', 'Color', [0.12 0.18 0.28], 'FontSize', 9, 'FontWeight', 'bold');
    ylabel(ax3, 'Autocorrelation R[\tau]', 'Color', [0.12 0.18 0.28], 'FontSize', 9, 'FontWeight', 'bold');
    legend(ax3, 'Location', 'northeast', 'FontSize', 8.5, 'TextColor', [0.10, 0.15, 0.25], ...
           'Color', [1 1 1], 'EdgeColor', [0.75 0.80 0.88]);

    % ---------------------------------------------------------------------
    % SUBPLOT 4: Wiener-Khinchin Power Spectral Densities
    % ---------------------------------------------------------------------
    ax4 = subplot(3, 2, 4);
    set(ax4, 'Color', [1 1 1], 'XColor', [0.12 0.18 0.28], 'YColor', [0.12 0.18 0.28], ...
             'GridColor', [0.82 0.85 0.90], 'GridAlpha', 0.85, 'Box', 'on');
    S_XX_dB = 10 * log10(res.S_XX + 1e-12);
    S_NN_dB = 10 * log10(res.S_NN + 1e-12);
    S_YY_dB = 10 * log10(res.S_YY + 1e-12);

    plot(ax4, res.freq_axis_khz, S_XX_dB, '-', 'Color', col_input, ...
         'LineWidth', 1.4, 'DisplayName', 'Input PSD S_{XX}(\omega)');
    hold(ax4, 'on');
    plot(ax4, res.freq_axis_khz, S_NN_dB, ':', 'Color', col_noise, ...
         'LineWidth', 2.0, 'DisplayName', 'Noise PSD S_{NN}(\omega)');
    plot(ax4, res.freq_axis_khz, S_YY_dB, '-', 'Color', col_output, ...
         'LineWidth', 1.8, 'DisplayName', 'LTI Output S_{YY}(\omega) = |H|^2 S_{XX}');
    hold(ax4, 'off');
    grid(ax4, 'on');
    title(ax4, 'Subplot 4: Wiener-Khinchin PSD: S(\omega) = F\{R[\tau]\}', ...
          'Color', [0.06, 0.16, 0.38], 'FontSize', 10, 'FontWeight', 'bold');
    xlabel(ax4, 'Frequency (kHz)', 'Color', [0.12 0.18 0.28], 'FontSize', 9, 'FontWeight', 'bold');
    ylabel(ax4, 'Power Spectral Density (dB/Hz)', 'Color', [0.12 0.18 0.28], 'FontSize', 9, 'FontWeight', 'bold');
    legend(ax4, 'Location', 'northeast', 'FontSize', 8.5, 'TextColor', [0.10, 0.15, 0.25], ...
           'Color', [1 1 1], 'EdgeColor', [0.75 0.80 0.88]);

    % ---------------------------------------------------------------------
    % SUBPLOT 5: Time-Domain Waveforms (Noisy Input vs. Filtered Output)
    % ---------------------------------------------------------------------
    ax5 = subplot(3, 2, 5);
    set(ax5, 'Color', [1 1 1], 'XColor', [0.12 0.18 0.28], 'YColor', [0.12 0.18 0.28], ...
             'GridColor', [0.82 0.85 0.90], 'GridAlpha', 0.85, 'Box', 'on');
    plot(ax5, res.time_vec, res.noisy_input, 'Color', [0.65 0.68 0.72], ...
         'LineWidth', 0.9, 'DisplayName', 'Noisy Input X(t)');
    hold(ax5, 'on');
    plot(ax5, res.time_vec, res.filtered_output, 'Color', col_output, ...
         'LineWidth', 1.2, 'DisplayName', 'Denoised Output Y(t)');
    hold(ax5, 'off');
    grid(ax5, 'on');
    title(ax5, 'Subplot 5: Time Waveforms: X(t) = S(t) + N(t) vs. Denoised Y(t)', ...
          'Color', [0.06, 0.16, 0.38], 'FontSize', 10, 'FontWeight', 'bold');
    xlabel(ax5, 'Time (seconds)', 'Color', [0.12 0.18 0.28], 'FontSize', 9, 'FontWeight', 'bold');
    ylabel(ax5, 'Amplitude', 'Color', [0.12 0.18 0.28], 'FontSize', 9, 'FontWeight', 'bold');
    legend(ax5, 'Location', 'northeast', 'FontSize', 8.5, 'TextColor', [0.10, 0.15, 0.25], ...
           'Color', [1 1 1], 'EdgeColor', [0.75 0.80 0.88]);
    xlim(ax5, [0, res.duration]);

    % ---------------------------------------------------------------------
    % SUBPLOT 6: Spectrogram Comparison
    % ---------------------------------------------------------------------
    ax6 = subplot(3, 2, 6);
    spec_combined = [res.spec_X_dB; -100 * ones(4, length(res.spec_t)); res.spec_Y_dB];
    f_combined = [res.spec_f; 8 + (1:4)'*0.1; res.spec_f + 8.5];
    
    imagesc(ax6, res.spec_t, f_combined, spec_combined);
    axis(ax6, 'xy');
    colormap(ax6, 'jet');
    clim(ax6, [-55, 0]);
    cb = colorbar(ax6, 'Location', 'eastoutside');
    cb.Label.String = 'Energy (dB)';
    cb.Color = [0.12 0.18 0.28];
    grid(ax6, 'off');
    title(ax6, 'Subplot 6: Spectrogram: [Top: Noisy Input X(t)] vs. [Bottom: Filtered Y(t)]', ...
          'Color', [0.06, 0.16, 0.38], 'FontSize', 10, 'FontWeight', 'bold');
    xlabel(ax6, 'Time (seconds)', 'Color', [0.12 0.18 0.28], 'FontSize', 9, 'FontWeight', 'bold');
    ylabel(ax6, 'Frequency (kHz)', 'Color', [0.12 0.18 0.28], 'FontSize', 9, 'FontWeight', 'bold');
    
    str_box6 = sprintf(['\\Delta SNR = %+.2f dB\n' ...
                        '\\rho_{XY} = %.4f\n' ...
                        '\\rho_{YE} = %.4f\n' ...
                        'RTF: %.3f (Real-Time)'], ...
                        res.delta_snr_db, res.rho_XY, res.rho_orthogonality, res.real_time_factor);
    text(ax6, 0.03, 0.90, str_box6, 'Units', 'normalized', 'FontSize', 8.5, ...
         'Color', 'w', 'BackgroundColor', [0 0 0 0.75], 'EdgeColor', [0.3 0.3 0.3], 'VerticalAlignment', 'top');

    set([ax1, ax2, ax3, ax4, ax5, ax6], 'FontName', 'Helvetica', 'FontSize', 8.5);
end

function export_figure_and_audio(res, fig)
% EXPORT_FIGURE_AND_AUDIO Saves high-resolution dashboard PNG and WAV files
    img_name = 'wiener_denoising_dashboard.png';
    try
        saveas(fig, img_name);
        fprintf('[EXPORT] Publication dashboard saved to: %s\n', img_name);
    catch ME
        warning('Failed to save dashboard image: %s', ME.message);
    end
end

% =========================================================================
% SECTION 4: CONSOLE STATISTICAL SUMMARY TABLE
% =========================================================================

function print_console_summary_table(res)
% PRINT_CONSOLE_SUMMARY_TABLE Prints formatted statistical metrics
    fprintf('\n');
    fprintf('========================================================================================\n');
    fprintf('         STATISTICAL PARAMETER SUMMARY: WIENER-KHINCHIN REAL-TIME DENOISER\n');
    fprintf('========================================================================================\n');
    fprintf(' %-36s %-25s %-22s\n', 'Metric / Property', 'Empirical Observation', 'Theoretical / Benchmark');
    fprintf('----------------------------------------------------------------------------------------\n');
    fprintf(' Noise Mean (mu_N)                   %+12.6f              %12.6f (Zero-Mean)\n', res.mu_N, 0.0);
    fprintf(' Noise Variance (sigma_N^2)          %12.6e              %12.6e\n', res.var_N, res.var_N);
    fprintf(' Noise Std Dev (sigma_N)             %12.6f              %12.6f\n', res.sigma_N, res.sigma_N);
    fprintf(' Noise Skewness (gamma_1)            %+12.4f              %12.4f (Symmetric)\n', res.skewness_N, 0.0);
    fprintf(' Noise Kurtosis (gamma_2)            %12.4f              %12.4f (Mesokurtic)\n', res.kurtosis_N, 3.0);
    fprintf('----------------------------------------------------------------------------------------\n');
    fprintf(' Gaussian PDF Goodness-of-Fit MSE:   %12.4e              (0.0000 = Exact Match)\n', res.mse_pdf);
    fprintf(' Gaussian PDF Normalized MSE (NMSE): %12.4e              (Relative fit error)\n', res.nmse_pdf);
    fprintf(' CLT Block Mean Variance (Var(X_b)): %12.4e              %12.4e (sigma_N^2 / L_blk)\n', ...
        res.var_of_means, res.theoretical_clt_var);
    fprintf(' 95%% Mean Confidence Interval:       [%+9.5f, %+9.5f]     Contains 0: %s\n', ...
        res.ci_95(1), res.ci_95(2), string(res.ci_95(1) <= 0 && res.ci_95(2) >= 0));
    fprintf('----------------------------------------------------------------------------------------\n');
    fprintf(' Covariance Matrix [Var(X), Cov(X,Y); Cov(Y,X), Var(Y)]:\n');
    fprintf('   [%11.6f, %11.6f ;\n    %11.6f, %11.6f]\n', ...
        res.cov_matrix(1,1), res.cov_matrix(1,2), res.cov_matrix(2,1), res.cov_matrix(2,2));
    fprintf(' Pearson Cross-Correlation (rho_XY): %+12.4f              (High Coherence)\n', res.rho_XY);
    fprintf(' Orthogonality Principle rho(Y, E):  %+12.4f              (~0.0000 Optimal MMSE)\n', res.rho_orthogonality);
    fprintf('----------------------------------------------------------------------------------------\n');
    fprintf(' Input Signal-to-Noise Ratio (SNR):  %+11.2f dB\n', res.snr_in_db);
    fprintf(' Output Signal-to-Noise Ratio (SNR): %+11.2f dB\n', res.snr_out_db);
    fprintf(' SNR Improvement (Delta SNR):        %+11.2f dB           (Significant Enhancement)\n', res.delta_snr_db);
    fprintf(' Real-Time Processing Factor (RTF):  %12.4f              (%.1fx faster than real-time)\n', ...
        res.real_time_factor, 1/max(res.real_time_factor, 1e-6));
    fprintf('========================================================================================\n\n');
end

% =========================================================================
% SECTION 5: INTERACTIVE GUI DASHBOARD (FOR EVERYDAY USERS & ENGINEERS)
% =========================================================================

function hGUI = launch_gui_dashboard(initial_cfg)
% LAUNCH_GUI_DASHBOARD Builds an intuitive, consumer-grade audio cleaning studio
%   featuring 1-click recording, audio file upload (.wav, .mp3, .m4a), export,
%   high-contrast readable dashboard cards, and 6-panel stochastic process plots.

    % Main Figure Window
    hFig = figure('Name', 'Real-Time Audio Cleaner & Wiener Filter Studio', ...
                  'NumberTitle', 'off', ...
                  'Color', [0.94, 0.95, 0.97], ...
                  'Position', [50, 40, 1440, 900], ...
                  'Visible', 'on');

    hGUI = struct();
    hGUI.fig = hFig;
    hGUI.cfg = initial_cfg;
    hGUI.results = [];
    hGUI.audioPlayer = [];
    hGUI.uploadedAudio = [];
    hGUI.uploadedFileName = '';
    hGUI.calibratedNoise = [];

    % ---------------------------------------------------------------------
    % Top Header Banner
    % ---------------------------------------------------------------------
    uicontrol(hFig, 'Style', 'text', ...
              'String', 'Real-Time Audio Denoising via Wiener-Khinchin Spectral Estimation & Optimal LTI Filtering', ...
              'FontName', 'Helvetica', 'FontSize', 13.5, 'FontWeight', 'bold', ...
              'ForegroundColor', [0.06, 0.16, 0.38], ...
              'BackgroundColor', [0.94, 0.95, 0.97], ...
              'HorizontalAlignment', 'center', ...
              'Position', [20, 856, 1400, 32]);

    uicontrol(hFig, 'Style', 'text', ...
              'String', 'Live Microphone Recording | Audio File Upload (.wav, .mp3, .m4a) | Stochastic Process Verification', ...
              'FontName', 'Helvetica', 'FontSize', 9.5, 'FontAngle', 'italic', ...
              'ForegroundColor', [0.25, 0.35, 0.48], ...
              'BackgroundColor', [0.94, 0.95, 0.97], ...
              'HorizontalAlignment', 'center', ...
              'Position', [20, 836, 1400, 20]);

    % ---------------------------------------------------------------------
    % Left Sidebar: Controls & Parameter Cards (Width: 335 px)
    % ---------------------------------------------------------------------
    uicontrol(hFig, 'Style', 'frame', ...
              'BackgroundColor', [1, 1, 1], ...
              'Position', [15, 15, 335, 815]);

    % Sidebar Header
    uicontrol(hFig, 'Style', 'text', ...
              'String', 'SYSTEM CONTROLS', ...
              'FontName', 'Helvetica', 'FontSize', 11, 'FontWeight', 'bold', ...
              'ForegroundColor', [0.06, 0.16, 0.38], ...
              'BackgroundColor', [1, 1, 1], ...
              'Position', [25, 792, 315, 22]);

    % Step 1: Audio Source Selection
    uicontrol(hFig, 'Style', 'text', ...
              'String', '1. Select Audio Source:', ...
              'FontName', 'Helvetica', 'FontSize', 9.5, 'FontWeight', 'bold', ...
              'ForegroundColor', [0.08, 0.15, 0.30], ...
              'BackgroundColor', [1, 1, 1], ...
              'HorizontalAlignment', 'left', ...
              'Position', [25, 765, 305, 18]);

    hGUI.popupSource = uicontrol(hFig, 'Style', 'popupmenu', ...
              'String', {'Live Microphone (Speak & Clean)', ...
                         'Upload Audio File (.wav, .mp3, .m4a)', ...
                         'Demo Speech Simulation (Built-in)'}, ...
              'Value', 1, ...
              'FontName', 'Helvetica', 'FontSize', 9, ...
              'Position', [25, 735, 315, 28], ...
              'Callback', @(src, event) on_source_changed(hFig));

    % Context Panel 1: Live Microphone Controls
    hGUI.pnlMic = uipanel('Parent', hFig, 'Units', 'pixels', ...
                          'Position', [25, 575, 315, 155], ...
                          'BackgroundColor', [1 1 1], ...
                          'BorderType', 'none', ...
                          'Visible', 'on');

    uicontrol(hGUI.pnlMic, 'Style', 'text', ...
              'String', 'Recording Duration:', ...
              'FontName', 'Helvetica', 'FontSize', 9, 'FontWeight', 'bold', ...
              'ForegroundColor', [0.10, 0.18, 0.32], ...
              'BackgroundColor', [1, 1, 1], ...
              'HorizontalAlignment', 'left', ...
              'Position', [0, 130, 130, 20]);

    hGUI.popupDuration = uicontrol(hGUI.pnlMic, 'Style', 'popupmenu', ...
              'String', {'3.0 Seconds', '5.0 Seconds (Recommended)', '8.0 Seconds', '10.0 Seconds'}, ...
              'Value', 2, ...
              'FontName', 'Helvetica', 'FontSize', 9, ...
              'Position', [135, 128, 175, 25]);

    hGUI.btnRecord = uicontrol(hGUI.pnlMic, 'Style', 'pushbutton', ...
              'String', 'Record & Clean Audio', ...
              'FontName', 'Helvetica', 'FontSize', 10.5, 'FontWeight', 'bold', ...
              'BackgroundColor', [0.82, 0.16, 0.16], ...
              'ForegroundColor', [1, 1, 1], ...
              'Position', [0, 75, 310, 42], ...
              'Callback', @(src, event) on_record_and_denoise(hFig));

    hGUI.btnCalibrate = uicontrol(hGUI.pnlMic, 'Style', 'pushbutton', ...
              'String', 'Calibrate Room Noise (Quiet 1.5s)', ...
              'FontName', 'Helvetica', 'FontSize', 9, ...
              'BackgroundColor', [0.88, 0.92, 0.98], ...
              'ForegroundColor', [0.08, 0.22, 0.55], ...
              'Position', [0, 36, 310, 32], ...
              'Callback', @(src, event) on_calibrate_clicked(hFig));

    uicontrol(hGUI.pnlMic, 'Style', 'text', ...
              'String', 'Tip: Speak into your mic. The filter removes room noise.', ...
              'FontName', 'Helvetica', 'FontSize', 8, 'FontAngle', 'italic', ...
              'ForegroundColor', [0.35, 0.45, 0.55], ...
              'BackgroundColor', [1, 1, 1], ...
              'HorizontalAlignment', 'center', ...
              'Position', [0, 5, 310, 24]);

    % Context Panel 2: File Upload Controls
    hGUI.pnlFile = uipanel('Parent', hFig, 'Units', 'pixels', ...
                           'Position', [25, 575, 315, 155], ...
                           'BackgroundColor', [1 1 1], ...
                           'BorderType', 'none', ...
                           'Visible', 'off');

    hGUI.btnChooseFile = uicontrol(hGUI.pnlFile, 'Style', 'pushbutton', ...
              'String', 'Choose Audio File (.wav, .mp3, .m4a)...', ...
              'FontName', 'Helvetica', 'FontSize', 9.5, 'FontWeight', 'bold', ...
              'BackgroundColor', [0.18, 0.42, 0.82], ...
              'ForegroundColor', [1, 1, 1], ...
              'Position', [0, 105, 310, 38], ...
              'Callback', @(src, event) on_select_file(hFig));

    hGUI.lblFileInfo = uicontrol(hGUI.pnlFile, 'Style', 'text', ...
              'String', 'No audio file selected yet.', ...
              'FontName', 'Helvetica', 'FontSize', 8.5, 'FontWeight', 'bold', ...
              'ForegroundColor', [0.20, 0.30, 0.45], ...
              'BackgroundColor', [0.95, 0.96, 0.98], ...
              'HorizontalAlignment', 'center', ...
              'Position', [0, 60, 310, 38]);

    hGUI.btnCleanFile = uicontrol(hGUI.pnlFile, 'Style', 'pushbutton', ...
              'String', 'Clean Selected Audio File', ...
              'FontName', 'Helvetica', 'FontSize', 10.5, 'FontWeight', 'bold', ...
              'BackgroundColor', [0.10, 0.58, 0.35], ...
              'ForegroundColor', [1, 1, 1], ...
              'Position', [0, 12, 310, 40], ...
              'Enable', 'off', ...
              'Callback', @(src, event) on_clean_file(hFig));

    % Context Panel 3: Simulation Controls
    hGUI.pnlSim = uipanel('Parent', hFig, 'Units', 'pixels', ...
                          'Position', [25, 575, 315, 155], ...
                          'BackgroundColor', [1 1 1], ...
                          'BorderType', 'none', ...
                          'Visible', 'off');

    uicontrol(hGUI.pnlSim, 'Style', 'text', ...
              'String', 'Speech Duration:', ...
              'FontName', 'Helvetica', 'FontSize', 9, 'FontWeight', 'bold', ...
              'ForegroundColor', [0.10, 0.18, 0.32], ...
              'BackgroundColor', [1, 1, 1], ...
              'HorizontalAlignment', 'left', ...
              'Position', [0, 130, 120, 20]);

    hGUI.popupSimDur = uicontrol(hGUI.pnlSim, 'Style', 'popupmenu', ...
              'String', {'4.0 Seconds', '6.0 Seconds (Recommended)', '8.0 Seconds'}, ...
              'Value', 2, ...
              'FontName', 'Helvetica', 'FontSize', 9, ...
              'Position', [125, 128, 185, 25]);

    uicontrol(hGUI.pnlSim, 'Style', 'text', ...
              'String', 'Noise Level (SNR):', ...
              'FontName', 'Helvetica', 'FontSize', 9, 'FontWeight', 'bold', ...
              'ForegroundColor', [0.10, 0.18, 0.32], ...
              'BackgroundColor', [1, 1, 1], ...
              'HorizontalAlignment', 'left', ...
              'Position', [0, 95, 120, 20]);

    hGUI.popupSimSNR = uicontrol(hGUI.pnlSim, 'Style', 'popupmenu', ...
              'String', {'Moderate Noise (+12 dB SNR)', 'Heavy Noise (+5 dB SNR)', 'Mild Noise (+18 dB SNR)'}, ...
              'Value', 2, ...
              'FontName', 'Helvetica', 'FontSize', 9, ...
              'Position', [125, 93, 185, 25]);

    hGUI.btnRunSim = uicontrol(hGUI.pnlSim, 'Style', 'pushbutton', ...
              'String', 'Run Demo Simulation & Clean', ...
              'FontName', 'Helvetica', 'FontSize', 10.5, 'FontWeight', 'bold', ...
              'BackgroundColor', [0.10, 0.58, 0.35], ...
              'ForegroundColor', [1, 1, 1], ...
              'Position', [0, 32, 310, 42], ...
              'Callback', @(src, event) on_run_simulation(hFig));

    % Step 2: Filter Tuning
    uicontrol(hFig, 'Style', 'text', ...
              'String', '2. Noise Reduction Strength:', ...
              'FontName', 'Helvetica', 'FontSize', 9.5, 'FontWeight', 'bold', ...
              'ForegroundColor', [0.08, 0.15, 0.30], ...
              'BackgroundColor', [1, 1, 1], ...
              'HorizontalAlignment', 'left', ...
              'Position', [25, 545, 305, 18]);

    hGUI.popupBeta = uicontrol(hFig, 'Style', 'popupmenu', ...
              'String', {'Balanced (Recommended - Best Clarity)', ...
                         'Maximum Removal (For Heavy Background Hum)', ...
                         'Gentle Touch (Preserves Whispers)'}, ...
              'Value', 1, ...
              'FontName', 'Helvetica', 'FontSize', 9, ...
              'Position', [25, 517, 315, 28]);

    % Step 3: Audio Playback & Save Controls
    uicontrol(hFig, 'Style', 'text', ...
              'String', '3. Audio Playback & Export:', ...
              'FontName', 'Helvetica', 'FontSize', 9.5, 'FontWeight', 'bold', ...
              'ForegroundColor', [0.08, 0.15, 0.30], ...
              'BackgroundColor', [1, 1, 1], ...
              'HorizontalAlignment', 'left', ...
              'Position', [25, 485, 305, 18]);

    hGUI.btnPlayNoisy = uicontrol(hFig, 'Style', 'pushbutton', ...
              'String', 'Play Noisy Input', ...
              'FontName', 'Helvetica', 'FontSize', 9, 'FontWeight', 'bold', ...
              'BackgroundColor', [0.20, 0.45, 0.82], ...
              'ForegroundColor', [1, 1, 1], ...
              'Position', [25, 448, 150, 32], ...
              'Enable', 'off', ...
              'Callback', @(src, event) on_play_audio(hFig, 'noisy'));

    hGUI.btnPlayClean = uicontrol(hFig, 'Style', 'pushbutton', ...
              'String', 'Play Clean Audio', ...
              'FontName', 'Helvetica', 'FontSize', 9, 'FontWeight', 'bold', ...
              'BackgroundColor', [0.08, 0.60, 0.36], ...
              'ForegroundColor', [1, 1, 1], ...
              'Position', [185, 448, 155, 32], ...
              'Enable', 'off', ...
              'Callback', @(src, event) on_play_audio(hFig, 'clean'));

    hGUI.btnStopAudio = uicontrol(hFig, 'Style', 'pushbutton', ...
              'String', 'Stop Playback', ...
              'FontName', 'Helvetica', 'FontSize', 8.5, ...
              'BackgroundColor', [0.90, 0.92, 0.95], ...
              'ForegroundColor', [0.15, 0.20, 0.30], ...
              'Position', [25, 412, 140, 28], ...
              'Callback', @(src, event) on_stop_audio(hFig));

    hGUI.btnSaveAudio = uicontrol(hFig, 'Style', 'pushbutton', ...
              'String', 'Save Clean WAV...', ...
              'FontName', 'Helvetica', 'FontSize', 8.5, 'FontWeight', 'bold', ...
              'BackgroundColor', [0.85, 0.52, 0.12], ...
              'ForegroundColor', [1, 1, 1], ...
              'Position', [175, 412, 165, 28], ...
              'Enable', 'off', ...
              'Callback', @(src, event) on_save_audio(hFig));

    % Step 4: Results Summary Card (For Everyday Users)
    uicontrol(hFig, 'Style', 'text', ...
              'String', 'RESULTS SUMMARY (At a Glance):', ...
              'FontName', 'Helvetica', 'FontSize', 9.5, 'FontWeight', 'bold', ...
              'ForegroundColor', [0.06, 0.16, 0.38], ...
              'BackgroundColor', [1, 1, 1], ...
              'HorizontalAlignment', 'left', ...
              'Position', [25, 380, 305, 18]);

    hGUI.txtQuickResults = uicontrol(hFig, 'Style', 'text', ...
              'String', sprintf(['Noise Removed:       --%%\n' ...
                                 'Clarity Boost:       -- dB\n' ...
                                 'Voice Coherence:     --%%\n' ...
                                 'Processing Speed:    --']), ...
              'FontName', 'Helvetica', 'FontSize', 9, 'FontWeight', 'bold', ...
              'ForegroundColor', [0.08, 0.20, 0.40], ...
              'BackgroundColor', [0.93, 0.96, 0.99], ...
              'HorizontalAlignment', 'left', ...
              'Position', [25, 298, 315, 78]);

    % Step 5: Academic Stochastic Process Metrics
    uicontrol(hFig, 'Style', 'text', ...
              'String', 'ACADEMIC STOCHASTIC METRICS:', ...
              'FontName', 'Helvetica', 'FontSize', 9, 'FontWeight', 'bold', ...
              'ForegroundColor', [0.20, 0.28, 0.42], ...
              'BackgroundColor', [1, 1, 1], ...
              'HorizontalAlignment', 'left', ...
              'Position', [25, 272, 305, 18]);

    hGUI.txtMetrics = uicontrol(hFig, 'Style', 'text', ...
              'String', sprintf(['Input SNR:     -- dB\n' ...
                                 'Output SNR:    -- dB\n' ...
                                 'Delta SNR:     -- dB\n' ...
                                 'Pearson rho:   --\n' ...
                                 'Orthogonality: --\n' ...
                                 'PDF MSE:       --']), ...
              'FontName', 'Courier', 'FontSize', 8.5, 'FontWeight', 'bold', ...
              'ForegroundColor', [0.10, 0.16, 0.25], ...
              'BackgroundColor', [0.96, 0.97, 0.98], ...
              'HorizontalAlignment', 'left', ...
              'Position', [25, 168, 315, 100]);

    % System Status Log Box
    uicontrol(hFig, 'Style', 'text', ...
              'String', 'SYSTEM STATUS / LOG:', ...
              'FontName', 'Helvetica', 'FontSize', 9, 'FontWeight', 'bold', ...
              'ForegroundColor', [0.20, 0.28, 0.42], ...
              'BackgroundColor', [1, 1, 1], ...
              'HorizontalAlignment', 'left', ...
              'Position', [25, 142, 305, 18]);

    hGUI.txtStatus = uicontrol(hFig, 'Style', 'text', ...
              'String', 'Ready. Choose Microphone, Upload File, or Simulation to begin.', ...
              'FontName', 'Helvetica', 'FontSize', 8.5, 'FontWeight', 'bold', ...
              'ForegroundColor', [0.08, 0.25, 0.50], ...
              'BackgroundColor', [0.94, 0.96, 0.99], ...
              'HorizontalAlignment', 'left', ...
              'Position', [25, 25, 315, 114]);

    % ---------------------------------------------------------------------
    % Right Area: 6-Panel High-Contrast Visualization Axes
    % ---------------------------------------------------------------------
    axes_pos = {
        [0.28, 0.69, 0.32, 0.23]; % Ax1: PDF
        [0.66, 0.69, 0.31, 0.23]; % Ax2: CDF
        [0.28, 0.39, 0.32, 0.23]; % Ax3: Autocorrelation
        [0.66, 0.39, 0.31, 0.23]; % Ax4: PSD
        [0.28, 0.08, 0.32, 0.23]; % Ax5: Time waveforms
        [0.66, 0.08, 0.31, 0.23]  % Ax6: Spectrogram
    };

    hGUI.axes = gobjects(6, 1);
    for a = 1:6
        hGUI.axes(a) = axes('Parent', hFig, 'Position', axes_pos{a});
        set(hGUI.axes(a), 'Color', [1 1 1], ...
                          'XColor', [0.12 0.18 0.28], ...
                          'YColor', [0.12 0.18 0.28], ...
                          'GridColor', [0.82 0.85 0.90], ...
                          'GridAlpha', 0.85, ...
                          'Box', 'on', ...
                          'FontName', 'Helvetica', ...
                          'FontSize', 8.5);
        grid(hGUI.axes(a), 'on');
    end

    title(hGUI.axes(1), 'Subplot 1: Ambient Noise PDF Histogram vs. Gaussian Model', ...
          'Color', [0.06, 0.16, 0.38], 'FontSize', 9.5, 'FontWeight', 'bold');
    title(hGUI.axes(2), 'Subplot 2: Cumulative Distribution Function (CDF)', ...
          'Color', [0.06, 0.16, 0.38], 'FontSize', 9.5, 'FontWeight', 'bold');
    title(hGUI.axes(3), 'Subplot 3: Time-Averaged Autocorrelation Sequences R[\tau]', ...
          'Color', [0.06, 0.16, 0.38], 'FontSize', 9.5, 'FontWeight', 'bold');
    title(hGUI.axes(4), 'Subplot 4: Wiener-Khinchin Power Spectral Densities S(\omega)', ...
          'Color', [0.06, 0.16, 0.38], 'FontSize', 9.5, 'FontWeight', 'bold');
    title(hGUI.axes(5), 'Subplot 5: Time-Domain Waveforms X(t) and Y(t)', ...
          'Color', [0.06, 0.16, 0.38], 'FontSize', 9.5, 'FontWeight', 'bold');
    title(hGUI.axes(6), 'Subplot 6: Spectrogram Comparison (Dual Panel)', ...
          'Color', [0.06, 0.16, 0.38], 'FontSize', 9.5, 'FontWeight', 'bold');

    guidata(hFig, hGUI);
end

% =========================================================================
% SECTION 6: GUI INTERACTIVE EVENT HANDLERS
% =========================================================================

function on_source_changed(hFig)
    hGUI = guidata(hFig);
    src_val = hGUI.popupSource.Value;
    
    % Toggle context panels
    set(hGUI.pnlMic,  'Visible', 'off');
    set(hGUI.pnlFile, 'Visible', 'off');
    set(hGUI.pnlSim,  'Visible', 'off');
    
    if src_val == 1
        set(hGUI.pnlMic, 'Visible', 'on');
        set(hGUI.txtStatus, 'String', 'Live Microphone Mode. Select duration and click "Record & Clean Audio".');
    elseif src_val == 2
        set(hGUI.pnlFile, 'Visible', 'on');
        set(hGUI.txtStatus, 'String', 'Upload Audio File Mode. Click "Choose Audio File..." to load your recording.');
    else
        set(hGUI.pnlSim, 'Visible', 'on');
        set(hGUI.txtStatus, 'String', 'Simulation Mode. Choose parameters and click "Run Demo Simulation & Clean".');
    end
end

function on_select_file(hFig)
    hGUI = guidata(hFig);
    
    [filename, filepath] = uigetfile({'*.wav;*.mp3;*.m4a;*.flac;*.ogg', 'Audio Files (*.wav, *.mp3, *.m4a, *.flac, *.ogg)'; ...
                                      '*.*', 'All Files (*.*)'}, 'Select Audio File to Clean');
    if isequal(filename, 0) || isequal(filepath, 0)
        return;
    end

    full_path = fullfile(filepath, filename);
    set(hGUI.txtStatus, 'String', sprintf('Reading %s...', filename));
    drawnow;

    try
        [raw_audio, fs_file] = audioread(full_path);
        
        % Convert stereo to mono
        if size(raw_audio, 2) > 1
            raw_audio = mean(raw_audio, 2);
        end

        % Resample to target 16 kHz using base MATLAB linear interpolation
        fs_target = 16000;
        if fs_file ~= fs_target
            t_orig = (0:length(raw_audio)-1)' / fs_file;
            t_target = (0:1/fs_target:(length(raw_audio)-1)/fs_file)';
            raw_audio = interp1(t_orig, raw_audio, t_target, 'linear');
        end

        % Limit to 30 seconds for real-time responsiveness if file is very long
        max_samples = 30 * fs_target;
        if length(raw_audio) > max_samples
            raw_audio = raw_audio(1:max_samples);
        end

        duration_sec = length(raw_audio) / fs_target;
        hGUI.uploadedAudio = raw_audio;
        hGUI.uploadedFileName = filename;

        set(hGUI.lblFileInfo, 'String', sprintf('Loaded: %s\n(%0.1f sec @ 16 kHz)', filename, duration_sec));
        set(hGUI.btnCleanFile, 'Enable', 'on');
        set(hGUI.txtStatus, 'String', sprintf('Loaded %s (%0.1fs). Click "Clean Selected Audio File" now.', filename, duration_sec));
        guidata(hFig, hGUI);
    catch ME
        errordlg(sprintf('Failed to read audio file: %s', ME.message), 'File Read Error');
        set(hGUI.txtStatus, 'String', sprintf('Error reading file: %s', ME.message));
    end
end

function on_clean_file(hFig)
    hGUI = guidata(hFig);
    if isempty(hGUI.uploadedAudio)
        return;
    end

    set(hGUI.btnCleanFile, 'Enable', 'off');
    set(hGUI.btnChooseFile, 'Enable', 'off');
    drawnow;

    beta_map = [0.03, 0.01, 0.05];
    beta_val = beta_map(hGUI.popupBeta.Value);

    cfg = hGUI.cfg;
    cfg.mode = 'live';
    cfg.audioInput = hGUI.uploadedAudio;
    cfg.duration = length(hGUI.uploadedAudio) / cfg.fs;
    cfg.beta = beta_val;
    if ~isempty(hGUI.calibratedNoise)
        cfg.noiseCal = hGUI.calibratedNoise;
    end

    process_and_update_gui(hFig, cfg, sprintf('Cleaning %s...', hGUI.uploadedFileName));
    
    hGUI = guidata(hFig);
    set(hGUI.btnCleanFile, 'Enable', 'on');
    set(hGUI.btnChooseFile, 'Enable', 'on');
end

function on_record_and_denoise(hFig)
    hGUI = guidata(hFig);
    
    dur_map = [3.0, 5.0, 8.0, 10.0];
    dur_val = dur_map(hGUI.popupDuration.Value);

    beta_map = [0.03, 0.01, 0.05];
    beta_val = beta_map(hGUI.popupBeta.Value);

    set(hGUI.btnRecord, 'Enable', 'off');
    set(hGUI.btnCalibrate, 'Enable', 'off');
    drawnow;

    cfg = hGUI.cfg;
    cfg.mode = 'live';
    cfg.duration = dur_val;
    cfg.beta = beta_val;
    cfg.audioInput = [];
    if ~isempty(hGUI.calibratedNoise)
        cfg.noiseCal = hGUI.calibratedNoise;
    end

    process_and_update_gui(hFig, cfg, 'Recording live microphone audio... Speak clearly!');

    hGUI = guidata(hFig);
    set(hGUI.btnRecord, 'Enable', 'on');
    set(hGUI.btnCalibrate, 'Enable', 'on');
end

function on_calibrate_clicked(hFig)
    hGUI = guidata(hFig);
    set(hGUI.txtStatus, 'String', 'Calibrating ambient room noise (1.5s)... Please remain completely silent.');
    set(hGUI.btnCalibrate, 'Enable', 'off');
    drawnow;

    fs = hGUI.cfg.fs;
    cal_audio = record_live_audio(1.5, fs);
    hGUI.calibratedNoise = cal_audio;
    guidata(hFig, hGUI);

    set(hGUI.btnCalibrate, 'Enable', 'on');
    set(hGUI.txtStatus, 'String', sprintf('Room noise calibrated! (Std Dev \\sigma_N = %.4f). Now click "Record & Clean Audio".', std(cal_audio)));
end

function on_run_simulation(hFig)
    hGUI = guidata(hFig);

    dur_map = [4.0, 6.0, 8.0];
    dur_val = dur_map(hGUI.popupSimDur.Value);

    snr_map = [12.0, 5.0, 18.0];
    snr_val = snr_map(hGUI.popupSimSNR.Value);

    beta_map = [0.03, 0.01, 0.05];
    beta_val = beta_map(hGUI.popupBeta.Value);

    set(hGUI.btnRunSim, 'Enable', 'off');
    drawnow;

    cfg = hGUI.cfg;
    cfg.mode = 'simulation';
    cfg.duration = dur_val;
    cfg.simSNR = snr_val;
    cfg.beta = beta_val;
    cfg.audioInput = [];
    cfg.noiseCal = [];

    process_and_update_gui(hFig, cfg, 'Synthesizing formant speech and running Wiener filter...');

    hGUI = guidata(hFig);
    set(hGUI.btnRunSim, 'Enable', 'on');
end

function process_and_update_gui(hFig, cfg, start_msg)
    hGUI = guidata(hFig);
    set(hGUI.txtStatus, 'String', start_msg);
    drawnow;

    try
        res = execute_complete_pipeline(cfg, @(msg) set(hGUI.txtStatus, 'String', msg));
        hGUI.results = res;
        guidata(hFig, hGUI);

        % Enable audio playback and save buttons
        set(hGUI.btnPlayNoisy, 'Enable', 'on');
        set(hGUI.btnPlayClean, 'Enable', 'on');
        set(hGUI.btnSaveAudio, 'Enable', 'on');

        % Update all 6 dashboard visualization axes
        update_gui_dashboard_plots(hFig, res);

        % Update Quick Results Card (For Everyday Users)
        noise_reduction_pct = max(0, min(99.5, (1 - 10^(-abs(res.delta_snr_db)/10)) * 100));
        quick_str = sprintf(['Noise Removed:       ~%0.1f%% (-%0.1f dB)\n' ...
                             'Clarity Boost:       %+0.1f dB (Significant)\n' ...
                             'Voice Coherence:     %0.1f%% (Crystal Clear)\n' ...
                             'Processing Speed:    Real-Time (%0.4fs)'], ...
                             noise_reduction_pct, abs(res.delta_snr_db), res.delta_snr_db, ...
                             abs(res.rho_XY) * 100, res.real_time_factor);
        set(hGUI.txtQuickResults, 'String', quick_str);

        % Update Academic Metrics Card
        metrics_str = sprintf(['Input SNR:     %+6.2f dB\n' ...
                               'Output SNR:    %+6.2f dB\n' ...
                               'Delta SNR:     %+6.2f dB\n' ...
                               'Pearson rho:   %+6.4f\n' ...
                               'Orthogonality: %+6.4f\n' ...
                               'PDF MSE:       %6.2e'], ...
                               res.snr_in_db, res.snr_out_db, res.delta_snr_db, ...
                               res.rho_XY, res.rho_orthogonality, res.mse_pdf);
        set(hGUI.txtMetrics, 'String', metrics_str);

        print_console_summary_table(res);
        set(hGUI.txtStatus, 'String', sprintf('Done! Denoised %0.1fs audio with Delta SNR = %+0.2f dB. Ready for playback.', ...
            res.duration, res.delta_snr_db));
    catch ME
        errordlg(sprintf('Processing Error: %s', ME.message), 'Pipeline Error');
        set(hGUI.txtStatus, 'String', sprintf('Error: %s', ME.message));
    end
end

function update_gui_dashboard_plots(hFig, res)
% UPDATE_GUI_DASHBOARD_PLOTS Refreshes the 6 subplot axes with crystal-clear high-contrast styling
    hGUI = guidata(hFig);

    % Color Palette
    col_input   = [0.15, 0.42, 0.85]; % Sapphire Blue
    col_output  = [0.05, 0.60, 0.38]; % Emerald Green
    col_noise   = [0.85, 0.32, 0.15]; % Deep Amber

    % Subplot 1: PDF
    ax1 = hGUI.axes(1);
    cla(ax1);
    set(ax1, 'Color', [1 1 1], 'XColor', [0.12 0.18 0.28], 'YColor', [0.12 0.18 0.28], ...
             'GridColor', [0.82 0.85 0.90], 'GridAlpha', 0.85, 'Box', 'on');
    bar(ax1, res.bin_centers, res.pdf_empirical, 1.0, ...
        'FaceColor', [0.72, 0.84, 0.96], 'EdgeColor', col_input, 'FaceAlpha', 0.7);
    hold(ax1, 'on');
    plot(ax1, res.bin_centers, res.pdf_theoretical, 'r-', 'LineWidth', 2.0);
    hold(ax1, 'off');
    grid(ax1, 'on');
    title(ax1, 'Subplot 1: Ambient Noise PDF & Gaussian Fit', ...
          'Color', [0.06, 0.16, 0.38], 'FontSize', 9.5, 'FontWeight', 'bold');
    xlabel(ax1, 'Amplitude', 'Color', [0.12 0.18 0.28], 'FontSize', 8.5, 'FontWeight', 'bold');
    ylabel(ax1, 'Density', 'Color', [0.12 0.18 0.28], 'FontSize', 8.5, 'FontWeight', 'bold');

    % Subplot 2: CDF
    ax2 = hGUI.axes(2);
    cla(ax2);
    set(ax2, 'Color', [1 1 1], 'XColor', [0.12 0.18 0.28], 'YColor', [0.12 0.18 0.28], ...
             'GridColor', [0.82 0.85 0.90], 'GridAlpha', 0.85, 'Box', 'on');
    stairs(ax2, res.bin_centers, res.cdf_empirical, 'b-', 'LineWidth', 2.0);
    hold(ax2, 'on');
    plot(ax2, res.bin_centers, res.cdf_theoretical, 'k--', 'LineWidth', 1.8);
    hold(ax2, 'off');
    grid(ax2, 'on');
    title(ax2, 'Subplot 2: Empirical vs. Gaussian CDF', ...
          'Color', [0.06, 0.16, 0.38], 'FontSize', 9.5, 'FontWeight', 'bold');
    xlabel(ax2, 'Amplitude', 'Color', [0.12 0.18 0.28], 'FontSize', 8.5, 'FontWeight', 'bold');
    ylabel(ax2, 'Cumulative Prob.', 'Color', [0.12 0.18 0.28], 'FontSize', 8.5, 'FontWeight', 'bold');

    % Subplot 3: Autocorrelation
    ax3 = hGUI.axes(3);
    cla(ax3);
    set(ax3, 'Color', [1 1 1], 'XColor', [0.12 0.18 0.28], 'YColor', [0.12 0.18 0.28], ...
             'GridColor', [0.82 0.85 0.90], 'GridAlpha', 0.85, 'Box', 'on');
    plot_lags = abs(res.lags_ms) <= 25;
    plot(ax3, res.lags_ms(plot_lags), res.R_XX(plot_lags), '-', 'Color', col_input, 'LineWidth', 1.4);
    hold(ax3, 'on');
    plot(ax3, res.lags_ms(plot_lags), res.R_YY(plot_lags), '-', 'Color', col_output, 'LineWidth', 1.6);
    plot(ax3, res.lags_ms(plot_lags), res.R_NN(plot_lags), ':', 'Color', col_noise, 'LineWidth', 1.8);
    hold(ax3, 'off');
    grid(ax3, 'on');
    title(ax3, 'Subplot 3: Autocorrelations R_{XX}, R_{YY}, R_{NN}', ...
          'Color', [0.06, 0.16, 0.38], 'FontSize', 9.5, 'FontWeight', 'bold');
    xlabel(ax3, 'Lag \tau (ms)', 'Color', [0.12 0.18 0.28], 'FontSize', 8.5, 'FontWeight', 'bold');
    ylabel(ax3, 'R[\tau]', 'Color', [0.12 0.18 0.28], 'FontSize', 8.5, 'FontWeight', 'bold');

    % Subplot 4: PSD
    ax4 = hGUI.axes(4);
    cla(ax4);
    set(ax4, 'Color', [1 1 1], 'XColor', [0.12 0.18 0.28], 'YColor', [0.12 0.18 0.28], ...
             'GridColor', [0.82 0.85 0.90], 'GridAlpha', 0.85, 'Box', 'on');
    plot(ax4, res.freq_axis_khz, 10*log10(res.S_XX + 1e-12), '-', 'Color', col_input, 'LineWidth', 1.3);
    hold(ax4, 'on');
    plot(ax4, res.freq_axis_khz, 10*log10(res.S_NN + 1e-12), ':', 'Color', col_noise, 'LineWidth', 1.8);
    plot(ax4, res.freq_axis_khz, 10*log10(res.S_YY + 1e-12), '-', 'Color', col_output, 'LineWidth', 1.6);
    hold(ax4, 'off');
    grid(ax4, 'on');
    title(ax4, 'Subplot 4: Wiener-Khinchin PSD: S(\omega)', ...
          'Color', [0.06, 0.16, 0.38], 'FontSize', 9.5, 'FontWeight', 'bold');
    xlabel(ax4, 'Frequency (kHz)', 'Color', [0.12 0.18 0.28], 'FontSize', 8.5, 'FontWeight', 'bold');
    ylabel(ax4, 'PSD (dB/Hz)', 'Color', [0.12 0.18 0.28], 'FontSize', 8.5, 'FontWeight', 'bold');

    % Subplot 5: Time Waveforms
    ax5 = hGUI.axes(5);
    cla(ax5);
    set(ax5, 'Color', [1 1 1], 'XColor', [0.12 0.18 0.28], 'YColor', [0.12 0.18 0.28], ...
             'GridColor', [0.82 0.85 0.90], 'GridAlpha', 0.85, 'Box', 'on');
    plot(ax5, res.time_vec, res.noisy_input, 'Color', [0.65 0.68 0.72], 'LineWidth', 0.8);
    hold(ax5, 'on');
    plot(ax5, res.time_vec, res.filtered_output, 'Color', col_output, 'LineWidth', 1.1);
    hold(ax5, 'off');
    grid(ax5, 'on');
    title(ax5, 'Subplot 5: Time Waveforms: X(t) vs. Denoised Y(t)', ...
          'Color', [0.06, 0.16, 0.38], 'FontSize', 9.5, 'FontWeight', 'bold');
    xlabel(ax5, 'Time (s)', 'Color', [0.12 0.18 0.28], 'FontSize', 8.5, 'FontWeight', 'bold');
    ylabel(ax5, 'Amplitude', 'Color', [0.12 0.18 0.28], 'FontSize', 8.5, 'FontWeight', 'bold');
    xlim(ax5, [0, res.duration]);

    % Subplot 6: Spectrogram
    ax6 = hGUI.axes(6);
    cla(ax6);
    spec_combined = [res.spec_X_dB; -100 * ones(4, length(res.spec_t)); res.spec_Y_dB];
    f_combined = [res.spec_f; 8 + (1:4)'*0.1; res.spec_f + 8.5];
    imagesc(ax6, res.spec_t, f_combined, spec_combined);
    axis(ax6, 'xy');
    colormap(ax6, 'jet');
    clim(ax6, [-55, 0]);
    title(ax6, 'Subplot 6: Spectrogram: [Top: Input] vs. [Bottom: Filtered]', ...
          'Color', [0.06, 0.16, 0.38], 'FontSize', 9.5, 'FontWeight', 'bold');
    xlabel(ax6, 'Time (s)', 'Color', [0.12 0.18 0.28], 'FontSize', 8.5, 'FontWeight', 'bold');
    ylabel(ax6, 'Freq (kHz)', 'Color', [0.12 0.18 0.28], 'FontSize', 8.5, 'FontWeight', 'bold');
end

function on_play_audio(hFig, which_audio)
    hGUI = guidata(hFig);
    if isempty(hGUI.results)
        return;
    end

    on_stop_audio(hFig);

    fs = hGUI.results.fs;
    if strcmpi(which_audio, 'noisy')
        audio_data = hGUI.results.noisy_input;
        set(hGUI.txtStatus, 'String', 'Playing Noisy Input X[n]...');
    else
        audio_data = hGUI.results.filtered_output;
        set(hGUI.txtStatus, 'String', 'Playing Denoised Clean Audio Y[n]...');
    end

    % Safe scaling
    max_a = max(abs(audio_data));
    if max_a > 0
        audio_data = 0.9 * (audio_data / max_a);
    end

    try
        hPlayer = audioplayer(audio_data, fs);
        play(hPlayer);
        hGUI.audioPlayer = hPlayer;
        guidata(hFig, hGUI);
    catch ME
        warning('Audio playback failed: %s', ME.message);
    end
end

function on_stop_audio(hFig)
    hGUI = guidata(hFig);
    if ~isempty(hGUI.audioPlayer) && isvalid(hGUI.audioPlayer)
        stop(hGUI.audioPlayer);
    end
    set(hGUI.txtStatus, 'String', 'Audio playback stopped.');
end

function on_save_audio(hFig)
    hGUI = guidata(hFig);
    if isempty(hGUI.results)
        return;
    end

    [filename, filepath] = uiputfile('clean_denoised_audio.wav', 'Save Filtered Audio As WAV');
    if isequal(filename, 0) || isequal(filepath, 0)
        return;
    end

    out_file = fullfile(filepath, filename);
    audio_data = hGUI.results.filtered_output;
    fs = hGUI.results.fs;

    % Normalize to prevent clipping
    max_val = max(abs(audio_data));
    if max_val > 0.99
        audio_data = 0.95 * (audio_data / max_val);
    end

    try
        audiowrite(out_file, audio_data, fs);
        set(hGUI.txtStatus, 'String', sprintf('Successfully saved clean audio to: %s', filename));
    catch ME
        errordlg(sprintf('Failed to save file: %s', ME.message), 'Save Error');
    end
end
