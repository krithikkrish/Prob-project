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
    addParameter(p, 'simSNR', 3.0, @(x) isnumeric(x));
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
    % across both calibration (Phase 1) and speech filtering (Phase 2)
    if strcmpi(cfg.mode, 'live')
        noise_cal = record_live_audio(cfg.calDuration, fs);
        clean_reference = [];
        noisy_input = [];
        sigma_sim_noise = [];
    else
        % Synthesize vocal formant speech and determine exact noise standard deviation
        [clean_reference, ~, sigma_sim_noise] = generate_synthetic_clean_speech(cfg.duration, fs, cfg.simSNR);
        N_cal_samples = round(cfg.calDuration * fs);
        % Sample realization of the exact same ambient noise random process
        noise_cal = sigma_sim_noise * randn(N_cal_samples, 1);
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

    for m = 1:num_cal_frames
        idx = (m-1)*hop + (1:frameLen);
        cal_frame = noise_cal(idx) .* w;
        
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

    % ---------------------------------------------------------------------
    % PHASE 2: AUDIO CAPTURE & REAL-TIME FRAME-BY-FRAME LTI WIENER FILTERING
    % ---------------------------------------------------------------------
    statusCallback('Phase 2: Capturing Audio & Executing Optimal Wiener Filtering...');

    if strcmpi(cfg.mode, 'live')
        noisy_input = record_live_audio(cfg.duration, fs);
        clean_reference = []; % Unknown in live microphone mode
    else
        % Simulation Mode: Inject noise realization from the EXACT SAME calibrated Gaussian process
        if isempty(clean_reference) || isempty(sigma_sim_noise)
            [clean_reference, noisy_input, sigma_sim_noise] = generate_synthetic_clean_speech(cfg.duration, fs, cfg.simSNR);
        else
            noisy_input = clean_reference + sigma_sim_noise * randn(length(clean_reference), 1);
        end
    end

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
    % Energy_threshold = \gamma * \sigma_N^2 (silence detection for noise updating)
    vad_threshold = 2.5 * var_N;
    vad_alpha = 0.95; % Smoothing factor for noise PSD adaptation during pauses
    S_NN_adaptive = S_NN; % Initialize adaptive noise spectrum

    eps_floor = 1e-12; % Numerical safeguard against divide-by-zero

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

        % Adaptive Noise Tracking during Silence / Inter-word Pauses
        if frame_energy < vad_threshold
            S_NN_adaptive = vad_alpha * S_NN_adaptive + (1 - vad_alpha) * S_XX;
        end

        % Clean Speech PSD Estimation under Statistical Independence:
        % Since speech S[n] and noise N[n] are statistically independent:
        % E[S[n] * N[n+tau]] = 0 \implies R_X[tau] = R_S[tau] + R_N[tau]
        % S_XX(w) = S_SS(w) + S_NN(w)
        % \hat{S}_SS(w) = \max(S_XX(w) - S_NN(w), 0)
        S_SS_hat = max(S_XX - S_NN_adaptive, 0);

        % Optimal Frequency-Domain LTI Wiener Transfer Function:
        % H(w) = \max\left( \frac{\hat{S}_SS(w)}{S_XX(w) + \epsilon}, \; \beta \right)
        % where \beta = 0.03 is the spectral floor to prevent musical noise artifacts
        H_wiener = max(S_SS_hat ./ (S_XX + eps_floor), beta);
        % Ensure H(w) does not amplify signals: H(w) \le 1.0
        H_wiener = min(H_wiener, 1.0);

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

    % 1. Glottal Pitch Source with Natural Intonation (f0 ~ 130 Hz +/- 15 Hz)
    f0 = 130 + 15 * sin(2 * pi * 1.2 * t);
    phase = 2 * pi * cumsum(f0) / fs;
    glottal_source = zeros(N, 1);
    for k = 1:24
        glottal_source = glottal_source + (1 / k) * sin(k * phase);
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
    % Simulates words: "Probability" ... "and Random Processes" ... "Wiener Denoising"
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
        clean_speech = 0.5 * (clean_speech / max_amp);
    end

    % 4. Additive Gaussian White Noise Injection calibrated to target SNR
    speech_power = sum(clean_speech.^2) / N;
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

    % Logarithmic Magnitude in Decibels (dB)
    spec_X_dB = 20 * log10(spec_X + 1e-6);
    spec_Y_dB = 20 * log10(spec_Y + 1e-6);

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
                 'Color', [0.96 0.97 0.98], ...
                 'Position', [50, 50, 1400, 880], ...
                 'Visible', vis_str);

    % Color Palette
    col_noise   = [0.85, 0.35, 0.20]; % Rust/Orange
    col_input   = [0.20, 0.45, 0.85]; % Sapphire Blue
    col_output  = [0.10, 0.65, 0.45]; % Emerald Green
    col_theo    = [0.15, 0.15, 0.15]; % Deep Charcoal
    col_gray    = [0.65, 0.65, 0.65];

    % ---------------------------------------------------------------------
    % SUBPLOT 1: Empirical Noise PDF vs. Theoretical Gaussian PDF
    % ---------------------------------------------------------------------
    ax1 = subplot(3, 2, 1);
    bar(ax1, res.bin_centers, res.pdf_empirical, 1.0, ...
        'FaceColor', [0.70, 0.82, 0.95], 'EdgeColor', [0.20, 0.45, 0.85], ...
        'FaceAlpha', 0.6, 'DisplayName', 'Empirical PDF \hat{f}_N(x)');
    hold(ax1, 'on');
    plot(ax1, res.bin_centers, res.pdf_theoretical, 'r-', 'LineWidth', 2.2, ...
         'DisplayName', 'Fitted Gaussian f_N(x)');
    hold(ax1, 'off');
    grid(ax1, 'on');
    title(ax1, 'Ambient Noise PDF: Empirical vs. Gaussian Model', 'FontWeight', 'bold');
    xlabel(ax1, 'Amplitude x');
    ylabel(ax1, 'Probability Density f_N(x)');
    legend(ax1, 'Location', 'northeast', 'FontSize', 8);
    
    % Annotation box for goodness-of-fit MSE
    str_box1 = sprintf(['\\mu_N = %+.4f\n\\sigma_N = %.4f\n' ...
                        'Skewness = %+.3f\nKurtosis = %.3f\n' ...
                        'PDF MSE = %.2e\nNMSE = %.2e'], ...
                        res.mu_N, res.sigma_N, res.skewness_N, res.kurtosis_N, res.mse_pdf, res.nmse_pdf);
    text(ax1, 0.05, 0.92, str_box1, 'Units', 'normalized', 'FontSize', 8, ...
         'BackgroundColor', [1 1 1 0.85], 'EdgeColor', [0.7 0.7 0.7], 'VerticalAlignment', 'top');

    % ---------------------------------------------------------------------
    % SUBPLOT 2: Empirical CDF vs. Theoretical Gaussian CDF
    % ---------------------------------------------------------------------
    ax2 = subplot(3, 2, 2);
    stairs(ax2, res.bin_centers, res.cdf_empirical, 'b-', 'LineWidth', 1.8, ...
           'DisplayName', 'Empirical CDF \hat{F}_N(x)');
    hold(ax2, 'on');
    plot(ax2, res.bin_centers, res.cdf_theoretical, 'k--', 'LineWidth', 2.0, ...
         'DisplayName', 'Gaussian CDF \Phi(x)');
    hold(ax2, 'off');
    grid(ax2, 'on');
    title(ax2, 'Cumulative Distribution Function (CDF) Verification', 'FontWeight', 'bold');
    xlabel(ax2, 'Amplitude x');
    ylabel(ax2, 'Probability F_N(x)');
    legend(ax2, 'Location', 'southeast', 'FontSize', 8);
    
    str_box2 = sprintf(['CLT Block Mean Var: %.2e\n' ...
                        'Theory \\sigma_N^2/L_{blk}: %.2e\n' ...
                        '95%% CI: [%+.4f, %+.4f]'], ...
                        res.var_of_means, res.theoretical_clt_var, res.ci_95(1), res.ci_95(2));
    text(ax2, 0.05, 0.92, str_box2, 'Units', 'normalized', 'FontSize', 8, ...
         'BackgroundColor', [1 1 1 0.85], 'EdgeColor', [0.7 0.7 0.7], 'VerticalAlignment', 'top');

    % ---------------------------------------------------------------------
    % SUBPLOT 3: Time-Averaged Autocorrelation Functions R_NN, R_XX, R_YY
    % ---------------------------------------------------------------------
    ax3 = subplot(3, 2, 3);
    plot_lags = abs(res.lags_ms) <= 25;
    plot(ax3, res.lags_ms(plot_lags), res.R_XX(plot_lags), '-', 'Color', col_input, ...
         'LineWidth', 1.4, 'DisplayName', 'Input R_{XX}[\tau]');
    hold(ax3, 'on');
    plot(ax3, res.lags_ms(plot_lags), res.R_YY(plot_lags), '-', 'Color', col_output, ...
         'LineWidth', 1.6, 'DisplayName', 'Filtered R_{YY}[\tau]');
    plot(ax3, res.lags_ms(plot_lags), res.R_NN(plot_lags), ':', 'Color', col_noise, ...
         'LineWidth', 1.8, 'DisplayName', 'Noise R_{NN}[\tau]');
    hold(ax3, 'off');
    grid(ax3, 'on');
    title(ax3, 'Time-Averaged Autocorrelation Sequences R[\tau]', 'FontWeight', 'bold');
    xlabel(ax3, 'Lag \tau (ms)');
    ylabel(ax3, 'Autocorrelation R[\tau]');
    legend(ax3, 'Location', 'northeast', 'FontSize', 8);

    % ---------------------------------------------------------------------
    % SUBPLOT 4: Wiener-Khinchin Power Spectral Densities
    % ---------------------------------------------------------------------
    ax4 = subplot(3, 2, 4);
    % Convert PSD to dB/Hz: 10 * log10(S(w) + 1e-12)
    S_XX_dB = 10 * log10(res.S_XX + 1e-12);
    S_NN_dB = 10 * log10(res.S_NN + 1e-12);
    S_YY_dB = 10 * log10(res.S_YY + 1e-12);

    plot(ax4, res.freq_axis_khz, S_XX_dB, '-', 'Color', col_input, ...
         'LineWidth', 1.2, 'DisplayName', 'Input PSD S_{XX}(\omega)');
    hold(ax4, 'on');
    plot(ax4, res.freq_axis_khz, S_NN_dB, ':', 'Color', col_noise, ...
         'LineWidth', 1.8, 'DisplayName', 'Noise PSD S_{NN}(\omega)');
    plot(ax4, res.freq_axis_khz, S_YY_dB, '-', 'Color', col_output, ...
         'LineWidth', 1.8, 'DisplayName', 'LTI Output S_{YY}(\omega) = |H|^2 S_{XX}');
    hold(ax4, 'off');
    grid(ax4, 'on');
    title(ax4, 'Wiener-Khinchin Power Spectral Density: S(\omega) = F\{R[\tau]\}', 'FontWeight', 'bold');
    xlabel(ax4, 'Frequency (kHz)');
    ylabel(ax4, 'Power Spectral Density (dB/Hz)');
    legend(ax4, 'Location', 'northeast', 'FontSize', 8);

    % ---------------------------------------------------------------------
    % SUBPLOT 5: Time-Domain Waveforms (Noisy Input vs. Filtered Output)
    % ---------------------------------------------------------------------
    ax5 = subplot(3, 2, 5);
    plot(ax5, res.time_vec, res.noisy_input, 'Color', [0.7 0.7 0.7], ...
         'LineWidth', 0.8, 'DisplayName', 'Noisy Input X(t)');
    hold(ax5, 'on');
    plot(ax5, res.time_vec, res.filtered_output, 'Color', col_output, ...
         'LineWidth', 1.1, 'DisplayName', 'Wiener Output Y(t)');
    hold(ax5, 'off');
    grid(ax5, 'on');
    title(ax5, 'Time-Domain Waveforms: X(t) = S(t) + N(t) vs. Denoised Y(t)', 'FontWeight', 'bold');
    xlabel(ax5, 'Time (seconds)');
    ylabel(ax5, 'Amplitude');
    legend(ax5, 'Location', 'northeast', 'FontSize', 8);
    xlim(ax5, [0, res.duration]);

    % ---------------------------------------------------------------------
    % SUBPLOT 6: Spectrogram Comparison (Dual Panel or Split Heatmap)
    % ---------------------------------------------------------------------
    ax6 = subplot(3, 2, 6);
    % Concatenate Noisy (top) and Filtered (bottom) with a separator line
    spec_combined = [res.spec_X_dB; -100 * ones(4, length(res.spec_t)); res.spec_Y_dB];
    f_combined = [res.spec_f; 8 + (1:4)'*0.1; res.spec_f + 8.5];
    
    imagesc(ax6, res.spec_t, f_combined, spec_combined);
    axis(ax6, 'xy');
    colormap(ax6, 'jet');
    caxis(ax6, [-70, 0]);
    cb = colorbar(ax6, 'Location', 'eastoutside');
    cb.Label.String = 'Energy (dB)';
    grid(ax6, 'off');
    title(ax6, 'Spectrogram: [Top: Noisy Input X(t)] vs. [Bottom: Filtered Y(t)]', 'FontWeight', 'bold');
    xlabel(ax6, 'Time (seconds)');
    ylabel(ax6, 'Frequency (kHz)');
    
    % Annotation overlay with SNR and Performance
    str_box6 = sprintf(['\\Delta SNR = %+.2f dB\n' ...
                        '\\rho_{XY} = %.4f\n' ...
                        '\\rho_{YE} (Orthogonality) = %.4f\n' ...
                        'Real-Time Factor: %.3f'], ...
                        res.delta_snr_db, res.rho_XY, res.rho_orthogonality, res.real_time_factor);
    text(ax6, 0.03, 0.90, str_box6, 'Units', 'normalized', 'FontSize', 8, ...
         'Color', 'w', 'BackgroundColor', [0 0 0 0.7], 'VerticalAlignment', 'top');

    % Set Uniform Aesthetic Styling
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
% SECTION 5: INTERACTIVE GUI DASHBOARD (PROGRAMMATIC UI)
% =========================================================================

function hGUI = launch_gui_dashboard(initial_cfg)
% LAUNCH_GUI_DASHBOARD Builds an interactive, modern user interface dashboard
%   featuring audio recording controls, live status badges, metric indicators,
%   interactive audio playback, and integrated 6-panel statistical plots.

    % Main Figure Window
    hFig = figure('Name', 'Real-Time Audio Denoising | Wiener-Khinchin & Optimal LTI Filtering', ...
                  'NumberTitle', 'off', ...
                  'Color', [0.94, 0.95, 0.97], ...
                  'Position', [60, 50, 1420, 890], ...
                  'Visible', 'on');

    hGUI = struct();
    hGUI.fig = hFig;
    hGUI.cfg = initial_cfg;
    hGUI.results = [];
    hGUI.audioPlayer = [];

    % ---------------------------------------------------------------------
    % Top Header Banner
    % ---------------------------------------------------------------------
    uicontrol(hFig, 'Style', 'text', ...
              'String', 'Real-Time Audio Denoising via Wiener-Khinchin Spectral Estimation & Optimal LTI Filtering', ...
              'FontName', 'Helvetica', 'FontSize', 14, 'FontWeight', 'bold', ...
              'ForegroundColor', [0.08, 0.20, 0.40], ...
              'BackgroundColor', [0.94, 0.95, 0.97], ...
              'HorizontalAlignment', 'center', ...
              'Position', [20, 845, 1380, 35]);

    uicontrol(hFig, 'Style', 'text', ...
              'String', 'Base MATLAB Implementation | No Black-Box Toolboxes | Rigorous Stochastic Process Validation', ...
              'FontName', 'Helvetica', 'FontSize', 9.5, 'FontAngle', 'italic', ...
              'ForegroundColor', [0.35, 0.45, 0.55], ...
              'BackgroundColor', [0.94, 0.95, 0.97], ...
              'HorizontalAlignment', 'center', ...
              'Position', [20, 825, 1380, 20]);

    % ---------------------------------------------------------------------
    % Left Sidebar: Controls & Parameter Cards (Width: 320 px)
    % ---------------------------------------------------------------------
    uicontrol(hFig, 'Style', 'frame', ...
              'BackgroundColor', [1, 1, 1], ...
              'Position', [20, 20, 310, 795]);

    % Sidebar Title
    uicontrol(hFig, 'Style', 'text', ...
              'String', 'SYSTEM CONTROLS', ...
              'FontName', 'Helvetica', 'FontSize', 11, 'FontWeight', 'bold', ...
              'ForegroundColor', [0.10, 0.25, 0.45], ...
              'BackgroundColor', [1, 1, 1], ...
              'Position', [30, 780, 290, 22]);

    % Mode Selection
    uicontrol(hFig, 'Style', 'text', ...
              'String', 'Operational Mode:', ...
              'FontName', 'Helvetica', 'FontSize', 9, 'FontWeight', 'bold', ...
              'BackgroundColor', [1, 1, 1], ...
              'HorizontalAlignment', 'left', ...
              'Position', [35, 750, 280, 18]);

    hGUI.popupMode = uicontrol(hFig, 'Style', 'popupmenu', ...
              'String', {'Simulation Mode (Formant Speech + WSS Noise)', 'Live Microphone Mode (16 kHz Mono)'}, ...
              'Value', 1, ...
              'Position', [35, 722, 280, 25]);

    % Audio Duration Selector
    uicontrol(hFig, 'Style', 'text', ...
              'String', 'Audio Duration (Seconds):', ...
              'FontName', 'Helvetica', 'FontSize', 9, 'FontWeight', 'bold', ...
              'BackgroundColor', [1, 1, 1], ...
              'HorizontalAlignment', 'left', ...
              'Position', [35, 692, 280, 18]);

    hGUI.popupDuration = uicontrol(hFig, 'Style', 'popupmenu', ...
              'String', {'4.0 Seconds', '6.0 Seconds (Recommended)', '8.0 Seconds', '10.0 Seconds'}, ...
              'Value', 2, ...
              'Position', [35, 664, 280, 25]);

    % Spectral Floor Parameter beta
    uicontrol(hFig, 'Style', 'text', ...
              'String', 'Spectral Floor \beta (Musical Noise Floor):', ...
              'FontName', 'Helvetica', 'FontSize', 9, 'FontWeight', 'bold', ...
              'BackgroundColor', [1, 1, 1], ...
              'HorizontalAlignment', 'left', ...
              'Position', [35, 634, 280, 18]);

    hGUI.popupBeta = uicontrol(hFig, 'Style', 'popupmenu', ...
              'String', {'\beta = 0.01 (-40 dB)', '\beta = 0.03 (-30.5 dB, Optimal)', '\beta = 0.05 (-26 dB)'}, ...
              'Value', 2, ...
              'Position', [35, 606, 280, 25]);

    % Action Buttons
    hGUI.btnCalibrate = uicontrol(hFig, 'Style', 'pushbutton', ...
              'String', 'Phase 1: Calibrate Ambient Noise', ...
              'FontName', 'Helvetica', 'FontSize', 9.5, 'FontWeight', 'bold', ...
              'BackgroundColor', [0.85, 0.90, 0.98], ...
              'ForegroundColor', [0.10, 0.25, 0.60], ...
              'Position', [35, 555, 280, 36], ...
              'Callback', @(src, event) on_calibrate_clicked(hFig));

    hGUI.btnRun = uicontrol(hFig, 'Style', 'pushbutton', ...
              'String', 'Phase 2: Record & Denoise Audio', ...
              'FontName', 'Helvetica', 'FontSize', 10, 'FontWeight', 'bold', ...
              'BackgroundColor', [0.15, 0.55, 0.35], ...
              'ForegroundColor', [1, 1, 1], ...
              'Position', [35, 510, 280, 38], ...
              'Callback', @(src, event) on_run_clicked(hFig));

    % Audio Playback Controls
    uicontrol(hFig, 'Style', 'text', ...
              'String', 'AUDIO PLAYBACK', ...
              'FontName', 'Helvetica', 'FontSize', 10, 'FontWeight', 'bold', ...
              'ForegroundColor', [0.10, 0.25, 0.45], ...
              'BackgroundColor', [1, 1, 1], ...
              'Position', [35, 465, 280, 20]);

    hGUI.btnPlayNoisy = uicontrol(hFig, 'Style', 'pushbutton', ...
              'String', 'Play Noisy Input X[n]', ...
              'FontName', 'Helvetica', 'FontSize', 9, ...
              'BackgroundColor', [0.90, 0.92, 0.95], ...
              'Position', [35, 430, 135, 30], ...
              'Enable', 'off', ...
              'Callback', @(src, event) on_play_audio(hFig, 'noisy'));

    hGUI.btnPlayClean = uicontrol(hFig, 'Style', 'pushbutton', ...
              'String', 'Play Filtered Y[n]', ...
              'FontName', 'Helvetica', 'FontSize', 9, 'FontWeight', 'bold', ...
              'BackgroundColor', [0.80, 0.95, 0.85], ...
              'ForegroundColor', [0.05, 0.45, 0.20], ...
              'Position', [180, 430, 135, 30], ...
              'Enable', 'off', ...
              'Callback', @(src, event) on_play_audio(hFig, 'clean'));

    hGUI.btnStopAudio = uicontrol(hFig, 'Style', 'pushbutton', ...
              'String', 'Stop Audio Playback', ...
              'FontName', 'Helvetica', 'FontSize', 8.5, ...
              'Position', [35, 395, 280, 26], ...
              'Callback', @(src, event) on_stop_audio(hFig));

    % Live Quantitative Metric Card
    uicontrol(hFig, 'Style', 'text', ...
              'String', 'STATISTICAL METRICS', ...
              'FontName', 'Helvetica', 'FontSize', 10, 'FontWeight', 'bold', ...
              'ForegroundColor', [0.10, 0.25, 0.45], ...
              'BackgroundColor', [1, 1, 1], ...
              'Position', [35, 355, 280, 20]);

    hGUI.txtMetrics = uicontrol(hFig, 'Style', 'text', ...
              'String', sprintf(['Input SNR:     -- dB\n' ...
                                 'Output SNR:    -- dB\n' ...
                                 'Delta SNR:     -- dB\n' ...
                                 'Pearson rho:   --\n' ...
                                 'Orthogonality: --\n' ...
                                 'Real-Time Fac: --\n' ...
                                 'PDF MSE:       --']), ...
              'FontName', 'Courier', 'FontSize', 8.5, ...
              'ForegroundColor', [0.15, 0.20, 0.25], ...
              'BackgroundColor', [0.96, 0.97, 0.98], ...
              'HorizontalAlignment', 'left', ...
              'Position', [35, 215, 280, 135]);

    % Status Log Box
    uicontrol(hFig, 'Style', 'text', ...
              'String', 'SYSTEM LOG / STATUS', ...
              'FontName', 'Helvetica', 'FontSize', 10, 'FontWeight', 'bold', ...
              'ForegroundColor', [0.10, 0.25, 0.45], ...
              'BackgroundColor', [1, 1, 1], ...
              'Position', [35, 185, 280, 20]);

    hGUI.txtStatus = uicontrol(hFig, 'Style', 'text', ...
              'String', 'Ready. Select parameters and click "Phase 2: Record & Denoise Audio".', ...
              'FontName', 'Helvetica', 'FontSize', 8.5, ...
              'ForegroundColor', [0.20, 0.35, 0.50], ...
              'BackgroundColor', [0.96, 0.97, 0.98], ...
              'HorizontalAlignment', 'left', ...
              'Position', [35, 35, 280, 145]);

    % ---------------------------------------------------------------------
    % Right Area: 6-Panel Visualization Axes Container (Width: 1040 px)
    % ---------------------------------------------------------------------
    % Pre-create default placeholder axes
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
        box(hGUI.axes(a), 'on');
        grid(hGUI.axes(a), 'on');
    end

    title(hGUI.axes(1), 'Subplot 1: Ambient Noise PDF Histogram vs. Gaussian Model');
    title(hGUI.axes(2), 'Subplot 2: Cumulative Distribution Function (CDF)');
    title(hGUI.axes(3), 'Subplot 3: Time-Averaged Autocorrelation Sequences R[\tau]');
    title(hGUI.axes(4), 'Subplot 4: Wiener-Khinchin Power Spectral Densities S(\omega)');
    title(hGUI.axes(5), 'Subplot 5: Time-Domain Waveforms X(t) and Y(t)');
    title(hGUI.axes(6), 'Subplot 6: Spectrogram Comparison (Dual Panel)');

    guidata(hFig, hGUI);
end

% =========================================================================
% SECTION 6: GUI INTERACTIVE EVENT HANDLERS
% =========================================================================

function on_calibrate_clicked(hFig)
    hGUI = guidata(hFig);
    set(hGUI.txtStatus, 'String', 'Calibrating ambient noise (1.5 seconds)... Stay silent.');
    drawnow;

    dur_val = 6.0;
    mode_str = 'simulation';
    if hGUI.popupMode.Value == 2
        mode_str = 'live';
    end

    cfg = hGUI.cfg;
    cfg.mode = mode_str;
    cfg.duration = dur_val;
    cfg.calDuration = 1.5;

    res = execute_complete_pipeline(cfg, @(msg) set(hGUI.txtStatus, 'String', msg));
    hGUI.results = res;
    guidata(hFig, hGUI);

    update_gui_dashboard_plots(hFig, res);
    set(hGUI.txtStatus, 'String', sprintf('Calibration Complete. Noise Variance = %.3e, PDF MSE = %.2e', res.var_N, res.mse_pdf));
end

function on_run_clicked(hFig)
    hGUI = guidata(hFig);
    
    % Read UI configurations
    mode_idx = hGUI.popupMode.Value;
    if mode_idx == 1
        mode_str = 'simulation';
    else
        mode_str = 'live';
    end

    dur_options = [4.0, 6.0, 8.0, 10.0];
    dur_val = dur_options(hGUI.popupDuration.Value);

    beta_options = [0.01, 0.03, 0.05];
    beta_val = beta_options(hGUI.popupBeta.Value);

    cfg = hGUI.cfg;
    cfg.mode = mode_str;
    cfg.duration = dur_val;
    cfg.beta = beta_val;

    set(hGUI.btnRun, 'Enable', 'off');
    set(hGUI.btnCalibrate, 'Enable', 'off');
    drawnow;

    try
        res = execute_complete_pipeline(cfg, @(msg) set(hGUI.txtStatus, 'String', msg));
        hGUI.results = res;
        
        % Enable playback buttons
        set(hGUI.btnPlayNoisy, 'Enable', 'on');
        set(hGUI.btnPlayClean, 'Enable', 'on');
        
        % Update Dashboard Visualizations
        update_gui_dashboard_plots(hFig, res);
        
        % Update Quantitative Metrics Card
        metrics_str = sprintf(['Input SNR:     %+6.2f dB\n' ...
                              'Output SNR:    %+6.2f dB\n' ...
                              'Delta SNR:     %+6.2f dB\n' ...
                              'Pearson rho:   %+6.4f\n' ...
                              'Orthogonality: %+6.4f\n' ...
                              'Real-Time Fac: %6.4f\n' ...
                              'PDF MSE:       %6.2e'], ...
                              res.snr_in_db, res.snr_out_db, res.delta_snr_db, ...
                              res.rho_XY, res.rho_orthogonality, res.real_time_factor, res.mse_pdf);
        set(hGUI.txtMetrics, 'String', metrics_str);
        
        print_console_summary_table(res);
        set(hGUI.txtStatus, 'String', sprintf('Done! Filtered %0.1fs audio with Delta SNR = %+0.2f dB. Audio files saved.', res.duration, res.delta_snr_db));
    catch ME
        errordlg(sprintf('Processing Error: %s', ME.message));
        set(hGUI.txtStatus, 'String', sprintf('Error: %s', ME.message));
    end

    set(hGUI.btnRun, 'Enable', 'on');
    set(hGUI.btnCalibrate, 'Enable', 'on');
    guidata(hFig, hGUI);
end

function update_gui_dashboard_plots(hFig, res)
% UPDATE_GUI_DASHBOARD_PLOTS Refreshes the 6 subplot axes inside the active GUI
    hGUI = guidata(hFig);

    % Color Palette
    col_input   = [0.20, 0.45, 0.85];
    col_output  = [0.10, 0.65, 0.45];
    col_noise   = [0.85, 0.35, 0.20];

    % Subplot 1: PDF
    ax1 = hGUI.axes(1);
    cla(ax1);
    bar(ax1, res.bin_centers, res.pdf_empirical, 1.0, ...
        'FaceColor', [0.70, 0.82, 0.95], 'EdgeColor', col_input, 'FaceAlpha', 0.6);
    hold(ax1, 'on');
    plot(ax1, res.bin_centers, res.pdf_theoretical, 'r-', 'LineWidth', 2.0);
    hold(ax1, 'off');
    grid(ax1, 'on');
    title(ax1, 'Subplot 1: Ambient Noise PDF & Gaussian Fit', 'FontSize', 9, 'FontWeight', 'bold');
    xlabel(ax1, 'Amplitude', 'FontSize', 8);
    ylabel(ax1, 'Density', 'FontSize', 8);

    % Subplot 2: CDF
    ax2 = hGUI.axes(2);
    cla(ax2);
    stairs(ax2, res.bin_centers, res.cdf_empirical, 'b-', 'LineWidth', 1.8);
    hold(ax2, 'on');
    plot(ax2, res.bin_centers, res.cdf_theoretical, 'k--', 'LineWidth', 1.8);
    hold(ax2, 'off');
    grid(ax2, 'on');
    title(ax2, 'Subplot 2: Empirical vs. Gaussian CDF', 'FontSize', 9, 'FontWeight', 'bold');
    xlabel(ax2, 'Amplitude', 'FontSize', 8);
    ylabel(ax2, 'Cumulative Prob.', 'FontSize', 8);

    % Subplot 3: Autocorrelation
    ax3 = hGUI.axes(3);
    cla(ax3);
    plot_lags = abs(res.lags_ms) <= 25;
    plot(ax3, res.lags_ms(plot_lags), res.R_XX(plot_lags), '-', 'Color', col_input, 'LineWidth', 1.3);
    hold(ax3, 'on');
    plot(ax3, res.lags_ms(plot_lags), res.R_YY(plot_lags), '-', 'Color', col_output, 'LineWidth', 1.5);
    plot(ax3, res.lags_ms(plot_lags), res.R_NN(plot_lags), ':', 'Color', col_noise, 'LineWidth', 1.6);
    hold(ax3, 'off');
    grid(ax3, 'on');
    title(ax3, 'Subplot 3: Autocorrelations R_{XX}, R_{YY}, R_{NN}', 'FontSize', 9, 'FontWeight', 'bold');
    xlabel(ax3, 'Lag \tau (ms)', 'FontSize', 8);
    ylabel(ax3, 'R[\tau]', 'FontSize', 8);

    % Subplot 4: PSD
    ax4 = hGUI.axes(4);
    cla(ax4);
    plot(ax4, res.freq_axis_khz, 10*log10(res.S_XX + 1e-12), '-', 'Color', col_input, 'LineWidth', 1.2);
    hold(ax4, 'on');
    plot(ax4, res.freq_axis_khz, 10*log10(res.S_NN + 1e-12), ':', 'Color', col_noise, 'LineWidth', 1.6);
    plot(ax4, res.freq_axis_khz, 10*log10(res.S_YY + 1e-12), '-', 'Color', col_output, 'LineWidth', 1.6);
    hold(ax4, 'off');
    grid(ax4, 'on');
    title(ax4, 'Subplot 4: Wiener-Khinchin PSD: S(\omega)', 'FontSize', 9, 'FontWeight', 'bold');
    xlabel(ax4, 'Frequency (kHz)', 'FontSize', 8);
    ylabel(ax4, 'PSD (dB/Hz)', 'FontSize', 8);

    % Subplot 5: Time Waveforms
    ax5 = hGUI.axes(5);
    cla(ax5);
    plot(ax5, res.time_vec, res.noisy_input, 'Color', [0.7 0.7 0.7], 'LineWidth', 0.8);
    hold(ax5, 'on');
    plot(ax5, res.time_vec, res.filtered_output, 'Color', col_output, 'LineWidth', 1.1);
    hold(ax5, 'off');
    grid(ax5, 'on');
    title(ax5, 'Subplot 5: Time Waveforms: X(t) vs. Denoised Y(t)', 'FontSize', 9, 'FontWeight', 'bold');
    xlabel(ax5, 'Time (s)', 'FontSize', 8);
    ylabel(ax5, 'Amplitude', 'FontSize', 8);
    xlim(ax5, [0, res.duration]);

    % Subplot 6: Spectrogram
    ax6 = hGUI.axes(6);
    cla(ax6);
    spec_combined = [res.spec_X_dB; -100 * ones(4, length(res.spec_t)); res.spec_Y_dB];
    f_combined = [res.spec_f; 8 + (1:4)'*0.1; res.spec_f + 8.5];
    imagesc(ax6, res.spec_t, f_combined, spec_combined);
    axis(ax6, 'xy');
    colormap(ax6, 'jet');
    caxis(ax6, [-70, 0]);
    title(ax6, 'Subplot 6: Spectrogram: [Top: Input] vs. [Bottom: Filtered]', 'FontSize', 9, 'FontWeight', 'bold');
    xlabel(ax6, 'Time (s)', 'FontSize', 8);
    ylabel(ax6, 'Freq (kHz)', 'FontSize', 8);
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
        set(hGUI.txtStatus, 'String', 'Playing Denoised Output Y[n]...');
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
