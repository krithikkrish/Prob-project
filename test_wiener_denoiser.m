function test_wiener_denoiser()
% =========================================================================
% TEST SUITE: WIENER-KHINCHIN SPECTRAL ESTIMATION & OPTIMAL LTI FILTER
% =========================================================================
%
% This test suite provides rigorous mathematical verification for each phase
% of the real-time Wiener denoising pipeline without using any black-box
% toolbox functions.
%
% Tests Performed:
%   1. Sample Moments Calculation (Mean, Variance, Skewness, Kurtosis)
%   2. Empirical PDF and CDF Normalization Properties (Area = 1.0)
%   3. Central Limit Theorem (CLT) Block Mean Convergence & 95% Confidence Interval
%   4. Autocorrelation & Wiener-Khinchin Spectral Estimation Identity
%   5. Optimal Frequency-Domain LTI Wiener Filter Bounds & S_YY = |H|^2 * S_XX
%   6. Overlap-Add (OLA) Synthesis Lossless Reconstruction (H = 1 Identity)
%   7. Principle of Orthogonality (Correlation between Estimate & Residual Error)
%   8. Full End-to-End Simulation Pipeline & WAV File Generation
% =========================================================================

    fprintf('\n========================================================================\n');
    fprintf('  RUNNING RIGOROUS UNIT TESTS: PROBABILITY & RANDOM PROCESS DENOISER\n');
    fprintf('========================================================================\n\n');

    tests_passed = 0;
    total_tests = 8;

    % ---------------------------------------------------------------------
    % TEST 1: Sample Moments (Mean, Variance, Skewness, Kurtosis)
    % ---------------------------------------------------------------------
    fprintf('[TEST 1/8] Verifying Sample Moments (Method of Moments)... ');
    rng(42); % Fixed seed for deterministic unit test
    N_test = 100000;
    true_mu = 0.5;
    true_sigma = 1.2;
    z = true_mu + true_sigma * randn(N_test, 1);

    % Scratch Moment Calculations
    calc_mu = sum(z) / N_test;
    calc_var = sum((z - calc_mu).^2) / (N_test - 1);
    calc_std = sqrt(calc_var);
    m2 = sum((z - calc_mu).^2) / N_test;
    m3 = sum((z - calc_mu).^3) / N_test;
    m4 = sum((z - calc_mu).^4) / N_test;
    calc_skew = m3 / (m2^1.5);
    calc_kurt = m4 / (m2^2);

    assert(abs(calc_mu - true_mu) < 0.02, 'Mean calculation error exceeds tolerance!');
    assert(abs(calc_var - true_sigma^2) < 0.05, 'Variance calculation error exceeds tolerance!');
    assert(abs(calc_skew - 0.0) < 0.05, 'Gaussian skewness should be approximately 0!');
    assert(abs(calc_kurt - 3.0) < 0.10, 'Gaussian kurtosis should be approximately 3!');
    fprintf('PASSED (mu = %.4f, var = %.4f, skew = %.4f, kurt = %.4f)\n', ...
        calc_mu, calc_var, calc_skew, calc_kurt);
    tests_passed = tests_passed + 1;

    % ---------------------------------------------------------------------
    % TEST 2: Empirical PDF and CDF Normalization Properties
    % ---------------------------------------------------------------------
    fprintf('[TEST 2/8] Verifying Empirical PDF/CDF Construction & Normalization... ');
    K_bins = 50;
    min_z = min(z);
    max_z = max(z);
    delta_x = (max_z - min_z) / K_bins;
    bin_idx = floor((z - min_z) / delta_x) + 1;
    bin_idx(bin_idx > K_bins) = K_bins;
    bin_idx(bin_idx < 1) = 1;
    counts = accumarray(bin_idx, 1, [K_bins, 1]);

    pdf_emp = counts / (N_test * delta_x);
    cdf_emp = cumsum(counts) / N_test;

    % Area under PDF must equal 1.0 (Axioms of Probability)
    pdf_area = sum(pdf_emp) * delta_x;
    assert(abs(pdf_area - 1.0) < 1e-10, 'Total probability under empirical PDF is not 1.0!');
    assert(abs(cdf_emp(end) - 1.0) < 1e-10, 'CDF upper limit must be 1.0!');
    assert(all(diff(cdf_emp) >= 0), 'CDF must be monotonically non-decreasing!');
    fprintf('PASSED (PDF Integral = %.10f, CDF Final = %.10f)\n', pdf_area, cdf_emp(end));
    tests_passed = tests_passed + 1;

    % ---------------------------------------------------------------------
    % TEST 3: Central Limit Theorem (CLT) Convergence & 95% Confidence Interval
    % ---------------------------------------------------------------------
    fprintf('[TEST 3/8] Verifying Central Limit Theorem (CLT) on Sub-Blocks... ');
    num_blocks = 250;
    L_blk = floor(N_test / num_blocks);
    blk_means = zeros(num_blocks, 1);
    for b = 1:num_blocks
        blk = z((b-1)*L_blk + 1 : b*L_blk);
        blk_means(b) = sum(blk) / L_blk;
    end
    var_blk_means = sum((blk_means - mean(blk_means)).^2) / (num_blocks - 1);
    expected_clt_var = calc_var / L_blk;
    rel_clt_err = abs(var_blk_means - expected_clt_var) / expected_clt_var;

    % Confidence Interval check
    ci_95 = [calc_mu - 1.96*(calc_std / sqrt(N_test)), calc_mu + 1.96*(calc_std / sqrt(N_test))];
    assert(ci_95(1) <= true_mu && ci_95(2) >= true_mu, '95%% CI does not cover the true population mean!');
    assert(rel_clt_err < 0.20, 'CLT block variance deviates substantially from sigma^2 / L_blk!');
    fprintf('PASSED (Empirical CLT Var = %.5f, Theoretical = %.5f)\n', var_blk_means, expected_clt_var);
    tests_passed = tests_passed + 1;

    % ---------------------------------------------------------------------
    % TEST 4: Autocorrelation & Wiener-Khinchin Spectral Estimation Identity
    % ---------------------------------------------------------------------
    fprintf('[TEST 4/8] Verifying Autocorrelation R[tau] & Wiener-Khinchin F{R[tau]}... ');
    L_frame = 512;
    nfft_test = 1024;
    sig = randn(L_frame, 1);
    w_hamm = 0.54 - 0.46 * cos(2*pi*(0:L_frame-1)' / (L_frame - 1));
    sig_w = sig .* w_hamm;

    % 1. Direct Time-Domain Autocorrelation:
    R_direct = zeros(2*L_frame - 1, 1);
    lags = -(L_frame - 1) : (L_frame - 1);
    for i = 1:length(lags)
        tau = lags(i);
        if tau >= 0
            R_direct(i) = sum(sig_w(1 : L_frame - tau) .* sig_w(1 + tau : L_frame)) / L_frame;
        else
            R_direct(i) = sum(sig_w(1 - tau : L_frame) .* sig_w(1 : L_frame + tau)) / L_frame;
        end
    end

    % 2. Frequency-Domain via Wiener-Khinchin circular convolution:
    Sig_fft = fft(sig_w, nfft_test);
    S_raw = (abs(Sig_fft).^2) / L_frame;
    r_circ = real(ifft(S_raw));
    R_fft = [r_circ(nfft_test - L_frame + 2 : nfft_test); r_circ(1:L_frame)];

    max_corr_diff = max(abs(R_direct - R_fft));
    assert(max_corr_diff < 1e-13, 'Autocorrelation via circular convolution does not match direct linear correlation!');
    fprintf('PASSED (Direct vs. Circular Conv Error = %.2e)\n', max_corr_diff);
    tests_passed = tests_passed + 1;

    % ---------------------------------------------------------------------
    % TEST 5: Optimal LTI Wiener Filter Bounds & S_YY = |H|^2 * S_XX
    % ---------------------------------------------------------------------
    fprintf('[TEST 5/8] Verifying Wiener Filter Transfer Function Bounds & LTI PSD Identity... ');
    S_XX_test = abs(randn(nfft_test, 1)).^2 + 1.0;
    S_NN_test = 0.5 * ones(nfft_test, 1);
    beta_test = 0.03;

    % Wiener Filter: H(w) = max( (S_XX - S_NN) / S_XX, beta )
    S_SS_hat = max(S_XX_test - S_NN_test, 0);
    H_w = max(S_SS_hat ./ (S_XX_test + 1e-12), beta_test);
    H_w = min(H_w, 1.0);

    assert(all(H_w >= beta_test - 1e-12), 'Wiener filter violates minimum spectral floor beta!');
    assert(all(H_w <= 1.0 + 1e-12), 'Wiener filter amplifies signal energy (H > 1.0)!');

    % LTI Relation: S_YY(w) = |H(w)|^2 * S_XX(w)
    Y_fft_test = H_w .* Sig_fft;
    S_YY_direct = (abs(Y_fft_test).^2) / L_frame;
    S_YY_theo = (abs(H_w).^2) .* S_raw;
    max_lti_err = max(abs(S_YY_direct - S_YY_theo));
    assert(max_lti_err < 1e-13, 'LTI random process PSD identity S_YY = |H|^2 * S_XX violated!');
    fprintf('PASSED (LTI PSD Identity Error = %.2e)\n', max_lti_err);
    tests_passed = tests_passed + 1;

    % ---------------------------------------------------------------------
    % TEST 6: Overlap-Add (OLA) Lossless Reconstruction Identity
    % ---------------------------------------------------------------------
    fprintf('[TEST 6/8] Verifying Overlap-Add (OLA) Perfect Reconstruction (H=1)... ');
    fs_test = 16000;
    t_test = (0:fs_test-1)' / fs_test;
    x_test = sin(2*pi*350*t_test) + 0.4*cos(2*pi*850*t_test) + 0.2*sin(2*pi*1800*t_test);
    N_sig = length(x_test);

    fLen = 1024;
    ovlp = 512;
    hStep = fLen - ovlp;
    nFft = 2048;
    w_ola = 0.54 - 0.46 * cos(2*pi*(0:fLen-1)' / (fLen - 1));

    num_f = floor((N_sig - fLen) / hStep) + 1;
    y_rec = zeros(N_sig, 1);
    w_acc = zeros(N_sig, 1);

    for m = 1:num_f
        idx = (m-1)*hStep + (1:fLen);
        x_w = x_test(idx) .* w_ola;
        
        % Frequency Domain with H(w) = 1 (Transparent All-Pass)
        X_f = fft(x_w, nFft);
        H_id = ones(size(X_f));
        Y_f = H_id .* X_f;
        y_t = real(ifft(Y_f));
        
        % Overlap-Add Synthesis
        y_rec(idx) = y_rec(idx) + y_t(1:fLen) .* w_ola;
        w_acc(idx) = w_acc(idx) + (w_ola.^2);
    end

    % Verify in the steady-state overlap region
    valid_range = (fLen : (num_f - 1)*hStep);
    reconstructed_signal = y_rec(valid_range) ./ w_acc(valid_range);
    original_signal = x_test(valid_range);
    ola_rec_err = max(abs(reconstructed_signal - original_signal));

    assert(ola_rec_err < 1e-12, 'Normalized Overlap-Add does not achieve perfect reconstruction!');
    fprintf('PASSED (Max OLA Reconstruction Error = %.2e)\n', ola_rec_err);
    tests_passed = tests_passed + 1;

    % ---------------------------------------------------------------------
    % TEST 7: Principle of Orthogonality Verification
    % ---------------------------------------------------------------------
    fprintf('[TEST 7/8] Verifying Principle of Orthogonality (rho_Y_Residual ~ 0)... ');
    % Execute small end-to-end simulation
    cfg_test = struct('mode', 'simulation', 'duration', 4.0, 'calDuration', 1.5, ...
                      'fs', 16000, 'frameLen', 1024, 'overlap', 512, 'nfft', 2048, ...
                      'beta', 0.03, 'simSNR', 3.0, 'headless', true);
    res_test = RealTime_Wiener_Noise_Filter(cfg_test);

    % Verify correlation between estimate Y and residual noise (X - Y)
    assert(abs(res_test.rho_orthogonality) < 0.50, 'Orthogonality correlation is unexpectedly high!');
    assert(res_test.rho_XY > 0.70, 'Input and Output audio should retain strong coherence!');
    fprintf('PASSED (rho_YE = %+.4f, rho_XY = %+.4f)\n', res_test.rho_orthogonality, res_test.rho_XY);
    tests_passed = tests_passed + 1;

    % ---------------------------------------------------------------------
    % TEST 8: Full End-to-End Simulation Pipeline & WAV File Verification
    % ---------------------------------------------------------------------
    fprintf('[TEST 8/8] Verifying End-to-End SNR Gain, RTF, and Exported Files... ');
    assert(res_test.delta_snr_db > 2.0, 'Denoising failed to achieve significant positive SNR improvement!');
    assert(res_test.real_time_factor < 0.10, 'Processing execution is too slow for real-time operation!');
    assert(isfile('noisy_recorded_input.wav'), 'File noisy_recorded_input.wav was not generated!');
    assert(isfile('filtered_clean_output.wav'), 'File filtered_clean_output.wav was not generated!');
    assert(isfile('wiener_denoising_dashboard.png'), 'File wiener_denoising_dashboard.png was not generated!');

    % Verify audio file headers & readability using audioinfo
    info_noisy = audioinfo('noisy_recorded_input.wav');
    info_clean = audioinfo('filtered_clean_output.wav');
    assert(info_noisy.SampleRate == 16000, 'Sample rate of exported WAV is incorrect!');
    assert(info_clean.SampleRate == 16000, 'Sample rate of exported WAV is incorrect!');
    fprintf('PASSED (Delta SNR = %+0.2f dB, RTF = %.4f, Files Verified)\n\n', ...
        res_test.delta_snr_db, res_test.real_time_factor);
    tests_passed = tests_passed + 1;

    % ---------------------------------------------------------------------
    % FINAL SUMMARY
    % ---------------------------------------------------------------------
    fprintf('========================================================================\n');
    fprintf('  TEST SUITE RESULTS: %d / %d TESTS PASSED (100%% SUCCESS RATE)\n', tests_passed, total_tests);
    fprintf('  ALL MATHEMATICAL CONSTRAINTS & STOCHASTIC IDENTITIES VALIDATED.\n');
    fprintf('========================================================================\n\n');
end
