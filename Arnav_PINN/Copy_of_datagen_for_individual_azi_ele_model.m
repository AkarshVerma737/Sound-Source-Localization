% FOR DATA GENERATION
clc;
clearvars;
mex -setup

addpath('D:\Projects\Learning based Beamforging\SH\Arnav_PINN\SMIR-Generator-master');
addpath('D:\Projects\Learning based Beamforging\SH\Arnav_PINN\matlab');

% Load clean audio files
startpath = 'librispeech_aud1';
filelist = dir('librispeech_aud1/*.flac');	

for loop = 1:10
    % Randomly select a file
    filelist_i = randi(length(filelist));
    f_name = filelist(filelist_i).name;
    f_path = fullfile(startpath, f_name);
    [sig_temp, f] = audioread(f_path);
     
    % Remove unvoiced parts
    envelope = imdilate(abs(sig_temp), true(1501, 1));
    quietParts = envelope < 0.13;
    s_temp = sig_temp;
    s_temp(quietParts) = [];

    % Clip signal to length 10000
    s_temp = s_temp(1:min(10000, length(s_temp)));

    % Parameters for spherical microphone array and simulation
    N = 5;
    sphRadius = 0.042;
    procFs = 16000;
    fs = procFs;
    c = 343;
    nsample = 256;
    N_harm = 40;
    K = 2;
    order = -1;
    sphType = 'rigid';

    % Load microphone positions
    load('em32micpos.mat');
    [M,~] = size(mic);
    mic = mic * pi/180;
    mic = mic(:, [2,1]);

    % Compile C++ function
    mex smir_generator_loop.cpp;

    % Fixed elevation
    theta_src = 45;
    theta_iter = theta_src * pi/180;

    % Azimuth sweep
    for phi_src = 0:2:90
        phi_iter = phi_src * pi/180;

        beta = 0.2 + rand() * (0.8 - 0.2);
        L = [5 4 6];
        sphLocation = [1.2, 2.3, 2];
        r_src = 1;

        % Source position
        pos_src = [
            r_src * sin(theta_iter) * cos(phi_iter),
            r_src * sin(theta_iter) * sin(phi_iter),
            r_src * cos(theta_iter)
        ] + sphLocation;

        % Generate RIR
        [h_tmp, ~, ~] = smir_generator(c, procFs, sphLocation, pos_src, L, beta, sphType, sphRadius, mic, N_harm, nsample, K, order);
        h1(1,1,:,:) = h_tmp;

        % Calculate spherical harmonics matrix Y_m
        N1 = 3;
        Y_m = zeros(M, (N1 + 1)^2);
        for j = 1:M
            i = 1;
            for n = 0:N1
                for m = -n:n
                    Y_m(j,i) = spherical_harmonics(n, m, mic(j,2), mic(j,1));
                    i = i + 1;
                end
            end
        end

        % Source audio convolved with RIRs
        for ind = 1:32
            audio_smir(ind,:) = conv2(h_tmp(:,ind), s_temp);
        end

        % Add noise
        snr = 5 + randi(15); % 5 to 20 dB
        audio_sig = awgn(audio_smir, snr, 'measured');

        % STFT parameters
        windowlength = 512;
        hop = 256;
        nfft = 512;
        win_name = 'hann';
        fs = 16000;

        % STFT
        for indstft = 1:32
            [tm, fq, Si(indstft,:,:) ] = stftt(audio_sig(indstft,:), windowlength, hop, nfft, fs, win_name);
        end
        % Mode Strength Matrix
        r_a = 0.042;
        r_s = 2;
        faxis = fs/2 * linspace(0, 1, nfft/2 + 1);
        nfreq = length(faxis);
        B = ModeStrengthMatrixAllFreq(N, faxis, r_a, r_s);

        count = 1;
        for Bind = 17:125        
            Bdiag = squeeze(B(Bind,:,:));
            Sifreq = squeeze(Si(:,Bind,:));
            invB = pinv(Bdiag);
            invB = invB(1:32,1:32);
            SH_B(:,count,:) = Y_m' * invB * Sifreq;
            SH_B_mag(:,count,:) = abs(SH_B(:,count,:));
            SH_B_ph(:,count,:) = angle(SH_B(:,count,:));
            count = count + 1;
        end

        % Stack features
        for l21 = 1:38
            %inputfeats_mag(:,:,:,l21) = SH_B_mag(:,:,l21);
            inputfeats_ph(:,:,:,l21) = SH_B_ph(:,:,l21);
        end
            
        % Save feature files
        %save(sprintf('train_datagen_individual_azi_ele_model/magnitude/elevation_feats_azi%dele%d_%d.mat', phi_src, theta_src, loop), 'inputfeats_mag'); 
        save(sprintf('D:/Projects/Learning based Beamforging/SH/Generated Audio/Octant/azimuth_feats_azi%dele%d_%d.mat', phi_src, theta_src, loop), 'inputfeats_ph'); 
    end
end