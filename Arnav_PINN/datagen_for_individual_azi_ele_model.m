% FOR DATA GENERATION
clc;
clearvars;
mex -setup

% Loading the clean audio files
startpath = 'librispeech_aud1';
filelist = dir('librispeech_aud1/*.flac');	
for loop = 1:100
    % Files for desired sound
    filelist_i = randi(length(filelist));
    f_name = filelist(filelist_i).name;
    f_path = strcat(startpath,'/',f_name);
    [sig_temp,f] = audioread(f_path);
     
    % Removing unvoiced signals
    envelope = imdilate(abs(sig_temp), true(1501, 1));
    quietParts = envelope < 0.13;
    s_temp = sig_temp; % Initialize
    s_temp(quietParts) = [];
    % Clipping signals upto 10K length
    s_temp = s_temp(1:10000,1);

% order of spherical microphone array
    N = 3;
    sphRadius = 0.042;
% settings for SMIR generator
    procFs = 16000;
    fs = procFs;
% Sampling frequency (Hz)
    c = 343;                            % Sound velocity (m/s)
    nsample = 256;                      % Length of desired RIR
    N_harm = 40;                        % Maximum order of harmonics to use in SHD
    K = 2;                              % Oversampling factor
    order = -1;                         % Reflection order (-1 is maximum reflection order)
    sphType = 'rigid';                  % Type of sphere (open/rigid)
% load microphone positions
    load('em32micpos.mat');
    [M,~] = size(mic);
    mic = mic*pi/180;
    mic = mic(:,[2,1]);
    mex smir_generator_loop.cpp;
% DOA angles for training data generation
% phi_src1 is desired source and phi_src2 is the interfering source
 for phi_src = 1:10:360
  theta_src = 45
                beta_min = 0.2;
                beta_max = 0.8;
                theta_iter = theta_src * pi/180;
                phi_iter = phi_src*pi/180;
                beta = beta_min + rand() * (beta_max-beta_min);
                L = [5 4 6];
                sphLocation = [1.2,2.3,2];
                r_src = 1;
% get source position
                pos_src(1) = r_src * sin(theta_iter) * cos(phi_iter);
                pos_src(2) = r_src * sin(theta_iter) * sin(phi_iter);
                pos_src(3) = r_src * cos(theta_iter);
                pos_src = sphLocation + pos_src;
                [h_tmp, ~, ~] = smir_generator(c, procFs, sphLocation, pos_src, L, beta, sphType, sphRadius, mic, N_harm, nsample, K, order);
                h1(1,1,:,:) = h_tmp;                
                % load microphone positions
                N1 = 3;
                load('em32micpos.mat');
                mic = mic*pi/180;
                mic = mic(:,[2,1]);
                % prepare Y_m
                Y_m=zeros(M,(N1+1)^2);
                for j=1:M
                    i=1;
                    for n=0:N1
                        for m = -n:1:n
                            Y_m(j,i)=spherical_harmonics(n,m,mic(j,2),mic(j,1));
                            i=i+1;
                        end
                    end
                end

% need B inverse for spherical harmonic frequency smoothing
        r_a = 0.042;
        r_s = 2;
        nfft = 512;
        fs = 16000;
        faxis = fs/2*linspace(0,1,nfft/2+1);
        nfreq = length(faxis);
% Getting the source with the location
                for ind = 1:32
                    audio_smir(ind,:) = conv2(h_tmp(:,ind),s_temp);
                end
                SNRmin = 5;
                SNRmax = 20;
                snr = SNRmin + randi(SNRmax-SNRmin);
% Adding AWGN noise
                audio_sig = awgn(audio_smir,snr,'measured');
% STFT requirements
                windowlength = 512; % 32 ms window (32ms*16khz)
                hop = 256; % 50% overlapping (windowlength/2)
                nfft = 512; % DFT length
                fs = 16000; % sampling frequency
                win_name = 'hann';
                % STFT
                for indstft = 1:32
                    [tm, fq, Si(indstft,:,:)] = stftt(audio_sig(indstft,:),windowlength,hop,nfft,fs, win_name);
                end
        B = ModeStrengthMatrixAllFreq(N,faxis,r_a,r_s);
        count=1;
        fcount = 1;
        for Bind = 17:125        
            Bdiag(:,:) = B(Bind,:,:);
            Sifreq(:,:) = Si(:,Bind,:);
            invB = pinv(Bdiag);
            invB = invB(1:32,1:32);
            SH_B(:,count,:) = (Y_m')*invB*Sifreq;
            SH_B_mag(:,count,:) = abs(SH_B(:,count,:));
            SH_B_ph(:,count,:) = angle(SH_B(:,count,:));
            count = count+1;
        end
        % Stacking the desired features to give as input to neural network
        for l21 = 1:38
            %inputfeats_mag(:,:,:,l21) = SH_B_mag(:,:,l21); % maps to elevation information
            inputfeats_ph(:,:,:,l21) = SH_B_ph(:,:,l21); % maps to azimuth information
        end
        % Save the mat files of the features for DOA estimation
        %save(sprintf('train_datagen_individual_azi_ele_model/magnitude/elevation_feats_azi%dele%d_%d.mat',phi_src,theta_src,loop),'inputfeats_mag'); 
        save(sprintf('train_datagen_individual_azi_ele_model/phase/azimuth_feats_azi%dele%d_%d.mat',phi_src,theta_src,loop),'inputfeats_ph'); 
        end
    % end
end
