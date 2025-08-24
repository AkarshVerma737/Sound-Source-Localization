clc
clear all
close all

% shuffle to avoid repetation
% 1 - training
% 2 - validation
% 3 - testing

diffssnnoise = 1;
voiced = 0;

rng(2);

% load SMIR data into h
load('/media/adityar/FFDOA-DANN/data/SMIR_validate_theta_rand.mat')
[ntheta_src,nphi_src,M,nsample] = size(h);
niterWithSpeech = 2;
nframes = 2;

% order of spherical microphone array
N = 4;
nfft = 256;
fselect = 1:(nfft/2 + 1);
SNRmin = 0;
SNRmax = 20;
% calculate the total number of rows for stft
rown = ceil((1+nfft)/2);

% load microphone positions
load('em32micpos.mat');
mic = mic*pi/180;
mic = mic(:,[2,1]);

% settings for SMIR generator
procFs = 16000;
fs = procFs;
signallength = uint32(0.016*fs);
signallimit = fs + (nframes + 5) * signallength;
wind = hann(signallength)';

% prepare Y_m
Y_m=zeros(M,(N+1)^2);
for j=1:M
    i=1;
    for n=0:N,
        for m = -n:1:n,
            Y_m(j,i)=spherical_harmonics(n,m,mic(j,2),mic(j,1));
            i=i+1;
        end
    end
end

% change here for increasing grid size
features = zeros(M,rown,nframes,niterWithSpeech,nphi_src,ntheta_src);

% changed
    for j=1:ntheta_src
	for k=1:nphi_src
            for l=1:niterWithSpeech
                j
                ntheta_src
                k
                nphi_src
                l
                niterWithSpeech
                test = '--------------------'
                snr = SNRmin + randi(SNRmax-SNRmin);
                conv_length = nsample + signallimit -1;
                recs = zeros(M, conv_length);

                % selection of file for speaker
		if(voiced == 1)
                   startpath = '/media/adityar/FFDOASpherical/voicetrain';
                   filelist = dir('/media/adityar/FFDOASpherical/voicetrain/*.flac');
		else
		   startpath = '/media/adityar/FFDOASpherical/LibriSpeech';
                   filelist = dir('/media/adityar/FFDOASpherical/LibriSpeech/*.flac');	
		end
                while(1)
                    filelist_i = randi(length(filelist));
                    f_name = filelist(filelist_i).name;
                    f_path = strcat(startpath,'/',f_name);
                    s_temp = audioread(f_path);
                    if(length(s_temp) > signallimit)
                        break;
                    end
                end
                sample_start = randi(length(s_temp) - signallimit);
                s_signal = s_temp(sample_start:(sample_start+signallimit-1));

		 % selection of noise file for recording
                if(diffssnnoise == 1)
                    noisedir = '/media/adityar/FFDOASpherical/diffssn17_em32'
                    d=dir([noisedir '/*.mat']);
                    f={d.name};
                    idx=randi(numel(f));
                    filename=[noisedir '/' f{idx}];
                    % loaded into parameter z
                    load(filename);
                    noise = z;
                    % channels are with columns
                    start = randi(size(noise,2) - conv_length);
                    noise = noise(:,start:start+conv_length-1);
                else
                    % go with just white noise
                    noise = randn(M,conv_length);
                end

                % prepare recording
                for indd = 1:M
		    % change here	
                    recs(indd,:) = transpose(conv(squeeze(h(j,k,indd,:)),s_signal));
                    sigpower = bandpower(recs(indd,:));
                    noisepower = sigpower/(10^(snr/10));
		    orig_noisepower = bandpower(noise(indd,:));
                    noise(indd,:) = sqrt(noisepower/orig_noisepower) * noise(indd,:);
                    % add noise to the recording as per SNR
                    recs(indd,:) = recs(indd,:) + noise(indd,:);	
                end


                % feature preparation from recording
                stft_mic = zeros(M,rown);
		for frame_i = 1:nframes
		    start = nsample + randi(conv_length-signallength-nsample);
                    for indd = 1:M
                    	tmp = fft(recs(indd,start:start+signallength-1).*wind,nfft);
                    	stft_mic(indd,:) = tmp(fselect);
		    end
		    features(:,:,frame_i,l,k,j) = stft_mic; 	
                end
            end
	end
    end

% save features into a mat file
save('/media/adityar/FFDOA-DANN/data/recording_validate_theta_rand.mat','features','-v7.3');
