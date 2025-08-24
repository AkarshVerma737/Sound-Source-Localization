clc
clear all
close all

% settings for different features
shmmagphaseshres = 1;

% frequency range interested
%freq_range_int = 17:167;
freq_range_int = 9:63;

% load microphone positions
load('em32micpos.mat');
mic = mic*pi/180;
mic = mic(:,[2,1]);
[M,~,~] = size(mic);
N = 3;
N_foa = 1;
%N = 4;

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

% need B inverse for spherical harmonic frequency smoothing
r_a = 0.042;
r_s = 2;
nfft = 256;
fs = 16000;
faxis = fs/2*linspace(0,1,nfft/2+1);
nfreq = length(faxis);
B = ModeStrengthMatrixAllFreq(N,faxis,r_a,r_s);

% we are in need of B inverse and not B
Binv = zeros(length(faxis),(N+1)^2,(N+1)^2);
for i=2:nfreq
    Binv(i,:,:) = pinv(squeeze(B(i,:,:)));
end

% load the complete feature set first : loads into features
load('/media/adityar/FFDOA-DANN/data/recording_validate_phi_rand.mat')
[M,rown,nframes,niterWithSpeech,ntheta_src,nphi_src] = size(features);
orig_features = features;
orig_features = orig_features./abs(orig_features);
features = [];

if(shmmagphaseshres == 1)
    % first prepare phasemaps features ready
    shmmagphaseshres_features = zeros(3*((N+1)^2 - 1),length(freq_range_int),nframes*niterWithSpeech*nphi_src*ntheta_src);
    saveshmmagphaseshres = '/media/adityar/FFDOA-DANN/data/shmmagphaseshres_validate_phi_rand.mat';
end

theta_labels = zeros(ntheta_src,nframes*niterWithSpeech*nphi_src*ntheta_src);
phi_labels = zeros(nphi_src,nframes*niterWithSpeech*nphi_src*ntheta_src);

f_index = 1;
for i=1:nphi_src
    i
    for j=1:ntheta_src
            for l=1:niterWithSpeech
	        for m=1:nframes
		    if(shmmagphaseshres == 1)
                       P_nm = Y_m' * squeeze(orig_features(:,:,m,l,j,i));
                       Q_nm = zeros((N+1)^2,nfreq);
                       for freq_index = freq_range_int(1):freq_range_int(end)
                           Q_nm(:,freq_index) = squeeze(Binv(freq_index,:,:)) * P_nm(:,freq_index);
                       end
                       % Q_nm = Q_nm(:,2:end);
		       Q_nm = Q_nm(:,freq_range_int);

                       tmp2 = bsxfun(@rdivide,Q_nm,Q_nm(1,:));
                       tmp2 = tmp2(2:end,:);
                       tmp3 = abs(tmp2);
                       norm_row = sqrt(sum(tmp3.^2));
                       tmp3 = bsxfun(@rdivide,tmp3,norm_row);
                       shmmagphaseshres_features(1:2:2*((N+1)^2 -1),:,f_index) = cos(angle(tmp2));
                       shmmagphaseshres_features(2:2:2*((N+1)^2 -1),:,f_index) = sin(angle(tmp2));
                       shmmagphaseshres_features(2*((N+1)^2 -1)+1:end,:,f_index) = tmp3;
                    end

		    phi_labels(i,f_index) = 1;
		    theta_labels(j,f_index) = 1; 	
            	    f_index = f_index + 1;
		end
            end
    end
end

% save features
if(shmmagphaseshres == 1)
     features = shmmagphaseshres_features;
     save(saveshmmagphaseshres,'features','phi_labels','theta_labels','-v7.3');
end
