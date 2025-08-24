clc
clear all
close all

% shuffle to avoid repetation
% 1 - training
% 2 - validation
% 3 - testing
rng(2);

% order of spherical microphone array
N = 6;
sphRadius = 0.042;
% settings for SMIR generator
procFs = 16000;
fs = procFs;
% Sampling frequency (Hz)
c = 343;                            % Sound velocity (m/s)
nsample = 256;                     % Length of desired RIR
N_harm = 40;                        % Maximum order of harmonics to use in SHD
K = 2;                              % Oversampling factor
order = -1;                         % Reflection order (-1 is maximum reflection order)
sphType = 'rigid';                  % Type of sphere (open/rigid)

% load microphone positions
load('em32micpos.mat');
[M,~] = size(mic);
mic = mic*pi/180;
mic = mic(:,[2,1]);
%disp(M);

% locations
phi_src = 0:2:358;
theta_src = 30:10:150;
beta_min = 0.250;
beta_max = 1;
h = zeros(length(phi_src),length(theta_src),M,nsample);

for i=1:length(phi_src)
    for j=1:length(theta_src)
            i
            length(phi_src)
            j
            length(theta_src)
            test = '---------------------'

	    theta_iter = theta_src(j) + 5 - 10 * rand;
	    theta_iter = theta_iter * pi/180;	

	    phi_iter = phi_src(i) + 1 - 2 * rand;
	    phi_iter = phi_iter*pi/180;	

	    beta = beta_min + rand() * (beta_max-beta_min);
            % beta = 0.25;
	 
            % get suitable center of sphere
            spherepos_ok = false;
            sourcepos_ok = false;
            while(~spherepos_ok || ~sourcepos_ok)
                L = [3 3 3] + 2*[rand rand 0];
                sphLocation = [rand()*L(1) rand()*L(2) rand()*L(3)];
                %sphLocation = L./2 + [2*rand-1 2*rand-1 2*rand-1];

		% select r_src
		lower = 1;
                upper = max(L);
                llower = log10(lower);
                lupper = log10(upper);
                r_src = 10 ^ genUniform(1,llower,lupper);
            
                % get source position
                pos_src(1) = r_src * sin(theta_iter) * cos(phi_iter);
                pos_src(2) = r_src * sin(theta_iter) * sin(phi_iter);
                pos_src(3) = r_src * cos(theta_iter);
                pos_src = sphLocation + pos_src;

		% translate now both source and microphone
                % translate center to this point
                %translate = [rand()*L(1) rand()*L(2) rand()*L(3)];
                %totranslate = sphLocation - translate;
                %pos_src = pos_src - totranslate;
                %sphLocation = translate;
            
                % verify requisites
                spherepos_ok = PosCheck(L,sphLocation);
                sourcepos_ok = PosCheck(L,pos_src);
            end
            [h_tmp, ~, ~] = smir_generator(c, procFs, sphLocation, pos_src, L, beta, sphType, sphRadius, mic, N_harm, nsample, K, order);
            h(i,j,:,:) = h_tmp;
        end    
end

save('/media/adityar/FFDOA-DANN/data/SMIR_validate_phi_rand.mat','h');
