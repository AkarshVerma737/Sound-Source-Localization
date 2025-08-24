clc;
clear all;
close all;

N = 6;
procFs = 16000;                      % Sampling frequency (Hz)
c = 343;                            % Sound velocity (m/s)
nsample = 256;                      % Length of desired RIR
N_harm = 40;                        % Maximum order of harmonics to use in SHD
K = 2;                              % Oversampling factor
order = -1;                         % Reflection order (-1 is maximum reflection order)
sphRadius = 0.042;                  % Radius of the sphere (m)
sphType = 'rigid';                  % Type of sphere (open/rigid)

L = [3 4 5];                        % Room dimensions (x,y,z) in m
sphLocation = [1.5 2 2.5];          % Receiver location (x,y,z) in m
beta = rand()*(0.9);
r_src = 1;
[sig, freq]= audioread('/home/vishnuv/Desktop/priya/clean/1.flac');
fs = 16000;
load('em32micpos.mat');
[M,~] = size(mic);
mic = mic*pi/180;
mic = mic(:,[2,1]);

phi_src = 0:45:345;
theta_src = 0:20:160;
h = zeros(length(phi_src),length(theta_src),M,nsample);

mex smir_generator_loop.cpp
count = 1;
for z = 1:length(phi_src)
    for w = 1:length(theta_src)
        theta_iter = theta_src(w);
	    theta_iter = theta_iter * pi/180;	

	    phi_iter = phi_src(z);
	    phi_iter = phi_iter*pi/180;
        
        pos_src(1) = r_src * sin(theta_iter) * cos(phi_iter);
        pos_src(2) = r_src * sin(theta_iter) * sin(phi_iter);
        pos_src(3) = r_src * cos(theta_iter);
        pos_src = pos_src + sphLocation;
        
        [h_tmp, ~, ~] = smir_generator(c, procFs, sphLocation, pos_src, L, beta, sphType, sphRadius, mic, N_harm, nsample, K, order);
        h(z,w,:,:) = h_tmp;         
%         save(sprintf('/media/adityar/FFDOA-DANN/data/SMIR_train_phi_rand_%d.mat',numfile),'h');
        save(sprintf('/home/vishnuv/Desktop/priya/smir_mat_files/angles_%d_%d.mat',phi_src(z),theta_src(w)),'h_tmp');
%         save(sprintf('/home/vishnuv/Desktop/priya/smir_mat_files/mat2/ang_%d.mat',count),'h_tmp'); 
        count = count+1;
    end 
end
% for indd = 1:32
% %     temp1(indd,:) = conv(squeeze(h(8,9,indd,:)),sig);
%     temp2(indd,:) = transpose(conv(squeeze(h(8,9,indd,:)),sig));
% end
% tempvar = conv2(h_tmp',sig);
