clc; clear; close all;
TIME_list=[];
% -------------------------------------------------------------------------------------------------------------------------------------------------------%
%Define the prefix and suffix of the file names
% data_prefix = 'subject001_frame01_slice';mask_prefix = 'subject001_frame01_slice';
% img_folder_path ='/home/vikas/Desktop/ICVS_FINAL/Data_Files/subject_1/frame_01/';
% mask_folder_path ='/home/vikas/Desktop/ICVS_FINAL/Mask_Files/mask_subject_1/frame_01/';

data_prefix = 'subject002_frame01_slice';mask_prefix = 'subject002_frame01_slice';
img_folder_path ='/home/vikas/Desktop/ICVS_FINAL/Data_Files/subject_2/frame_01/';
mask_folder_path ='/home/vikas/Desktop/ICVS_FINAL/Mask_Files/mask_subject_2/frame_01/';

data_suffix = '.png';mask_suffix = '_LV.png';

%% -------------------------------------------------------------------------------------------------------------------------------------------------------%
% Define the number of files
numFiles = 5;
% Create an empty cell array to store the file names of images and masks
data_fileNames = cell(1, numFiles);
mask_fileNames = cell(1, numFiles);
% Generate the file names using a for loop
for i = 1:numFiles
    data_fileNames{i} = [data_prefix, num2str(i, '%02d'), data_suffix];
    mask_fileNames{i} = [mask_prefix, num2str(i, '%02d'), mask_suffix];
end
% Create an empty cell array to store the images and masks
images = cell(1, numFiles);
masks = cell(1, numFiles);
for i = 1:numFiles
    full_image_path = strcat(img_folder_path, data_fileNames{i});
    full_mask_path = strcat(mask_folder_path, mask_fileNames{i});
   
    images{i} = imread(full_image_path);
    masks{i} = imread(full_mask_path);
end
% Create an empty cell array to store ROI of the images and masks 
ROI_images = cell(1, numFiles);
ROI_masks = cell(1, numFiles);
% Get [rx_roi, ry_roi]
imshow(images{1});
title('Select ROI','FontSize',50);
% Wait for the user to click on the image
[rx_roi, ry_roi] = ginput(1); pause(0.1);
close;
for i = 1:numFiles
    ROI_images{i} = imcrop(images{i}, [rx_roi-50 ry_roi-50 99 99]);
    ROI_masks{i} = imcrop(masks{i}, [rx_roi-50 ry_roi-50 99 99]);
end
% Create an empty cell array to store preprocessed ROI of the images
ROI_images_pp = cell(1, numFiles);
for i = 1:numFiles
    ROI_images_pp{i} = Preprocessing(ROI_images{i});
end
%% -------------------------------------------------------------------------------------------------------------------------------------------------------%
%% -------------------------------------------------------------------------------------------------------------------------------------------------------%
% User Inputs
[alpha_list, beta_list, gamma_list, delta_list, mu_list, Px, Py, nc, nl, max_iter, lr, init_rad]= user_inputs();
% Parametric Representation  
s = linspace(0, 1, nc+1); % contour parameter
% Some Matrices
x=linspace(1,Px,Px);y=linspace(Py,1,Py);
% Mesh Grid: Mat:1,2
[X,Y]=meshgrid(x,y); 
% Regularization Matrices: Mat:  3-7
M = periodic_block(s);M_inv = inv(M);
R = spline_RHS_block(s);
[B, dBds, d2Bds2] = interpolation_blocks(s,nl);
%% -------------------------------------------------------------------------------------------------------------------------------------------------------%
IoU_list = zeros(numFiles,1);
% Slice number-1
fprintf('SLICE-%d\n',1)
%% Weight Initialization
img=ROI_images_pp{1};
[Cx, Cy] = Get_Center(img,init_rad);Cy=Py-Cy;
[x0, y0] = create_init_circle(Cx, Cy, init_rad,nc);
w_init = [x0 y0];
%% Optimization
% Predictor Step
tic;
[w_pred, J_hist,Jr_hist,Js_hist,Jiv_hist,Jig_hist,break_iter] =...
Optimize_via_Adam(w_init,M_inv, R, B, dBds, d2Bds2,img,X,Y,alpha_list,beta_list,gamma_list,delta_list,mu_list,max_iter,lr);
elapsed_time = toc;
TIME_list=[TIME_list,elapsed_time];
disp(['Elapsed Time: ', num2str(elapsed_time), ' seconds']);
% Difference between prediction with and without shape prior 
% Ground Truth
GT=ROI_masks{1};
GT=im2gray(GT);
GT=GT./max(max(GT));
% Get masks with and without shape prior
nw=length(w_pred);
xf_pred=w_pred(1:0.5*nw);yf_pred=w_pred(1+0.5*nw:nw);
[coeff_x, coeff_y] = spline_coefficients(M_inv, R, xf_pred, yf_pred);
xsi_pred = B * coeff_x;ysi_pred = B * coeff_y;

% With shape prior
[in,on] = inpolygon (X, Y, xsi_pred, ysi_pred);
chi_pred = in & ~on;
    
wopt=w_pred;
save('wopt_1.mat', 'wopt');%<----------->%
% IoU score
IoU = jaccard( double(GT),double(chi_pred));
DIC = dice( double(GT),double(chi_pred));
IoU_list(1,1)=IoU;
DIC_list(1,1)=DIC;
IoU = sprintf('%0.3g', IoU);
%figure()
figure(11)
subplot(1,5,1)
imshowpair(chi_pred,GT);axis('tight');
pause(1)
title(['Dice=' num2str(DIC),],'FontSize',20)
%title(['IoU=' num2str(IoU),],'FontSize',20)
%plotname=sprintf('slice_%d.fig',1);
%savefig(plotname);
% Slice number-2 to 5
%% Weight Initialization (Transfer Learning)
for image_id = 2:numFiles
fprintf('SLICE-%d\n',image_id);    
w_init=wopt;
img=ROI_images_pp{image_id};
%% Optimization
tic;
[w_pred, J_hist,Jr_hist,Js_hist,Jiv_hist,Jig_hist,break_iter] =...
Optimize_via_Adam(w_init,M_inv, R, B, dBds, d2Bds2,img,X,Y,alpha_list,beta_list,gamma_list,delta_list,mu_list,max_iter,lr);
elapsed_time = toc;
TIME_list=[TIME_list elapsed_time];
disp(['Elapsed Time: ', num2str(elapsed_time), ' seconds']);
% Difference between prediction with and without shape prior 
% Ground Truth
GT=ROI_masks{image_id};
GT=im2gray(GT);
GT=GT./max(max(GT));
% Get masks with and without shape prior
nw=length(w_pred);
xf_pred=w_pred(1:0.5*nw);yf_pred=w_pred(1+0.5*nw:nw);
[coeff_x, coeff_y] = spline_coefficients(M_inv, R, xf_pred, yf_pred);
xsi_pred = B * coeff_x;ysi_pred = B * coeff_y;
[in,on] = inpolygon (X, Y, xsi_pred, ysi_pred);
chi_pred = in & ~on;
wopt=w_pred;
wopt_filename = sprintf('wopt_%d.mat', image_id);
save(wopt_filename, 'wopt')
% IoU score
IoU = jaccard( double(GT),double(chi_pred));
DIC = dice( double(GT),double(chi_pred));
IoU_list(image_id,1)=IoU;
DIC_list(image_id,1)=DIC;
IoU = sprintf('%0.3g', IoU);
%figure()
figure(11)
subplot(1,5,image_id)
imshowpair(chi_pred,GT);axis('tight');
pause(1)
title(['Dice=' num2str(DIC),],'FontSize',20)
%title(['IoU=' num2str(IoU),],'FontSize',20)
%plotname=sprintf('slice_%d.fig',image_id);
%savefig(plotname);
end
save('IoU.mat', 'IoU_list');
save('DICE.mat', 'DIC_list');

figure (21)
% Sample data
categories = {'Slice-1', 'Slice-2', 'Slice-3', 'Slice-4', 'Slice-5'};
values = TIME_list; % Random values for each category

% Create a bar plot
bar(values);

% Customize the plot
%title('Segmentation Time Per Slice','FontSize',20);
xlabel('Slice No.','FontSize',20);
ylabel('Convergence Time (Sec)','FontSize',20);
set(gca, 'XTickLabel', categories);

% Display the values on top of each bar
text(1:length(categories), values, num2str(values'), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');

% Show grid
grid on;
%------------------------------------------------------------------------------------------%
%                                    LIST OF MODULES                                        %
%------------------------------------------------------------------------------------------%
%% 01
function [alpha_list, beta_list, gamma_list, delta_list, mu_list, Px, Py, nc, nl, max_iter, lr, init_rad]= user_inputs()
% Domain 
Px=100; % Pixels in x-direction
Py=100; % Pixels in y-direction

% Approximate number of points in x and y direction
nc=10;
% Number of points within each local spline (including end points)
nl=4;
init_rad=15;
% Training params
max_iter=1000; 
lr=5e-2;

% Loss coefficients
% Regularization
alpha_list=1e-1*ones(max_iter,1); % Stretching Loss
beta_list=10*1e-2*ones(max_iter,1); % Bending Loss (100x for weak boundary)
gamma_list=10*1e7*ones(max_iter,1); % Shape Prior Loss(10x for thick muscles)

delta_list=10*1e-1*ones(max_iter,1); % Image Gradient Loss (100x for weak boundary)

% Chan-Vese Loss Coefficient
mu_list=1e4*ones(max_iter,1); % Data Loss      
end
%% 02
function M=periodic_block(s)
n = length(s)-1;
M=zeros(4*n,4*n);
for i=1:n
    j=1+4*(i-1);
    if i==n
    % first intersection
    M(j,j)=s(i)^3;M(j,j+1)=s(i)^2;M(j,j+2)=s(i);M(j,j+3)=1;
    % second intersection
    M(j+1,j)=s(i+1)^3;M(j+1,j+1)=s(i+1)^2;M(j+1,j+2)=s(i+1);M(j+1,j+3)=1;
    % first derivatives matching
    M(j+2,j)  = 3*s(i+1)^2;M(j+2,j+1)= 2*s(i+1);M(j+2,j+2)= 1;
    M(j+2,1)=-3*s(1)^2;M(j+2,2)=-2*s(1);M(j+2,3)=-1;
    % second derivatives matching
    M(j+3,j)  = 6*s(i+1);M(j+3,j+1)= 2;
    M(j+3,1)=-6*s(1);M(j+3,2)=-2;
    else
    % first intersection
    M(j,j)=s(i)^3;M(j,j+1)=s(i)^2;M(j,j+2)=s(i);M(j,j+3)=1;
    % second intersection
    M(j+1,j)=s(i+1)^3;M(j+1,j+1)=s(i+1)^2;M(j+1,j+2)=s(i+1);M(j+1,j+3)=1;
    % first derivatives matching
    M(j+2,j)  = 3*s(i+1)^2;M(j+2,j+1)= 2*s(i+1);M(j+2,j+2)= 1;
    M(j+2,j+4)=-3*s(i+1)^2;M(j+2,j+5)=-2*s(i+1);M(j+2,j+6)=-1;
    % second derivatives matching
    M(j+3,j)  = 6*s(i+1);M(j+3,j+1)= 2;
    M(j+3,j+4)=-6*s(i+1);M(j+3,j+5)=-2;
    end
end
end
%% 03
function RHS = spline_RHS_block(s)
    n = length(s)-1;
    RHS = zeros(4*n, n);
    for i = 1:(n-1)
        j=1+4*(i-1);
        RHS(j,i) = 1;
        RHS(j+1,i+1) = 1;
    end
    RHS(4*(n-1)+1,n) = 1;
    RHS(4*(n-1)+2,1) = 1;
end
%% 04
function [B, dBds, d2Bds2] = interpolation_blocks(s,nl)
    nc = length(s)-1;
    block1 = eye(nc);
    block2 = ones(nl,4);
    
    B=kron(block1, block2);
    dBds=0*B;
    d2Bds2=0*B;
    
    for i = 1:nc
        ss = linspace(s(i), s(i+1), nl);  % interpolated points within each spline
        for j = 1:nl
            % for interpolated values 
            B((i-1)*nl+j,4*(i-1)+1)=ss(j)^3;
            B((i-1)*nl+j,4*(i-1)+2)=ss(j)^2;
            B((i-1)*nl+j,4*(i-1)+3)=ss(j);
            B((i-1)*nl+j,4*(i-1)+4)=1;
            % for interpolated first derivs
            dBds((i-1)*nl+j,4*(i-1)+1)=3*ss(j)^2;
            dBds((i-1)*nl+j,4*(i-1)+2)=2*ss(j);
            dBds((i-1)*nl+j,4*(i-1)+3)=1;
            % for interpolated second derivs
            d2Bds2((i-1)*nl+j,4*(i-1)+1)=6*ss(j);
            d2Bds2((i-1)*nl+j,4*(i-1)+2)=2;        
        end
    end
end
%% 05
function [coeff_x, coeff_y] = spline_coefficients(M_inv, R, x, y)
    coeff_x = M_inv * R * reshape(x, [length(x), 1]);
    coeff_y = M_inv * R * reshape(y, [length(y), 1]);
end
%% 06
function [ROI,mask_ROI]= Get_ROI(img,mask)
% Display the image in a new figure
imshow(img);
title('Select ROI','FontSize',50);
% Wait for the user to click on the image
[x, y] = ginput(1); 
% Crop a 100x100 ROI
ROI = imcrop(img, [x-50 y-50 99 99]);
mask_ROI = imcrop(mask, [x-50 y-50 99 99]);
close
end
%% 07
function [Cx, Cy] = Get_Center(img,init_rad)
    % Display the image in a new figure
    figure();
    imshow(img,'InitialMagnification',250);hold on;%<------------
    
    % Initialize the circle around the initial position
    r = init_rad;  % radius of the circle
    t = 0:pi/20:2*pi;
    Cx = 0;
    Cy = 0;
    xc = Cx + r*cos(t);
    yc = Cy + r*sin(t);
    h = plot(xc, yc, 'r--', 'LineWidth', 4);
    
    % Set up the mouse movement callback function
    set(gcf, 'WindowButtonMotionFcn', @mousemove_callback);
    
    % Wait for the user to click on the image
    title('Weight Initialization','FontSize',50);
    waitforbuttonpress;
    
    % Get the coordinates of the clicked point
    point = get(gca, 'CurrentPoint');
    Cx = round(point(1,1));
    Cy = round(point(1,2));
    
    % Remove the mouse movement callback function
    set(gcf, 'WindowButtonMotionFcn', '');
    
    % Remove the circle around the clicked point
    delete(h);
    
    % Define the mouse movement callback function
    function mousemove_callback(~, ~)
        % Get the coordinates of the current point
        point = get(gca, 'CurrentPoint');
        Cx_new = round(point(1,1));
        Cy_new = round(point(1,2));
        
        % Update the circle around the current point
        if Cx_new ~= Cx || Cy_new ~= Cy
            Cx = Cx_new;
            Cy = Cy_new;
            xc = Cx + r*cos(t);
            yc = Cy + r*sin(t);
            set(h, 'XData', xc, 'YData', yc);
        end
    end
pause(0.5)
close
end
%% 08
function ROI=Preprocessing(ROI)
%ROI = imdiffusefilt(ROI); % Anisotropic Diffusion
ROI = adapthisteq(im2gray(ROI)); % Histogram Matching
ROI = imadjust(im2gray(ROI)); % Imadjust
ROI = im2double(ROI);
clc;
end
%% 09
function [x0, y0] = create_init_circle(Cx, Cy, Rad,nc)
ne = nc+1;
t = linspace(0, 1, ne);
t = t(1:end-1);

x0 = Cx + Rad * cos(2 * pi * t);
y0 = Cy + Rad * sin (2 * pi * t);
       
end
%% 10
function [Jr, Js] = Regularization(xs, ys, M_inv, R, dBds, d2Bds2, alpha, beta, gamma)
    [coeff_x,coeff_y] = spline_coefficients(M_inv, R, xs, ys);
    dxdsi = dBds * coeff_x; dydsi = dBds * coeff_y;
    d2xds2i = d2Bds2 * coeff_x; d2yds2i = d2Bds2 * coeff_y;
    kappa=(d2yds2i.*dxdsi-d2xds2i.*dydsi)./((sqrt(dxdsi.^2+dydsi.^2)).^3);

    Jr = alpha * mean((dxdsi.^2 + dydsi.^2)) + beta * mean((d2xds2i.^2 + d2yds2i.^2)); % Shape Regularization
    Js = gamma*mean(kappa.^2); % Shape Prior
end
%% 11
function [Jiv,Jig] = Data_Loss (xs,ys,img,X,Y,delta, mu)
[in,on] = inpolygon (X, Y, xs, ys);
chi_1 = in & ~on; 
chi_2 = ~in;

c1 = mean(img(chi_1==1));  
c2 = mean(img(chi_2==1));

[img_x,img_y] = gradient(img);                     
E0 = img_x.^2+img_y.^2;

% Total loss
Jiv = mu*(sum(((img-c1).*chi_1).^2,'all')+sum(((img-c2).*chi_2).^2,'all'));
Jig = delta*sum((E0.*chi_1).^2,'all');
end
%% 13
function [J,dJdw,Jr,Js,Jiv,Jig]=Loss_Function(w,M_inv, R, dBds, d2Bds2,img,X,Y,alpha,beta,gamma,delta,mu)
% cost function
nw=length(w);
xs=w(1:0.5*nw);ys=w(1+0.5*nw:nw);

[Jr, Js] = Regularization(xs, ys, M_inv, R, dBds, d2Bds2, alpha, beta, gamma);
[Jiv,Jig] = Data_Loss (xs,ys,img,X,Y,delta,mu);
J = Jr+Js+Jiv+Jig; 
% gradients
dw=0.5;
dJdw = 0*w;
for j=1:nw
     wf = w; wb=w;
     % perturb the weights     
     wf(j) = w(j)+dw; wb(j)= w(j)-dw;
     xf =wf(1:0.5*nw); xb=wb(1:0.5*nw);
     yf =wf(1+0.5*nw:nw);yb=wb(1+0.5*nw:nw);
     
     % calculate losses:
     [Jr_f, Js_f] = Regularization(xf, yf, M_inv, R, dBds, d2Bds2, alpha, beta, gamma);
     [Jiv_f,Jig_f] = Data_Loss (xf,yf,img,X,Y,delta,mu);
     Jf = Jr_f+Js_f+Jiv_f+Jig_f; 
     
     [Jr_b, Js_b] = Regularization(xb, yb, M_inv, R, dBds, d2Bds2, alpha, beta, gamma);
     [Jiv_b,Jig_b] = Data_Loss (xb,yb,img,X,Y,delta,mu);
     Jb = Jr_b+Js_b+Jiv_b+Jig_b; 
     
     % grads using CDS
     dJdw(j)=(Jf-Jb)/(2*dw);
end
end
%% 14
function [wopt, J_hist,Jr_hist,Js_hist,Jiv_hist,Jig_hist,break_iter] = Optimize_via_Adam(w,M_inv, R, B, dBds, d2Bds2,img,X,Y,alpha_list,beta_list,gamma_list,delta_list,mu_list,max_iter,lr)

% Initialize variables
J_hist = zeros(max_iter, 1);
Jr_hist = zeros(max_iter, 1); 
Js_hist = zeros(max_iter, 1); 
Jiv_hist = zeros(max_iter, 1);
Jig_hist = zeros(max_iter, 1);


v = zeros(size(w));% 1st moment
s = zeros(size(w));% 2nd moment
beta1 = 0.9;% decay rate for 2nd moment
beta2 = 0.999;% decay rate for 2nd moment
eps = 10^-8;%stability term

% Parameters for early stopping
patience = 50; % number of iterations to wait before stopping
best_w = w; % best parameters found so far
best_J = inf; % best cost found so far
wait = 0; % counter for number of iterations without improvement
break_iter=max_iter;

for iter = 1:max_iter
    alpha=alpha_list(iter);
    beta=beta_list(iter);
    gamma=gamma_list(iter);
    delta=delta_list(iter);
    mu=mu_list(iter);
    % Compute cost and gradients
    [J,dJdw,Jr,Js,Jiv,Jig]=Loss_Function(w,M_inv, R, dBds, d2Bds2,img,X,Y,alpha,beta,gamma,delta,mu);
    
    % Update moment estimates
    v = beta1 * v + (1-beta1) * dJdw;
    s = beta2 * s + (1-beta2) * dJdw.^2;
    
    % Correct bias
    v_hat = v / (1-beta1^iter);
    s_hat = s / (1-beta2^iter);
    
    % Update parameters
    w = w - lr * v_hat ./ (sqrt(s_hat) + eps);
    
    % Save cost
    J_hist(iter) = J;
    Jr_hist(iter) = Jr; 
    Js_hist(iter) = Js; 
    Jiv_hist(iter) = Jiv;
    Jig_hist(iter) = Jig;
%---------------------------------------------------------------------------------------------------%
%     % Visualization while training
    plot_freq=100;%50;
    if iter==1 ||mod(iter,plot_freq)==0
        figure(1)
        nw=length(w);
        x0=w(1:0.5*nw);y0=w(1+0.5*nw:nw);
        [coeff_x, coeff_y] = spline_coefficients(M_inv, R, x0, y0);
        xsi = B * coeff_x;ysi = B * coeff_y;
        imshow(img,'InitialMagnification',250);hold on;
        plot(xsi,100-ysi,'-g','LineWidth',5); hold on;
        plot(x0,100-y0,'ro','LineWidth',5); hold off;
        %title(['Iter=' num2str(iter),', ','Loss=' num2str(round(J,2))],'FontSize',15);
        title(['Iter=' num2str(iter)],'FontSize',25)
    end
%---------------------------------------------------------------------------------------------------%
    % Check for improvement
    if J < best_J
        best_J = J;
        best_w = w;
        wait = 0;
    else
        wait = wait + 1;
    end
    % Check for early stopping
    if wait >= patience
        disp("Early stopping at iteration " + iter + " with cost " + best_J);
        w = best_w;
        break_iter=iter;
        break;
    end

end
wopt=w;
end