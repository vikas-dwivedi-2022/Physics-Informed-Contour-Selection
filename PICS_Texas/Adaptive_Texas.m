clc; clear; close all;
% User Inputs
[alpha_list,beta_list,mu_list,Px,Py,Nx,Ny,nl,max_iter,lr]= user_inputs();
nc=2*(Nx+Ny-2)-1;         % number of control knots 
s = linspace(0, 1, nc+1); % contour parameter

% Some Matrices
x=linspace(1,Px,Px);y=linspace(Py,1,Py);

% Mesh Grid: Mat:1,2
[X,Y]=meshgrid(x,y); 

% Regularization Matrices: Mat:  3-7
M = periodic_block(s);M_inv = inv(M);
R = spline_RHS_block(s);
[B, dBds, d2Bds2] = interpolation_blocks(s,nl);
load("TC_USA_map.mat");
img=resized_img/255;

Cx=2*64-20;Cy=64;
[x0, y0] = create_init_circle(Cx, Cy, 0.03*Px,nc);

%----------------------------------------------------------%
[coeff_x, coeff_y] = spline_coefficients(M_inv, R, x0, y0);
xsi = B * coeff_x;ysi = B * coeff_y;


figure(1)
subplot(1,2,1)
imshow(img,'InitialMagnification',250);hold on;
plot(xsi,Py-ysi,'-g','LineWidth',5); hold on;
plot(x0,Py-y0,'r+','LineWidth',5); hold off;
title('Initial Position','FontSize',25);
tic;
w_init = [x0 y0];
[w, loss_history,J1_history,J2_history,opi_history,break_iter,mu_list] = Optimize_via_Adam(w_init,M_inv, R, B, dBds, d2Bds2,img,X,Y,alpha_list,beta_list,mu_list,max_iter,lr);
elapsed_time = toc;
fprintf('Elapsed time: %.4f seconds\n', elapsed_time);



%% Plots
J1=J1_history(1:break_iter);
J2=J2_history(1:break_iter);
opi=opi_history(1:break_iter);
ind=find(opi<0.8);

% Define the figure size
figure('Units', 'inches', 'Position', [1 1 8 6]);

% Define the subplots
subplot(2, 2, 1);
plot(log10(J1(1:break_iter)),'-k','LineWidth',3);
xlabel('Iterations','FontSize',20);
ylabel('log_{10}(J_{int})','FontSize',20);
title('Internal Energy','FontSize',20);

subplot(2, 2, 2);
plot(log10(J2(1:break_iter)),'-k','LineWidth',3);
xlabel('Iterations','FontSize',20);
ylabel('log_{10}(J_{ext})','FontSize',20);
title('External Energy','FontSize',20);


subplot(2, 2, 3);
plot(log10(J1(1:break_iter)+J2(1:break_iter)),'-k','LineWidth',3);
xlabel('Iterations','FontSize',20);
ylabel('log_{10}(J_{int}+J_{ext})','FontSize',20);
title('Total Energy','FontSize',20);


subplot(2, 2, 4);
plot(opi(1:break_iter),'-k','LineWidth',3);hold on;
plot(0.8*ones(1,break_iter),'--r','LineWidth',3); hold off;
axis([1,break_iter,0,1])
xlabel('Iterations','FontSize',20);
ylabel('OPI','FontSize',20);
title('Operation Performance Index','FontSize',20);


% Customize the appearance
% xlim([0 10]);
% ylim([-1.5 1.5]);
% xticks(0:2:10);
% yticks(-1:0.5:1);
% set(gca, 'FontSize', 20);

% Save the figure
saveas(gcf, 'training_summary.pdf');
%------------------------------------------------------------------------------------------%
%                                    LIST OF MODULES                                        %
%------------------------------------------------------------------------------------------%
%% F-1
function [alpha_list,beta_list,mu_list,Px,Py,Nx,Ny,nl,max_iter,lr]= user_inputs()
% Domain 
Px=256; % Pixels in x-direction
Py=256; % Pixels in y-direction

% Approximate number of points in x and y direction
Nx=7;
Ny=7;

% Number of points within each local spline (including end points)
nl=5;

% Training params
max_iter=1000; 
lr=1e-1;

% Loss Coefficients
% Regularization

% TC-3B
alpha_list=8*1e-1*ones(max_iter,1);
beta_list=1e-3*ones(max_iter,1); 

% Initial Chan-Vese Loss Coefficient
mu_list=1e3*ones(max_iter,1);    
end

%% F-2
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
%% F-3
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
%% F-4
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
%% F-5
function [x0, y0] = create_init_circle(Cx, Cy, Rad,nc)
ne = nc+1;
t = linspace(0, 1, ne);
t = t(1:end-1);

x0 = Cx + Rad * cos(2 * pi * t);
y0 = Cy + Rad * sin (2 * pi * t);
       
end
%% F-6
function [coeff_x, coeff_y] = spline_coefficients(M_inv, R, x, y)
    coeff_x = M_inv * R * reshape(x, [length(x), 1]);
    coeff_y = M_inv * R * reshape(y, [length(y), 1]);
end

%% F-7
function loss = chan_vese_loss (xs,ys,img,X,Y,mu)
[in,on] = inpolygon (X, Y, xs, ys);
chi_1 = in & ~on; 
chi_2 = ~in;

c1 = mean(img(chi_1==1));  
c2 = mean(img(chi_2==1));

%  Area loss terms
loss_in= sum(((img-c1).*chi_1).^2,'all');
loss_out=sum(((img-c2).*chi_2).^2,'all');

% Total loss
loss=mu*(loss_in+loss_out);
end

%% F-8
function regu = regularization(xs, ys, M_inv, R, B, dBds, d2Bds2, alpha, beta)
    [coeff_x,coeff_y] = spline_coefficients(M_inv, R, xs, ys);
    %xsi = B * coeff_x;
    %ysi = B * coeff_y;
    dxdsi = dBds * coeff_x;
    dydsi = dBds * coeff_y;
    d2xds2i = d2Bds2 * coeff_x;
    d2yds2i = d2Bds2 * coeff_y;
    regu = alpha * mean((dxdsi.^2 + dydsi.^2)) + beta * mean((d2xds2i.^2 + d2yds2i.^2));
end
%% F-9
function [J,dJdw,J1,J2]=loss_function(w,M_inv, R, B, dBds, d2Bds2,img,X,Y,alpha,beta,mu)
% cost function
nw=length(w);
xs=w(1:0.5*nw);ys=w(1+0.5*nw:nw);
J1 = regularization(xs, ys, M_inv, R, B, dBds, d2Bds2, alpha, beta);
J2 = chan_vese_loss (xs,ys,img,X,Y,mu);
J = J1+J2; 
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
     J1f = regularization(xf, yf, M_inv, R, B, dBds, d2Bds2, alpha, beta);
     J2f = chan_vese_loss (xf,yf,img,X,Y,mu);
     Jf = J1f+J2f; 
     
     J1b = regularization(xb, yb, M_inv, R, B, dBds, d2Bds2, alpha, beta);
     J2b = chan_vese_loss (xb,yb,img,X,Y,mu);
     Jb = J1b+J2b;      
     
     % grads using CDS
     dJdw(j)=(Jf-Jb)/(2*dw);
end
end
%---------------------------------------------------------------------------------------------------------------->
function [wopt, loss_history,J1_history,J2_history,opi_history,break_iter,mu_list] = Optimize_via_Adam(w,M_inv, R, B, dBds, d2Bds2,img,X,Y,alpha_list,beta_list,mu_list,max_iter,lr)
% Initialize variables
win_size=5; adjust_count=0;max_adjust=15;
opi_history = zeros(max_iter, 1);
loss_history = zeros(max_iter, 1);
mu_list_orig=mu_list;

J1_history = zeros(max_iter, 1); % regularization
J2_history = zeros(max_iter, 1); % edge loss


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
    mu=mu_list(iter);
    % Compute cost and gradients
    [J, dJdw,J1,J2] = loss_function(w,M_inv, R, B, dBds, d2Bds2,img,X,Y,alpha,beta,mu);
    
    % Update moment estimates
    v = beta1 * v + (1-beta1) * dJdw;
    s = beta2 * s + (1-beta2) * dJdw.^2;
    
    % Correct bias
    v_hat = v / (1-beta1^iter);
    s_hat = s / (1-beta2^iter);
    
    % Update parameters
    w = w - lr * v_hat ./ (sqrt(s_hat) + eps);
    
    % Save cost
    loss_history(iter) = J;
    J1_history(iter) = J1;J2_history(iter) = J2;
    
    if iter>win_size+1
        opi_history(iter)=get_opi_score(J1_history,J2_history,iter,win_size,adjust_count,max_adjust);
        % Adaptive Chan-Vese Coefficient
        if adjust_count<max_adjust && opi_history(iter)<0.8 
           %fprintf("Adjustment no. %d\n",adjust_count+1);
           %fprintf("Chan-Vese coefficient adjusted at iter number %d\n",iter);
           pow=adjust_count+floor(log10(J2/J1));
           mu_list(iter+1:max_iter)=(2^pow)*ones(max_iter-iter,1)+mu_list_orig(iter+1:max_iter);
           iter_adjust=iter;
           adjust_count=adjust_count+1;
        end
    end
  
%---------------------------------------------------------------------------------------------------%
%   % Visualization while training
    plot_freq=50;
    if iter==1 ||mod(iter,plot_freq)==0
        %fprintf("Completed iterations = %d ...\n",iter)        
        figure(1)
        subplot(1,2,2)
        nw=length(w);
        x0=w(1:0.5*nw);y0=w(1+0.5*nw:nw);
        [coeff_x, coeff_y] = spline_coefficients(M_inv, R, x0, y0);
        xsi = B * coeff_x;ysi = B * coeff_y;
        pause(1)
        imshow(img,'InitialMagnification',250);hold on;
        plot(xsi,256-ysi,'-g','LineWidth',5); hold on;
        plot(x0,256-y0,'r+','LineWidth',5); hold off;
        %title(['Iter=' num2str(iter),', ','Loss=' num2str(round(J,2))],'FontSize',15);   
        title('Final Position','FontSize',25);
    end
%---------------------------------------------------------------------------------------------------%
    % Check for improvement
    if iter>=2 
       deltaJ=abs(loss_history(iter)-loss_history(iter-1))/loss_history(iter);
       deltaJ= log2(deltaJ);
       if deltaJ<-20
            best_J=J;
            best_w=w;
            wait=wait+1;
%             fprintf("patience=%d\n",patience-wait);
       end
    end
    %disp("wait=");disp(wait);
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
%% F-12
function opi_score=get_opi_score(J1_history,J2_history,iter,win_size,adjust_count,max_adjust)
omg=2;
J1_ref=J1_history(iter-win_size+1:iter);
J2_ref=J2_history(iter-win_size+1:iter);
rel_perf=1-floor(0.5*(1+0.5*sign(diff(J2_ref))-0.5*sign(diff(J1_ref))));
obt=exp(linspace(1,omg,win_size-1)).*rel_perf;
des=exp(linspace(1,omg,win_size-1)).*ones(size(rel_perf));
score=sum(obt)/sum(des);
if adjust_count<max_adjust
opi_score=score;
else
opi_score=min(1,ceil(score));%ceil(score-0.25);    
end
end




















