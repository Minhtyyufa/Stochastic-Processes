%%
%Project 2
%Do Hyung (Dave), Brian, Minh

%% P1
% Bayes MMSE estimator

% Number of trials
N= 10000;

% Making 
Y = (2*rand(N,1)-1);
W = (4*rand(N,1)-2);

X = Y + W;

x_ltneg1 = X<-1;
x_btwn = X < 1 & X > -1;
x_else = ~(x_ltneg1 | x_btwn);

MSE = zeros(N,1);
MSE(x_ltneg1) = (Y(x_ltneg1)-(X(x_ltneg1)+1)/2).^2;
MSE(x_btwn) = Y(x_btwn).^2;
MSE(x_else) = (Y(x_else)-(X(x_else)-1)/2).^2;

var_names = {'Experimental_MMSE', 'Theoretical_MMSE'};
MMSE_emp = mean(MSE);
MMSE_theo = .25;

% Bayes MMSE estimator in example 8.5 results
bayes_results_table = table(MMSE_emp,MMSE_theo , 'VariableNames', var_names)

% Linear MMSE estimator
x_mean = mean(X);
y_mean = mean(Y);
w_var = var(W);
Cxy = cov(X,Y);
x_var = Cxy(1,1);
y_var = Cxy(2,2);
c_xy = Cxy(1,2);

y_hat = y_mean + (c_xy/(x_var))*(X-x_mean);
MMSE_emp = mean((Y-y_hat).^2);
MMSE_theo = 4/15;

% Linear MMSE estimator in example 8.6 results
lin_mmse_results_table = table(MMSE_emp,MMSE_theo, 'VariableNames', var_names)

%% P2
clear all;
y_var = [1;1;2;2];
r_var = [1;2;1;2];
N_r = 2;
N = 5000;
y_mean = 1;
size_y = size(y_var);
size_y = size_y(1);

MMSE_emp = zeros(size_y,1);
MMSE_theo = zeros(size_y,1);

for i = 1:size_y
    Y = repmat(sqrt(y_var(i)).*randn(N,1)+y_mean, 1, N_r);
    R = sqrt(r_var(i)).*randn(N,N_r);

    X = Y + R;
    Cxx = cov(X);
    a = (inv(Cxx))*(y_var(i)*ones(N_r,1));

    y_hat = y_mean*(1-a(1)-a(2)) + a(1)*X(:,1) + a(2)*X(:,2);

    MMSE_emp(i) = mean((Y(:,1)-y_hat).^2);
    MMSE_theo(i) = (y_var(i))*(r_var(i))/(2*y_var(i)+r_var(i));
end

var_names = {'Var_y', 'Var_r', 'Experimental_MMSE', 'Theoretical_MMSE'};

% Results of using the linear estimator for multiple noisy observations
lin_mult_obs_results_table = table(y_var, r_var, MMSE_emp,MMSE_theo, 'VariableNames', var_names)



