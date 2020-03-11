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
y_var = [1 5 10 1 5 10 1 5 10].';
r_var = [1 1 1 5 5 5 10 10 10].';
N = 10000;
y_mean = 1;
size_y = size(y_var);
size_y = size_y(1);
N_r = 1000;

emp_message = "var_y=%d var_r=%d empirical";
theo_message = "var_y=%d var_r=%d theoretical";

figure;
legend_labels = [];
for i = 1:size_y
    legend_labels = [legend_labels sprintf(emp_message, y_var(i), r_var(i))];
    legend_labels = [legend_labels sprintf(theo_message, y_var(i), r_var(i))];
    MMSE_emp = zeros(N_r,1);
    MMSE_theo = zeros(N_r,1);
    for num_obs = 1:N_r
        Y = repmat(sqrt(y_var(i)).*randn(N,1)+y_mean, 1, num_obs);
        R = sqrt(r_var(i)).*randn(N,num_obs);

        X = Y + R;
        Cxx = cov(X);
        a = (inv(Cxx))*(y_var(i)*ones(num_obs,1));
                
        y_hat = y_mean;
        for j = 1:num_obs
            y_hat = y_hat - a(j)*(y_mean-X(:, j));
        end
        MMSE_emp(num_obs) = mean((Y(:,1)-y_hat).^2, 'all');
        MMSE_theo(num_obs) = (y_var(i))*(r_var(i))/(num_obs*y_var(i)+r_var(i));
    end
    plot(1:N_r, MMSE_emp);
    hold on;
    plot(1:N_r, MMSE_theo);
    hold on; 
    % var_names = {'N_r', 'Var_y', 'Var_r', 'Experimental_MMSE', 'Theoretical_MMSE'};
    % lin_mult_obs_results_table = table(ones(size_y,1)*N_r, y_var, r_var, MMSE_emp, MMSE_theo, 'VariableNames', var_names)
end
legend(legend_labels)


% Results of using the linear estimator for multiple noisy observations



