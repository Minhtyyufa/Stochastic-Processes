%%
%Project 3
%Do Hyung (Dave), Brian, Minh

%% P2
% Bayes MMSE estimator
clc; close all; clear all;
% Number of trials
Ns = round(linspace(1, 1e4, 100));
sz = [Ns(end), 1];
mu = 10;
variance = 10;
n_trial = 1000;

mses_exp = zeros(size(Ns,2), n_trial);
mses_ray = zeros(size(Ns,2), n_trial);

lambda_exp = 1/mu;
lambda_ray = sqrt(variance);

for i = 1:size(Ns,2)
    for j = 1:n_trial
        X_exp = exprnd(mu, Ns(i), 1);
        X_ray = raylrnd(sqrt(variance), Ns(i), 1);
        
        % for an exponential distribution, a lambda for the mle estimator would
        % simply be the reciprocal of sample mean
        mses_exp(i, j) = 1/mean(X_exp);
        mses_ray(i, j) = sqrt(sum(X_ray.*X_ray)/(2*Ns(i)));
    end
end

bias_exp = mean(mses_exp, 2) - lambda_exp;
bias_ray = mean(mses_ray, 2) - lambda_ray;

var_exp = var(mses_exp, 0, 2);
var_ray = var(mses_ray, 0, 2);

mses_exp = vecnorm(mses_exp-lambda_exp, 2, 2);
mses_ray = vecnorm(mses_ray-lambda_ray, 2, 2);

figure;
semilogy(Ns, mses_exp);
hold on;
semilogy(Ns, mses_ray);

title('MSE vs Number of Observations for Exponential and Rayleigh Estimator');
legend('Exponential MLE', 'Rayleigh MLE');
xlabel('Number of Observations');
ylabel('MSE');

figure;
plot(Ns, bias_exp);
hold on;
plot(Ns, bias_ray);

title('Bias vs Number of Observations for Exponential and Rayleigh Estimator');
legend('Exponential MLE Bias', 'Rayleigh MLE Bias');
xlabel('Number of Observations');
ylabel('Bias');

figure;
semilogy(Ns, var_exp);
hold on;
semilogy(Ns, var_ray);

title('Variance vs Number of Observations for Exponential and Rayleigh Estimator');
legend('Exponential MLE Variance', 'Rayleigh MLE Variance');
xlabel('Number of Observations');
ylabel('Variance');

%% P3
clc; close all; clear all;

X = load('data.mat').data;
X_size = size(X,2);

lambda_exp = 1/mean(X);
lambda_ray = sqrt(sum(X.*X)/(2*X_size));

l_exp = sum(log(lambda_exp*exp(-lambda_exp*X)))
l_ray = sum(log(X/(lambda_ray*lambda_ray).*exp(-(X.*X)/(2*lambda_ray*lambda_ray))))

% Since likelihood estimate of Rayleigh MLE is higher than that of
% Exponential MLE, the data is more likely to be drawn from a Rayleigh
% Distribution. 