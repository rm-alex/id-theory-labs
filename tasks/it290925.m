%% QR
% Создаем вектор значений t
t = linspace(0, 2*pi, 24);

% Формируем матрицу, где каждый столбец = вектор v для t(k)
A = zeros(length(t), 4);
for k = 1:length(t)
    A(k, :) = [sin(t(k)), cos(t(k)), sin(t(k) + pi/4), 1];
end

% Применяем QR-разложение
[Q, R, E] = qr(A);

% Q — ортонормированная матрица
% R — верхнетреугольная матрица
E
R
A*E-Q*R

%% Ridge
X = [1, 2, 3, 4, 5, 6];
det(X'*X)
% y = [1, 1.8, 2.6, 3.4, 6, 4];
y = [1, 1.8, 2.4, 2.8, 3.0, 3.1];

lambda_values = 0:0.1:10;
J = zeros(size(lambda_values)); % для хранения функции потерь

for i = 1:length(lambda_values)
    lambda_2 = lambda_values(i);
    % Ridge regression
    hat_theta = inv((X'*X + lambda_2*eye(size(X,2)))) * (X'*y);
    % MSE
    J(i) = (1/2) * sum((X*hat_theta - y).^2);
end

figure;
plot(lambda_values, J, 'LineWidth', 2);
xlabel('lambda_2');
ylabel('J(theta estimated)');
title('Функция потерь J(theta estimated) от lambda_2');
grid on;

%% 17:34

% t=(0:0.1:10)';
t=(0:0.1:1000)';
% X = [sin(t) cos(t) ones(size(t))];
% X=[sin(t) exp(t/100)];
% X=[sin(t) exp(-t/100)];
X = [sin(t) cos(t)];
% X = [sin(t) t/1000]
% X = [sin(t) ones(size(t))];
% theta=[1;2;3];
theta=[1;2];
Y0=X*theta;
% Y=Y0+rand(length(t), 1);
Y=Y0+rand(length(t), 1)-0.5;
% S=rand(length(t), 1)-0.5;
% S=rand(length(t), 1);
% Y=Y0+S.*sin(t);
hat_theta=inv(X'*X)*X'*Y
E=Y-X*hat_theta;
mean(E)
corr(E,X(:,1))
corr(E,X(:,2))

% возможно выполнялись для помехи (если выполнились)
% точно нет (если не выполнились)
% стат тест -?- дисперсия постоянная
% наличие трендов. вероятность роста больше чем убывания при росте.
% вероятность падения больше чем убывания если падает. корреляция.

%% new
t=(0:0.1:2*pi)';

a = sin(t) + rand(length(t), 1);
b = sin(t) .* rand(length(t), 1);

var_a = var(a)
var_b = var(b)

figure;

subplot(2,1,1)
% autocorr(a)
autocorr(a, 'NumLags', length(a)-1)
title('Автокорреляция a')

subplot(2,1,2)
% autocorr(b)
autocorr(b, 'NumLags', length(a)-1)
title('Автокорреляция b')

