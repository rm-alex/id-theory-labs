dataPath = 'C:\Users\alexe\study\id-theory-labs\tasks\data2.csv';
t1Path = 'C:\Users\alexe\study\id-theory-labs\tasks\t1.txt';

% === Чтение данных ===
data = readmatrix(dataPath);       % [t, phi1, phi2, phi3, y]
t = data(:, 1);
phi = data(:, 2:4);                % регрессор размера N x 3
y = data(:, 5);

% === Чтение t1 ===
t1_val = str2double(fileread(t1Path));
[~, idx_t1] = min(abs(t - t1_val)); % ближайший индекс к t1
% t(idx_t1)

% === Инициализация ===
theta_hat = zeros(3, 1);           % нулевые начальные условия
gamma = 1;

% === Градиентный спуск (дискретная версия) ===
dt = mean(diff(t));                % предполагаем равномерную дискретизацию

for k = 1:length(t)
    phi_k = phi(k, :)';
    y_k = y(k);
    e_k = y_k - phi_k' * theta_hat;
    theta_hat = theta_hat + gamma * phi_k * e_k * dt; % dt -- приближение интеграла
end

% === Вывод оценок в момент t1 ===
% Поскольку градиентный спуск реализован последовательно, оценка на шаге idx_t1:
theta_hat_t1 = zeros(3, 1);
theta_hat_curr = zeros(3, 1);
for k = 1:idx_t1
    phi_k = phi(k, :)';
    y_k = y(k);
    e_k = y_k - phi_k' * theta_hat_curr;
    theta_hat_curr = theta_hat_curr + gamma * phi_k * e_k * dt;
end
theta_hat_t1 = theta_hat_curr;

% === Ответы ===
fprintf('Оценка theta_1 в момент t1: %.6f\n', theta_hat_t1(1));
fprintf('Оценка theta_2 в момент t1: %.6f\n', theta_hat_t1(2));
fprintf('Оценка theta_3 в момент t1: %.6f\n', theta_hat_t1(3));
