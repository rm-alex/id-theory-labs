%% experiment 1
t=(0:0.1:10)';
X=[sin(t) cos(t)];
% X=[sin(t) sin(t)];
theta=[1;2];
Y=X*theta;

cvx_begin
    variable hat_theta(2)       
    minimize( norm(Y - X * hat_theta, 2) )  
cvx_end

disp('Оптимальное hat_theta:');
disp(hat_theta);

theta1_vals = -10:0.1:10;
theta2_vals = -10:0.1:10;
[J1, J2] = meshgrid(theta1_vals, theta2_vals);

J = zeros(size(J1));
for i = 1:numel(J1)
    theta_test = [J1(i); J2(i)];
    % J(i) = norm(Y - X*theta_test, 2)^2;
    J(i) = norm(Y+10*(rand(size(t))-0.5) - X*theta_test, 2)^2;
end

figure;
surf(J1, J2, J);
xlabel('hat\_theta_1');
ylabel('hat\_theta_2');
zlabel('J(hat\_theta)');
title('Параболоид функции стоимости');
% title('Параболический цилиндр');
shading interp;
colorbar;

%% expermiment 2
t=(0:0.1:10)';
X2=[sin(t) cos(t) sin(t+pi/4)]; % третий синус ЛЗ
theta2=[1;2;3];
Y2=X2*theta2;
hat_theta2=(X2'*X2)^-1*X2'*Y2

lambda1=0.28;
lambda2=0.5;
cvx_begin
    variable hat_theta(3)
    J1=norm(Y2 - X2 * hat_theta, 2)
    J2=lambda1*norm(hat_theta,1)
    J3=lambda2*norm(hat_theta,2)
    J=J1 + J2 + J3
    minimize(J)  
cvx_end

disp(J1)
disp(J2)
disp(J3)
disp(J)

disp('Оптимальное hat_theta:');
disp(hat_theta);

% barX2 = X2(:, 1:2);
% hat_theta22=(barX2'*barX2)^-1*barX2'*Y2
% 
% bar_theta_1=theta2(1)+sqrt(2)/2*theta2(3)
% bar_theta_2=theta2(2)+sqrt(2)/2*theta2(3)
% 
% hat_theta2_0=hat_theta2(1)
% hat_theta2_1= L1
% hat_theta2_2=norm(lsqlin(X2, Y2))