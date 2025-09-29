# Создам код, который вычисляет оценки по L2 (среднеквадратичная) и по L1 (сумма модулей),
# строит предсказания hat_y2 (L2) и hat_y1 (L1), считает значения J1 и J2 в найденных точках,
# и показывает графики: данные + обе линии аппроксимации и функции потерь J1(b), J2(b).
import numpy as np
import matplotlib.pyplot as plt
from statistics import median

# Данные
x = np.array([1, 2, 3, 4, 5, 6], dtype=float)
y = np.array([1, 1.8, 2.6, 3.4, 6, 4], dtype=float)
log_x = np.log(x)
n = len(y)

# --- Оценка L2 (нормальная регрессия) ---
# Модель: y = a + b * log_x
X = np.vstack([np.ones(n), log_x]).T
theta_l2, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
a_l2, b_l2 = theta_l2[0], theta_l2[1]

# Предсказания и значения функций потерь в theta_hat_L2
yhat_l2 = a_l2 + b_l2 * log_x
J2_at_l2 = np.sum((y - yhat_l2)**2)          # L2 loss (sum of squared errors)
J1_at_l2 = np.sum(np.abs(y - yhat_l2))       # L1 loss at L2 estimate

# --- Поиск оценки L1 (минимизация суммы абсолютных отклонений) ---
# Для фиксированного b оптимальный a — медиана(y - b*log_x).
# Перебор по b: грубый сет, потом уточнение.
def evaluate_Js_for_b(b_vals):
    a_vals = np.array([median((y - b * log_x).tolist()) for b in b_vals])
    residuals = y.reshape(-1,1) - (a_vals + b_vals * log_x.reshape(-1,1))
    J1_vals = np.sum(np.abs(residuals), axis=0)
    J2_vals = np.sum(residuals**2, axis=0)
    return a_vals, J1_vals, J2_vals

# Грубый поиск
b_grid1 = np.linspace(-5, 5, 4001)
a_grid1, J1_grid1, J2_grid1 = evaluate_Js_for_b(b_grid1)
idx1 = np.argmin(J1_grid1)
b_est1 = b_grid1[idx1]
a_est1 = a_grid1[idx1]
# Уточним локально два раза
for span, pts in [(0.5,2001), (0.05,2001)]:
    b_grid_local = np.linspace(b_est1-span, b_est1+span, pts)
    a_grid_local, J1_grid_local, J2_grid_local = evaluate_Js_for_b(b_grid_local)
    idx_local = np.argmin(J1_grid_local)
    b_est1 = b_grid_local[idx_local]
    a_est1 = a_grid_local[idx_local]

a_l1, b_l1 = a_est1, b_est1
yhat_l1 = a_l1 + b_l1 * log_x
J1_at_l1 = np.sum(np.abs(y - yhat_l1))
J2_at_l1 = np.sum((y - yhat_l1)**2)

# Также подготовим J1(b) и J2(b) в широком диапазоне для отображения графиков
b_plot = np.linspace(b_grid1[0], b_grid1[-1], 8001)
a_plot = np.array([median((y - b * log_x).tolist()) for b in b_plot])
res_plot = y.reshape(-1,1) - (a_plot + b_plot * log_x.reshape(-1,1))
J1_plot = np.sum(np.abs(res_plot), axis=0)
# Для J2: оптимальный a_for_b (минимизирует SSE) это mean(y - b*log_x)
a_plot_mean = np.array([np.mean(y - b * log_x) for b in b_plot])
res_plot2 = y.reshape(-1,1) - (a_plot_mean + b_plot * log_x.reshape(-1,1))
J2_plot = np.sum(res_plot2**2, axis=0)

# --- Вывод числовых результатов ---
print("Оценки L2 (минимизация суммы квадратов): a = {:.6f}, b = {:.6f}".format(a_l2, b_l2))
print("J2(theta_hat_L2) = {:.6f}".format(J2_at_l2))
print("J1(theta_hat_L2) = {:.6f}".format(J1_at_l2))
print()
print("Оценки L1 (минимизация суммы модулей): a = {:.6f}, b = {:.6f}".format(a_l1, b_l1))
print("J1(theta_hat_L1) = {:.6f}".format(J1_at_l1))
print("J2(theta_hat_L1) = {:.6f}".format(J2_at_l1))
print()
print("hat_y (L2):", np.round(yhat_l2, 6))
print("hat_y (L1):", np.round(yhat_l1, 6))

# --- График 1: данные + обе аппроксимации ---
plt.figure(figsize=(9,6))
plt.scatter(x, y)
x_smooth = np.linspace(x.min(), x.max(), 200)
plt.plot(x_smooth, a_l2 + b_l2 * np.log(x_smooth), linewidth=2, label='L2 fit (least squares)')
plt.plot(x_smooth, a_l1 + b_l1 * np.log(x_smooth), linestyle='--', linewidth=2, label='L1 fit (least absolute deviations)')
plt.legend()
plt.title('Data and fits: L2 vs L1')
plt.xlabel('x'); plt.ylabel('y')
plt.grid(True, alpha=0.3)
plt.show()

# # --- График 2: J1(b) и J2(b) как функции b (с оптимальным a для каждого b) ---
# plt.figure(figsize=(9,6))
# plt.plot(b_plot, J1_plot, linewidth=2, label='J1(b) = sum |y - (a_med(b) + b ln x)|')
# plt.plot(b_plot, J2_plot, linewidth=2, label='J2(b) = min_a sum (y - (a + b ln x))^2 (a = mean residual)')
# # Отметим найденные точки
# plt.scatter([b_l1], [J1_at_l1], s=80)
# plt.scatter([b_l2], [J2_at_l2], s=80)
# plt.axvline(b_l1, linestyle=':', linewidth=1)
# plt.axvline(b_l2, linestyle=':', linewidth=1)
# plt.legend()
# plt.title('Loss curves J1(b) и J2(b) (оптимальный a для каждого b)')
# plt.xlabel('b'); plt.ylabel('Loss')
# plt.grid(True, alpha=0.3)
# plt.show()

# Чтобы пользователь видел числа в удобном виде — создадим словарь результатов
results = {
    'L2_theta': (a_l2, b_l2),
    'J2_at_L2': float(J2_at_l2),
    'J1_at_L2': float(J1_at_l2),
    'L1_theta': (a_l1, b_l1),
    'J1_at_L1': float(J1_at_l1),
    'J2_at_L1': float(J2_at_l1),
    'yhat_L2': yhat_l2.tolist(),
    'yhat_L1': yhat_l1.tolist(),
}
results

# Остатки
res_L2 = y - yhat_l2
res_L1 = y - yhat_l1

# Локальные L2 и L1 нормы (нарастающие)
cum_L2 = [np.linalg.norm(res_L2[:i+1], 2) for i in range(len(res_L2))]
cum_L1 = [np.linalg.norm(res_L1[:i+1], 1) for i in range(len(res_L1))]

# Построение графика
plt.figure(figsize=(9, 6))
plt.plot(x, cum_L2, "o-", label="||Y - Ŷ_L2||_2")
plt.plot(x, cum_L1, "s--", label="||Y - Ŷ_L1||_1")
plt.xlabel("x")
plt.ylabel("Накопленная ошибка")
plt.title("Накопленные нормы ошибок L1 и L2")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

