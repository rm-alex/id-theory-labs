import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import warnings
import os

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------- Candidate functions (each signature f(t, a, b, c, d)) ----------
def f_sin(t, a, b, c, d):
    return a * np.sin(b * t + c) + d

def f_linear(t, a, b, c, d):
    # at + b  -> use a->a, b->b, ignore c,d
    return a * t + b

def f_quadratic_form(t, a, b, c, d):
    # (a t - d)^2 + b t + c
    return (a * t - d) ** 2 + b * t + c

def f_exp(t, a, b, c, d):
    # a * exp(b t + d) + c
    return a * np.exp(b * t + d) + c

def f_sqrt(t, a, b, c, d):
    # a * sqrt(b t + d) + c ; ensure domain
    val = b * t + d
    # clip small negatives to small positive to allow fit attempts
    val = np.where(val <= 0, 1e-9, val)
    return a * np.sqrt(val) + c

def f_reciprocal(t, a, b, c, d):
    # a / (b t + d) + c
    val = b * t + d
    val = np.where(np.abs(val) < 1e-12, np.sign(val) * 1e-12 + 1e-12, val)
    return a / val + c

def f_inv_sqrt(t, a, b, c, d):
    # a / sqrt(b t + d) + c
    val = b * t + d
    val = np.where(val <= 0, 1e-9, val)
    return a / np.sqrt(val) + c

def f_inv_square(t, a, b, c, d):
    # a / (b t + d)**2 + c
    val = b * t + d
    val = np.where(np.abs(val) < 1e-12, np.sign(val) * 1e-12 + 1e-12, val)
    return a / (val ** 2) + c

# mapping
CANDIDATES = {
    "a*sin(bt+c)+d": f_sin,
    "at+b": f_linear,
    "(at-d)^2 + bt + c": f_quadratic_form,
    "a*exp(bt+d)+c": f_exp,
    "a*sqrt(bt+d)+c": f_sqrt,
    "a/(bt+d)+c": f_reciprocal,
    "a/sqrt(bt+d)+c": f_inv_sqrt,
    "a/(bt+d)^2 + c": f_inv_square,
}


# ---------- Fitting helper ----------
def try_fit(func, t, x, p0=None, bounds=(-np.inf, np.inf)):
    """
    Try to fit func(t, a,b,c,d) to data (t,x) with curve_fit.
    Returns dict with success flag, params, y_pred, r2 and RSS.
    """
    result = {"success": False, "params": None, "y_pred": None, "r2": -np.inf, "rss": np.inf}
    try:
        popt, pcov = curve_fit(func, t, x, p0=p0, bounds=bounds, maxfev=5000)
        y_pred = func(t, *popt)
        r2 = r2_score(x, y_pred)
        rss = np.sum((x - y_pred) ** 2)
        result.update({"success": True, "params": popt, "y_pred": y_pred, "r2": r2, "rss": rss})
    except Exception as e:
        # fitting failed
        result["error"] = str(e)
    return result


# ---------- Special-case faster fits (linear, polynomial) ----------
def fit_linear_least_squares(t, x):
    # fit x = a t + b
    A = np.vstack([t, np.ones_like(t)]).T
    sol, *_ = np.linalg.lstsq(A, x, rcond=None)
    a, b = sol[0], sol[1]
    y_pred = a * t + b
    r2 = r2_score(x, y_pred)
    rss = np.sum((x - y_pred) ** 2)
    return {"success": True, "params": np.array([a, b, 0, 0]), "y_pred": y_pred, "r2": r2, "rss": rss}


def fit_quadratic_as_poly(t, x):
    # Fit degree-2 polynomial: x = p2 t^2 + p1 t + p0
    p = np.polyfit(t, x, 2)
    p2, p1, p0 = p
    y_pred = np.polyval(p, t)
    r2 = r2_score(x, y_pred)
    rss = np.sum((x - y_pred) ** 2)
    # map to params as a, b, c, d approx is not exact, but we return poly fit for scoring
    return {"success": True, "params": np.array([p2, p1, p0, 0]), "y_pred": y_pred, "r2": r2, "rss": rss}


# ---------- Wrapper that attempts all candidate fits for one regressor ----------
def identify_regressor_type(t, x, verbose=False):
    """
    Try all candidate models; return best model name and details.
    """
    fits = {}
    # quick guesses for initial params
    t_mean = np.mean(t)
    x_mean = np.mean(x)
    x_std = np.std(x) if np.std(x) > 1e-9 else 1.0
    # try linear and quadratic (fast)
    fits["at+b"] = fit_linear_least_squares(t, x)
    fits["(at-d)^2 + bt + c"] = fit_quadratic_as_poly(t, x)

    # try nonlinear candidates with curve_fit
    # reasonable initial guesses / bounds for stability
    for name, func in CANDIDATES.items():
        if name in fits:
            continue  # already did linear/quadratic
        p0 = [x_std, 0.0, 0.0, 0.0]  # a,b,c,d default guess
        # tweak p0 for some functions
        if name == "a*sin(bt+c)+d":
            p0 = [x_std, 1.0, 0.0, x_mean]
            bounds = ([-np.inf, -10.0, -2*np.pi, -np.inf], [np.inf, 10.0, 2*np.pi, np.inf])
        elif name == "a*exp(bt+d)+c":
            p0 = [x_mean if x_mean!=0 else 1.0, 0.0, 0.0, 0.0]  # roughly
            bounds = ([-np.inf, -3.0, -np.inf, -np.inf], [np.inf, 3.0, np.inf, np.inf])
        elif name.startswith("a*sqrt") or "sqrt" in name:
            p0 = [x_std, 1e-3 if np.mean(t)==0 else 1.0, 0.0, 1.0]
            bounds = ([-np.inf, -10.0, -np.inf, -np.inf], [np.inf, 10.0, np.inf, np.inf])
        else:
            bounds = (-np.inf, np.inf)

        try:
            res = try_fit(func, t, x, p0=p0, bounds=bounds)
            fits[name] = res
        except Exception as e:
            fits[name] = {"success": False, "error": str(e)}

    # choose best by R^2 (also prefer simpler if ties)
    best_name = None
    best_r2 = -np.inf
    for name, info in fits.items():
        if info.get("success", False):
            r2 = info.get("r2", -np.inf)
            if r2 > best_r2 + 1e-9:
                best_r2 = r2
                best_name = name

    # prepare result
    best_info = fits[best_name] if best_name is not None else None
    if verbose:
        print("Tried fits and R^2 scores:")
        for name, info in sorted(fits.items(), key=lambda p: -p[1].get("r2", -np.inf)):
            print(f"  {name:30s}: success={info.get('success',False):5}  R2={info.get('r2',None)}")
    return best_name, best_info, fits


# ---------- PE test (numerical) ----------
def numeric_pe_test(phi, window_size=None, eig_threshold=1e-4):
    """
    Check persistent excitation numerically.
    phi: (N x m) regressor matrix
    window_size: length of window for sliding-window Gram check; if None use m*5 or 50 min
    Returns:
      - global_eigvals: eigenvalues of (Phi^T Phi)/N over whole dataset
      - min_window_eig: minimum eigenvalue found over sliding windows
      - pe_decision: boolean whether min_window_eig > eig_threshold
    """
    N, m = phi.shape
    if window_size is None:
        window_size = max(50, 5 * m)
        window_size = min(window_size, N // 2)  # sensible limit

    # global Gram
    G_global = (phi.T @ phi) / float(N)
    evals_global = np.linalg.eigvalsh(G_global)

    # sliding windows
    min_eig = np.inf
    step = max(1, window_size // 10)
    for start in range(0, N - window_size + 1, step):
        window = phi[start : start + window_size, :]
        G = (window.T @ window) / float(window.shape[0])
        eigs = np.linalg.eigvalsh(G)
        min_eig = min(min_eig, np.min(eigs))

    pe = min_eig > eig_threshold
    return {"evals_global": evals_global, "min_window_eig": min_eig, "pe": pe, "threshold": eig_threshold, "window_size": window_size}


# ---------- Main: load data, identify types, run PE test ----------
def main(path="data2.csv", verbose=True):
    df = pd.read_csv(path)
    # ensure expected columns
    expected = ["t", "x_1", "x_2", "x_3", "y"]
    for c in expected:
        if c not in df.columns:
            raise RuntimeError(f"File {path} missing column '{c}'")

    t = df["t"].values
    regs = {}
    detailed_fits = {}

    for i, col in enumerate(["x_1", "x_2", "x_3"], start=1):
        x = df[col].values
        name, best_info, all_fits = identify_regressor_type(t, x, verbose=False)
        regs[col] = {"type": name, "fit": best_info}
        detailed_fits[col] = all_fits
        if verbose:
            print("\n" + "-" * 60)
            print(f"Regressor {col}: best-fit type -> {name}")
            if best_info and best_info.get("success", False):
                print(f"  R² = {best_info['r2']:.6f}, RSS = {best_info['rss']:.6g}")
                print(f"  Fitted params (a,b,c,d) ~ {best_info['params']}")
            else:
                print("  Fit failed for all candidates.")

    # Form regressor matrix phi (N x 3)
    phi = df[["x_1", "x_2", "x_3"]].values

    # PE test
    pe_res = numeric_pe_test(phi, window_size=None, eig_threshold=1e-5)

    if verbose:
        print("\n" + "=" * 60)
        print("Persistent Excitation (PE) numeric test results:")
        print(f"  Global Gram eigenvalues: {pe_res['evals_global']}")
        print(f"  Minimum sliding-window eigenvalue: {pe_res['min_window_eig']:.6g}")
        print(f"  Window size used for sliding check: {pe_res['window_size']}")
        print(f"  Threshold for 'significant' eigenvalue: {pe_res['threshold']}")
        if pe_res["pe"]:
            print("  -> Decision: regressor looks persistently exciting (PE) -> gradient descent/LMS can converge (noise-free, proper step-size).")
        else:
            print("  -> Decision: NOT persistently exciting (PE) -> gradient descent may NOT drive parameter error to zero.")
        print("=" * 60)

    # Return structured result
    return {"regressors": regs, "pe_test": pe_res, "detailed_fits": detailed_fits, "df": df}


if __name__ == "__main__":
    res = main(os.path.abspath(os.path.join("tasks", "data2.csv")), verbose=True)
    # If you want to inspect the detailed fits programmatically:
    # for col in ["x_1","x_2","x_3"]:
    #     for name, fit in res['detailed_fits'][col].items():
    #         print(col, name, fit.get('r2'))
