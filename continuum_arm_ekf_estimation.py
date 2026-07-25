import numpy as np
from scipy.linalg import expm, block_diag
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
from continuum_arm_kinematics_and_dynamics  import (L, r, alpha, k_t, M, B,
                               arc_from_tendon_lengths)
 
rng = np.random.default_rng(7)
 
k_arm   = 300.0                  # backbone elastic restoring per tendon [N/m]
gamma   = r * np.tan(alpha) / L  # twist per unit bend [rad/rad] (helix model)
K_m     = 0.05                   # motor torque constant [Nm/A]
r_spool = 0.01                   # spool radius [m]
c_F     = r_spool / K_m          # current per tension [A/N]  (I = c_F * F)
 
Ts      = 0.05                   # EKF sample time [s]
dt_true = 1e-3                   # truth-model integration step [s]
T_END   = 6.0
 
NX, NZ, NU = 12, 6, 3
IDX_L  = [0, 3, 6]
IDX_V  = [1, 4, 7]
IDX_F  = [2, 5, 8]
IDX_ARC = [9, 10, 11]            # phi, theta, Phi
 
A_t = np.array([[0.0,      1.0,    0.0],
                [-k_arm/M, -B/M,  -1.0/M],
                [0.0,      k_t,    0.0]])
B_t = np.array([[0.0], [0.0], [k_t]])
w_t = np.array([[0.0], [k_arm*L/M], [0.0]])   # affine offset from (L - l)
 
# augmented exponential -> exact ZOH discretization incl. offset
Maug = np.zeros((5, 5))
Maug[:3, :3] = A_t
Maug[:3, 3:4] = B_t
Maug[:3, 4:5] = w_t
Md = expm(Maug * Ts)
A_dt, B_dt, w_dt = Md[:3, :3], Md[:3, 3:4], Md[:3, 4:5].ravel()
 
# full 12x12 A_d: tendon blocks are exact; arc rows filled by the EKF
# Jacobian each step (nonlinear kinematic map). B_d is 12x3.
A_d = block_diag(A_dt, A_dt, A_dt, np.zeros((3, 3)))
B_d = np.zeros((NX, NU))
for i in range(3):
    B_d[3*i:3*i+3, i] = B_dt.ravel()
 
C = np.zeros((NZ, NX))
for i in range(3):
    C[i,     IDX_L[i]] = 1.0        # encoder: z_i = l_i
    C[3 + i, IDX_F[i]] = c_F        # current: z_{3+i} = c_F * F_i
 
sigma_enc = 1e-4                 # encoder noise 0.1 mm
sigma_cur = 0.05                 # current noise 0.05 A  (~0.25 N of tension)
R = np.diag([sigma_enc**2]*3 + [sigma_cur**2]*3)
 
Q = np.diag(([1e-10, 1e-6, 1e-2] * 3) +      # per tendon: l, v, F
            [1e-6, 1e-6, 1e-8])              # phi, theta, Phi
# process disturbance actually injected into the truth model (force jitter)
sigma_Fdist = 0.05               # [N/sqrt-step] force disturbance
 
def arc_states(l):
    """g: tendon lengths -> [phi, theta, Phi]."""
    kappa, phi = arc_from_tendon_lengths(l)
    theta = kappa * L
    return np.array([phi, theta, gamma * theta])
 
def arc_jacobian(l, eps=1e-7):
    """Numerical 3x3 Jacobian d[phi,theta,Phi]/d[l1,l2,l3]."""
    J = np.zeros((3, 3))
    g0 = arc_states(l)
    for j in range(3):
        lp = l.copy(); lp[j] += eps
        gp = arc_states(lp)
        d = gp - g0
        d[0] = (d[0] + np.pi) % (2*np.pi) - np.pi     # wrap phi difference
        J[:, j] = d / eps
    return J
 
def f_discrete(x, u):
    """Nonlinear discrete process model over one Ts."""
    xn = x.copy()
    for i in range(3):
        s = x[3*i:3*i+3]
        xn[3*i:3*i+3] = A_dt @ s + B_dt.ravel() * u[i] + w_dt
    xn[IDX_ARC] = arc_states(xn[IDX_L])
    return xn
 
def F_jacobian(x_pred):
    """Discrete Jacobian: exact tendon blocks + numeric arc rows."""
    A = A_d.copy()
    Jg = arc_jacobian(x_pred[IDX_L])
    # arc rows depend on the *predicted* tendon lengths -> chain through A_dt
    for j in range(3):
        A[np.ix_(IDX_ARC, [3*j, 3*j+1, 3*j+2])] = np.outer(Jg[:, j],
                                                           A_dt[0, :])
    return A
 
def ekf_predict(x, P, u):
    x_pred = f_discrete(x, u)
    A = F_jacobian(x_pred)
    P_pred = A @ P @ A.T + Q
    return x_pred, P_pred
 
def ekf_update(x_pred, P_pred, z):
    y = z - C @ x_pred                       # innovation
    S = C @ P_pred @ C.T + R
    K = P_pred @ C.T @ np.linalg.solve(S, np.eye(NZ)).T
    x_new = x_pred + K @ y
    P_new = (np.eye(NX) - K @ C) @ P_pred
    x_new[IDX_ARC] = arc_states(x_new[IDX_L])   # keep arc consistent
    return x_new, P_new, y
 
def u_cmd(t):
    """Step in tendon-velocity commands: bend, hold, release."""
    if 0.5 <= t < 2.5:
        return np.array([0.004, -0.002, -0.002])    # bend toward tendon 1
    if 4.0 <= t < 5.0:
        return np.array([-0.004, 0.002, 0.002])     # release
    return np.zeros(3)
 
def truth_deriv(s, u, w):
    ds = np.zeros(9)
    for i in range(3):
        l, v, F = s[3*i:3*i+3]
        ds[3*i]     = v
        ds[3*i + 1] = (-F - B*v + k_arm*(L - l)) / M
        ds[3*i + 2] = k_t * (u[i] + v) + w[i]
    return ds
 
def simulate_truth():
    n = int(T_END / dt_true) + 1
    t = np.linspace(0, T_END, n)
    S = np.zeros((n, 9))
    S[0, [0, 3, 6]] = L                      # start straight, slack-free
    for k in range(n - 1):
        u = u_cmd(t[k])
        w = rng.normal(0, sigma_Fdist, 3) / np.sqrt(dt_true)
        s = S[k]
        k1 = truth_deriv(s,               u, w)
        k2 = truth_deriv(s + dt_true/2*k1, u, w)
        k3 = truth_deriv(s + dt_true/2*k2, u, w)
        k4 = truth_deriv(s + dt_true*k3,   u, w)
        S[k+1] = s + dt_true/6*(k1 + 2*k2 + 2*k3 + k4)
    return t, S
 
def main():
    t_true, S_true = simulate_truth()
    step = int(Ts / dt_true)
    t_k = t_true[::step]
    S_k = S_true[::step]                     # true tendon states at EKF rate
    nk = len(t_k)
 
    # true full state (arc states from true tendon lengths)
    X_true = np.zeros((nk, NX))
    X_true[:, :9] = S_k
    for k in range(nk):
        X_true[k, IDX_ARC] = arc_states(S_k[k, [0, 3, 6]])
 
    # measurements
    Z = np.zeros((nk, NZ))
    Z[:, :3] = S_k[:, [0, 3, 6]] + rng.normal(0, sigma_enc, (nk, 3))
    Z[:, 3:] = c_F * S_k[:, [2, 5, 8]] + rng.normal(0, sigma_cur, (nk, 3))
 
    # EKF init: deliberately offset to show convergence
    x = np.zeros(NX)
    x[IDX_L] = L - 0.003                     # 3 mm length error
    x[IDX_F] = 2.0                           # 2 N tension error
    x[IDX_ARC] = arc_states(x[IDX_L])
    P = np.diag(([1e-5, 1e-4, 4.0] * 3) + [0.1, 0.1, 0.01])
 
    X_est = np.zeros((nk, NX))
    P_diag = np.zeros((nk, NX))
    for k in range(nk):
        if k > 0:
            x, P = ekf_predict(x, P, u_cmd(t_k[k-1]))
        x, P, _ = ekf_update(x, P, Z[k])
        X_est[k], P_diag[k] = x, np.diag(P)
 
    # errors
    rmse_l = np.sqrt(np.mean((X_est[:, IDX_L] - X_true[:, IDX_L])**2, axis=0))
    rmse_F = np.sqrt(np.mean((X_est[:, IDX_F] - X_true[:, IDX_F])**2, axis=0))
    print("EKF results (Ts = 50 ms, {} updates)".format(nk))
    print(f"  RMSE tendon lengths [mm]: {rmse_l*1000}")
    print(f"  RMSE tendon tension [N] : {rmse_F}")
    print(f"  final arc estimate  phi={np.rad2deg(X_est[-1,9]):.2f} deg, "
          f"theta={np.rad2deg(X_est[-1,10]):.2f} deg, "
          f"Phi={np.rad2deg(X_est[-1,11]):.3f} deg")
    print(f"  final arc truth     phi={np.rad2deg(X_true[-1,9]):.2f} deg, "
          f"theta={np.rad2deg(X_true[-1,10]):.2f} deg, "
          f"Phi={np.rad2deg(X_true[-1,11]):.3f} deg")
 
    colors = ["#d62728", "#2ca02c", "#1f77b4"]
 
    # ---------- Figure 1: tension, estimated vs true ------------------
    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for i in range(3):
        ax = axs[i]
        ax.plot(t_k, Z[:, 3+i]/c_F, ".", ms=2.5, color="0.65",
                label="from current  I/c_F")
        ax.plot(t_true, S_true[:, 3*i+2], "-", color=colors[i], lw=1.6,
                label="true tension")
        ax.plot(t_k, X_est[:, IDX_F[i]], "k--", lw=1.4, label="EKF estimate")
        sd = np.sqrt(P_diag[:, IDX_F[i]])
        ax.fill_between(t_k, X_est[:, IDX_F[i]]-2*sd,
                        X_est[:, IDX_F[i]]+2*sd, color="k", alpha=0.12,
                        label=r"$\pm 2\sigma$")
        ax.set_ylabel(f"F{i+1} [N]"); ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axs[0].set_title("Tendon tension: EKF estimate vs truth (step velocity command)")
    axs[2].set_xlabel("time [s]")
    fig.tight_layout(); fig.savefig("ekf_tension.png", dpi=150)
 
    # ---------- Figure 2: tendon positions ----------------------------
    fig2, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for i in range(3):
        ax = axs[i]
        ax.plot(t_k, Z[:, i]*1000, ".", ms=2.5, color="0.65", label="encoder")
        ax.plot(t_true, S_true[:, 3*i]*1000, "-", color=colors[i], lw=1.6,
                label="true length")
        ax.plot(t_k, X_est[:, IDX_L[i]]*1000, "k--", lw=1.4, label="EKF estimate")
        ax.set_ylabel(f"l{i+1} [mm]"); ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axs[0].set_title("Tendon lengths: EKF estimate vs truth")
    axs[2].set_xlabel("time [s]")
    fig2.tight_layout(); fig2.savefig("ekf_position.png", dpi=150)
 
    # ---------- Figure 3: arc states + estimation errors --------------
    fig3, (a1, a2, a3) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    labels = [r"$\phi$", r"$\theta$", r"$\Phi$"]
    for j, idx in enumerate(IDX_ARC):
        a1.plot(t_k, np.rad2deg(X_true[:, idx]), "-", lw=1.6,
                label=labels[j] + " true")
        a1.plot(t_k, np.rad2deg(X_est[:, idx]), "--", lw=1.2,
                label=labels[j] + " est")
    a1.set_ylabel("angle [deg]"); a1.set_title("Arc states (bending plane, bend, twist)")
    a1.grid(alpha=0.3); a1.legend(loc="upper right", fontsize=8, ncol=3)
    for i in range(3):
        a2.plot(t_k, (X_est[:, IDX_L[i]]-X_true[:, IDX_L[i]])*1000,
                color=colors[i], label=f"l{i+1}")
        a3.plot(t_k, X_est[:, IDX_F[i]]-X_true[:, IDX_F[i]],
                color=colors[i], label=f"F{i+1}")
    a2.set_ylabel("length error [mm]"); a2.grid(alpha=0.3); a2.legend(fontsize=8)
    a3.set_ylabel("tension error [N]"); a3.set_xlabel("time [s]")
    a3.grid(alpha=0.3); a3.legend(fontsize=8)
    fig3.tight_layout(); fig3.savefig("ekf_arc_and_errors.png", dpi=150)
 
    print("\nSaved: ekf_tension.png, ekf_position.png, ekf_arc_and_errors.png")
 
if __name__ == "__main__":
    main()

