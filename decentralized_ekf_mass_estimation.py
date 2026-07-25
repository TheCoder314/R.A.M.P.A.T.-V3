import numpy as np
from collections import deque
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
rng = np.random.default_rng(3)
 
SPEC_LITERAL = False        # True -> a from state (degenerate, diverges)
 
G_VEC  = np.array([0.0, 0.0, -9.81])
M_TRUE = 0.8                        # object mass [kg]
Ts     = 0.02                       # filter step [s]
T_END  = 10.0
TAU    = 0.10                       # neighbor transport delay [s]
DLY    = int(round(TAU / Ts))       # = 5 steps
 
R_OWN      = 0.01 * np.eye(6)       # trust in own arm sensors
R_NEIGHBOR = 0.05 * np.eye(6)       # trust in neighbor's shared estimate
Q = np.diag([2e-5,                  # M   (small: near-constant parameter)
             1e-6, 1e-6, 1e-6,      # r
             1e-4, 1e-4, 1e-4])     # v
 
# imperfect knowledge of the applied arm force (bias + noise)
B_SELF = np.array([0.02, -0.015, 0.03])  # small residual bias [N]
S_SELF = 0.02                            # noise std [N]
S_IMU  = 0.05                            # IMU accel noise [m/s^2]
 
NX, NZ = 7, 6
 
AMP  = np.array([0.15, 0.10, 0.05])      # drift amplitude [m]
FREQ = np.array([0.6, 0.9, 0.4])         # drift frequency [Hz]
R0   = np.array([0.0, 0.0, 1.0])         # hover point [m]
W    = 2*np.pi*FREQ
 
def truth_kinematics(t):
    r = R0 + AMP*np.sin(W*t)
    v = AMP*W*np.cos(W*t)
    a = -AMP*W**2*np.sin(W*t)
    return r, v, a
 
def f_self_true(t):
    """True arm force on the object: M a = F_self + M g."""
    _, _, a = truth_kinematics(t)
    return M_TRUE*(a - G_VEC)
 
def f_self_believed(t):
    """What the VTOL thinks it is applying (bias + noise)."""
    return f_self_true(t) + B_SELF + rng.normal(0, S_SELF, 3)
def predict_state(x, u, dt=Ts):
    M, r, v = x[0], x[1:4], x[4:7]
    xn = x.copy()
    xn[1:4] = r + v*dt
    xn[4:7] = v + (u + M*G_VEC)/M*dt
    return xn
 
def predict_jacobian(x, u, dt=Ts):
    M = x[0]
    F = np.eye(NX)
    F[1:4, 4:7] = np.eye(3)*dt          # dr/dv
    F[4:7, 0]   = -u/M**2*dt            # dv/dM   (analytic)
    return F
 
def h_and_H(x, a_meas, v_prev, dt=Ts):
    """Observation function and analytic Jacobian."""
    M, r, v = x[0], x[1:4], x[4:7]
    H = np.zeros((NZ, NX))
    H[0:3, 1:4] = np.eye(3)                     # dp/dr
    if SPEC_LITERAL:                            # degenerate variant
        a = (v - v_prev)/dt
        H[3:6, 4:7] = -(M/dt)*np.eye(3)         # dF/dv
    else:
        a = a_meas                              # exogenous (IMU)
    z_hat = np.concatenate([r, M*(G_VEC - a)])
    H[3:6, 0] = G_VEC - a                       # dF/dM
    return z_hat, H
 
def ekf_update(x, P, z, a_meas, v_prev, R):
    z_hat, H = h_and_H(x, a_meas, v_prev)
    y = z - z_hat
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x_new = x + K @ y
    x_new[0] = max(x_new[0], 0.05)              # keep mass physical
    I_KH = np.eye(NX) - K @ H
    P_new = I_KH @ P @ I_KH.T + K @ R @ K.T     # Joseph form
    return x_new, P_new
 
class DecentralizedEKF:
    def __init__(self, x0, P0, use_neighbor=True):
        self.x, self.P = x0.copy(), P0.copy()
        self.v_prev = x0[4:7].copy()
        self.use_neighbor = use_neighbor
        self.buf = deque(maxlen=DLY + 2)
 
    def step(self, u, z_own, a_meas, z_nb):
        self.buf.append((self.x.copy(), self.P.copy(), self.v_prev.copy(),
                         u.copy(), z_own.copy(), a_meas.copy()))
        v_prev = self.x[4:7].copy()
        F = predict_jacobian(self.x, u)
        x = predict_state(self.x, u)
        P = F @ self.P @ F.T + Q
        x, P = ekf_update(x, P, z_own, a_meas, v_prev, R_OWN)
        self.x, self.P, self.v_prev = x, P, v_prev
 
        if self.use_neighbor and z_nb is not None and len(self.buf) > DLY:
            xb, Pb, vpb, _, _, amb = self.buf[-DLY-1]
            x, P = ekf_update(xb.copy(), Pb.copy(), z_nb, amb, vpb,
                              R_NEIGHBOR)
            for j in range(len(self.buf)-DLY, len(self.buf)):
                _, _, _, uj, zj, amj = self.buf[j]
                vp = x[4:7].copy()
                Fj = predict_jacobian(x, uj)
                xp = predict_state(x, uj)
                Pp = Fj @ P @ Fj.T + Q
                x, P = ekf_update(xp, Pp, zj, amj, vp, R_OWN)
                self.v_prev = vp
            self.x, self.P = x, P
        return self.x.copy(), np.diag(self.P).copy()
 
def make_measurement(t, R):
    """Arm measurement of the true object: tip pose + tip reaction force."""
    r, _, a = truth_kinematics(t)
    z = np.concatenate([r, M_TRUE*(G_VEC - a)])
    return z + rng.normal(0, np.sqrt(np.diag(R)))
 
def run():
    n = int(T_END/Ts) + 1
    t = np.arange(n)*Ts
 
    r0, v0, _ = truth_kinematics(0.0)
    x0 = np.concatenate([[0.4], r0 + 0.05, v0 + 0.05])   # M0 = 0.4 kg
    P0 = np.diag([0.02, 0.02, 0.02, 0.02, 0.05, 0.05, 0.05])
 
    ekf_dec = DecentralizedEKF(x0, P0, use_neighbor=True)
    ekf_own = DecentralizedEKF(x0, P0, use_neighbor=False)
 
    nbq = deque()
    X_dec = np.zeros((n, NX)); Pd_dec = np.zeros((n, NX))
    X_own = np.zeros((n, NX)); Pd_own = np.zeros((n, NX))
    R_t = np.zeros((n, 3)); V_t = np.zeros((n, 3)); Z_log = np.zeros((n, NZ))
 
    for k in range(n):
        u = f_self_believed(t[k])
        _, _, a_true = truth_kinematics(t[k])
        a_meas = a_true + rng.normal(0, S_IMU, 3)
        z_own = make_measurement(t[k], R_OWN)
        nbq.append(make_measurement(t[k], R_NEIGHBOR))
        z_nb = nbq.popleft() if len(nbq) > DLY else None
 
        X_dec[k], Pd_dec[k] = ekf_dec.step(u, z_own, a_meas, z_nb)
        X_own[k], Pd_own[k] = ekf_own.step(u, z_own, a_meas, None)
        R_t[k], V_t[k], _ = truth_kinematics(t[k])
        Z_log[k] = z_own
 
    tail = t >= 5.0
    def rmse(X): return np.sqrt(np.mean((X[tail, 0]-M_TRUE)**2))
    within = np.abs(X_dec[:, 0]-M_TRUE) < 0.05*M_TRUE
    t95 = t[np.argmax(within)] if within.any() else np.inf
    print(f"Decentralized EKF | M_true = {M_TRUE} kg | 10 s | Ts = {Ts*1000:.0f} ms")
    print(f"  mass, own+neighbor : final {X_dec[-1,0]:.4f} kg, "
          f"steady RMSE {rmse(X_dec):.4f} kg")
    print(f"  mass, own only     : final {X_own[-1,0]:.4f} kg, "
          f"steady RMSE {rmse(X_own):.4f} kg")
    print(f"  time to within 5%  : {t95:.2f} s")
    e_r = np.linalg.norm(X_dec[:, 1:4]-R_t, axis=1)
    e_v = np.linalg.norm(X_dec[:, 4:7]-V_t, axis=1)
    print(f"  CoG error  final {e_r[-1]*1000:.2f} mm, "
          f"steady RMS {np.sqrt(np.mean(e_r[tail]**2))*1000:.2f} mm")
    print(f"  vel error  steady RMS {np.sqrt(np.mean(e_v[tail]**2)):.4f} m/s")
 
    fig, (ax, axt, axz) = plt.subplots(3, 1, figsize=(10, 10),
                                       gridspec_kw={"height_ratios": [2, 1.2, 1.2]})
    sd = np.sqrt(Pd_dec[:, 0])
    for a in (ax, axt):
        a.axhline(M_TRUE, color="k", lw=1.2, ls=":", label="true mass 0.8 kg")
        a.fill_between(t, X_dec[:, 0]-2*sd, X_dec[:, 0]+2*sd,
                       color="#1f77b4", alpha=0.18, label=r"$\pm2\sigma$ (dec.)")
        a.plot(t, X_dec[:, 0], color="#1f77b4", lw=1.8,
               label="decentralized (own + delayed neighbor)")
        a.plot(t, X_own[:, 0], color="#d62728", lw=1.3, ls="--",
               label="own measurements only")
        a.grid(alpha=0.3)
    ax.axhline(x0[0], color="0.5", lw=1, ls="-.", label="initial guess 0.4 kg")
    ax.set_ylabel("estimated mass [kg]")
    ax.set_title("Grasped-object mass estimate convergence "
                 "(VTOL drifting sinusoidally, $M_0$ = 0.4 kg)")
    ax.legend(loc="lower right", fontsize=8)
    axt.set_xlim(0, 0.5); axt.set_ylim(0.3, 0.95)
    axt.set_ylabel("mass [kg]")
    axt.set_title("Transient detail (first 0.5 s): mass is directly observable "
                  "in the force channel, so convergence takes ~1-2 samples")
 
    axz.axhline(0, color="k", lw=1.2, ls=":")
    axz.plot(t, (X_dec[:, 0]-M_TRUE)*1000, color="#1f77b4", lw=1.4,
             label="decentralized")
    axz.plot(t, (X_own[:, 0]-M_TRUE)*1000, color="#d62728", lw=1.1, ls="--",
             label="own only")
    axz.set_xlabel("time [s]"); axz.set_ylabel("mass error [g]")
    axz.set_ylim(-40, 40); axz.grid(alpha=0.3); axz.legend(fontsize=8)
    axz.set_title("Steady-state detail")
    fig.tight_layout(); fig.savefig("dekf_mass_convergence.png", dpi=150)
 
    fig2, (a1, a2, a3) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for i, lab in enumerate("xyz"):
        a1.plot(t, (X_dec[:, 1+i]-R_t[:, i])*1000, label=f"r{lab}")
        a2.plot(t, X_dec[:, 4+i]-V_t[:, i], label=f"v{lab}")
    for i, lab in enumerate("xyz"):
        a3.plot(t, Z_log[:, 3+i], ".", ms=2, alpha=0.35, color=f"C{i}")
        a3.plot(t, X_dec[:, 0]*(G_VEC[i]+AMP[i]*W[i]**2*np.sin(W[i]*t)),
                lw=1.3, color=f"C{i}", label=f"F{lab} est")
    a1.set_ylabel("CoG error [mm]"); a1.set_title("CoG position error")
    a2.set_ylabel("velocity error [m/s]"); a2.set_title("Velocity error")
    a3.set_ylabel("tip force [N]"); a3.set_xlabel("time [s]")
    a3.set_title("Force channel: measurements (dots) vs estimated $M(g-a)$")
    for a in (a1, a2, a3):
        a.grid(alpha=0.3); a.legend(loc="upper right", fontsize=8, ncol=3)
    fig2.tight_layout(); fig2.savefig("dekf_errors.png", dpi=150)
 
    print("\nSaved: dekf_mass_convergence.png, dekf_errors.png")
 
if __name__ == "__main__":
    run()

