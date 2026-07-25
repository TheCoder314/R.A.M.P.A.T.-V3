import numpy as np
from collections import deque
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


rng = np.random.default_rng(42)


G_VEC = np.array([0.0, 0.0, -9.81])
M_TRUE = 0.8                        # True object mass [kg]
Ts = 0.05                           # Controller & EKF step [s] (20 Hz)
T_END = 10.0
DLY = 1                             # 50ms latency at 20Hz = 1 step delay


K_SHARE = 0.2                       # Coupling gain
T_BIAS = 3.0                        # Tendon bias tension [N]
SHARE_1 = 0.6                       # VTOL 1 is the 'anchor' (closer to CoG)
SHARE_2 = 0.4                       # VTOL 2 takes the remaining load


# Mock Jacobian mapping 3D Cartesian tip force to 3 Tendon tensions
# (Simplified nominal straight-arm Jacobian transpose)
J_T = np.array([
    [ 1.0,  0.0, -0.5],
    [-0.5,  0.866, -0.5],
    [-0.5, -0.866, -0.5]
])


NX, NZ = 7, 6




AMP = np.array([0.05, 0.05, 0.0])   # 5cm drift in XY
FREQ = 0.5                          # 0.5 Hz
W = 2 * np.pi * FREQ


def vtol_kinematics(t, phase_offset=0.0):
    """Simulates the sinusoidal drift of the VTOL bases."""
    pos = AMP * np.sin(W * t + phase_offset)
    vel = AMP * W * np.cos(W * t + phase_offset)
    acc = -AMP * W**2 * np.sin(W * t + phase_offset)
    return pos, vel, acc


class ObjectEKF:
    def __init__(self, x0, P0):
        self.x = x0.copy()
        self.P = P0.copy()
        self.Q = np.diag([1e-4, 1e-6,1e-6,1e-6, 1e-4,1e-4,1e-4])
        self.R = 0.01 * np.eye(NZ)
   
    def predict(self, u, dt=Ts):
        M, r, v = self.x[0], self.x[1:4], self.x[4:7]
        # State transition
        self.x[1:4] = r + v * dt
        self.x[4:7] = v + (u + M * G_VEC) / M * dt
       
        # Jacobian F
        F = np.eye(NX)
        F[1:4, 4:7] = np.eye(3) * dt
        F[4:7, 0] = -u / (M**2) * dt
       
        self.P = F @ self.P @ F.T + self.Q


    def update(self, z, a_meas):
        M, r, v = self.x[0], self.x[1:4], self.x[4:7]
       
        # Observation model h(x)
        z_hat = np.concatenate([r, M * (G_VEC - a_meas)])
       
        # Observation Jacobian H
        H = np.zeros((NZ, NX))
        H[0:3, 1:4] = np.eye(3)
        H[3:6, 0] = G_VEC - a_meas
       
        # EKF Update equations
        y = z - z_hat
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
       
        self.x = self.x + K @ y
        self.x[0] = max(self.x[0], 0.1)  # Constrain mass > 0
        self.P = (np.eye(NX) - K @ H) @ self.P


def run_simulation():
    n_steps = int(T_END / Ts)
    t_arr = np.arange(n_steps) * Ts
   
    x0 = np.array([0.3, 0,0,0, 0,0,0]) # Initial guess: 0.3 kg
    P0 = np.diag([0.1, 0.01,0.01,0.01, 0.05,0.05,0.05])
   
    ekf1 = ObjectEKF(x0, P0)
    ekf2 = ObjectEKF(x0, P0)
   
    f1_tx = deque([np.zeros(3)] * DLY, maxlen=DLY)
    f2_tx = deque([np.zeros(3)] * DLY, maxlen=DLY)
   
    log_M = np.zeros((n_steps, 2))
    log_Fz = np.zeros((n_steps, 2))
    log_T = np.zeros((n_steps, 6)) # Tensions (3 for V1, 3 for V2)
   
    F_apply_1 = np.zeros(3)
    F_apply_2 = np.zeros(3)
   
    for k, t in enumerate(t_arr):
        _, _, a1 = vtol_kinematics(t, phase_offset=0.0)
        _, _, a2 = vtol_kinematics(t, phase_offset=np.pi)
       
        a_obj = (a1 + a2) / 2.0  # Rigidly linked, avg acceleration
        a_meas1 = a_obj + rng.normal(0, 0.02, 3)
        a_meas2 = a_obj + rng.normal(0, 0.02, 3)
       
        F2_shared = f2_tx.popleft()
        F1_shared = f1_tx.popleft()
       
        M1_est = ekf1.x[0]
        M2_est = ekf2.x[0]
       
        F_des1 = SHARE_1 * M1_est * (-G_VEC) + K_SHARE * (F2_shared - F_apply_1)
        F_des2 = SHARE_2 * M2_est * (-G_VEC) + K_SHARE * (F1_shared - F_apply_2)
       
        T1 = J_T @ F_des1 + T_BIAS
        T2 = J_T @ F_des2 + T_BIAS
       
        F_apply_1 = F_des1 + rng.normal(0, 0.1, 3)
        F_apply_2 = F_des2 + rng.normal(0, 0.1, 3)
       
        # Enqueue for next step (Network TX)
        f1_tx.append(F_apply_1.copy())
        f2_tx.append(F_apply_2.copy())
       
        ekf1.predict(F_apply_1, Ts)
        ekf2.predict(F_apply_2, Ts)
       
        z1 = np.concatenate([np.zeros(3), F_apply_1]) # Ignoring position z for brevity
        z2 = np.concatenate([np.zeros(3), F_apply_2])
       
        ekf1.update(z1, a_meas1)
        ekf2.update(z2, a_meas2)
       
        # Log data
        log_M[k] = [M1_est, M2_est]
        log_Fz[k] = [F_apply_1[2], F_apply_2[2]]
        log_T[k, :3] = T1
        log_T[k, 3:] = T2
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
   
    # Plot 1: Mass Convergence
    axs[0].axhline(M_TRUE, color='k', ls='--', label='True Mass (0.8kg)')
    axs[0].plot(t_arr, log_M[:, 0], lw=2, color='#1f77b4', label='VTOL 1 Estimate (Anchor)')
    axs[0].plot(t_arr, log_M[:, 1], lw=2, color='#ff7f0e', label='VTOL 2 Estimate')
    axs[0].set_title('Decentralized EKF Mass Estimation')
    axs[0].set_ylabel('Mass [kg]')
    axs[0].grid(alpha=0.3)
    axs[0].legend()
   
    # Plot 2: Load Distribution (Fz)
    axs[1].plot(t_arr, log_Fz[:, 0], color='#1f77b4', label=f'VTOL 1 Z-Force (Target: {SHARE_1*100}%)')
    axs[1].plot(t_arr, log_Fz[:, 1], color='#ff7f0e', label=f'VTOL 2 Z-Force (Target: {SHARE_2*100}%)')
    axs[1].plot(t_arr, log_Fz[:, 0] + log_Fz[:, 1], color='k', ls=':', label='Total System Force')
    axs[1].set_title('Consensus Load Balancing via Delayed Wireless Sharing')
    axs[1].set_ylabel('Force Z [N]')
    axs[1].grid(alpha=0.3)
    axs[1].legend()
   
    # Plot 3: Tendon Tensions
    axs[2].plot(t_arr, log_T[:, 0], color='r', label='V1 Tendon 1')
    axs[2].plot(t_arr, log_T[:, 1], color='g', label='V1 Tendon 2')
    axs[2].plot(t_arr, log_T[:, 2], color='b', label='V1 Tendon 3')
    axs[2].axhline(T_BIAS, color='k', ls=':', label='Bias Tension (3N)')
    axs[2].set_title('VTOL 1: Commanded Tendon Tensions ($T = J^T F_{des} + T_{bias}$)')
    axs[2].set_xlabel('Time [s]')
    axs[2].set_ylabel('Tension [N]')
    axs[2].grid(alpha=0.3)
    axs[2].legend(loc='lower right', ncol=4)
   
    fig.tight_layout()
    fig.savefig("dual_vtol_consensus.png", dpi=150)
    print("Simulation complete. Saved: dual_vtol_consensus.png")


if __name__ == "__main__":
    run_simulation()

