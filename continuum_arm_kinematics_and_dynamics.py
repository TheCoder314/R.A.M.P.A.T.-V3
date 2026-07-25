import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


L       = 0.5                 # arm (backbone) length [m]
r       = 0.02                # tendon routing radius [m]
alpha   = np.deg2rad(45.0)    # helix angle [rad]
r_eff   = r * np.cos(alpha)   # effective moment arm of helical tendons [m]


k_t     = 28000.0             # tendon stiffness [N/m]
m_t     = 0.1                 # tendon mass [kg]  (folded into actuator inertia)
M       = 0.5 + m_t           # effective moving mass per tendon [kg]
B       = 2.0                 # viscous damping [N s/m]


PSI = np.array([0.0, 2*np.pi/3, 4*np.pi/3])   # tendon angular positions


def tendon_lengths_from_arc(kappa, phi):
    return L - r_eff * (kappa * L) * np.cos(PSI - phi)


def arc_from_tendon_lengths(l):
    l = np.asarray(l, dtype=float)
    dl = L - l                                    # contractions
    # dl_i = r_eff*kappa*L*cos(psi_i - phi)  ->  resolve into two components
    a = (2.0/3.0) * np.sum(dl * np.cos(PSI))      # = r_eff*kappa*L*cos(phi)
    b = (2.0/3.0) * np.sum(dl * np.sin(PSI))      # = r_eff*kappa*L*sin(phi)
    mag = np.hypot(a, b)
    kappa = mag / (r_eff * L)
    phi = np.arctan2(b, a) if mag > 1e-12 else 0.0
    return kappa, phi


def rot_z(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rot_y(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rpy_from_R(R):
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    if np.abs(np.cos(pitch)) > 1e-9:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw  = np.arctan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        roll = 0.0
        yaw  = np.arctan2(-R[0, 1], R[1, 1])
    return roll, pitch, yaw


def forward_kinematics(l):
    """
    Tendon lengths (l1,l2,l3) -> tip pose (x,y,z, roll,pitch,yaw)
    plus the arc parameters for convenience.
    """
    kappa, phi = arc_from_tendon_lengths(l)
    theta = kappa * L
    if kappa < 1e-9:                      # straight configuration
        p = np.array([0.0, 0.0, L])
        R = np.eye(3)
    else:
        rad = 1.0 / kappa
        p = np.array([rad * (1 - np.cos(theta)) * np.cos(phi),
                      rad * (1 - np.cos(theta)) * np.sin(phi),
                      rad * np.sin(theta)])
        R = rot_z(phi) @ rot_y(theta) @ rot_z(-phi)
    roll, pitch, yaw = rpy_from_R(R)
    return np.array([*p, roll, pitch, yaw]), (kappa, phi)


def inverse_kinematics(x, y, z):
    """
    Desired tip position -> tendon lengths (l1,l2,l3).
    Uses the closed-form CC inversion; orientation follows from position
    for a single CC section (only 2 DoF: kappa, phi).
    Raises ValueError if the point is unreachable by a CC arc of length L.
    """
    rho2 = x*x + y*y
    if rho2 < 1e-14:                      # straight arm
        if not np.isclose(z, L, atol=1e-6):
            raise ValueError("On-axis points are only reachable at z = L.")
        return tendon_lengths_from_arc(0.0, 0.0), (0.0, 0.0)
    kappa = 2.0 * np.sqrt(rho2) / (rho2 + z*z)
    phi = np.arctan2(y, x)
    theta = np.arctan2(kappa * z, 1.0 - kappa * np.sqrt(rho2))
    if theta < 0:
        theta += 2*np.pi
    if not np.isclose(theta / kappa, L, rtol=1e-3):
        raise ValueError("Point not on a constant-curvature arc of length L.")
    return tendon_lengths_from_arc(kappa, phi), (kappa, phi)


def backbone_points(kappa, phi, n=40):
    """Points along the backbone arc (for drawing arm shapes)."""
    s = np.linspace(0, L, n)
    if kappa < 1e-9:
        return np.column_stack([np.zeros(n), np.zeros(n), s])
    th = kappa * s
    rad = 1.0 / kappa
    return np.column_stack([rad * (1 - np.cos(th)) * np.cos(phi),
                            rad * (1 - np.cos(th)) * np.sin(phi),
                            rad * np.sin(th)])


def tendon_dynamics(state, x_des):
    """state = [x1,x2,x3, v1,v2,v3]; returns d(state)/dt."""
    x, v = state[:3], state[3:]
    F_spring = k_t * (x_des - x)          # spring pulls x toward command
    acc = (F_spring - B * v) / M
    return np.concatenate([v, acc])


def rk4_step(state, x_des, dt):
    k1 = tendon_dynamics(state,             x_des)
    k2 = tendon_dynamics(state + dt/2 * k1, x_des)
    k3 = tendon_dynamics(state + dt/2 * k2, x_des)
    k4 = tendon_dynamics(state + dt * k3,   x_des)
    return state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)


def commanded_arc(t, t_bend=1.0, t_total=5.0, theta_max=np.deg2rad(60)):
    """Smoothly bend up, then rotate the bending plane 360 degrees."""
    if t < t_bend:                        # quintic-ish ramp in curvature
        s = t / t_bend
        theta = theta_max * (10*s**3 - 15*s**4 + 6*s**5)
        phi = 0.0
    else:
        theta = theta_max
        phi = 2*np.pi * (t - t_bend) / (t_total - t_bend)
    return theta / L, phi                 # kappa, phi


def simulate(t_total=5.0, dt=1e-3):
    n = int(t_total / dt) + 1
    t_arr = np.linspace(0, t_total, n)


    x_log  = np.zeros((n, 3))   # tendon displacements  (L - l)
    v_log  = np.zeros((n, 3))   # tendon velocities
    F_log  = np.zeros((n, 3))   # tendon forces
    l_log  = np.zeros((n, 3))   # tendon lengths
    ld_log = np.zeros((n, 3))   # desired tendon lengths
    pose_log = np.zeros((n, 6)) # x,y,z,roll,pitch,yaw


    state = np.zeros(6)         # start straight, at rest
    for i, t in enumerate(t_arr):
        kappa_d, phi_d = commanded_arc(t, t_total=t_total)
        l_des = tendon_lengths_from_arc(kappa_d, phi_d)
        x_des = L - l_des                       # commanded contraction


        x, v = state[:3], state[3:]
        F = k_t * (x_des - x)                   # tendon (spring) force
        l_act = L - x                           # actual tendon lengths
        pose, _ = forward_kinematics(l_act)


        x_log[i], v_log[i], F_log[i] = x, v, F
        l_log[i], ld_log[i], pose_log[i] = l_act, l_des, pose


        state = rk4_step(state, x_des, dt)


    return t_arr, x_log, v_log, F_log, l_log, ld_log, pose_log


def main():
    l_test = tendon_lengths_from_arc(2.0, np.deg2rad(30))
    pose, (k_, p_) = forward_kinematics(l_test)
    l_back, _ = inverse_kinematics(*pose[:3])
    print("FK/IK round-trip check")
    print(f"  tendon lengths in : {l_test}")
    print(f"  tip pose (x,y,z)  : {pose[:3]}")
    print(f"  tip rpy [deg]     : {np.rad2deg(pose[3:])}")
    print(f"  tendon lengths out: {l_back}")
    print(f"  max error         : {np.max(np.abs(l_back - l_test)):.2e} m\n")
    t, x, v, F, l, ld, pose = simulate()
    print("Simulation complete: 5 s, dt = 1 ms")
    print(f"  final tip pose  (x,y,z) [m]  : {pose[-1, :3]}")
    print(f"  final tip rpy [deg]          : {np.rad2deg(pose[-1, 3:])}")
    print(f"  peak tendon force [N]        : {np.max(np.abs(F)):.2f}")
    print(f"  peak tendon speed [m/s]      : {np.max(np.abs(v)):.3f}")


    colors = ["#d62728", "#2ca02c", "#1f77b4"]


    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(pose[:, 0], pose[:, 1], pose[:, 2], "b-", lw=2, label="tip trajectory")
    ax.scatter(*pose[0, :3],  c="g", s=60, label="start")
    ax.scatter(*pose[-1, :3], c="r", s=60, label="end")
    for frac in np.linspace(0, 1, 8, endpoint=False):     # arm snapshots
        idx = int(frac * (len(t) - 1))
        kappa_i, phi_i = arc_from_tendon_lengths(l[idx])
        bb = backbone_points(kappa_i, phi_i)
        ax.plot(bb[:, 0], bb[:, 1], bb[:, 2], "k-", alpha=0.25, lw=1)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title("3-Tendon Continuum Arm: 3D Tip Trajectory\n(bend to 60°, then sweep bending plane 360°)")
    ax.legend()
    lim = 0.55
    ax.set_xlim(-lim/2, lim/2); ax.set_ylim(-lim/2, lim/2); ax.set_zlim(0, lim)
    fig.tight_layout()
    fig.savefig("tip_trajectory_3d.png", dpi=150)


    fig2, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for i in range(3):
        axs[0].plot(t, l[:, i]*1000, color=colors[i], label=f"l{i+1} actual")
        axs[0].plot(t, ld[:, i]*1000, "--", color=colors[i], alpha=0.5)
        axs[1].plot(t, v[:, i]*1000, color=colors[i], label=f"tendon {i+1}")
        axs[2].plot(t, F[:, i], color=colors[i], label=f"tendon {i+1}")
    axs[0].set_ylabel("tendon length [mm]")
    axs[0].set_title("Tendon lengths (solid = actual, dashed = commanded)")
    axs[1].set_ylabel("velocity [mm/s]"); axs[1].set_title("Tendon velocities")
    axs[2].set_ylabel("force [N]"); axs[2].set_title("Tendon forces")
    axs[2].set_xlabel("time [s]")
    for a in axs:
        a.grid(alpha=0.3); a.legend(loc="upper right", fontsize=8)
    fig2.tight_layout()
    fig2.savefig("tendon_states.png", dpi=150)


    fig3, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for i, lab in enumerate(["x", "y", "z"]):
        a1.plot(t, pose[:, i], label=lab)
    a1.set_ylabel("position [m]"); a1.set_title("Tip position")
    for i, lab in enumerate(["roll", "pitch", "yaw"]):
        a2.plot(t, np.rad2deg(pose[:, 3 + i]), label=lab)
    a2.set_ylabel("angle [deg]"); a2.set_title("Tip orientation (ZYX RPY)")
    a2.set_xlabel("time [s]")
    for a in (a1, a2):
        a.grid(alpha=0.3); a.legend(loc="upper right")
    fig3.tight_layout()
    fig3.savefig("tip_pose_time.png", dpi=150)


    print("\nSaved: tip_trajectory_3d.png, tendon_states.png, tip_pose_time.png")


if __name__ == "__main__":
    main()

