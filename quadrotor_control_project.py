"""
-------------------------------------------------------------------------------------------
Planar Quadrotor: Nonlinear Simulation, PID Autopilot, Frequency Analysis, Monte Carlo
-------------------------------------------------------------------------------------------
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# 1. Nominal parameters and controller gains
# ----------------------------------------------------------------------------
G = 9.81
NOM = dict(m=1.0, J=0.02, d=0.1)          # mass [kg], inertia [kg m^2], linear drag [N s/m]

# Attitude loop (fast): wn=15 rad/s, zeta=0.8, small integral for CG-torque bias
KP_TH, KD_TH, KI_TH = 15.0**2, 2*0.8*15.0, 20.0
# Position loop (slow, ~5x slower than attitude)
KP_X, KD_X, KI_X = 2.5**2, 2*0.9*2.5, 1.0
# Altitude loop
KP_Z, KD_Z, KI_Z = 5.0**2, 2*0.9*5.0, 5.0

TH_MAX = 0.35        # tilt command limit [rad]
T_MAX = 25.0         # max thrust [N]
TAU_MAX = 2.0        # max torque [N m]


# ----------------------------------------------------------------------------
# 2. Plant + controller (state includes the controller integrators)
#    s = [x, z, th, vx, vz, w, int_z, int_th, int_x]
# ----------------------------------------------------------------------------
def deriv(s, ref, p, dist):
    x, z, th, vx, vz, w, iz, ith, ix = s
    xr, zr = ref
    m, J, d = p["m"], p["J"], p["d"]
    fw, tau_cg = dist                          # wind force [N], CG-offset torque [N m]

    # --- outer loops ---
    ax_cmd = KP_X*(xr - x) - KD_X*vx + KI_X*ix
    th_cmd = np.clip(-ax_cmd / G, -TH_MAX, TH_MAX)       # +x accel needs negative pitch here
    az_cmd = KP_Z*(zr - z) - KD_Z*vz + KI_Z*iz
    T = np.clip(m*(G + az_cmd)/max(np.cos(th), 0.5), 0.0, T_MAX)

    # --- inner loop ---
    e_th = th_cmd - th
    tau = np.clip(J*(KP_TH*e_th - KD_TH*w + KI_TH*ith), -TAU_MAX, TAU_MAX)

    # --- rigid-body dynamics ---
    ax = (-T*np.sin(th) - d*vx + fw) / m
    az = (T*np.cos(th) - m*G - d*vz) / m
    alpha = (tau + tau_cg) / J
    # Conditional-integration anti-windup: the x-integrator only accumulates near the target,
    # so the large step (and tilt saturation) cannot wind it up.
    dix = (xr - x) if abs(xr - x) < 1.0 else 0.0
    return np.array([vx, vz, w, ax, az, alpha, zr - z, e_th, dix])


def simulate(p, dist_fn, t_end=15.0, dt=0.005, ref=(5.0, 2.0)):
    n = int(t_end/dt)
    s = np.zeros(9)
    out = np.zeros((n, 9))
    ts = np.arange(n)*dt
    for k in range(n):
        dist = dist_fn(ts[k])
        k1 = deriv(s, ref, p, dist)
        k2 = deriv(s + 0.5*dt*k1, ref, p, dist)
        k3 = deriv(s + 0.5*dt*k2, ref, p, dist)
        k4 = deriv(s + dt*k3, ref, p, dist)
        s = s + dt*(k1 + 2*k2 + 2*k3 + k4)/6
        out[k] = s
    return ts, out


def metrics(ts, out, t_gust, xr=5.0, zr=2.0):
    """Step-response metrics are taken BEFORE the gust; gust effect is reported separately."""
    pre = ts < t_gust
    x, z, th = out[:, 0], out[:, 1], out[:, 2]
    overshoot = max(0.0, (x[pre].max() - xr)/xr*100)
    outside = np.where(np.abs(x[pre] - xr) > 0.05*xr)[0]          # 5 % band
    settle = ts[pre][outside[-1]] if len(outside) else 0.0
    gust_dev = np.abs(x[~pre] - xr).max()                            # worst deviation after gust hits
    return dict(overshoot=overshoot, settle=settle, gust_dev=gust_dev,
                z_err=abs(z[-1] - zr), x_err=abs(x[-1] - xr), peak_theta=np.abs(th).max())


# ----------------------------------------------------------------------------
# 3. Linearization and frequency-domain analysis (attitude loop)
#    Hover: theta'' = tau/J.  With C(s) = kp + kd s + ki/s on the normalised plant 1/s^2:
#       L(s) = (kd s^2 + kp s + ki) / s^3
# ----------------------------------------------------------------------------
def attitude_L(w):
    s = 1j*w
    return (KD_TH*s**2 + KP_TH*s + KI_TH) / s**3


def margins():
    w = np.logspace(-1, 3, 200000)
    L = attitude_L(w)
    mag = np.abs(L)
    ph = (np.angle(L, deg=True) % 360) - 360                  # keep phase in (-360, 0]
    i_gc = np.where(np.diff(np.sign(mag - 1)))[0][0]          # gain crossover (|L| = 1)
    pm = 180 + ph[i_gc]
    cross = np.where(np.diff(np.sign(ph + 180)))[0]           # phase crosses -180 deg
    gm = None
    if len(cross):
        # With an integrator the loop is conditionally stable: |L| > 1 where phase = -180,
        # so the meaningful number is how much the loop gain can be REDUCED (dB).
        gm = 20*np.log10(mag[cross[0]])
    return w, mag, ph, w[i_gc], pm, gm


# ----------------------------------------------------------------------------
# 4. Run everything
# ----------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(42)

    # --- nominal run with a wind gust and small CG offset ---
    T_GUST = 10.0
    gust = lambda t: (1.0 if T_GUST <= t < T_GUST + 1.0 else 0.0, 0.03)
    ts, out = simulate(NOM, gust, t_end=20.0)
    nom = metrics(ts, out, T_GUST)
    print("NOMINAL  overshoot %.1f %% | 5%% settle %.2f s | worst deviation after 1 N gust %.2f m"
          " | final x/z error %.3f / %.3f m | peak pitch %.1f deg"
          % (nom["overshoot"], nom["settle"], nom["gust_dev"], nom["x_err"], nom["z_err"],
             np.degrees(nom["peak_theta"])))

    # --- frequency analysis ---
    w, mag, ph, wc, pm, gm = margins()
    print("ATTITUDE LOOP  crossover %.1f rad/s | phase margin %.1f deg | gain-reduction margin %s"
          % (wc, pm, "n/a" if gm is None else "%.1f dB (loop is conditionally stable because of the integrator)" % gm))

    # --- Monte Carlo ---
    N = 200
    rows = []
    for _ in range(N):
        p = dict(m=NOM["m"]*rng.uniform(0.9, 1.1),
                 J=NOM["J"]*rng.uniform(0.85, 1.15),
                 d=NOM["d"]*rng.uniform(0.5, 1.5))
        fw0 = rng.uniform(-1.0, 1.0)                       # steady wind force
        tcg = rng.uniform(-0.05, 0.05)                     # CG-offset torque
        gd = lambda t, fw0=fw0, tcg=tcg: (fw0 + (1.0 if T_GUST <= t < T_GUST + 1 else 0.0), tcg)
        t2, o2 = simulate(p, gd, t_end=20.0)
        rows.append(metrics(t2, o2, T_GUST))
    ov = np.array([r["overshoot"] for r in rows])
    st = np.array([r["settle"] for r in rows])
    gdv = np.array([r["gust_dev"] for r in rows])
    ze = np.array([r["z_err"] for r in rows])
    xe = np.array([r["x_err"] for r in rows])
    print("MONTE CARLO (%d runs)  overshoot mean %.1f %% / 95th %.1f %% | 5%% settle mean %.2f s / 95th %.2f s"
          % (N, ov.mean(), np.percentile(ov, 95), st.mean(), np.percentile(st, 95)))
    print("                       worst gust deviation %.2f m | worst final x error %.3f m | worst final z error %.3f m"
          % (gdv.max(), xe.max(), ze.max()))

    # --- plots ---
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    ax[0, 0].plot(ts, out[:, 0], label="x"); ax[0, 0].axhline(5, ls="--", c="k", lw=.7)
    ax[0, 0].plot(ts, out[:, 1], label="z"); ax[0, 0].axhline(2, ls="--", c="k", lw=.7)
    ax[0, 0].set(title="Position response (gust at 10-11 s)", xlabel="t [s]", ylabel="m"); ax[0, 0].legend()
    ax[0, 1].plot(ts, np.degrees(out[:, 2])); ax[0, 1].set(title="Pitch angle", xlabel="t [s]", ylabel="deg")
    ax[0, 2].semilogx(w, 20*np.log10(mag)); ax[0, 2].axhline(0, c="k", lw=.7)
    ax[0, 2].set(title="Bode magnitude, attitude loop", xlabel="rad/s", ylabel="dB")
    ax[1, 0].semilogx(w, ph); ax[1, 0].axhline(-180, c="r", ls="--", lw=.7)
    ax[1, 0].set(title="Bode phase (PM = %.0f deg)" % pm, xlabel="rad/s", ylabel="deg")
    Lw = attitude_L(np.logspace(-1, 3, 5000))
    ax[1, 1].plot(Lw.real, Lw.imag); ax[1, 1].plot(Lw.real, -Lw.imag, ls=":")
    ax[1, 1].plot(-1, 0, "rx"); ax[1, 1].set(title="Nyquist, attitude loop", xlim=(-4, 2), ylim=(-4, 4))
    ax[1, 2].hist(ov, bins=20); ax[1, 2].set(title="Monte Carlo: x overshoot [%]", xlabel="%", ylabel="runs")
    plt.tight_layout(); plt.savefig("quadrotor_results.png", dpi=110)
    print("Saved quadrotor_results.png")


if __name__ == "__main__":
    main()
