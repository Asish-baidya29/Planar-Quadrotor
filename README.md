What this demonstrates (maps to a Control & Simulation role):
  1. Rigid-body dynamics model (planar 3-DOF: x, z, pitch) with drag, wind, CG offset
  2. Cascaded autopilot: altitude PID -> position PD -> attitude PID
  3. Linearization about hover -> transfer function -> Bode / Nyquist / margins
  4. Monte Carlo robustness: mass, inertia, CG offset, wind randomised
