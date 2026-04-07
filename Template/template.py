import numpy as np
import matplotlib.pyplot as plt

class SmartIrrigation:
    def __init__(self, a=0.5, b=1.0, t_max=20, dt=0.01):
        self.a = a
        self.b = b
        self.t_max = t_max
        self.dt = dt
        self.t = np.arange(0, t_max, dt)

    def u_step(self):
        return np.ones_like(self.t)

    def u_ramp(self):
        return self.t.copy()

    def u_sin(self, omega=1.0):
        return np.sin(omega * self.t)

    def u_exponential(self, decay=0.5): 
        # return np.exp(-decay * self.t)
        """Exponential Input: u(t) = 1 - e^(-0.3t) [cite: 39-40]."""
        return 1 - np.exp(-0.3 * self.t)

    def u_pulse(self, start=2.0, end=5.0):
        u = np.zeros_like(self.t)
        u[(self.t >= start) & (self.t <= end)] = 1.0
        return u

    def laplace_transform(self, f, s):
        """Numerical Laplace transform: F(s) = Integral[ f(t)*exp(-s*t) dt ]"""
        return np.sum(f * np.exp(-s * self.t)) * self.dt

    def inverse_laplace(self, s_list, F_s_values):
        """Numerical Inverse Laplace transform using Bromwich integral over the contour"""
        ds = s_list[1] - s_list[0]
        h_t = np.zeros_like(self.t)
        for i, t_val in enumerate(self.t):
            integrand = F_s_values * np.exp(s_list * t_val)
            h_t[i] = np.real((1 / (2j * np.pi)) * np.sum(integrand * ds))
        return h_t

    def H_s(self, s, U_s):
        """Transfer function H(s) = b / (s + a) => out = H(s) * U(s)"""
        return (self.b / (s + self.a)) * U_s

    def steady_state(self, h):
        """Mean of last 5% of signal."""
        n_tail = max(1, int(0.05 * len(h)))
        return np.mean(h[-n_tail:])

    def time_constant(self, h):
        """Time to first reach 63.2% of steady-state."""
        h_ss = self.steady_state(h)
        target = 0.632 * h_ss
        idx = np.searchsorted(h, target)
        if idx < len(self.t):
            return self.t[idx]
        return None

    def rise_time(self, h):
        """Time to go from 10% to 90% of steady-state."""
        h_ss = self.steady_state(h)
        idx_10 = np.searchsorted(h, 0.1 * h_ss)
        idx_90 = np.searchsorted(h, 0.9 * h_ss)
        if idx_10 < len(self.t) and idx_90 < len(self.t):
            return self.t[idx_90] - self.t[idx_10]
        return None

    def settling_time(self, h):
        """Time after which h(t) stays permanently within ±2% of h_ss."""
        h_ss = self.steady_state(h)
        if h_ss == 0:
            return None
        # Error array
        err = np.abs((h - h_ss) / h_ss)
        # Find the last index where error > 0.02
        out_of_bounds = np.where(err > 0.02)[0]
        if len(out_of_bounds) == 0:
            return 0.0
        last_out_idx = out_of_bounds[-1]
        if last_out_idx + 1 < len(self.t):
            return self.t[last_out_idx + 1]
        return None

    def overshoot(self, h):
        """Percentage overshoot: (h_max - h_ss) / h_ss * 100."""
        h_ss = self.steady_state(h)
        if h_ss == 0:
            return 0.0
        h_max = np.max(h)
        os = (h_max - h_ss) / h_ss * 100.0
        return max(0.0, os)

    def compute_metrics(self, h):
        return {
            "steady_state":  self.steady_state(h),
            "time_constant": self.time_constant(h),
            "rise_time":     self.rise_time(h),
            "settling_time": self.settling_time(h),
            "overshoot_%":   self.overshoot(h),
        }

    def euler_simulate(self, u):
        """
        Euler method for dh/dt = -a*h(t) + b*u(t)
        h[n+1] = h[n] + dt * (-a*h[n] + b*u[n])
        """
        h = np.zeros_like(self.t)
        for n in range(len(self.t) - 1):
            dhdt = -self.a * h[n] + self.b * u[n]
            h[n + 1] = h[n] + self.dt * dhdt
        return h


#Change values of a, b to experiment with different system dynamics
system = SmartIrrigation(a=0.5, b=1.0, t_max=20, dt=0.01)

inputs = {
    "Step Input":        system.u_step(),
    "Ramp Input":        system.u_ramp(),
    "Sinusoidal Input":  system.u_sin(omega=0.5),
    "Exponential Input": system.u_exponential(decay=0.8),
    "Pulse Input":       system.u_pulse(start=2.0, end=6.0),
}

# Bromwich contour parameters, set these values
c = 0.1 # Real part, greater than all real parts of singularities
W = 100.0 # Frequency boundary
N = 5000 # Number of points
s_list = c + 1j * np.linspace(-W, W, N)

colors = ['#2196F3', '#4CAF50', '#FF5722', '#9C27B0', '#FF9800']

for idx, (name, u) in enumerate(inputs.items()):
    print(f"Processing: {name}...")

    # --- Laplace ---
    U_s_vals = np.array([system.laplace_transform(u, s) for s in s_list])
    H_s_vals = np.array([system.H_s(s, U_s) for s, U_s in zip(s_list, U_s_vals)])
    
    h_laplace = system.inverse_laplace(s_list, H_s_vals)

    print(f"\n  ► {name}")
    metrics = system.compute_metrics(h_laplace)
    for k, v in metrics.items():
        v_str = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"      {k.replace('_',' ').title():<22}: {v_str}")

    # --- Euler ---
    h_euler = system.euler_simulate(u)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    fig.suptitle(f"Smart Irrigation — {name}", fontsize=13, fontweight='bold')

    # Laplace subplot
    axes[0].plot(system.t, u, 'b--', lw=1.8, label="Input u(t)")
    axes[0].plot(system.t, h_laplace, color=colors[idx], lw=2.2, label="Output h(t)")
    axes[0].set_title("Laplace Transform Simulation", fontweight='bold')
    axes[0].set_xlabel("Time (s)", fontsize=11)
    axes[0].set_ylabel("Water Level / Input", fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Euler subplot
    axes[1].plot(system.t, u, 'b--', lw=1.8, label="Input u(t)")
    axes[1].plot(system.t, h_euler, color='tomato', lw=2.2, label="Output h(t)")
    axes[1].set_title("Euler Method Simulation", fontweight='bold')
    axes[1].set_xlabel("Time (s)", fontsize=11)
    axes[1].set_ylabel("Water Level / Input", fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
