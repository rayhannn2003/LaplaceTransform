import numpy as np
import matplotlib.pyplot as plt

class SmartIrrigation:
    def __init__(self, a=0.5, b=1.0, t_max=20, dt=0.01):
        pass

    def u_step(self):
        pass
    def u_ramp(self):
        pass
    def u_sin(self):
        pass
    def u_exponential(self): 
        pass
    def u_pulse(self):
        pass

    def laplace_transform(self, f, s):
        pass

    def inverse_laplace(self, s_list, F_s_values):
        pass

    def H_s(self, s, U_s):
        pass
    def steady_state(self, h):
        """Mean of last 5% of signal."""
        pass

    def time_constant(self, h):
        """Time to first reach 63.2% of steady-state."""
        pass

    def rise_time(self, h):
        """Time to go from 10% to 90% of steady-state."""
        pass

    def settling_time(self, h):
        """Time after which h(t) stays permanently within ±2% of h_ss."""
        pass

    def overshoot(self, h):
        """Percentage overshoot: (h_max - h_ss) / h_ss * 100."""
        pass

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
    "Sinusoidal Input":  system.u_sin(),
    "Exponential Input": system.u_exponential(),
    "Pulse Input":       system.u_pulse(),
}

# Bromwich contour parameters, set these values
c = None 
W = None
N = None
s_list = None

colors = ['#2196F3', '#4CAF50', '#FF5722', '#9C27B0', '#FF9800']

for idx, (name, u) in enumerate(inputs.items()):
    print(f"Processing: {name}...")

    # --- Laplace --- set these values
    U_s_vals = None 
    H_s_vals = None
    h_laplace = None
    print(f"\n  ► {name}")
    metrics = None
    for k, v in metrics.items():
        print(f"      {k.replace('_',' ').title():<22}: {v}")

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

    """

     fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Smart Irrigation — {name}", fontsize=13, fontweight='bold')

    for ax, data, title, color in zip(axes, [h_laplace, h_euler], 
                                     ["Laplace Transform Simulation", "Euler Method Simulation"],
                                     [colors[idx], 'tomato']):
        ax.plot(system.t, u, 'b--', lw=1.2, label="Input u(t)")
        ax.plot(system.t, data, color=color, lw=2.2, label="Output h(t)")
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Water Level")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
 
 
     """