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
        return 0.1 * self.t

    def u_sin(self):
        return np.sin(0.5 * self.t)

    def u_exponential(self): 
        # Exponential response
        random_decay = -0.3
        return 1 - np.exp(random_decay * self.t)

    def u_pulse(self):
        return np.where((self.t >= 0) & (self.t < 5), 1.0, 0.0)
    
    def u_corrupted(self,u):
        e_t = -0.02*self.t + 0.08*np.sin(8*self.t)
        return u + e_t

    def laplace_transform(self, f, s):
        integran = f * np.exp(-s * self.t)
        integral = (self.dt / 2.0) * (integran[0] + 2 * np.sum(integran[1:-1]) + integran[-1])
        return integral

    def H_s(self, s, U_s):
        return (self.b / (s + self.a)) * U_s

    def inverse_laplace(self, s_list, H_s_values):
        h_t = np.zeros_like(self.t, dtype=float)
        N = len(s_list)
        W = np.imag(s_list[-1])
        dw = 2 * W / N 
        
        for i, t_val in enumerate(self.t):
            summation_total = np.sum(H_s_values * np.exp(s_list * t_val))
            h_t[i] = (dw / (2 * np.pi)) * np.real(summation_total)
        return h_t


    def steady_state(self, h):
        """Mean of last 5% of signal."""
        last_5_percent = int(len(h) * 0.05)
        return np.mean(h[-last_5_percent:]) if last_5_percent > 0 else h[-1]

    def time_constant(self, h):
        """Time to reach 63.2% of steady-state."""
        h_ss = self.steady_state(h)
        target = 0.632 * h_ss
        idx = np.where(h >= target)[0]
        return self.t[idx[0]] if len(idx) > 0 else 0.0

    def rise_time(self, h):
        """Time from 10% to 90% of steady-state."""
        h_ss = self.steady_state(h)
        t10 = self.t[np.where(h >= 0.1 * h_ss)[0][0]]
        t90 = self.t[np.where(h >= 0.9 * h_ss)[0][0]]
        return t90 - t10

    def settling_time(self, h):
        """Time after which h(t) stays within +/- 2% of h_ss."""
        h_ss = self.steady_state(h)
        if h_ss == 0: return 0.0
        # Find all indices where it's outside the bound
        outside = np.where(np.abs(h - h_ss) > 0.02 * h_ss)[0]
        return self.t[outside[-1]] if len(outside) > 0 else 0.0

    def overshoot(self, h):
        """Percentage overshoot."""
        h_ss = self.steady_state(h)
        if h_ss == 0: return 0.0
        h_max = np.max(h)
        os = ((h_max - h_ss) / h_ss) * 100
        return max(0.0, os) 
    def ISE(self, h, u):
        """Integral of Squared Error."""
        h_ss = self.steady_state(h)
        error = h - h_ss
        return np.sum(error**2) * self.dt
    def response_energy(self, h):     
        """Energy of the response."""
        return np.sum(h**2) * self.dt   
    
    # ==========================================
    # 1. EXTENDED INPUTS & DISTURBANCES
    # ==========================================
    
    def u_impulse(self):
        """Approximation of Dirac Delta (area = 1) over a very short time."""
        u = np.zeros_like(self.t)
        # Apply a high value over the first two time steps
        u[0] = 1.0 / self.dt
        u[1] = 1.0 / self.dt
        return u

    def u_damped_sin(self):
        """Exponentially decaying sine wave: e^(-0.2*t) * sin(2*t)"""
        return np.exp(-0.2 * self.t) * np.sin(2 * self.t)
        
    def u_square_wave(self):
        """Periodic pump toggling on and off every 2 seconds."""
        # square wave using modulo arithmetic
        return np.where(self.t % 4 < 2, 1.0, 0.0)

    def error_white_noise(self, u, noise_level=0.05):
        """Adds Gaussian white noise (e.g., erratic sensor readings)."""
        noise = np.random.normal(0, noise_level, len(self.t))
        return u + noise

    def error_sensor_drift(self, u, drift_rate=0.01, bias=0.1):
        """Adds a constant offset + a slow linear drift to the input."""
        drift = bias + (drift_rate * self.t)
        return u + drift

    # ==========================================
    # 2. EXTENDED PERFORMANCE METRICS
    # ==========================================

    def steady_state(self, h):
        """Mean of last 5% of signal."""
        last_5_percent = int(len(h) * 0.05)
        return np.mean(h[-last_5_percent:]) if last_5_percent > 0 else h[-1]

    def integral_absolute_error(self, h):
        """IAE: Integral of |h_ss - h(t)| dt"""
        h_ss = self.steady_state(h)
        error = np.abs(h_ss - h)
        # Using Trapezoidal rule for integration (similar to Laplace requirement)
        return np.trapz(error, dx=self.dt)

    def integral_time_absolute_error(self, h):
        """ITAE: Integral of t * |h_ss - h(t)| dt"""
        h_ss = self.steady_state(h)
        time_weighted_error = self.t * np.abs(h_ss - h)
        return np.trapz(time_weighted_error, dx=self.dt)
        
    def integral_squared_error(self, h):
        """ISE: Integral of (h_ss - h(t))^2 dt (From your friend's exam)"""
        h_ss = self.steady_state(h)
        squared_error = (h_ss - h)**2
        return np.trapz(squared_error, dx=self.dt)

    def response_energy(self, h):
        """Energy E: Integral of h(t)^2 dt (From your friend's exam)"""
        return np.trapz(h**2, dx=self.dt)

    def peak_time(self, h):
        """Time at which the maximum overshoot occurs."""
        max_idx = np.argmax(h)
        return self.t[max_idx]

    def delay_time(self, h):
        """Time to reach 50% of the steady-state value."""
        h_ss = self.steady_state(h)
        target = 0.5 * h_ss
        idx = np.where(h >= target)[0]
        return self.t[idx[0]] if len(idx) > 0 else 0.0
    
    
    def compute_metrics(self, h):
        return {
            "steady_state":  self.steady_state(h),
            "time_constant": self.time_constant(h),
            "rise_time":     self.rise_time(h),
            "settling_time": self.settling_time(h),
            "overshoot_%":   self.overshoot(h),
            "ISE":           self.ISE(h, None),  # Placeholder for u, can be computed if needed
            "Response_Energy": self.response_energy(h),
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

# --- Bromwich Contour Parameters 
c = 0.42 #0.5 
W = 50
N = 5000
omega = np.linspace(-W, W, N)
s_list = c + 1j * omega

system = SmartIrrigation(a=0.5, b=1.0, t_max=20, dt=0.01)

inputs = {
    "Step Input":        system.u_step(),
    # "Ramp Input":        system.u_ramp(),
    # "Sinusoidal Input":  system.u_sin(),
    "Exponential Input": system.u_exponential(),
    # "Pulse Input":       system.u_pulse(),
}

colors = ['#2196F3', '#4CAF50', "#FF3C00", "#FF0000FF", '#FF9800']

for idx, (name, u) in enumerate(inputs.items()):
    print(f"Processing: {name}...")
    # --- Laplace Analysis ---
    # Compute U(s) for each s in s_list
    U_s_vals = np.array([system.laplace_transform(u, s) for s in s_list])
    H_s_vals = system.H_s(s_list, U_s_vals)
    h_laplace = system.inverse_laplace(s_list, H_s_vals)
    
    print(f"\n  ► {name}")
    metrics = system.compute_metrics(h_laplace)
    for k, v in metrics.items():
        print(f"      {k.replace('_',' ').title():<22}: {v:.4f}")

    # --- Euler ---
    # h_euler = system.euler_simulate(u)
    u_corrupted = system.u_corrupted(u)
    U_s_vals1 = np.array([system.laplace_transform(u_corrupted, s) for s in s_list])
    H_s_vals1 = system.H_s(s_list, U_s_vals1)
    h_euler = system.inverse_laplace(s_list, H_s_vals1)

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
    axes[1].plot(system.t, u_corrupted, 'b--', lw=1.8, label="Input u(t)")
    axes[1].plot(system.t, h_euler, color='tomato', lw=2.2, label="Output h(t)")
    axes[1].set_title("Euler Method Simulation", fontweight='bold')
    axes[1].set_xlabel("Time (s)", fontsize=11)
    axes[1].set_ylabel("Water Level / Input", fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
   

    plt.tight_layout()
    plt.show()