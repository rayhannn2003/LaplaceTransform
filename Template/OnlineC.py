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
    def calculate_ise(self, h):
        """Task 2A: Integral of Squared Error."""
        h_ss = self.steady_state(h)
        error_sq = (h_ss - h)**2
        # Numerical integration using trapezoidal rule [cite: 60]
        return (self.dt / 2.0) * (error_sq[0] + 2 * np.sum(error_sq[1:-1]) + error_sq[-1])

    def calculate_energy(self, h):
        """Task 2B: Response Energy."""
        h_sq = h**2
        return (self.dt / 2.0) * (h_sq[0] + 2 * np.sum(h_sq[1:-1]) + h_sq[-1])
    
    #all other performance metrics 
    def calculate_iae(self, h):
        """
        Integral of Absolute Error (IAE):
        IAE = ∫ |h_ss - h(t)| dt
        
        Intuition: 
        IAE treats all errors equally regardless of magnitude. It is a 
        good measure of the 'total accumulated error'. Unlike ISE, it 
        is more sensitive to small, persistent errors that last a long time.
        """
        h_ss = self.steady_state(h)
        absolute_error = np.abs(h_ss - h)
        # Trapezoidal integration across the time vector
        return (self.dt / 2.0) * (absolute_error[0] + 2 * np.sum(absolute_error[1:-1]) + absolute_error[-1])

    def calculate_itae(self, h):
        """
        Integral of Time-weighted Absolute Error (ITAE):
        ITAE = ∫ t * |h_ss - h(t)| dt
        
        Intuition:
        The 't' multiplier heavily penalizes errors that occur LATE in the 
        simulation. It is the best metric for determining if a system 
        actually settles properly. Early errors (like the initial rise) 
        are ignored, while late oscillations are heavily punished.
        """
        h_ss = self.steady_state(h)
        # Multiply the absolute error by the time vector 'self.t'
        weighted_error = self.t * np.abs(h_ss - h)
        return (self.dt / 2.0) * (weighted_error[0] + 2 * np.sum(weighted_error[1:-1]) + weighted_error[-1])

    def calculate_itse(self, h):
        """
        Integral of Time-weighted Squared Error (ITSE):
        ITSE = ∫ t * (h_ss - h(t))^2 dt
        
        Intuition:
        Similar to ITAE, but uses squared error. It is extremely 
        aggressive against large errors that happen late in time.
        """
        h_ss = self.steady_state(h)
        weighted_sq_error = self.t * (h_ss - h)**2
        return (self.dt / 2.0) * (weighted_sq_error[0] + 2 * np.sum(weighted_sq_error[1:-1]) + weighted_sq_error[-1])

    def peak_value(self, h):
        """
        Maximum Overshoot Value:
        The highest water level reached during the simulation.
        Useful for hardware safety to ensure the tank doesn't overflow.
        """
        return np.max(h)

    def peak_time(self, h):
        """
        Time to Peak:
        The time at which h(t) reaches its maximum value.
        In a first-order system with a step, this is usually t_max, 
        but for a pulse or sine wave, this identifies the system's delay.
        """
        idx_max = np.argmax(h)
        return self.t[idx_max]

    def delay_time(self, h):
        """
        Delay Time (t_d):
        The time required for the response to reach 50% of its 
        final steady-state value for the first time.
        """
        h_ss = self.steady_state(h)
        target = 0.5 * h_ss
        idx = np.where(h >= target)[0]
        return self.t[idx[0]] if len(idx) > 0 else 0.0

    def rms_error(self, h):
        """
        Root Mean Square Error (RMSE):
        RMSE = sqrt( (1/T) * ∫ (h_ss - h(t))^2 dt )
        
        Intuition:
        Provides a 'standard deviation' style measure of how far 
        the water level is from the goal on average.
        """
        ise = self.calculate_ise(h) # Using the ISE method from previous task
        return np.sqrt(ise / self.t_max)

    def total_variation_output(self, h):
        """
        Total Variation (TV) of the Output:
        TV = Σ |h[n] - h[n-1]|
        
        Intuition:
        Measures the 'smoothness' of the water level. High TV 
        indicates excessive jitter or oscillation in the tank.
        """
        return np.sum(np.abs(np.diff(h)))
    def u_chatter(self, u, prob=0.1):
        """Task 1: Simulate a failing relay (randomly dropping to 0)."""
        u_noisy = np.copy(u)
        for i in range(len(u_noisy)):
            # If the pump should be on, it has a 10% chance to fail/chatter
            if u_noisy[i] > 0 and np.random.rand() < prob:
                u_noisy[i] = 0.0
        return u_noisy

    def calculate_total_variation(self, u):
        """Task 2: Measure pump wear and tear (Total Variation)."""
        # Sum of absolute differences between consecutive points
        return np.sum(np.abs(np.diff(u)))

    def peak_time(self, h):
        """Task 3: Find the time at which the output reaches its maximum."""
        idx_max = np.argmax(h)
        return self.t[idx_max]
    def u_delayed(self, u, delay_time=2.0):
        """Task 1: Shift the input signal by delay_time."""
        # Calculate how many indices represent the delay_time
        shift_indices = int(delay_time / self.dt)
        u_delayed = np.zeros_like(u)
        
        # Shift the values; everything before shift_indices remains 0
        if shift_indices < len(u):
            u_delayed[shift_indices:] = u[:-shift_indices]
        return u_delayed

    def euler_simulate_saturated(self, u, limit=1.5):
        """Task 2: Euler method with a physical saturation cap at 1.5 units."""
        h = np.zeros_like(self.t)
        for n in range(len(self.t) - 1):
            dhdt = -self.a * h[n] + self.b * u[n]
            next_h = h[n] + self.dt * dhdt
            
            # Apply the saturation cap (Non-linear constraint)
            h[n + 1] = min(next_h, limit)
        return h

    def calculate_iae(self, h):
        """Task 3: Integral of Absolute Error."""
        h_ss = self.steady_state(h)
        absolute_error = np.abs(h_ss - h)
        # Numerical integration
        return (self.dt / 2.0) * (absolute_error[0] + 2 * np.sum(absolute_error[1:-1]) + absolute_error[-1])


    def compute_metrics(self, h):
        return {
            "steady_state":  self.steady_state(h),
            "time_constant": self.time_constant(h),
            "rise_time":     self.rise_time(h),
            "settling_time": self.settling_time(h),
            "overshoot_%":   self.overshoot(h),
            "ISE":           self.ISE(h, None),  # Placeholder for u, can be computed if needed
            "Response_Energy": self.response_energy(h),
            "ISE_Trapz":     self.calculate_ise(h),
            "Energy_Trapz":  self.calculate_energy(h),
            "IAE":             self.calculate_iae(h),
            "ITAE":            self.calculate_itae(h),
            "ITSE":            self.calculate_itse(h),
            "Peak_Value":      self.peak_value(h),
            "Peak_Time":       self.peak_time(h),
            "Delay_Time":      self.delay_time(h),
            "RMSE":            self.rms_error(h),
            "Total_Variation": self.total_variation_output(h),
            "Total_Variation_Input": self.calculate_total_variation(u),
            "u_Chatter": self.u_chatter(u),
            "u_Delayed": self.u_delayed(u),
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
    axes[0].set_ylabel("Water Level", fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Corrupted Laplace subplot
    axes[1].plot(system.t, u_corrupted, 'g--', lw=1.8, label="Input u_corrupted(t)")
    axes[1].plot(system.t, h_euler, color='tomato', lw=2.2, label="Corrupted Output h(t)")
    axes[1].set_title("Corrupted Input Simulation", fontweight='bold')
    axes[1].set_xlabel("Time (s)", fontsize=11)
    axes[1].set_ylabel("Water Level", fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
   
    plt.tight_layout()
    plt.show()
