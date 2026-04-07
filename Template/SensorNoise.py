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
        random_decay = -0.3
        return 1 - np.exp(random_decay * self.t)

    def u_pulse(self):
        return np.where((self.t >= 0) & (self.t < 5), 1.0, 0.0)

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

    # --- Extension Problem 1 Methods ---
    def add_sensor_noise(self, h):
        """Simulates a faulty ultrasonic sensor by injecting high-frequency ripples and a static offset."""
        return h + 0.1 * np.sin(25 * self.t) + 0.05

    def apply_moving_average(self, h_noisy, window_size=10):
        """Smooths the noisy signal using fundamental discrete convolution (summation)."""
        # The filter 'impulse response' h[n]
        window = np.ones(window_size) / window_size
        
        N = len(h_noisy)
        M = len(window)
        
        # 1. Full discrete convolution from scratch
        # The resulting array length is N + M - 1
        y_full = np.zeros(N + M - 1)
        
        for n in range(N):
            for k in range(M):
                # Integrating (summing) the overlapping products
                y_full[n + k] += h_noisy[n] * window[k]
                
        # 2. Extract the 'same' mode to match the original array size
        # This trims the padding to keep the filtered signal aligned with the original time axis
        start_idx = M // 2
        y_same = y_full[start_idx : start_idx + N]
        
        return y_same

    def MAE(self, h_clean, h_other):
        """Calculates Mean Absolute Error between the clean signal and another signal."""
        return np.mean(np.abs(h_clean - h_other))

# --- Bromwich Contour Parameters ---
c = 0.42
W = 50
N = 5000
omega = np.linspace(-W, W, N)
s_list = c + 1j * omega

system = SmartIrrigation(a=0.5, b=1.0, t_max=20, dt=0.01)

# Apply to Ramp and Sinusoidal inputs
inputs = {
    "Ramp Input":        system.u_ramp(),
    "Sinusoidal Input":  system.u_sin(),
}

colors = ['#4CAF50', '#FF5722']

for idx, (name, u) in enumerate(inputs.items()):
    print(f"Processing: {name}...")
    
    # --- Laplace Analysis for Clean Signal ---
    U_s_vals = np.array([system.laplace_transform(u, s) for s in s_list])
    H_s_vals = system.H_s(s_list, U_s_vals)
    h_clean = system.inverse_laplace(s_list, H_s_vals)
    
    # --- Task 1: Add Sensor Noise ---
    h_noisy = system.add_sensor_noise(h_clean)
    
    # --- Task 2: Apply Moving Average Filter ---
    h_filtered = system.apply_moving_average(h_noisy, window_size=10)
    
    # --- Task 3: Calculate MAE ---
    mae_noisy = system.MAE(h_clean, h_noisy)
    mae_filtered = system.MAE(h_clean, h_filtered)
    
    print(f"\n  ► {name}")
    print(f"      MAE (Noisy)           : {mae_noisy:.4f}")
    print(f"      MAE (Filtered)        : {mae_filtered:.4f}")

    # --- Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    fig.suptitle(f"Sensor Noise & Digital Filtering — {name}", fontsize=14, fontweight='bold')

    # Left: Clean vs Noisy
    axes[0].plot(system.t, h_clean, 'b-', lw=2.2, label="Clean h(t)")
    axes[0].plot(system.t, h_noisy, color=colors[idx], lw=1.2, alpha=0.8, label="Noisy h(t)")
    axes[0].set_title("Clean vs Noisy Signal", fontweight='bold')
    axes[0].set_xlabel("Time (s)", fontsize=11)
    axes[0].set_ylabel("Water Level", fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Right: Clean vs Filtered
    axes[1].plot(system.t, h_clean, 'b-', lw=2.2, label="Clean h(t)")
    axes[1].plot(system.t, h_filtered, color='tomato', lw=2.2, label="Filtered h(t)")
    axes[1].set_title(f"Clean vs Filtered (Window={10})", fontweight='bold')
    axes[1].set_xlabel("Time (s)", fontsize=11)
    axes[1].set_ylabel("Water Level", fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
   
    plt.tight_layout()
    plt.show()
