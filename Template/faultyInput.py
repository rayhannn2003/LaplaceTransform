import numpy as np
import matplotlib.pyplot as plt

class SmartIrrigation:
    def __init__(self, a=0.5, b=1.0, t_max=20, dt=0.01):
        self.a = a
        self.b = b
        self.t_max = t_max
        self.dt = dt
        self.t = np.arange(0, t_max + dt, dt)

    def u_step(self):
        return np.ones(len(self.t))

    # --- PRACTICE TASK 7: Transport Delay ---
    def u_delay(self, u, delay_time=3.0):
        """
        Shifts the signal in time to simulate water traveling through pipes.
        Uses array indexing to shift values and pads the beginning with zeros.
        """
        delay_index = int(delay_time / self.dt)
        u_delayed = np.zeros_like(u)
        if delay_index < len(u):
            u_delayed[delay_index:] = u[:-delay_index]
        return u_delayed

    # --- PRACTICE TASK 8: Intermittent Dropout ---
    def u_dropout(self, u):
        """
        Simulates a loose wire where the pump loses power every alternate 2 seconds.
        We use a square wave logic based on modulo division.
        """
        # True if t is in [0,2), [4,6) etc., False if in [2,4), [6,8)
        working_mask = (self.t % 4.0) < 2.0
        # Multiply signal by boolean mask (True=1, False=0)
        return u * working_mask

    # --- PRACTICE TASK 9: Maximum Rate of Change ---
    def max_rate_of_change(self, h):
        """
        Finds the peak velocity of the water level changes (dh/dt).
        Uses np.gradient for numerical differentiation.
        """
        dh_dt = np.gradient(h, self.t)
        return np.max(np.abs(dh_dt))

    # --- Laplace Transform Core ---
    def laplace_transform(self, f, s):
        integrand = f * np.exp(-s * self.t)
        return self.dt * (0.5 * integrand[0] + 0.5 * integrand[-1] + integrand[1:-1].sum())

    def inverse_laplace(self, s_list, F_s_values):
        h_t = np.zeros_like(self.t)
        omega = np.imag(s_list)
        del_omega = omega[1] - omega[0]
        for k in range(len(self.t)):
            integrand = F_s_values * np.exp(s_list * self.t[k])
            h_t[k] = del_omega / (2 * np.pi) * np.real(np.sum(integrand))
        return h_t

    def H_s(self, s, U_s):
        return self.b * U_s / (s + self.a)


# ==========================================
# Simulation Block
# ==========================================
system = SmartIrrigation(a=0.5, b=1.0, t_max=20, dt=0.01)

# Baseline
u_clean = system.u_step()

# New Fault Conditions
u_delayed = system.u_delay(u_clean, delay_time=4.0)  # 4 second delay
u_dropped = system.u_dropout(u_clean)                # 2s on / 2s off

# Inverse Laplace setup
c = 0.05
W = 100
N = 5000
del_omega = (2*W) / N
omega_k = np.arange(-W, W, del_omega)
s_list = c + 1j * omega_k

print("Simulating Laplace transforms...")

# Simulate Clean
U_s_clean = np.array([system.laplace_transform(u_clean, s) for s in s_list]) 
h_clean = system.inverse_laplace(s_list, system.H_s(s_list, U_s_clean))

# Simulate Transport Delay
U_s_delayed = np.array([system.laplace_transform(u_delayed, s) for s in s_list]) 
h_delayed = system.inverse_laplace(s_list, system.H_s(s_list, U_s_delayed))

# Simulate Dropout
U_s_dropped = np.array([system.laplace_transform(u_dropped, s) for s in s_list]) 
h_dropped = system.inverse_laplace(s_list, system.H_s(s_list, U_s_dropped))

# Print Task 9 Metrics (Differentiation)
print("\n--- Max Rate of Level Change (dh/dt) ---")
print(f"Ideal System    : {system.max_rate_of_change(h_clean):.4f} units/sec")
print(f"Delayed System  : {system.max_rate_of_change(h_delayed):.4f} units/sec")
print(f"Dropout System  : {system.max_rate_of_change(h_dropped):.4f} units/sec")

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Advanced System Faults: Transport Delay & Signal Dropout", fontsize=14, fontweight='bold')

# Delay Plot
axes[0].plot(system.t, u_delayed, 'k--', lw=1.5, label="Input u(t) (Delayed by 4s)")
axes[0].plot(system.t, h_delayed, color='purple', lw=2.2, label="Output h(t)")
axes[0].set_title("System with Transport Delay")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Water Level")
axes[0].legend(loc="lower right")
axes[0].grid(True, alpha=0.3)

# Dropout Plot
axes[1].plot(system.t, u_dropped, 'k--', lw=1.5, label="Input u(t) (Intermittent)")
axes[1].plot(system.t, h_dropped, color='teal', lw=2.2, label="Output h(t)")
axes[1].set_title("System with Loose Wire (Signal Dropout)")
axes[1].set_xlabel("Time (s)")
axes[1].legend(loc="lower right")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()