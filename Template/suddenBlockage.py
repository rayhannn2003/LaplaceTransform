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
    
    def u_exponential(self): 
        return 1 - np.exp(-0.5 * self.t)
        
    # --- TASK 1: Gradual Pump Error ---
    def u_error_A(self, u):
        """Gradual wear (-0.02t) and vibration noise (0.08*sin(8t))."""
        e_t = -0.02 * self.t + 0.08 * np.sin(8 * self.t)
        return u + e_t

    # --- PRACTICE TASK 4: Sudden Blockage ---
    def u_blockage(self, u):
        """
        Models debris hitting the pump at t=10s.
        Flow remains normal (100%) before t=10, but drops to 30% after.
        """
        # np.where(condition, true_value, false_value)
        return np.where(self.t < 10, u, u * 0.3)

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
        
    def steady_state(self, h):
        n_last = int(0.05 * len(h))
        return np.mean(h[-n_last:])

    # --- TASK 2: Advanced Metrics ---
    def ISE(self, h):
        """Integral of Squared Error relative to steady-state."""
        h_ss = self.steady_state(h)
        integrand = (h_ss - h)**2
        return np.trapz(integrand, dx=self.dt)
        
    def response_energy(self, h):
        """Response Energy = ∫ h(t)² dt"""
        integrand = h**2
        return np.trapz(integrand, dx=self.dt)

    # --- PRACTICE TASK 5 & 6: New Cost & Error Metrics ---
    def total_water_used(self, u):
        """Calculates total volume of water requested: ∫ u(t) dt"""
        return np.trapz(u, dx=self.dt)
        
    def RMSE(self, h_clean, h_corrupted):
        """Root Mean Square Error between the ideal output and corrupted output."""
        return np.sqrt(np.mean((h_clean - h_corrupted)**2))


# ==========================================
# Main Execution / Simulation Block
# ==========================================
system = SmartIrrigation(a=0.5, b=1.0, t_max=20, dt=0.01)

# We will apply the new Practice Tasks to the Step Input
u_clean = system.u_step()

# Generate the two different error scenarios
u_wear_noise = system.u_error_A(u_clean)       # Task 1 error
u_blocked = system.u_blockage(u_clean)         # Task 4 error

# Bromwich contour parameters for Inverse Laplace
c = 0.05
W = 100
N = 5000
del_omega = (2*W) / N
omega_k = np.arange(-W, W, del_omega)
s_list = c + 1j * omega_k

print("Simulating Laplace transforms... (this may take a few seconds)")

# 1. Simulate Clean
U_s_clean = np.array([system.laplace_transform(u_clean, s) for s in s_list]) 
H_s_clean = system.H_s(s_list, U_s_clean)
h_clean = system.inverse_laplace(s_list, H_s_clean)

# 2. Simulate Gradual Wear + Noise (Task 1)
U_s_wear = np.array([system.laplace_transform(u_wear_noise, s) for s in s_list]) 
H_s_wear = system.H_s(s_list, U_s_wear)
h_wear = system.inverse_laplace(s_list, H_s_wear)

# 3. Simulate Sudden Blockage (Task 4)
U_s_blocked = np.array([system.laplace_transform(u_blocked, s) for s in s_list]) 
H_s_blocked = system.H_s(s_list, U_s_blocked)
h_blocked = system.inverse_laplace(s_list, H_s_blocked)


# --- Print Output Metrics ---
print("\n======================================")
print(" WATER USAGE & ERROR METRICS ")
print("======================================")

print(f"1. Ideal System:")
print(f"   Total Water Used   : {system.total_water_used(u_clean):.2f} Liters/Units")
print(f"   Response Energy    : {system.response_energy(h_clean):.2f}")

print(f"\n2. System with Gradual Pump Wear (Task 1):")
print(f"   Total Water Used   : {system.total_water_used(u_wear_noise):.2f} Liters/Units")
print(f"   Deviation (RMSE)   : {system.RMSE(h_clean, h_wear):.4f} units from ideal")

print(f"\n3. System with Sudden Pipe Blockage (Practice Task 4):")
print(f"   Total Water Used   : {system.total_water_used(u_blocked):.2f} Liters/Units")
print(f"   Deviation (RMSE)   : {system.RMSE(h_clean, h_blocked):.4f} units from ideal")


# --- Plotting 3 Scenarios Side-by-Side ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
fig.suptitle("Smart Irrigation System — Failure Mode Analysis", fontsize=15, fontweight='bold')

# Subplot 1: Clean
axes[0].plot(system.t, u_clean, 'k--', lw=1.5, label="Input u(t)")
axes[0].plot(system.t, h_clean, color='#4CAF50', lw=2.2, label="Output h(t)")
axes[0].set_title("1. Ideal Conditions", fontweight='bold')
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Water Level")
axes[0].legend(loc="lower right")
axes[0].grid(True, alpha=0.3)

# Subplot 2: Gradual Wear
axes[1].plot(system.t, u_wear_noise, 'k--', lw=1.5, alpha=0.7, label="Input u(t) [Corrupted]")
axes[1].plot(system.t, h_wear, color='#FF9800', lw=2.2, label="Output h(t)")
axes[1].set_title("2. Gradual Pump Wear + Noise", fontweight='bold')
axes[1].set_xlabel("Time (s)")
axes[1].legend(loc="lower right")
axes[1].grid(True, alpha=0.3)

# Subplot 3: Sudden Blockage
axes[2].plot(system.t, u_blocked, 'k--', lw=1.5, alpha=0.7, label="Input u(t) [Blocked]")
axes[2].plot(system.t, h_blocked, color='#F44336', lw=2.2, label="Output h(t)")
axes[2].set_title("3. Sudden Pipe Blockage at t=10s", fontweight='bold')
axes[2].set_xlabel("Time (s)")
axes[2].legend(loc="lower right")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()