import numpy as np
import matplotlib.pyplot as plt

class LaplacePropertiesVerifier:
    def __init__(self, t_max=20, dt=0.01):
        self.t_max = t_max
        self.dt = dt
        self.t = np.arange(0, t_max, dt)

    def laplace_transform(self, f, s):
        """
        Numerical Laplace Transform using Trapezoidal Rule.
        Same implementation as the assignment.
        """
        integran = f * np.exp(-s * self.t)
        integral = (self.dt / 2.0) * (integran[0] + 2 * np.sum(integran[1:-1]) + integran[-1])
        return integral

    def transform_array(self, f, s_list):
        """Helper to compute F(s) for an array of s values."""
        return np.array([self.laplace_transform(f, s) for s in s_list])

    # --- BASE SIGNALS ---
    def f_base(self):
        """Base signal f(t) = sin(t)"""
        return np.sin(self.t)

    def g_base(self):
        """Secondary base signal g(t) = t (ramp)"""
        return 0.1 * self.t

    # --- PROPERTY VERIFICATIONS ---
    
    def verify_linearity(self, s_list, a=2.0, b=3.0):
        """ L{a*f(t) + b*g(t)} = a*F(s) + b*G(s) """
        f_t = self.f_base()
        g_t = self.g_base()
        
        # LHS: Transform the combined time signal
        time_signal_combined = a * f_t + b * g_t
        LHS = self.transform_array(time_signal_combined, s_list)
        
        # RHS: Combine the individual transforms
        F_s = self.transform_array(f_t, s_list)
        G_s = self.transform_array(g_t, s_list)
        RHS = a * F_s + b * G_s
        
        return LHS, RHS, time_signal_combined

    def verify_time_shifting(self, s_list, shift=2.0):
        """ L{f(t-a)*u(t-a)} = e^(-as) * F(s) """
        # LHS: Create shifted time signal (using np.where for the step function)
        f_shifted_t = np.where(self.t >= shift, np.sin(self.t - shift), 0.0)
        LHS = self.transform_array(f_shifted_t, s_list)
        
        # RHS: Multiply original F(s) by e^(-as)
        F_s = self.transform_array(self.f_base(), s_list)
        RHS = np.exp(-shift * s_list) * F_s
        
        return LHS, RHS, f_shifted_t

    def verify_frequency_shifting(self, s_list, a=1.5):
        """ L{e^(at) * f(t)} = F(s - a) """
        # We use a negative 'a' so the signal decays and doesn't blow up to infinity
        shift = -a 
        
        # LHS: Transform e^(at) * f(t)
        f_exp_t = np.exp(shift * self.t) * self.f_base()
        LHS = self.transform_array(f_exp_t, s_list)
        
        # RHS: Compute F(s - a)
        s_shifted_list = s_list - shift
        RHS = self.transform_array(self.f_base(), s_shifted_list)
        
        return LHS, RHS, f_exp_t

    def verify_time_scaling(self, s_list, scale=2.0):
        """ L{f(at)} = (1/a) * F(s/a) """
        # LHS: Transform f(a*t)
        f_scaled_t = np.sin(scale * self.t)
        LHS = self.transform_array(f_scaled_t, s_list)
        
        # RHS: Compute (1/a) * F(s/a)
        s_scaled_list = s_list / scale
        F_s_scaled = self.transform_array(self.f_base(), s_scaled_list)
        RHS = (1.0 / scale) * F_s_scaled
        
        return LHS, RHS, f_scaled_t


# ==========================================
# MAIN EXECUTION & PLOTTING BLOCK
# ==========================================
if __name__ == "__main__":
    system = LaplacePropertiesVerifier(t_max=20, dt=0.01)
    
    # Define an s-domain vector (real values for simplicity in magnitude plotting)
    # Note: Complex Bromwich contour s_list can also be used, but real 's' is easier to visualize.
    s_list = np.linspace(0.5, 5, 100) 
    
    properties = {
        "Linearity": system.verify_linearity(s_list, a=2.0, b=3.0),
        "Time Shifting": system.verify_time_shifting(s_list, shift=3.0),
        "Frequency Shifting": system.verify_frequency_shifting(s_list, a=1.0),
        "Time Scaling": system.verify_time_scaling(s_list, scale=2.0)
    }

    colors = ['#2196F3', '#4CAF50', "#FF3C00", '#FF9800']

    for idx, (name, data) in enumerate(properties.items()):
        LHS, RHS, time_signal = data
        
        # Calculate Mean Absolute Error between LHS and RHS to prove they match
        error = np.mean(np.abs(LHS - RHS))
        print(f"► {name} | Mean Numerical Difference: {error:.2e}")

        # --- Plot ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Laplace Property Verification: {name}", fontsize=13, fontweight='bold')

        # Left: Time Domain Subplot
        axes[0].plot(system.t, time_signal, color=colors[idx], lw=2)
        axes[0].set_title(f"Modified Time Signal $f(t)$", fontweight='bold')
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Amplitude")
        axes[0].grid(True, alpha=0.3)

        # Right: S-Domain Subplot (LHS vs RHS)
        axes[1].plot(s_list, np.real(LHS), 'b-', lw=4, alpha=0.5, label="LHS (Direct Transform)")
        axes[1].plot(s_list, np.real(RHS), 'r--', lw=2, label="RHS (Applied Property)")
        axes[1].set_title("S-Domain Comparison $F(s)$", fontweight='bold')
        axes[1].set_xlabel("s (Real)")
        axes[1].set_ylabel("Magnitude")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()