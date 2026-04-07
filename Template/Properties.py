#Linearity
import numpy as np

"""
1
Linearity
L{af+bg} = aF(s)+bG(s)
▶
superposition
scaling
The Laplace transform is linear — you can scale signals and add them before or after transforming. This is why we can handle differential equations term by term.
L{ a·f(t) + b·g(t) } = a·F(s) + b·G(s)

Where F(s) = L{f(t)},  G(s) = L{g(t)}"""

def laplace_linearity(f, g, a, b, s, t, dt):
    """
    Verifies: L{a*f + b*g}(s) == a*L{f}(s) + b*L{g}(s)
    Uses trapezoidal integration over t array.
    """
    def L(fn):
        integrand = fn * np.exp(-s * t)
        return (dt/2) * (integrand[0] + 2*np.sum(integrand[1:-1]) + integrand[-1])

    lhs = L(a*f + b*g)
    rhs = a*L(f) + b*L(g)
    return lhs, rhs

t  = np.linspace(0, 10, 1000); dt = t[1]-t[0]
f  = np.exp(-t);  g = np.sin(t)
a, b, s = 2.0, 3.0, 1.5+0j
lhs, rhs = laplace_linearity(f, g, a, b, s, t, dt)
print(f"LHS = {lhs:.6f}")
print(f"RHS = {rhs:.6f}")
print(f"Match: {np.isclose(lhs, rhs)}")

"""
2
Time shifting (delay)
L{f(t-a)u(t-a)} = e^(-as)·F(s)
▶
delay
unit step
causality
Delaying a signal by 'a' seconds in time multiplies its Laplace transform by e^(−as). Used in pulse modelling (like your u_pulse input).
L{ f(t-a)·u(t-a) } = e^(-as) · F(s),   a ≥ 0

u(t-a) = Heaviside unit step at t=a

"""

def laplace_time_shift(f_original, delay, s, t, dt):
    """
    L{f(t - delay) * u(t - delay)} = exp(-delay*s) * F(s)
    f_original: signal evaluated on t (no delay)
    Returns lhs (direct), rhs (formula), and relative error.
    """
    def L(fn):
        integrand = fn * np.exp(-s * t)
        return (dt/2)*(integrand[0] + 2*np.sum(integrand[1:-1]) + integrand[-1])

    # Shifted signal with unit step (causal)
    f_shifted = np.where(t >= delay, np.interp(t - delay, t, f_original), 0.0)

    F_s = L(f_original)
    lhs = L(f_shifted)
    rhs = np.exp(-delay * s) * F_s
    return lhs, rhs, abs(lhs-rhs)/abs(rhs)

t = np.linspace(0, 15, 2000); dt = t[1]-t[0]
f = np.exp(-0.5*t)
lhs, rhs, err = laplace_time_shift(f, delay=2.0, s=1.0+0j, t=t, dt=dt)
print(f"LHS={lhs:.5f}  RHS={rhs:.5f}  rel_err={err:.2e}")


"""
3
Frequency shifting (s-shifting)
L{e^(at)·f(t)} = F(s-a)
▶
modulation
exponential
Multiplying a time-domain signal by an exponential e^(at) shifts the entire transform left/right in the s-plane. Explains exponentially-modulated sinusoids.
L{ e^(at) · f(t) } = F(s - a)

Multiplying by e^(at) in time shifts the s-domain by +a.
"""

def laplace_freq_shift(f, a, s, t, dt):
    """
    L{e^(at)*f(t)}(s) should equal F(s-a).
    Verifies numerically for a given s.
    """
    def L(fn, s_val):
        integrand = fn * np.exp(-s_val * t)
        return (dt/2)*(integrand[0] + 2*np.sum(integrand[1:-1]) + integrand[-1])

    lhs = L(np.exp(a*t) * f, s)     # L{e^(at)*f(t)} at s
    rhs = L(f, s - a)               # F(s-a)
    return lhs, rhs

t  = np.linspace(0, 8, 1000); dt = t[1]-t[0]
f  = np.sin(2*t)          # f(t) = sin(2t)
a  = 1.0                  # shift amount
s  = 3.0 + 0j

lhs, rhs = laplace_freq_shift(f, a, s, t, dt)
print(f"LHS = {lhs:.6f}")
print(f"RHS = {rhs:.6f}")
print(f"Match: {np.isclose(lhs, rhs, atol=1e-4)}")













"""

Time scaling
L{f(at)} = (1/|a|)·F(s/a)
▶
compression
dilation
Scaling time compresses or stretches the signal. The s-domain is inversely scaled. Useful when changing simulation speed or sampling rate.
L{ f(a·t) } = (1/|a|) · F(s/a),   a > 0

Compressing time (a>1) expands the s-domain and vice versa.

"""

def laplace_time_scale(f_fn, a, s, t_max=10, N=2000):
    """
    L{f(a*t)}(s)  ==  (1/a) * F(s/a)
    f_fn: callable, e.g. lambda t: np.exp(-t)
    """
    t  = np.linspace(0, t_max, N)
    dt = t[1] - t[0]

    def L(fn, s_val):
        vals = fn(t) * np.exp(-s_val * t)
        return (dt/2)*(vals[0] + 2*np.sum(vals[1:-1]) + vals[-1])

    lhs = L(lambda u: f_fn(a * u), s)
    rhs = (1.0/a) * L(f_fn, s/a)
    return lhs, rhs

f   = lambda t: np.exp(-t)  # f(t) = e^(-t)
a   = 2.0
s   = 1.5 + 0j
lhs, rhs = laplace_time_scale(f, a, s)
print(f"LHS = {lhs:.6f}")
print(f"RHS = {rhs:.6f}")
print(f"Match: {np.isclose(lhs, rhs, atol=1e-4)}")


"""

Differentiation in time
L{f'(t)} = sF(s) - f(0)
▶
ODE
derivative
initial condition
The key property that converts ODEs to algebra. Each derivative introduces a factor of s and subtracts the initial conditions. This is exactly how your irrigation ODE dh/dt + a·h = b·u(t) becomes (s+a)H(s) = b·U(s).
L{ f'(t)  } = s·F(s) - f(0⁻)
L{ f''(t) } = s²F(s) - s·f(0⁻) - f'(0⁻)
"""
def laplace_differentiation(f, f0, s, t, dt):
    """
    L{f'(t)}(s) = s*F(s) - f(0)
    f   : signal array
    f0  : initial value f(0)
    s   : complex frequency
    """
    def L(fn):
        integrand = fn * np.exp(-s * t)
        return (dt/2)*(integrand[0] + 2*np.sum(integrand[1:-1]) + integrand[-1])

    # Numerical derivative using central differences
    f_prime = np.gradient(f, dt)

    lhs = L(f_prime)          # L{f'} computed directly
    F_s = L(f)
    rhs = s * F_s - f0        # s*F(s) - f(0)

    return lhs, rhs

t  = np.linspace(0, 10, 2000); dt = t[1]-t[0]
f  = np.exp(-0.5*t)           # f(t) = e^(-0.5t)
f0 = 1.0                      # f(0) = 1
s  = 1.0 + 0j

lhs, rhs = laplace_differentiation(f, f0, s, t, dt)
print(f"LHS = {lhs:.6f}")
print(f"RHS = {rhs:.6f}")
print(f"Match: {np.isclose(lhs, rhs, atol=1e-3)}")





"""
6
Integration in time
L{∫f dt} = F(s)/s
▶
integral
accumulation
Integration in time divides the Laplace transform by s. The dual of differentiation. Useful for modelling cumulative quantities (e.g. total water delivered).
L{ ∫₀ᵗ f(τ) dτ } = F(s) / s

Dividing by s in the s-domain = integrating in time.
"""




def laplace_integration(f, s, t, dt):
    """
    L{ integral_0_to_t f(tau) dtau }(s) = F(s)/s
    """
    def L(fn):
        integrand = fn * np.exp(-s * t)
        return (dt/2)*(integrand[0] + 2*np.sum(integrand[1:-1]) + integrand[-1])

    # Cumulative integral using cumulative trapezoidal sum
    f_int = np.cumsum(f) * dt   # approximates ∫₀ᵗ f dτ

    lhs = L(f_int)              # L of the integral
    rhs = L(f) / s              # F(s)/s

    return lhs, rhs

t  = np.linspace(0, 10, 2000); dt = t[1]-t[0]
f  = np.sin(t)
s  = 1.0 + 1j

lhs, rhs = laplace_integration(f, s, t, dt)
print(f"LHS = {lhs:.5f}")
print(f"RHS = {rhs:.5f}")
print(f"Match: {np.isclose(lhs, rhs, atol=1e-3)}")


"""
7
Convolution theorem
L{ (f * g)(t) } = F(s) · G(s)

(f * g)(t) = ∫₀ᵗ f(τ)·g(t-τ) dτ   (convolution)

Convolution in time  ↔  multiplication in s-domain
"""


def laplace_convolution(f, g, s, t, dt):
    """
    L{f*g}(s) == F(s)*G(s)
    Uses scipy.signal.fftconvolve for the time-domain convolution.
    """
    from scipy.signal import fftconvolve

    def L(fn):
        integrand = fn * np.exp(-s * t)
        return (dt/2)*(integrand[0] + 2*np.sum(integrand[1:-1]) + integrand[-1])

    # Convolution (only keep 'same' length)
    f_conv_g = fftconvolve(f, g, mode='full')[:len(t)] * dt

    lhs = L(f_conv_g)
    rhs = L(f) * L(g)
    return lhs, rhs

t  = np.linspace(0, 10, 1000); dt = t[1]-t[0]
f  = np.exp(-t)
g  = np.exp(-2*t)
s  = 1.5 + 0j

lhs, rhs = laplace_convolution(f, g, s, t, dt)
print(f"LHS = {lhs:.5f}")
print(f"RHS = {rhs:.5f}")
print(f"Match: {np.isclose(lhs, rhs, atol=1e-3)}")













"""
8
Multiplication by t (frequency differentiation)

L{ t · f(t) }   = -dF(s)/ds
L{ tⁿ · f(t) } = (-1)ⁿ · dⁿF(s)/dsⁿ

Example: L{t} = 1/s²,  L{t²} = 2/s³

"""

def laplace_mult_by_t(f, s, t, dt, ds=1e-5):
    """
    L{t*f(t)}(s) = -d/ds [F(s)]
    Approximates d/ds using central finite difference.
    """
    def L(fn, sv):
        integrand = fn * np.exp(-sv * t)
        return (dt/2)*(integrand[0] + 2*np.sum(integrand[1:-1]) + integrand[-1])

    lhs = L(t * f, s)
    # Central difference: dF/ds ≈ [F(s+ds) - F(s-ds)] / (2*ds)
    dF_ds = (L(f, s+ds) - L(f, s-ds)) / (2*ds)
    rhs   = -dF_ds

    return lhs, rhs

t  = np.linspace(0, 10, 2000); dt = t[1]-t[0]
f  = np.exp(-t)          # F(s) = 1/(s+1),  so -dF/ds = 1/(s+1)²
s  = 2.0 + 0j

lhs, rhs = laplace_mult_by_t(f, s, t, dt)
print(f"LHS = {lhs:.6f}")
print(f"RHS = {rhs:.6f}")
print(f"Analytic check 1/(s+1)² = {1/(s+1)**2:.6f}")
print(f"Match: {np.isclose(lhs, rhs, atol=1e-5)}")







"""
9
Division by t (frequency integration)
L{ f(t)/t } = ∫ₛ^∞ F(σ) dσ

Requires: lim_{t→0} f(t)/t  exists (finite).

Example: L{sin(t)/t} = π/2 - arctan(s)
"""
import numpy as np

def laplace_div_by_t(f, s_val, t, dt, s_max=100, ds=0.01):
    """
    L{f(t)/t}(s) ≈ ∫_s^s_max F(σ) dσ
    Avoids t=0 division by starting from dt.
    """
    def L(fn, sv):
        integrand = fn * np.exp(-sv * t)
        return (dt/2)*(integrand[0] + 2*np.sum(integrand[1:-1]) + integrand[-1])

    # LHS: integrate f(t)/t directly (skip t[0] to avoid /0)
    t_safe = t.copy(); t_safe[0] = t[1]
    f_over_t = f / t_safe; f_over_t[0] = f_over_t[1]
    lhs = L(f_over_t, s_val)

    # RHS: integrate F(σ) from s to s_max
    sigmas = np.arange(float(np.real(s_val)), s_max, ds)
    F_vals = np.array([L(f, sig) for sig in sigmas])
    rhs = np.trapz(F_vals, sigmas)
    return lhs, rhs

t  = np.linspace(1e-6, 10, 3000); dt = t[1]-t[0]
f  = np.sin(t)           # f(t)/t = sinc-like, L = π/2 - arctan(s)
s  = 1.0

lhs, rhs = laplace_div_by_t(f, s, t, dt)
analytic = np.pi/2 - np.arctan(s)
print(f"LHS      = {lhs:.5f}")
print(f"RHS      = {rhs:.5f}")
print(f"Analytic = {analytic:.5f}")








"""
10
Initial value theorem:

f(0⁺) = lim_{s→∞} s · F(s)

Directly reads the initial value from F(s) — no need to invert.

"""
import numpy as np

def initial_value_theorem(F_fn, s_values=None):
    """
    f(0+) = lim_{s->inf} s*F(s)
    F_fn: callable that takes s (complex) and returns F(s).
    Approximates the limit using large s values.
    """
    if s_values is None:
        s_values = np.logspace(1, 6, 200)  # 10 to 10^6

    products = s_values * np.array([F_fn(s) for s in s_values])
    f0_est = np.real(products[-1])   # value at largest s
    return f0_est

# Example: F(s) = 1/(s+2),  f(t) = e^(-2t),  f(0+) = 1
F = lambda s: 1.0/(s + 2.0)
f0 = initial_value_theorem(F)
print(f"Estimated f(0+) = {f0:.6f}")
print(f"Exact      f(0+) = 1.000000")

# Irrigation system: H(s) = b/((s+a)*(s)) for step input
# F(s) = b/(s*(s+a)), f(0+) should be 0 (water starts at 0)
a, b = 0.5, 1.0
F2 = lambda s: b / (s * (s + a))
print(f"\nIrrigation step h(0+) = {initial_value_theorem(F2):.6f}")
print(f"Expected: 0.0")







"""
11
Final value theorem
f(∞) = lim_{s→0⁺} s · F(s)

Valid only if f(t) has a finite limit as t→∞.
All poles of s·F(s) must be in the left half-plane.
"""

import numpy as np

def final_value_theorem(F_fn, s_values=None):
    """
    f(inf) = lim_{s->0+} s*F(s)
    Approximates by evaluating s*F(s) at very small positive s.
    WARNING: only valid if all poles of s*F(s) are in LHP.
    """
    if s_values is None:
        s_values = np.logspace(-6, -1, 200)  # s from ~1e-6 to 0.1

    products = s_values * np.array([F_fn(s) for s in s_values])
    f_inf = np.real(products[0])  # value as s→0+
    return f_inf

# Example: step input, H(s) = b/(s*(s+a))
# h_ss should be b/a = 1.0/0.5 = 2.0
a, b = 0.5, 1.0
F_step = lambda s: b / (s * (s + a))

hss = final_value_theorem(F_step)
print(f"Final value (FVT): h(inf) = {hss:.4f}")
print(f"Analytic   (b/a):  h(inf) = {b/a:.4f}")

# Exponential input: u(t) = 1 - e^(-0.3t)
# U(s) = 1/s - 1/(s+0.3)
F_exp = lambda s: b * (1/s - 1/(s+0.3)) / (s + a)
hss_exp = final_value_theorem(lambda s: s * F_exp(s) / s)
print(f"\nExponential input h(inf) ≈ {final_value_theorem(F_exp):.4f}")






"""
12
Periodicity
For a periodic signal with period T:

L{ f(t) } = [∫₀ᵀ f(t)e^(-st) dt] / (1 - e^(-Ts))

Only need to integrate over ONE period!
"""

import numpy as np

def laplace_periodic(f_one_period, T, s, dt):
    """
    L{periodic f}(s) = L{one period}(s) / (1 - exp(-T*s))
    f_one_period: signal array over [0, T]
    T           : period
    """
    # Integrate one period only
    integrand = f_one_period * np.exp(-s * np.linspace(0, T, len(f_one_period)))
    F1 = (dt/2)*(integrand[0] + 2*np.sum(integrand[1:-1]) + integrand[-1])

    denom = 1.0 - np.exp(-T * s)
    if np.abs(denom) < 1e-12:
        return np.nan   # s=0 or s=2πi/T (resonance)
    return F1 / denom

# Verify against full signal for sin(t), T = 2π
T  = 2 * np.pi
t1 = np.linspace(0, T, 500); dt1 = t1[1]-t1[0]
f1 = np.sin(t1)              # one period

# Full signal (10 periods)
t_full = np.linspace(0, 10*T, 5000); dt_f = t_full[1]-t_full[0]
f_full = np.sin(t_full)

s = 0.5 + 0j

# Using periodicity formula
F_periodic = laplace_periodic(f1, T, s, dt1)

# Direct transform of full signal
integrand = f_full * np.exp(-s * t_full)
F_direct = (dt_f/2)*(integrand[0] + 2*np.sum(integrand[1:-1]) + integrand[-1])

print(f"Periodic formula : {F_periodic:.4f}")
print(f"Direct transform : {F_direct:.4f}")
print(f"Match: {np.isclose(F_periodic, F_direct, atol=1e-2)}")