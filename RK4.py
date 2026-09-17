import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from matplotlib.animation import FuncAnimation

m1, m2, l1, l2, g = sp.symbols("m1 m2 l1 l2 g")

theta1, theta2 = sp.symbols("theta1 theta2")
theta1_dot, theta2_dot = sp.symbols("theta1_dot theta2_dot")
theta1_ddot, theta2_ddot = sp.symbols("theta1_ddot theta2_ddot")

eq1 = sp.Eq(
    theta1_ddot,
    (
        -m2 * l1 * l2 * theta2_ddot * sp.cos(theta1 - theta2)
        - m2 * l1 * l2 * theta2_dot**2 * sp.sin(theta1 - theta2)
        - (m1 + m2) * g * l1 * sp.sin(theta1)
    )
    / ((m1 + m2) * l1**2),
)

eq2 = sp.Eq(
    theta2_ddot,
    (
        -l1 * theta1_ddot * sp.cos(theta1 - theta2)
        + l1 * theta1_dot**2 * sp.sin(theta1 - theta2)
        - g * sp.sin(theta2)
    )
    / l2,
)

solution = sp.solve((eq1, eq2), (theta1_ddot, theta2_ddot), dict=True)[0]

theta1_ddot_explicit = sp.simplify(solution[theta1_ddot])
theta2_ddot_explicit = sp.simplify(solution[theta2_ddot])

acceleration_inputs = (theta1, theta2, theta1_dot, theta2_dot, m1, m2, l1, l2, g)

theta1_ddot_func = sp.lambdify(
    acceleration_inputs,
    theta1_ddot_explicit,
    "numpy",
)

theta2_ddot_func = sp.lambdify(
    acceleration_inputs,
    theta2_ddot_explicit,
    "numpy",
)

m1_val = 50.0
m2_val = 25.0
l1_val = 1.0
l2_val = 1.0
g_val = 9.81

theta1_0 = np.deg2rad(120.0)
theta2_0 = np.deg2rad(30.0)
theta1_dot_0 = 0.0
theta2_dot_0 = 0.0

initial_state = np.array([theta1_0, theta1_dot_0, theta2_0, theta2_dot_0])


def derivatives(state):
    theta1_val, theta1_dot_val, theta2_val, theta2_dot_val = state

    theta1_ddot_val = theta1_ddot_func(
        theta1_val,
        theta2_val,
        theta1_dot_val,
        theta2_dot_val,
        m1_val,
        m2_val,
        l1_val,
        l2_val,
        g_val,
    )

    theta2_ddot_val = theta2_ddot_func(
        theta1_val,
        theta2_val,
        theta1_dot_val,
        theta2_dot_val,
        m1_val,
        m2_val,
        l1_val,
        l2_val,
        g_val,
    )

    return np.array([theta1_dot_val, theta1_ddot_val, theta2_dot_val, theta2_ddot_val])


def rk4_step(state, dt):
    k1 = derivatives(state)
    k2 = derivatives(state + 0.5 * dt * k1)
    k3 = derivatives(state + 0.5 * dt * k2)
    k4 = derivatives(state + dt * k3)

    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


dt = 0.01
t_end = 10.0
time = np.arange(0.0, t_end + dt, dt)

states = np.zeros((len(time), 4))
states[0] = initial_state

for i in range(1, len(time)):
    states[i] = rk4_step(states[i - 1], dt)

theta1_values = states[:, 0]
theta1_dot_values = states[:, 1]
theta2_values = states[:, 2]
theta2_dot_values = states[:, 3]

x1_values = l1_val * np.sin(theta1_values)
y1_values = -l1_val * np.cos(theta1_values)
x2_values = x1_values + l2_val * np.sin(theta2_values)
y2_values = y1_values - l2_val * np.cos(theta2_values)

kinetic_energy = (
    0.5 * m1_val * l1_val**2 * theta1_dot_values**2
    + 0.5
    * m2_val
    * (
        l1_val**2 * theta1_dot_values**2
        + l2_val**2 * theta2_dot_values**2
        + 2
        * l1_val
        * l2_val
        * theta1_dot_values
        * theta2_dot_values
        * np.cos(theta1_values - theta2_values)
    )
)

potential_energy = (
    -(m1_val + m2_val) * g_val * l1_val * np.cos(theta1_values)
    - m2_val * g_val * l2_val * np.cos(theta2_values)
)

total_energy = kinetic_energy + potential_energy
energy_drift = total_energy - total_energy[0]

fig, ax = plt.subplots()
limit = l1_val + l2_val + 0.2
ax.set_xlim(-limit, limit)
ax.set_ylim(-limit, limit)
ax.set_aspect("equal")
ax.grid(True)

rod_line, = ax.plot([], [], "o-", linewidth=2)
trace_line, = ax.plot([], [], "-", linewidth=1)
energy_text = ax.text(
    0.02,
    0.98,
    "",
    transform=ax.transAxes,
    verticalalignment="top",
    fontfamily="monospace",
)

frame_step = 3
playback_speed = 1.0
animation_interval = 1000 * dt * frame_step / playback_speed


def update(frame):
    x = [0.0, x1_values[frame], x2_values[frame]]
    y = [0.0, y1_values[frame], y2_values[frame]]

    rod_line.set_data(x, y)
    trace_line.set_data(x2_values[:frame], y2_values[:frame])
    energy_text.set_text(
        f"KE:    {kinetic_energy[frame]: .4f} J\n"
        f"PE:    {potential_energy[frame]: .4f} J\n"
        f"Total: {total_energy[frame]: .4f} J\n"
        f"Drift: {energy_drift[frame]: .4e} J"
    )

    return rod_line, trace_line, energy_text


animation = FuncAnimation(
    fig,
    update,
    frames=range(0, len(time), frame_step),
    interval=animation_interval,
    blit=False,
)

plt.show()
