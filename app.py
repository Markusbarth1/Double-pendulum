import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import streamlit as st
import streamlit.components.v1 as components
from matplotlib.animation import FuncAnimation


st.set_page_config(page_title="Double Pendulum", layout="centered")
st.title("Double Pendulum")


# -------------------------------------------------
# Symbolic setup
# -------------------------------------------------

@st.cache_resource
def build_acceleration_functions():

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

    solution = sp.solve(
        (eq1, eq2),
        (theta1_ddot, theta2_ddot),
        dict=True,
    )[0]

    theta1_ddot_explicit = sp.simplify(solution[theta1_ddot])
    theta2_ddot_explicit = sp.simplify(solution[theta2_ddot])

    acceleration_inputs = (
        theta1,
        theta2,
        theta1_dot,
        theta2_dot,
        m1,
        m2,
        l1,
        l2,
        g,
    )

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

    return theta1_ddot_func, theta2_ddot_func


theta1_ddot_func, theta2_ddot_func = build_acceleration_functions()


# -------------------------------------------------
# Numerical functions
# -------------------------------------------------

def derivatives(state, m1_val, m2_val, l1_val, l2_val, g_val):

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

    return np.array(
        [theta1_dot_val, theta1_ddot_val, theta2_dot_val, theta2_ddot_val]
    )


def rk4_step(state, dt, m1_val, m2_val, l1_val, l2_val, g_val):

    k1 = derivatives(state, m1_val, m2_val, l1_val, l2_val, g_val)

    k2 = derivatives(
        state + 0.5 * dt * k1,
        m1_val, m2_val, l1_val, l2_val, g_val,
    )

    k3 = derivatives(
        state + 0.5 * dt * k2,
        m1_val, m2_val, l1_val, l2_val, g_val,
    )

    k4 = derivatives(
        state + dt * k3,
        m1_val, m2_val, l1_val, l2_val, g_val,
    )

    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def run_simulation(
    m1_val,
    m2_val,
    l1_val,
    l2_val,
    g_val,
    theta1_0,
    theta2_0,
    theta1_dot_0,
    theta2_dot_0,
    dt,
    t_end,
):

    initial_state = np.array(
        [theta1_0, theta1_dot_0, theta2_0, theta2_dot_0]
    )

    time = np.arange(0.0, t_end + dt, dt)

    states = np.zeros((len(time), 4))
    states[0] = initial_state

    for i in range(1, len(time)):
        states[i] = rk4_step(
            states[i - 1],
            dt,
            m1_val,
            m2_val,
            l1_val,
            l2_val,
            g_val,
        )

    return time, states


def calculate_results(
    states,
    m1_val,
    m2_val,
    l1_val,
    l2_val,
    g_val,
):

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
        -(m1_val + m2_val)
        * g_val
        * l1_val
        * np.cos(theta1_values)
        - m2_val
        * g_val
        * l2_val
        * np.cos(theta2_values)
    )

    total_energy = kinetic_energy + potential_energy
    energy_drift = total_energy - total_energy[0]

    return (
        x1_values,
        y1_values,
        x2_values,
        y2_values,
        kinetic_energy,
        potential_energy,
        total_energy,
        energy_drift,
    )


def make_initial_figure(
    m1_val,
    m2_val,
    l1_val,
    l2_val,
    g_val,
    theta1_0,
    theta2_0,
    theta1_dot_0,
    theta2_dot_0,
):
    """Show the pendulum immediately, before Start is pressed."""

    x1 = l1_val * np.sin(theta1_0)
    y1 = -l1_val * np.cos(theta1_0)

    x2 = x1 + l2_val * np.sin(theta2_0)
    y2 = y1 - l2_val * np.cos(theta2_0)

    kinetic_energy = (
        0.5 * m1_val * l1_val**2 * theta1_dot_0**2
        + 0.5
        * m2_val
        * (
            l1_val**2 * theta1_dot_0**2
            + l2_val**2 * theta2_dot_0**2
            + 2
            * l1_val
            * l2_val
            * theta1_dot_0
            * theta2_dot_0
            * np.cos(theta1_0 - theta2_0)
        )
    )

    potential_energy = (
        -(m1_val + m2_val)
        * g_val
        * l1_val
        * np.cos(theta1_0)
        - m2_val
        * g_val
        * l2_val
        * np.cos(theta2_0)
    )

    total_energy = kinetic_energy + potential_energy

    fig, ax = plt.subplots(figsize=(6, 6))

    limit = l1_val + l2_val + 0.2
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.grid(True)

    ax.plot(
        [0.0, x1, x2],
        [0.0, y1, y2],
        "o-",
        linewidth=2,
    )

    ax.text(
        0.02,
        0.98,
        (
            f"KE:    {kinetic_energy: .4f} J\n"
            f"PE:    {potential_energy: .4f} J\n"
            f"Total: {total_energy: .4f} J\n"
            f"Drift: {0.0: .4e} J"
        ),
        transform=ax.transAxes,
        verticalalignment="top",
        fontfamily="monospace",
    )

    return fig


def make_animation_html(
    time,
    x1_values,
    y1_values,
    x2_values,
    y2_values,
    kinetic_energy,
    potential_energy,
    total_energy,
    energy_drift,
    l1_val,
    l2_val,
    dt,
    playback_speed,
):

    fig, ax = plt.subplots(figsize=(6, 6))

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

    max_frames = 400
    frame_step = max(1, int(np.ceil(len(time) / max_frames)))

    animation_interval = (
        1000 * dt * frame_step / playback_speed
    )

    def update(frame):

        x = [0.0, x1_values[frame], x2_values[frame]]
        y = [0.0, y1_values[frame], y2_values[frame]]

        rod_line.set_data(x, y)

        trace_line.set_data(
            x2_values[: frame + 1],
            y2_values[: frame + 1],
        )

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

    fps = max(
        1,
        int(round(1000 / animation_interval)),
    )

    html = animation.to_jshtml(fps=fps)

    plt.close(fig)

    return html


# -------------------------------------------------
# Input boxes
# -------------------------------------------------

with st.form("inputs"):

    m1_val = st.number_input(
        "Mass 1 [kg]",
        min_value=0.001,
        value=50.0,
    )

    m2_val = st.number_input(
        "Mass 2 [kg]",
        min_value=0.001,
        value=25.0,
    )

    l1_val = st.number_input(
        "Length 1 [m]",
        min_value=0.001,
        value=1.0,
    )

    l2_val = st.number_input(
        "Length 2 [m]",
        min_value=0.001,
        value=1.0,
    )

    g_val = st.number_input(
        "Gravity [m/s²]",
        min_value=0.001,
        value=9.81,
    )

    theta1_deg_val = st.number_input(
        "Initial angle 1 [degrees]",
        value=120.0,
    )

    theta2_deg_val = st.number_input(
        "Initial angle 2 [degrees]",
        value=30.0,
    )

    theta1_dot_0 = st.number_input(
        "Initial angular velocity 1 [rad/s]",
        value=0.0,
    )

    theta2_dot_0 = st.number_input(
        "Initial angular velocity 2 [rad/s]",
        value=0.0,
    )

    dt = st.number_input(
        "Timestep [s]",
        min_value=0.0001,
        value=0.01,
        format="%.4f",
    )

    t_end = st.number_input(
        "Simulation time [s]",
        min_value=0.1,
        value=10.0,
    )

    playback_speed = st.number_input(
        "Playback speed",
        min_value=0.1,
        value=1.0,
    )

    start = st.form_submit_button("Start")


stop = st.button("Stop / reset")


if stop:
    st.session_state["show_simulation"] = False


theta1_0 = np.deg2rad(theta1_deg_val)
theta2_0 = np.deg2rad(theta2_deg_val)


if start:

    time, states = run_simulation(
        m1_val,
        m2_val,
        l1_val,
        l2_val,
        g_val,
        theta1_0,
        theta2_0,
        theta1_dot_0,
        theta2_dot_0,
        dt,
        t_end,
    )

    results = calculate_results(
        states,
        m1_val,
        m2_val,
        l1_val,
        l2_val,
        g_val,
    )

    st.session_state["simulation_data"] = (
        time,
        results,
        l1_val,
        l2_val,
        dt,
        playback_speed,
    )

    st.session_state["show_simulation"] = True


# -------------------------------------------------
# Display
# -------------------------------------------------

if st.session_state.get("show_simulation", False):

    (
        time,
        results,
        l1_val_sim,
        l2_val_sim,
        dt_sim,
        playback_speed_sim,
    ) = st.session_state["simulation_data"]

    (
        x1_values,
        y1_values,
        x2_values,
        y2_values,
        kinetic_energy,
        potential_energy,
        total_energy,
        energy_drift,
    ) = results

    animation_html = make_animation_html(
        time,
        x1_values,
        y1_values,
        x2_values,
        y2_values,
        kinetic_energy,
        potential_energy,
        total_energy,
        energy_drift,
        l1_val_sim,
        l2_val_sim,
        dt_sim,
        playback_speed_sim,
    )

    components.html(
        animation_html,
        height=650,
        scrolling=False,
    )

else:
    # Before Start (and after reset), show the pendulum at its initial position.
    initial_fig = make_initial_figure(
        m1_val,
        m2_val,
        l1_val,
        l2_val,
        g_val,
        theta1_0,
        theta2_0,
        theta1_dot_0,
        theta2_dot_0,
    )

    st.pyplot(initial_fig)
    plt.close(initial_fig)
