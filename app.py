import json
import numpy as np
import sympy as sp
import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(page_title="Assignment 1 MAS313 - Double Pendulum", layout="wide")
st.title("Assignment 1 MAS313 - Double Pendulum")
st.write(
    "This model uses the Euler–Lagrange equations to derive the equations of motion "
    "and a fourth-order Runge–Kutta (RK4) method to simulate the double pendulum."
)

if "show_simulation" not in st.session_state:
    st.session_state["show_simulation"] = False


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
        sp.simplify(solution[theta1_ddot]),
        "numpy",
    )

    theta2_ddot_func = sp.lambdify(
        acceleration_inputs,
        sp.simplify(solution[theta2_ddot]),
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

    return {
        "x1": x1_values.tolist(),
        "y1": y1_values.tolist(),
        "x2": x2_values.tolist(),
        "y2": y2_values.tolist(),
        "ke": kinetic_energy.tolist(),
        "pe": potential_energy.tolist(),
        "te": total_energy.tolist(),
        "drift": energy_drift.tolist(),
    }


# -------------------------------------------------
# Browser-rendered pendulum
# -------------------------------------------------

def build_pendulum_html(
    data,
    l1_val,
    l2_val,
    playback_speed,
    animate,
    dt,
):
    limit = float(l1_val + l2_val + 0.2)

    payload = {
        "x1": data["x1"],
        "y1": data["y1"],
        "x2": data["x2"],
        "y2": data["y2"],
        "ke": data["ke"],
        "pe": data["pe"],
        "te": data["te"],
        "drift": data["drift"],
        "playback_speed": float(playback_speed),
        "limit": limit,
        "animate": bool(animate),
        "dt": float(dt),
    }

    payload_json = json.dumps(payload)

    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{
    margin: 0;
    font-family: Arial, sans-serif;
    background: white;
  }}
  .wrap {{
    width: 100%;
    display: flex;
    justify-content: center;
  }}
  canvas {{
    border: 0;
    display: block;
  }}
</style>
</head>
<body>
<div class="wrap">
  <canvas id="pendulumCanvas" width="700" height="700"></canvas>
</div>

<script>
const data = {payload_json};

const canvas = document.getElementById("pendulumCanvas");
const ctx = canvas.getContext("2d");

const width = canvas.width;
const height = canvas.height;
const margin = 44;
const limit = data.limit;

function mapX(x) {{
  return margin + (x + limit) * (width - 2 * margin) / (2 * limit);
}}

function mapY(y) {{
  return height - (margin + (y + limit) * (height - 2 * margin) / (2 * limit));
}}

function drawGrid() {{
  ctx.strokeStyle = "#dddddd";
  ctx.lineWidth = 1;

  const step = 0.5;
  for (let v = -limit; v <= limit + 1e-9; v += step) {{
    const px = mapX(v);
    const py = mapY(v);

    ctx.beginPath();
    ctx.moveTo(px, margin);
    ctx.lineTo(px, height - margin);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(margin, py);
    ctx.lineTo(width - margin, py);
    ctx.stroke();
  }}
}}

function drawEnergyText(i) {{
  ctx.fillStyle = "black";
  ctx.font = "16px monospace";
  const lines = [
    `KE:    ${{data.ke[i].toFixed(4)}} J`,
    `PE:    ${{data.pe[i].toFixed(4)}} J`,
    `Total: ${{data.te[i].toFixed(4)}} J`,
    `Drift: ${{data.drift[i].toExponential(4)}} J`
  ];
  let y = 28;
  for (const line of lines) {{
    ctx.fillText(line, 12, y);
    y += 20;
  }}
}}

function drawFrame(i) {{
  ctx.clearRect(0, 0, width, height);
  drawGrid();

  const x0 = mapX(0.0);
  const y0 = mapY(0.0);
  const x1 = mapX(data.x1[i]);
  const y1 = mapY(data.y1[i]);
  const x2 = mapX(data.x2[i]);
  const y2 = mapY(data.y2[i]);

  // Trace
  ctx.strokeStyle = "#3b82f6";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let k = 0; k <= i; k++) {{
    const tx = mapX(data.x2[k]);
    const ty = mapY(data.y2[k]);
    if (k === 0) {{
      ctx.moveTo(tx, ty);
    }} else {{
      ctx.lineTo(tx, ty);
    }}
  }}
  ctx.stroke();

  // Rods
  ctx.strokeStyle = "black";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();

  // Pivot and masses
  ctx.fillStyle = "black";
  for (const [px, py, r] of [[x0, y0, 5], [x1, y1, 9], [x2, y2, 9]]) {{
    ctx.beginPath();
    ctx.arc(px, py, r, 0, 2 * Math.PI);
    ctx.fill();
  }}

  drawEnergyText(i);
}}

const n = data.x1.length;

if (!data.animate || n <= 1) {{
  drawFrame(0);
}} else {{
  let startTime = null;

  function step(ts) {{
    if (startTime === null) {{
      startTime = ts;
    }}

    const elapsedSeconds = (ts - startTime) / 1000.0;
    const simulatedTime = elapsedSeconds * data.playback_speed;
    let index = Math.floor(simulatedTime / data.dt);
    if (index >= n) {{
      index = n - 1;
    }}

    drawFrame(index);

    if (index < n - 1) {{
      window.requestAnimationFrame(step);
    }}
  }}

  window.requestAnimationFrame(step);
}}
</script>
</body>
</html>
"""


def build_static_data(
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

    return {
        "x1": [float(x1)],
        "y1": [float(y1)],
        "x2": [float(x2)],
        "y2": [float(y2)],
        "ke": [float(kinetic_energy)],
        "pe": [float(potential_energy)],
        "te": [float(total_energy)],
        "drift": [0.0],
    }


# -------------------------------------------------
# Side-by-side layout
# -------------------------------------------------

left_col, right_col = st.columns([0.9, 1.5], gap="large")

with right_col:
    display_area = st.empty()

with left_col:
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

    st.session_state["simulation_payload"] = {
        "results": results,
        "l1_val": float(l1_val),
        "l2_val": float(l2_val),
        "playback_speed": float(playback_speed),
        "dt": float(dt),
    }

    st.session_state["show_simulation"] = True


# -------------------------------------------------
# Display in right column
# -------------------------------------------------

if st.session_state["show_simulation"]:

    payload = st.session_state["simulation_payload"]

    html = build_pendulum_html(
        payload["results"],
        payload["l1_val"],
        payload["l2_val"],
        payload["playback_speed"],
        animate=True,
        dt=payload["dt"],
    )

    with display_area.container():
        components.html(
            html,
            height=720,
            scrolling=False,
        )

else:
    static_data = build_static_data(
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

    html = build_pendulum_html(
        static_data,
        l1_val,
        l2_val,
        playback_speed,
        animate=False,
        dt=float(dt),
    )

    with display_area.container():
        components.html(
            html,
            height=720,
            scrolling=False,
        )
