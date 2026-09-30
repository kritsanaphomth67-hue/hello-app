import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------
# การตั้งค่าหน้าเว็บ (Page Configuration)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Euler's Method - Initial Value Problem",
    page_icon="📈",
    layout="wide",
)

st.title("Project 1: Euler's Method Approximation")

# ---------------------------------------------------------
# ข้อกำหนดข้อ 5: Layout Organization (ใช้ st.sidebar รับค่า)
# ข้อกำหนดข้อ 1: Interactive Controls (ใช้ Widgets อย่างน้อย 3 ชนิด)
# ---------------------------------------------------------
st.sidebar.header("⚙️ การตั้งค่าพารามิเตอร์ (Input)")

# Widget ชนิดที่ 1: st.number_input สำหรับรับค่า step size (h)
h = st.sidebar.number_input(
    "1. กำหนดขนาดก้าว (Step size: h)",
    min_value=0.001,
    max_value=0.5,
    value=0.1,
    step=0.01,
    format="%.4f",
)

# Widget ชนิดที่ 2: st.slider สำหรับปรับทศนิยมในตาราง[cite: 2]
precision = st.sidebar.slider(
    "2. จำนวนตำแหน่งทศนิยมในตาราง (Precision)",
    min_value=2,
    max_value=8,
    value=7,
)

# Widget ชนิดที่ 3: st.selectbox สำหรับเลือกสไตล์ของกราฟ[cite: 2]
chart_theme = st.sidebar.selectbox(
    "3. เลือกรูปแบบธีมกราฟ (Chart Theme)",
    ["plotly_white", "plotly_dark", "ggplot2", "seaborn"],
)

# ---------------------------------------------------------
# ส่วนการคำนวณทางคณิตศาสตร์ (Core Logic)
# ---------------------------------------------------------
# ODE: y' = y/t - (y^2 / t^2)[cite: 1]
def f(t, y):
    return (y / t) - ((y / t) ** 2)


# Exact Solution: y(t) = t / (1 + ln(t))[cite: 1]
def exact_solution(t):
    return t / (1 + np.log(t))


# ช่วงข้อมูล t จาก 1 ถึง 2[cite: 1]
t_start = 1.0
t_end = 2.0
t_values = np.arange(t_start, t_end + h / 2, h)

# คำนวณ Euler's Method[cite: 1]
euler_values = [1.0]  # เงื่อนไขเริ่มต้น y(1) = 1[cite: 1]
for i in range(len(t_values) - 1):
    t_curr = t_values[i]
    y_curr = euler_values[-1]
    y_next = y_curr + h * f(t_curr, y_curr)
    euler_values.append(y_next)

euler_values = np.array(euler_values)
exact_values = exact_solution(t_values)
errors = np.abs(exact_values - euler_values)

# สร้าง DataFrame สำหรับแสดงผล
df = pd.DataFrame(
    {
        "t_i": t_values,
        "Euler's": euler_values,
        "Exact": exact_values,
        "Error": errors,
    }
)

# ---------------------------------------------------------
# ข้อกำหนดข้อ 5: Layout Organization (ใช้ st.tabs แบ่งเนื้อหา)[cite: 2]
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    [
        "📘 ทฤษฎี (Theory)",
        "📊 ตัวจำลองและกราฟ (Simulation & Chart)",
        "📋 ตารางสรุปผล (Results Table)",
    ]
)

# ---------------------------------------------------------
# Tab 1: ทฤษฎีและสูตรคณิตศาสตร์
# ข้อกำหนดข้อ 2: Mathematical Notation (ใช้ st.latex)[cite: 2]
# ---------------------------------------------------------
with tab1:
    st.subheader("โจทย์ปัญหาค่าเริ่มต้น (Initial-Value Problem)")
    st.write("สมการเชิงอนุพันธ์ที่ต้องการหาคำตอบแบบประมาณค่า:")
    st.latex(
        r"y' = \frac{y}{t} - \frac{y^2}{t^2}, \quad 1 \le t \le 2, \quad y(1)"
        r" = 1"
    )

    st.subheader("คำตอบที่แท้จริง (Exact Solution)")
    st.latex(r"y(t) = \frac{t}{1 + \ln(t)}")

    st.subheader("ระเบียบวิธีของออยเลอร์ (Euler's Method)")
    st.latex(r"y_{i+1} = y_i + h \cdot f(t_i, y_i)")
    st.latex(
        r"y_{i+1} = y_i + h \left( \frac{y_i}{t_i} - \frac{y_i^2}{t_i^2}"
        r" \right)"
    )

# ---------------------------------------------------------
# Tab 2: ตัวจำลองและแสดงกราฟปฏิสัมพันธ์
# ข้อกำหนดข้อ 3: Dynamic Visualization (ใช้ st.plotly_chart)[cite: 2]
# ข้อกำหนดข้อ 4: Step-by-Step Calculation (ใช้ st.metric)[cite: 2]
# ---------------------------------------------------------
with tab2:
    st.subheader("เปรียบเทียบ Exact Solution และ Numerical Solution")

    # แสดงผลสรุปด้วย st.metric[cite: 2]
    m1, m2, m3 = st.columns(3)
    m1.metric("ค่า Step Size (h)", f"{h}")
    m2.metric("Error สูงสุด (Max Error)", f"{np.max(errors):.{precision}f}")
    m3.metric("Error เฉลี่ย (Mean Error)", f"{np.mean(errors):.{precision}f}")

    # สร้างกราฟเปรียบเทียบด้วย Plotly[cite: 2]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t_values,
            y=exact_values,
            mode="lines",
            name="Exact Solution",
            line=dict(color="#1f77b4", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t_values,
            y=euler_values,
            mode="lines+markers",
            name="Euler's Method",
            line=dict(color="#ff7f0e", dash="dash", width=2),
            marker=dict(size=6),
        )
    )

    fig.update_layout(
        title=f"การประมาณค่าด้วยวิธี Euler (h = {h})",
        xaxis_title="t",
        yaxis_title="y(t)",
        template=chart_theme,
        hovermode="x unified",
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# Tab 3: ตารางเปรียบเทียบผลลัพธ์
# ข้อกำหนดข้อ 4: Step-by-Step Calculation (ใช้ st.dataframe)[cite: 2]
# ---------------------------------------------------------
with tab3:
    st.subheader(f"ตารางเปรียบเทียบค่าที่คำนวณได้ (h = {h})")

    # จัดการรูปแบบทศนิยมตามที่ผู้ใช้เลือก
    df_display = df.copy()
    df_display["t_i"] = df_display["t_i"].map(lambda x: f"{x:.2f}")
    df_display["Euler's"] = df_display["Euler's"].map(
        lambda x: f"{x:.{precision}f}"
    )
    df_display["Exact"] = df_display["Exact"].map(
        lambda x: f"{x:.{precision}f}"
    )
    df_display["Error"] = df_display["Error"].map(
        lambda x: f"{x:.{precision}f}"
    )

    # แสดงตารางเปรียบเทียบ[cite: 1, 2]
    st.dataframe(df_display, use_container_width=True)
