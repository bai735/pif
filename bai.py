import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
# 配置中文字体
matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# --- 系统参数 ---
K = 1.0          # 增益
tau = 10.0       # 时间常数（秒）
dt = 0.1         # 仿真步长
t_end = 100      # 仿真总时长

# 参考设定点
setpoint = 50.0  # ℃

# PID 参数
Kp = 2.0
Ki = 0.2
Kd = 1.0

# --- 初始化 ---
n_steps = int(t_end / dt)
time = np.linspace(0, t_end, n_steps)
y = np.zeros(n_steps)      # 系统输出（温度）
u = np.zeros(n_steps)      # 控制量
e = np.zeros(n_steps)      # 误差

integral = 0.0             # 积分项累积
prev_error = 0.0           # 上一时刻误差

# 初始温度假设为环境温度（0℃），也可设为其他值
y[0] = 0.0

# --- 主仿真循环 ---
for i in range(1, n_steps):
    # 当前误差
    e[i] = setpoint - y[i-1]
    
    # PID 计算（增量式位置型）
    integral += e[i] * dt
    derivative = (e[i] - prev_error) / dt
    
    u[i] = Kp * e[i] + Ki * integral + Kd * derivative
    
    # 控制量限幅（例如 0~100% 对应 0~100 单位加热功率）
    u[i] = max(0.0, min(100.0, u[i]))
    
    # 一阶系统差分方程（零阶保持离散化）
    # 传递函数 G(s)=K/(tau*s+1) -> 差分方程: y[k] = y[k-1] + (dt/tau)*(K*u[k-1] - y[k-1])
    y[i] = y[i-1] + (dt / tau) * (K * u[i-1] - y[i-1])
    
    # 更新上一时刻误差
    prev_error = e[i]

# --- 绘图 ---



plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time, y, label='系统输出（温度）')
plt.plot(time, [setpoint]*n_steps, 'r--', label='设定点')
plt.ylabel('温度 (℃)')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(time, u, label='控制量')
plt.xlabel('时间 (s)')
plt.ylabel('控制量 (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
