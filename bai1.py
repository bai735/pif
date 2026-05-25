
import numpy as np
import matplotlib
# 强制使用 Wayland 兼容的后端，避免卡死
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.integrate import odeint
matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  
# ==============================
# 一阶温度系统模型
# ==============================
class FirstOrderSystem:
    def __init__(self, K=1.0, tau=10.0, T0=0.0):
        self.K = K
        self.tau = tau
        self.T0 = T0

    def dynamics(self, T, t, u):
        return (self.K * u - T) / self.tau

    def step_response(self, u, t_span):
        t = np.linspace(t_span[0], t_span[1], int((t_span[1]-t_span[0])/0.01))
        T = odeint(self.dynamics, self.T0, t, args=(u,))
        return t, T.flatten()

# ==============================
# PID 控制器
# ==============================
class PIDController:
    def __init__(self, Kp, Ki, Kd, dt, output_limits=(0, 100), 
                 integral_sep_thresh=None, anti_windup=True):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.output_limits = output_limits
        self.integral_sep_thresh = integral_sep_thresh
        self.anti_windup = anti_windup
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, setpoint, measurement):
        error = setpoint - measurement

        if (self.integral_sep_thresh is not None 
                and abs(error) > self.integral_sep_thresh):
            Ki_eff = 0.0
        else:
            Ki_eff = self.Ki

        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        output = self.Kp * error + Ki_eff * self.integral + self.Kd * derivative

        if self.output_limits is not None:
            if output < self.output_limits[0]:
                output = self.output_limits[0]
                if self.anti_windup and Ki_eff != 0:
                    self.integral -= error * self.dt
            elif output > self.output_limits[1]:
                output = self.output_limits[1]
                if self.anti_windup and Ki_eff != 0:
                    self.integral -= error * self.dt

        self.prev_error = error
        return output

# ==============================
# 闭环仿真
# ==============================
def run_closed_loop_simulation(plant, pid, setpoint_profile, t_span, dt, 
                               disturbance=None):
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt) + 1
    t = np.linspace(t_start, t_end, n_steps)

    T = np.zeros(n_steps)
    u = np.zeros(n_steps)
    setpoint_arr = np.zeros(n_steps)

    T[0] = plant.T0
    pid.reset()

    for i in range(n_steps - 1):
        if callable(setpoint_profile):
            sp = setpoint_profile(t[i])
        else:
            sp = setpoint_profile[i] if i < len(setpoint_profile) else setpoint_profile[-1]
        setpoint_arr[i] = sp

        u[i] = pid.compute(sp, T[i])
        u_eff = u[i] + disturbance(t[i]) if disturbance is not None else u[i]
        dT = plant.dynamics(T[i], t[i], u_eff)
        T[i+1] = T[i] + dT * dt

    setpoint_arr[-1] = setpoint_profile(t[-1]) if callable(setpoint_profile) else setpoint_profile[-1]
    u[-1] = u[-2]
    return t, T, u, setpoint_arr

# ==============================
# 主程序：仿真 + 画图
# ==============================
if __name__ == "__main__":
    # 1. 配置系统和控制器
    plant = FirstOrderSystem(K=1.0, tau=10.0, T0=25.0)
    pid = PIDController(Kp=2.0, Ki=0.1, Kd=0.5, dt=0.1, output_limits=(0, 100))

    # 设定值：30秒后从25阶跃到100
    def setpoint_profile(t):
        return 25.0 if t < 30 else 100.0

    # 2. 运行仿真
    t_span = (0, 100)
    dt = 0.1
    t, T, u, setpoint_arr = run_closed_loop_simulation(
        plant, pid, setpoint_profile, t_span, dt
    )

    # 3. 绘图（保存为图片，避免 Wayland 交互卡死）
    plt.figure(figsize=(10, 8))

    # 子图1：温度响应与设定值
    plt.subplot(2, 1, 1)
    plt.plot(t, setpoint_arr, 'r--', label='设定值 (SP)')
    plt.plot(t, T, 'b-', linewidth=1.5, label='温度响应 (PV)')
    plt.ylabel('温度 (°C)')
    plt.title('一阶系统 PID 闭环阶跃响应')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图2：控制量输出
    plt.subplot(2, 1, 2)
    plt.plot(t, u, 'g-', linewidth=1.5, label='PID 控制量')
    plt.xlabel('时间 (s)')
    plt.ylabel('控制信号')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图片（不会弹出窗口，不会卡死）
    plt.savefig('pid_step_response.png', dpi=300, bbox_inches='tight')
    print("✅ 图片已保存为 pid_step_response.png，可直接打开查看")
