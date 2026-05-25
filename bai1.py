import numpy as np
# --------------------------
# 关键配置：解决 Wayland + 中文 + 直接显示
# --------------------------
import matplotlib
# 改用 TkAgg 后端（支持弹出窗口），如果还是卡死，可以换成 'Qt5Agg'
matplotlib.use('TkAgg')
# 中文配置（确保安装了文泉驿字体）
matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
matplotlib.rcParams['axes.unicode_minus'] = False  # 负号正常显示

import matplotlib.pyplot as plt
from scipy.integrate import odeint

# ==============================
# 一阶温度系统模型（微分方程）
# ==============================
class FirstOrderSystem:
    """一阶系统：tau * dT/dt + T = K * u"""
    def __init__(self, K=1.0, tau=10.0, T0=0.0):
        self.K = K      # 增益
        self.tau = tau  # 时间常数 (秒)
        self.T0 = T0    # 初始温度

    def dynamics(self, T, t, u):
        """微分方程 dT/dt = (K*u - T)/tau"""
        return (self.K * u - T) / self.tau

    def step_response(self, u, t_span):
        """给定阶跃输入 u（常数），返回温度响应"""
        t = np.linspace(t_span[0], t_span[1], int((t_span[1]-t_span[0])/0.01))
        T = odeint(self.dynamics, self.T0, t, args=(u,))
        return t, T.flatten()

# ==============================
# PID 控制器（增强版）
# ==============================
class PIDController:
    """
    位置式 PID 控制器，带抗积分饱和 (clamping) 和积分分离选项
    """
    def __init__(self, Kp, Ki, Kd, dt, output_limits=(0, 100), 
                 integral_sep_thresh=None, anti_windup=True):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.output_limits = output_limits   # (min, max)
        self.integral_sep_thresh = integral_sep_thresh  # 积分分离阈值，None 表示禁用
        self.anti_windup = anti_windup       # 输出限幅时是否停止积分累加

        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_output = 0.0

    def compute(self, setpoint, measurement):
        error = setpoint - measurement

        # 积分分离：当误差绝对值大于阈值时，不累积积分项
        if (self.integral_sep_thresh is not None 
                and abs(error) > self.integral_sep_thresh):
            Ki_eff = 0.0
        else:
            Ki_eff = self.Ki

        # 积分项
        self.integral += error * self.dt

        # 微分项（实际微分，非理想）
        derivative = (error - self.prev_error) / self.dt

        # PID 输出
        output = self.Kp * error + Ki_eff * self.integral + self.Kd * derivative

        # 输出限幅 + 抗积分饱和
        if self.output_limits is not None:
            if output < self.output_limits[0]:
                output = self.output_limits[0]
                if self.anti_windup and Ki_eff != 0:
                    # 停止积分增长，使积分项不再增大
                    self.integral -= error * self.dt
            elif output > self.output_limits[1]:
                output = self.output_limits[1]
                if self.anti_windup and Ki_eff != 0:
                    self.integral -= error * self.dt

        # 保存误差供下次微分用
        self.prev_error = error
        self.prev_output = output
        return output

# ==============================
# 闭环仿真器
# ==============================
def run_closed_loop_simulation(plant, pid, setpoint_profile, t_span, dt, 
                               disturbance=None):
    """
    执行闭环仿真
    参数：
        plant: FirstOrderSystem 对象
        pid: PIDController 对象
        setpoint_profile: 函数或数组，给定时间返回设定值
        t_span: (t_start, t_end)
        dt: 步长
        disturbance: 可选扰动函数 d(t)，作用于系统输入端
    返回：
        t, T, u, setpoint_arr
    """
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt) + 1
    t = np.linspace(t_start, t_end, n_steps)

    T = np.zeros(n_steps)
    u = np.zeros(n_steps)
    setpoint_arr = np.zeros(n_steps)

    # 初始条件
    T[0] = plant.T0
    pid.reset()

    for i in range(n_steps - 1):
        # 当前设定值
        if callable(setpoint_profile):
            sp = setpoint_profile(t[i])
        else:
            sp = setpoint_profile[i] if i < len(setpoint_profile) else setpoint_profile[-1]
        setpoint_arr[i] = sp

        # PID 计算控制量
        u[i] = pid.compute(sp, T[i])

        # 加入外部扰动（直接作用于控制输入）
        if disturbance is not None:
            u_eff = u[i] + disturbance(t[i])
        else:
            u_eff = u[i]

        # 一阶系统递推（欧拉法）
        dT = plant.dynamics(T[i], t[i], u_eff)
        T[i+1] = T[i] + dT * dt

    # 最后一步记录设定值
    setpoint_arr[-1] = setpoint_profile(t[-1]) if callable(setpoint_profile) else setpoint_profile[-1]
    u[-1] = u[-2]   # 保持一致

    return t, T, u, setpoint_arr

# ==============================
# 主程序：仿真 + 直接显示图片
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

    # 3. 绘图（直接弹出窗口显示）
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

    # 关键：直接显示图片
    print("正在打开绘图窗口...")
    plt.show()
