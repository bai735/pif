import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.fft import fft, fftfreq

# 设置中文显示
matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# ==================== 1. 四分之一车体模型参数 ====================
class QuarterCarModel:
    """四分之一车体模型"""
    def __init__(self):
        # 簧上质量（车身）[kg]
        self.ms = 400.0
        # 簧下质量（车轮）[kg]  
        self.mu = 40.0
        # 悬架刚度 [N/m]
        self.ks = 25000.0
        # 悬架阻尼系数 [N·s/m]
        self.cs = 1500.0
        # 轮胎刚度 [N/m]
        self.kt = 200000.0
    
    def get_params(self):
        """获取所有参数"""
        return {
            'ms': self.ms, 'mu': self.mu,
            'ks': self.ks, 'cs': self.cs, 'kt': self.kt
        }

# ==================== 2. B级路面激励生成 ====================
def generate_b_road_profile(v, t, random_seed=42):
    """
    使用谐波叠加法生成B级路面激励
    
    参数:
        v: 车速 [m/s]
        t: 时间数组 [s]
        random_seed: 随机种子
    
    返回:
        road_profile: 路面不平度序列 [m]
    """
    # B级路面参数（ISO 8608标准）
    Gd_n0 = 64e-6      # B级路面的路面不平度系数 [m^3]
    n0 = 0.1           # 参考空间频率 [cycles/m]
    w = 2              # 频率指数
    
    # 空间频率范围
    n_min = 0.011      # 最低空间频率 [cycles/m]
    n_max = 2.83       # 最高空间频率 [cycles/m]
    n_points = 200     # 频率采样点数
    
    # 生成空间频率点（对数分布）
    n = np.logspace(np.log10(n_min), np.log10(n_max), n_points)
    dn = np.diff(np.append([n_min], n))
    
    # 计算功率谱密度
    Gd = Gd_n0 * (n / n0) ** (-w)
    
    # 计算振幅
    val = 2 * Gd *dn
    val = np.maximum (val, 0)
    A = np.sqrt(val)
    
    # 随机相位角
    np.random.seed(random_seed)
    phi = 2 * np.pi * np.random.rand(n_points)
    
    # 生成路面不平度时域信号
    road_profile = np.zeros(len(t))
    for i, freq in enumerate(n):
        omega = 2 * np.pi * freq * v  # 时间角频率
        road_profile += A[i] * np.sin(omega * t + phi[i])
    
    return road_profile

# ==================== 3. 运动微分方程 ====================
def quarter_car_ode(t, y, params, road_func):
    """
    四分之一车体模型的微分方程
    
    状态变量 y = [zs, zu, dzs, dzu]
    zs: 簧上质量位移
    zu: 簧下质量位移
    dzs: 簧上质量速度
    dzu: 簧下质量速度
    """
    zs, zu, dzs, dzu = y
    
    ms = params['ms']
    mu = params['mu']
    ks = params['ks']
    cs = params['cs']
    kt = params['kt']
    
    # 获取当前时刻的路面激励
    r = road_func(t)
    
    # 计算悬架力和轮胎力
    F_spring = ks * (zs - zu)
    F_damper = cs * (dzs - dzu)
    F_tire = kt * (zu - r)
    
    # 加速度方程
    ddzs = (-F_spring - F_damper) / ms
    ddzu = (F_spring + F_damper - F_tire) / mu
    
    return [dzs, dzu, ddzs, ddzu]

# ==================== 4. 性能指标计算 ====================
def calculate_metrics(t, zs, zu, r, params):
    """计算振动响应指标"""
    # 车身加速度
    ddzs = np.gradient(np.gradient(zs, t), t)
    
    # 悬架动挠度
    suspension_deflection = zs - zu
    
    # 轮胎动载荷
    kt = params['kt']
    tire_load = kt * (zu - r)
    
    # 均方根值
    rms_acc = np.sqrt(np.mean(ddzs**2))
    rms_deflection = np.sqrt(np.mean(suspension_deflection**2))
    rms_tire_load = np.sqrt(np.mean(tire_load**2))
    
    # 最大值
    max_acc = np.max(np.abs(ddzs))
    max_deflection = np.max(np.abs(suspension_deflection))
    max_tire_load = np.max(np.abs(tire_load))
    
    return {
        'time': t,
        'acceleration': ddzs,
        'deflection': suspension_deflection,
        'tire_load': tire_load,
        'rms_acc': rms_acc,
        'rms_deflection': rms_deflection,
        'rms_tire_load': rms_tire_load,
        'max_acc': max_acc,
        'max_deflection': max_deflection,
        'max_tire_load': max_tire_load
    }

# ==================== 5. 主程序 ====================
def main():
    print("=" * 60)
    print("四分之一车体模型 - B级路面振动响应计算")
    print("=" * 60)
    
    # 5.1 设置仿真参数
    v = 20.0                    # 车速 20 m/s = 72 km/h
    duration = 10.0             # 仿真时长 10 秒
    dt = 0.01                   # 时间步长 0.01 秒
    
    t = np.arange(0, duration, dt)
    
    print(f"\n仿真参数:")
    print(f"  车速: {v} m/s ({v*3.6:.0f} km/h)")
    print(f"  时长: {duration} s")
    print(f"  步长: {dt} s")
    car = QuarterCarModel()
    params = car.get_params()
    
    print(f"\n车辆参数:")
    print(f"  簧上质量: {params['ms']} kg")
    print(f"  簧下质量: {params['mu']} kg")
    print(f"  悬架刚度: {params['ks']} N/m")
    print(f"  悬架阻尼: {params['cs']} N·s/m")
    print(f"  轮胎刚度: {params['kt']} N/m")
    
    # 计算固有频率和阻尼比
    wn = np.sqrt(params['ks'] / params['ms'])
    fn = wn / (2 * np.pi)
    zeta = params['cs'] / (2 * np.sqrt(params['ks'] * params['ms']))
    print(f"  车身固有频率: {fn:.2f} Hz")
    print(f"  阻尼比: {zeta:.3f}")
    
    # 5.3 生成B级路面激励（插值函数）
    print("\n生成B级路面激励...")
    road_raw = generate_b_road_profile(v, t)
    
    # 创建插值函数供ODE求解器使用
    from scipy.interpolate import interp1d
    road_func = interp1d(t, road_raw, kind='linear', 
                         fill_value=(road_raw[0], road_raw[-1]), 
                         bounds_error=False)
    
    # 5.4 求解微分方程
    print("求解振动响应方程...")
    y0 = [0.0, 0.0, 0.0, 0.0]  # 初始状态：位移和速度均为0
    
    # 使用solve_ivp求解
    solution = solve_ivp(
        lambda t, y: quarter_car_ode(t, y, params, road_func),
        [0, duration], y0, 
        t_eval=t,
        method='RK45',
        rtol=1e-6, atol=1e-8
    )
    
    if not solution.success:
        print(f"求解失败: {solution.message}")
        return
    
    # 提取结果
    zs = solution.y[0]    # 车身位移
    zu = solution.y[1]    # 车轮位移
    dzs = solution.y[2]   # 车身速度
    dzu = solution.y[3]   # 车轮速度
    
    # 5.5 计算性能指标
    print("计算性能指标...")
    metrics = calculate_metrics(t, zs, zu, road_raw, params)
    
    # 输出结果
    print("\n" + "=" * 60)
    print("计算结果")
    print("=" * 60)
    print(f"\n车身加速度:")
    print(f"  均方根值 (RMS): {metrics['rms_acc']:.3f} m/s²")
    print(f"  最大值: {metrics['max_acc']:.3f} m/s²")
    
    print(f"\n悬架动挠度:")
    print(f"  均方根值 (RMS): {metrics['rms_deflection']*1000:.2f} mm")
    print(f"  最大值: {metrics['max_deflection']*1000:.2f} mm")
    
    print(f"\n轮胎动载荷:")
    print(f"  均方根值 (RMS): {metrics['rms_tire_load']:.1f} N")
    print(f"  最大值: {metrics['max_tire_load']:.1f} N")
    print("\n生成可视化图表...")
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    # 图1: 路面激励
    axes[0, 0].plot(t, road_raw * 1000, 'g-', linewidth=0.8)
    axes[0, 0].set_xlabel('时间 (s)')
    axes[0, 0].set_ylabel('路面不平度 (mm)')
    axes[0, 0].set_title(f'B级路面激励 (v={v*3.6:.0f} km/h)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim([0, duration])
    
    # 图2: 车身和车轮位移
    axes[0, 1].plot(t, zs * 1000, 'b-', label='车身位移', linewidth=1)
    axes[0, 1].plot(t, zu * 1000, 'r-', label='车轮位移', linewidth=1)
    axes[0, 1].set_xlabel('时间 (s)')
    axes[0, 1].set_ylabel('位移 (mm)')
    axes[0, 1].set_title('车身与车轮位移响应')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim([0, duration])
    
    # 图3: 车身加速度
    axes[1, 0].plot(t, metrics['acceleration'], 'b-', linewidth=0.8)
    axes[1, 0].set_xlabel('时间 (s)')
    axes[1, 0].set_ylabel('加速度 (m/s²)')
    axes[1, 0].set_title(f'车身加速度响应 (RMS: {metrics["rms_acc"]:.3f} m/s²)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim([0, duration])
    
    # 图4: 悬架动挠度
    axes[1, 1].plot(t, metrics['deflection'] * 1000, 'purple', linewidth=0.8)
    axes[1, 1].set_xlabel('时间 (s)')
    axes[1, 1].set_ylabel('动挠度 (mm)')
    axes[1, 1].set_title(f'悬架动挠度响应 (RMS: {metrics["rms_deflection"]*1000:.2f} mm)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim([0, duration])
    
    # 图5: 轮胎动载荷
    axes[2, 0].plot(t, metrics['tire_load'] / 1000, 'orange', linewidth=0.8)
    axes[2, 0].set_xlabel('时间 (s)')
    axes[2, 0].set_ylabel('动载荷 (kN)')
    axes[2, 0].set_title(f'轮胎动载荷响应 (RMS: {metrics["rms_tire_load"]/1000:.2f} kN)')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].set_xlim([0, duration])
    
    # 图6: 加速度功率谱密度
    freqs = fftfreq(len(metrics['acceleration']), dt)[:len(metrics['acceleration'])//2]
    acc_fft = np.abs(fft(metrics['acceleration'])[:len(metrics['acceleration'])//2])
    
    axes[2, 1].loglog(freqs[1:], acc_fft[1:], 'b-', linewidth=0.8)
    axes[2, 1].set_xlabel('频率 (Hz)')
    axes[2, 1].set_ylabel('加速度幅值 (m/s²/Hz)')
    axes[2, 1].set_title('车身加速度功率谱密度')
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].axvline(fn, color='r', linestyle='--', alpha=0.5, 
                       label=f'固有频率 {fn:.2f} Hz')
    axes[2, 1].legend()
    
    plt.tight_layout()
    plt.savefig('quarter_car_response.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n图表已保存为 'quarter_car_response.png'")
    print("\n计算完成！")

if __name__ == "__main__":
    main()