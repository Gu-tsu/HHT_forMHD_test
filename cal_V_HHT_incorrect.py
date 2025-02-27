import os
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from PyEMD import EMD
from scipy.signal import hilbert

def read_all_time_steps(file_path, nr=257, nz=321, choice_nr=90):
    file_list = sorted(os.listdir(file_path))
    wave_data_list = []
    valid_file_count = 0
    for filename in file_list:
        file_full_path = os.path.join(file_path, filename)
        if os.path.isfile(file_full_path):
            with open(file_full_path, 'r') as f:
                data_str = f.read().replace('D', 'E')
            data = np.loadtxt(StringIO(data_str))
            if data.size != nr * nz:
                print(f"数据长度与二维模式 (NR={nr}, nz={nz}) 不符: {file_full_path}")
                continue  # 或者直接 exit()
            array2d = data.reshape(nr, nz)
            adjust_nr = choice_nr - 1
            if adjust_nr < 0 or adjust_nr >= nr:
                print(f"choice_nr 超出范围 (1, {nr}).")
                continue
            selected_line = array2d[adjust_nr, :]
            wave_data_list.append(selected_line)
            valid_file_count += 1
    wave_data = np.array(wave_data_list)
    print(f"读取完成，共 {valid_file_count} 个有效文件，wave_data 形状: {wave_data.shape}")
    return wave_data


def compute_instant_freq(signal_1d, dt=1.0):
#    对单个一维信号(signal_1d)做 EMD 分解，然后对主要IMF做 Hilbert 变换，
#    得到瞬时相位phi(t)，再通过相位随时间的导数得到瞬时频率omega(t)。
#    返回:inst_phase:  瞬时相位(已unwrap), inst_freq:   瞬时频率 (omega(t))，长度与signal_1d相同
    emd = EMD()
    imfs = emd.emd(signal_1d)

    # 这里演示只取“能量最大”的IMF
    imf_energies = [np.sum(imf ** 2) for imf in imfs]
    max_imf_index = np.argmax(imf_energies)
    main_imf = imfs[max_imf_index]

    # Hilbert 变换
    analytic_signal = hilbert(main_imf)
    inst_phase_raw = np.angle(analytic_signal)
    inst_phase = np.unwrap(inst_phase_raw)

    # 瞬时频率 = d(phi)/dt
    inst_freq = np.gradient(inst_phase, dt)
    return inst_phase, inst_freq

def compute_wave_velocity(wave_data, physical_Z=2.24, dt=2.5142456E-7, z_index_1=100, z_index_2=150):
#    在 wave_data(形状: [time_steps, nz]) 上，
#    1. 分别对 z_index_1 和 z_index_2 两个空间点的时间信号做 EMD + Hilbert，
#    2. 取它们的瞬时相位差来估计 k(t)，取频率平均来估计 omega(t)，
#    3. 得到相速度 v(t) = omega(t) / k(t)。
    time_steps, nz = wave_data.shape
    if z_index_1 < 0 or z_index_1 >= nz or z_index_2 < 0 or z_index_2 >= nz:
        raise ValueError("z_index_1 或 z_index_2 超出范围")
    if z_index_1 == z_index_2:
        raise ValueError("需要两个不同的空间点以计算波数")

    # 计算网格间距
    dx = physical_Z / (nz - 1)

    # 分别拿到这两个点的时间序列
    signal_1 = wave_data[:, z_index_1]
    signal_2 = wave_data[:, z_index_2]

    # 得到它们的瞬时相位 & 瞬时频率
    phase_1, freq_1 = compute_instant_freq(signal_1, dt=dt)
    phase_2, freq_2 = compute_instant_freq(signal_2, dt=dt)

    # 波数 k(t) ~= (phase_2 - phase_1)/dx
    phase_diff = phase_2 - phase_1

    # 这里假设逐点 diff
    grid_dist = (z_index_2 - z_index_1) * dx
    k_of_t = phase_diff / grid_dist

    # 频率可取 freq_1, freq_2 的平均
    freq_avg = 0.5 * (freq_1 + freq_2)

    # 相速度 v(t) = freq_avg / k_of_t
    v_of_t = np.zeros(time_steps)
    eps = 1e-30
    for i in range(time_steps):
        if abs(k_of_t[i]) < eps:
            v_of_t[i] = np.nan  # 或者 0.0
        else:
            v_temp = freq_avg[i] / k_of_t[i]
            # v_temp < 0，则置为 NaN
            if v_temp < 0:
                v_of_t[i] = np.nan
            else:
                v_of_t[i] = v_temp

    # 时间数组
    t_array = np.arange(time_steps) * dt
    return t_array, v_of_t


def main():
    file_path = r"C:\Users\Administrator\Desktop\正面bth0+5数据1周期"
    NR = 257
    nz = 321
    choice_nr = 218
    physical_Z = 2.24
    dt = 2.5142456e-7

    # 选用两个空间下标来估计波数
    z_index_1 = 120
    z_index_2 = 320

    # 读取数据
    wave_data = read_all_time_steps(file_path, nr=NR, nz=nz, choice_nr=choice_nr)
    if wave_data.size == 0:
        print("无有效数据")
        return

    # 计算波速
    t_array, v_of_t = compute_wave_velocity(wave_data,physical_Z=physical_Z,dt=dt,z_index_1=z_index_1,z_index_2=z_index_2)

    # 波速随时间的演变可视化
    plt.figure(figsize=(8, 4))
    plt.plot(t_array, v_of_t, '-o', markersize=3)
    plt.xlabel("Time (s)")
    plt.ylabel("Instantaneous Phase Velocity (m/s)")
    plt.title("Wave Velocity vs Time (HHT-based)")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
