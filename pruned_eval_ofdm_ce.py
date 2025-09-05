import numpy as np
from tqdm import tqdm
import tensorflow as tf
from pathlib import Path

tf.get_logger().setLevel('ERROR')
try:
    import sionna as sn
except AttributeError:
    import sionna as sn

from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEInterpolator
from sionna.channel import GenerateOFDMChannel, OFDMChannel, gen_single_sector_topology
from sionna.channel.tr38901 import UMi, Antenna, PanelArray
from sionna.utils import QAMSource
import matplotlib.pyplot as plt
from dataset_classes.ofdm_channel_estimation import OfdmChannelEstimation
import models_ofdm_ce
import torch

from sionna_use_case.channel_estimation_util import estimate_covariance_matrices

import timm

from pruned_engine_finetune_regression import forward
 
def calculate_mse(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) ** 2)


def calculate_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


normalized = False
# load model
device = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_keys = ['small_ce_weighted_l2']
model_type = 'ce_small_patch16'
ckpt = '%s_checkpoint-230.pth'


models = list()

model = torch.load('/home/ict317-3/Mohammad/Tiny-WFMs/pruning_results/ofdm_ce/best_model.pth', weights_only=False)
for m in model.modules():
    if isinstance(m, timm.models.vision_transformer.Attention):
        m.forward = forward.__get__(m, timm.models.vision_transformer.Attention)


model = model.to(device)

models.append(model)

# system parameters
subcarrier_spacing = 30e3  # Hz
carrier_frequency = 3.5e9  # Hz
speed = 3.  # m/s
fft_size = 12 * 4   # 4 PRBs
num_ofdm_symbols = 14
num_rx_ant = 16

# The user terminals (UTs) are equipped with a single antenna
# with vertial polarization.
ut_antenna = Antenna(polarization='single',
                     polarization_type='V',
                     antenna_pattern='omni',  # Omnidirectional antenna pattern
                     carrier_frequency=carrier_frequency)


bs_array = PanelArray(num_rows_per_panel=4,
                      num_cols_per_panel=2,
                      polarization='dual',
                      polarization_type='cross',
                      antenna_pattern='38.901',  # 3GPP 38.901 antenna pattern
                      carrier_frequency=carrier_frequency)

qam_source = QAMSource(num_bits_per_symbol=2)
rg = ResourceGrid(num_ofdm_symbols=num_ofdm_symbols,
                  fft_size=fft_size,
                  subcarrier_spacing=subcarrier_spacing,
                  num_tx=1,
                  pilot_pattern="kronecker",
                  pilot_ofdm_symbol_indices=[2, 11])
rg_mapper = ResourceGridMapper(rg)
# 3GPP UMi channel model is considered
channel_model = UMi(carrier_frequency=carrier_frequency,
                    o2i_model='low',
                    ut_array=ut_antenna,
                    bs_array=bs_array,
                    direction='uplink',
                    enable_shadow_fading=False,
                    enable_pathloss=False)
channel = OFDMChannel(channel_model, rg, return_channel=True)
channel_sampler = GenerateOFDMChannel(channel_model, rg)
sn.config.xla_compat = True  # Enable Sionna's support of XLA
freq_cov_mat, time_cov_mat, space_cov_mat = (
    estimate_covariance_matrices(
        100, 1000, fft_size, num_ofdm_symbols, num_rx_ant, channel_model, channel_sampler, speed))
sn.config.xla_compat = False  # Disable Sionna's support of XLA

# freq_cov_mat : [fft_size, fft_size]
# time_cov_mat : [num_ofdm_symbols, num_ofdm_symbols]
# space_cov_mat : [num_rx_ant, num_rx_ant]

# freq_cov_mat = tf.constant(np.load('sionna_use_case/freq_cov_mat.npy'), tf.complex64)
# time_cov_mat = tf.constant(np.load('sionna_use_case/time_cov_mat.npy'), tf.complex64)
# space_cov_mat = tf.constant(np.load('sionna_use_case/space_cov_mat.npy'), tf.complex64)
ls_estimator = LSChannelEstimator(rg, interpolation_type='nn')
lmmse_int_freq_first = LMMSEInterpolator(rg.pilot_pattern, time_cov_mat, freq_cov_mat, space_cov_mat, order='t-f-s')
lmmse_estimator = LSChannelEstimator(rg, interpolator=lmmse_int_freq_first)


dataset = OfdmChannelEstimation(Path('/home/ict317-3/Mohammad/Tiny-WFMs/downstream_tasks_datasets/channel_estimation_dataset_-10,20/val_preprocessed'))
all_snr_db = range(-10, 21, 2)
mse_models = np.zeros((len(models), len(all_snr_db),))
mse_ls = np.zeros((len(all_snr_db),))
mse_lmmse = np.zeros((len(all_snr_db),))

batch_size = 64
num_it = 5

with torch.no_grad():
    for i, snr_db in enumerate(all_snr_db):
        tqdm.write(f"SNR = {snr_db}\n\r")
        no = tf.pow(10.0, -snr_db / 10.0)
        for _ in tqdm(range(num_it), total=num_it, desc='Iteration'):
            x = qam_source([batch_size, 1, 1, rg.num_data_symbols])
            x_rg = rg_mapper(x)
            topology = gen_single_sector_topology(batch_size, 1, 'umi', min_ut_velocity=speed, max_ut_velocity=speed)
            channel_model.set_topology(*topology)
            y_rg, h_freq = channel((x_rg, no))
            h_ls = np.squeeze(ls_estimator((y_rg, no))[0].numpy())
            h_lmmse = np.squeeze(lmmse_estimator((y_rg, no))[0].numpy())
            h_freq = np.squeeze(h_freq.numpy())
            mse_ls[i] += calculate_mse(h_freq, h_ls)
            mse_lmmse[i] += calculate_mse(h_freq, h_lmmse)
            x_rg = np.squeeze(x_rg.numpy())
            y_rg = np.squeeze(y_rg.numpy())
            x_model = dataset.create_sample(x_rg, y_rg).to(device)
            h_freq = np.concatenate([h_freq[:, :, i] for i in range(14)], axis=-1)
            for j, model in enumerate(models):
                if normalized:
                    h_model = dataset.reverse_normalize(model(x_model).cpu().numpy())
                else:
                    h_model = model(x_model).cpu().numpy()
                h_model = h_model[:, 0] + 1j * h_model[:, 1]
                mse_models[j, i] += calculate_mse(h_freq, h_model)

mse_models /= num_it
mse_ls /= num_it
mse_lmmse /= num_it

model_colors = ['r', 'c', 'm', 'y']
model_name = 'ViT-S' if model_type == 'ce_small_patch16' else model_type

plt.rcParams['font.family'] = 'serif'
fig, ax = plt.subplots(1, 1)
# ax.set_title(f'MIMO OFDM Channel Estimation\nFinetuning {model_name} ({ckpt})')
ax.semilogy(all_snr_db, mse_ls, label='LS Estimator', color='b', linewidth=2, marker='*')
ax.semilogy(all_snr_db, mse_lmmse, label='LMMSE Estimator', color='g', linewidth=2, marker='s')
for i, model in enumerate(models):
    ax.semilogy(all_snr_db, mse_models[i], label='SRFM', color=model_colors[i], linewidth=2, marker='o')

ax.legend(loc='lower left', fontsize=13)
ax.grid(True)
ax.set_xlabel('SNR (dB)', fontsize=16)
ax.set_ylabel('MSE', fontsize=16)
plt.tight_layout()
plt.savefig('/home/ict317-3/Mohammad/Tiny-WFMs/pruning_results/ofdm_ce/ofdm_ce_mse.png')
plt.show()
test = []
