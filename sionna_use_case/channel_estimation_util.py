from sionna.channel import gen_single_sector_topology
import tensorflow as tf


def sample_channel(batch_size, channel_model, channel_sampler, speed):
    # Sample random topologies
    topology = gen_single_sector_topology(batch_size, 1, 'umi', min_ut_velocity=speed, max_ut_velocity=speed)
    channel_model.set_topology(*topology)

    # Sample channel frequency responses
    # [batch size, 1, num_rx_ant, 1, 1, num_ofdm_symbols, fft_size]
    h_freq = channel_sampler(batch_size)
    # [batch size, num_rx_ant, num_ofdm_symbols, fft_size]
    h_freq = h_freq[:, 0, :, 0, 0]

    return h_freq


@tf.function(jit_compile=True)  # Use XLA for speed-up
def estimate_covariance_matrices(num_it, batch_size, fft_size, num_ofdm_symbols, num_rx_ant, channel_model, channel_sampler, speed):
    freq_cov_mat = tf.zeros([fft_size, fft_size], tf.complex64)
    time_cov_mat = tf.zeros([num_ofdm_symbols, num_ofdm_symbols], tf.complex64)
    space_cov_mat = tf.zeros([num_rx_ant, num_rx_ant], tf.complex64)
    for _ in tf.range(num_it):
        # [batch size, num_rx_ant, num_ofdm_symbols, fft_size]
        h_samples = sample_channel(batch_size, channel_model, channel_sampler, speed)

        #################################
        # Estimate frequency covariance
        #################################
        # [batch size, num_rx_ant, fft_size, num_ofdm_symbols]
        h_samples_ = tf.transpose(h_samples, [0, 1, 3, 2])
        # [batch size, num_rx_ant, fft_size, fft_size]
        freq_cov_mat_ = tf.matmul(h_samples_, h_samples_, adjoint_b=True)
        # [fft_size, fft_size]
        freq_cov_mat_ = tf.reduce_mean(freq_cov_mat_, axis=(0,1))
        # [fft_size, fft_size]
        freq_cov_mat += freq_cov_mat_

        ################################
        # Estimate time covariance
        ################################
        # [batch size, num_rx_ant, num_ofdm_symbols, fft_size]
        time_cov_mat_ = tf.matmul(h_samples, h_samples, adjoint_b=True)
        # [num_ofdm_symbols, num_ofdm_symbols]
        time_cov_mat_ = tf.reduce_mean(time_cov_mat_, axis=(0,1))
        # [num_ofdm_symbols, num_ofdm_symbols]
        time_cov_mat += time_cov_mat_

        ###############################
        # Estimate spatial covariance
        ###############################
        # [batch size, num_ofdm_symbols, num_rx_ant, fft_size]
        h_samples_ = tf.transpose(h_samples, [0,2,1,3])
        # [batch size, num_ofdm_symbols, num_rx_ant, num_rx_ant]
        space_cov_mat_ = tf.matmul(h_samples_, h_samples_, adjoint_b=True)
        # [num_rx_ant, num_rx_ant]
        space_cov_mat_ = tf.reduce_mean(space_cov_mat_, axis=(0,1))
        # [num_rx_ant, num_rx_ant]
        space_cov_mat += space_cov_mat_

    freq_cov_mat /= tf.complex(tf.cast(num_ofdm_symbols * num_it, tf.float32), 0.0)
    time_cov_mat /= tf.complex(tf.cast(fft_size * num_it, tf.float32), 0.0)
    space_cov_mat /= tf.complex(tf.cast(fft_size * num_it, tf.float32), 0.0)

    return freq_cov_mat, time_cov_mat, space_cov_mat


def evaluate_mse(model, snr_dbs, batch_size, num_it):

    # Casting model inputs to TensorFlow types to avoid
    # re-building of the graph
    snr_dbs = tf.cast(snr_dbs, tf.float32)
    batch_size = tf.cast(batch_size, tf.int32)

    mses = []
    for snr_db in snr_dbs:

        mse_ = 0.0
        for _ in range(num_it):
            mse_ += model(batch_size, snr_db).numpy()
        # Averaging over the number of iterations
        mse_ /= float(num_it)
        mses.append(mse_)

    return mses
