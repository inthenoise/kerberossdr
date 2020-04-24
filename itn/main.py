import sys
import os
import time

import numpy as np
import scipy

from scipy import fft,ifft
from scipy import signal
from scipy.signal import correlate
from pyargus import directionEstimation as de

currentPath = os.path.dirname(os.path.realpath(__file__))
rootPath = os.path.dirname(currentPath)
receiverPath        = os.path.join(rootPath, "_receiver")
signalProcessorPath = os.path.join(rootPath, "_signalProcessing")

sys.path.insert(0, receiverPath)
sys.path.insert(0, signalProcessorPath)

from hydra_receiver import ReceiverRTLSDR

GAIN = "42.1"
CENTER_FREQ = "381.1875"
SAMPLE_RATE = "1.024"
BUF_SIZE = 256
ENABLE_DC_COMP = False
FIR_SIZE = 0
DECIMATION_RATIO = 1
DOA_SAMPLE_SIZE = 2**15
X_CORR_SAMPLE_SIZE = 2**18
DOA_INTER_ELEM_SPACE = 0.12
DOA_ANT_ALIGNMENT = "ULA"
PHASOR_WIN = 2**10
NUM_CHANNELS = 4


class SignalProcessor(object):
    def __init__(self):
        self.receiver = ReceiverRTLSDR()
        self.receiver.block_size = BUF_SIZE * 1024
        self.delay_log= np.array([[0],[0],[0]])
        self.phase_log= np.array([[0],[0],[0]])
        self.doa_bartlett_res = np.ones(181)
        self.doa_capon_res = np.ones(181)
        self.doa_mem_res = np.ones(181)
        self.doa_music_res = np.ones(181)
        self.doa_theta = np.arange(0,181,1)

        center_freq = float(CENTER_FREQ) *10**6
        sample_rate = float(SAMPLE_RATE) *10**6
        gain_val = 10*float(GAIN)
        gain = [
            gain_val,
            gain_val,
            gain_val,
            gain_val,
        ]
        self.receiver.receiver_gain = gain_val
        self.receiver.receiver_gain_2 = gain_val
        self.receiver.receiver_gain_3 = gain_val
        self.receiver.receiver_gain_4 = gain_val
        self.receiver.fs = sample_rate
        self.receiver.en_dc_compensation = ENABLE_DC_COMP
        self.receiver.reconfigure_tuner(center_freq, sample_rate, gain)

        self.test = None
        self.spectrum_sample_size = 2**14 #2**14
        self.spectrum = np.ones((NUM_CHANNELS + 1, self.spectrum_sample_size), dtype=np.float32)
        self.doa_sample_size = 2**15 # Connect to GUI value??
        self.xcorr_sample_size = 2**18 

    def sync(self):
        N = self.xcorr_sample_size
        iq_samples = self.receiver.iq_samples[:, 0:N]
       
        delays = np.array([[0],[0],[0]])
        phases = np.array([[0],[0],[0]])

        # Channel matching
        np_zeros = np.zeros(N, dtype=np.complex64)
        x_padd = np.concatenate([iq_samples[0, :], np_zeros])
        x_fft = np.fft.fft(x_padd)

        for m in np.arange(1, NUM_CHANNELS):
            y_padd = np.concatenate([np_zeros, iq_samples[m, :]])
            y_fft = np.fft.fft(y_padd)
            self.xcorr[m-1] = np.fft.ifft(x_fft.conj() * y_fft)
            delay = np.argmax(np.abs(self.xcorr[m-1])) - N
            phase = np.rad2deg(np.angle(self.xcorr[m-1, N]))
            
            delays[m-1,0] = delay
            phases[m-1,0] = phase

        self.delay_log = np.concatenate((self.delay_log, delays),axis=1)
        self.phase_log = np.concatenate((self.phase_log, phases),axis=1)
        self.receiver.set_sample_offsets(self.delay_log[:,-1])

    def calibrate_iq(self):
        for m in range(NUM_CHANNELS):
            self.receiver.iq_corrections[m] *= np.size(
                self.receiver.iq_samples[0, :]) / (np.dot(
                    self.receiver.iq_samples[m, :],
                    self.receiver.iq_samples[0, :].conj()
                ))
        c = np.sqrt(np.sum(np.abs(self.receiver.iq_corrections)**2))
        self.receiver.iq_corrections = np.divide(self.receiver.iq_corrections, c)

    def get_doa_estimate(self):
        self.spectrum[1,:] = 10*np.log10(np.fft.fftshift(np.abs(np.fft.fft(self.receiver.iq_samples[0, 0:self.spectrum_sample_size]))))
        iq_samples = self.receiver.iq_samples[:, 0:self.doa_sample_size]
        R = de.corr_matrix_estimate(iq_samples.T, imp="fast")
        M = np.size(iq_samples, 0)
        self.doa_theta = np.linspace(0,360,361)
        x = DOA_INTER_ELEM_SPACE * np.cos(2*np.pi/M * np.arange(M))
        y = DOA_INTER_ELEM_SPACE * np.sin(-2*np.pi/M * np.arange(M)) # For this specific array only
        scanning_vectors = de.gen_scanning_vectors(M, x, y, self.doa_theta)

        self.doa_bartlett_res = de.DOA_Bartlett(R, scanning_vectors)

        thetas = self.doa_theta
        bartlett = self.doa_bartlett_res
        doa = 0
        doa_results = []
        combined = np.zeros_like(thetas, dtype=np.complex)
        plt = self.doa_plot_helper(bartlett, thetas)
        combined += np.divide(np.abs(bartlett), np.max(np.abs(bartlett)))
        doa_results.append(thetas[np.argmax(bartlett)])

        if len(doa_results) != 0:
            combined_log = self.doa_plot_helper(combined, thetas)
            confidence = scipy.signal.find_peaks_cwt(combined_log, np.arange(10,30), min_snr=1)
            maxIndex = confidence[np.argmax(combined_log[confidence])]
            confidence_sum = 0

            for val in confidence:
               if (val != maxIndex and np.abs(combined_log[val] - min(combined_log)) > np.abs(min(combined_log))*0.25):
                  confidence_sum += 1/(np.abs(combined_log[val]))
               elif val == maxIndex:
                  confidence_sum += 1/np.abs(min(combined_log))
            max_power_level = np.max(self.spectrum[1,:])
            confidence_sum = 10 / confidence_sum
            doa_results = np.array(doa_results)
            doa_results_c = np.exp(1j*np.deg2rad(doa_results))
            doa_avg_c = np.average(doa_results_c)
            doa = np.rad2deg(np.angle(doa_avg_c))

            if doa < 0:
                doa += 360
            print("---------- DOA Estimate ----------")
            print(str(int(doa)))
            print(str(int(confidence_sum)))
            print(str(np.maximum(0, max_power_level)))


    def doa_plot_helper(self, doa_data, incident_angles, log_scale_min=-50):
        doa_data = np.divide(np.abs(doa_data), np.max(np.abs(doa_data)))
        if log_scale_min is not None:
            DOA_data = 10*np.log10(doa_data)
            theta_index = 0
            for theta in incident_angles:
                if doa_data[theta_index] < log_scale_min:
                    doa_data[theta_index] = log_scale_min
                theta_index += 1
        return doa_data

    def run(self):
        while True:
            start_time = time.time()
            self.receiver.download_iq_samples()
            self.doa_sample_size = self.receiver.iq_samples[0,:].size
            self.xcorr_sample_size = self.receiver.iq_samples[0,:].size
            self.xcorr = np.ones((NUM_CHANNELS-1, self.xcorr_sample_size*2), dtype=np.complex64)        

            self.sync()
            self.calibrate_iq()
            self.get_doa_estimate()

            time.sleep(.5)



if __name__ == '__main__':
    sp = SignalProcessor()
    sp.run()