import lws
import numpy as np
import librosa
import scipy

def stft(x,fft_size=512,window_length=None,hop_length=None,analyse_window=None,perfectrec=True):
    # STFT with a fixed frame shift
    # Assumes that the frame shift is an integer ratio of the frame size for simplicity

    if len(np.shape(x)) != 1: # no multi-channel input
        raise ValueError('We only deal with single channel signals here')

    if fft_size % 2 ==1:
        raise ValueError('FFT size must be power of 2!')

    if window_length is None:
        window_length = fft_size

    if hop_length is None:
        hop_length = window_length // 4
        
    if window_length % hop_length != 0:
        raise ValueError('Frame shift should be integer ratio of frame size.')

    if analyse_window is None:
        analyse_window = lws.hann(window_length, True, False)
        if fft_size > window_length:
            pad = np.zeros((fft_size - window_length) // 2)
            analyse_window = np.hstack((pad, analyse_window, pad))
    
    if perfectrec is True:
        x = np.pad(x, int(fft_size // 2), mode='reflect')
    
    num_frame = 1 + int((len(x) - fft_size) / hop_length)

    frame_starts = hop_length * np.arange(num_frame)

    spec=np.zeros([num_frame, fft_size//2+1]).astype('complex128')
    
    for m in range(num_frame):
        frame = x[frame_starts[m]:frame_starts[m] + fft_size]*analyse_window
        temp  = scipy.fft(np.squeeze(frame),n=fft_size)
        spec[m]=temp[:fft_size//2+1]

    return spec

def istft(stft, hop_length=None, window_length=None, analyse_window=None, synthesis_window=None, perfectrec=True):
    
    # iSsignal_lengthFsignal_length with a fixed frame shift
    # Assumes that the frame shift is an integer ratio of the frame size for simplicity

    if len(np.shape(stft)) != 2: # no multi-channel input
        raise ValueError('We only deal with single channel signals here')

    num_frames, N = np.shape(stft)
    if N % 2 != 1:
        raise ValueError('We expect the stft to only have non-negative frequencies')

    fft_size = (N - 1) * 2

    if window_length % hop_length != 0:
        raise ValueError('Frame shift should be integer ratio of frame size.')

    if window_length is None:
        window_length = fft_size

    if hop_length is None:
        hop_length = int(window_length // 4)
    
    if analyse_window is None:
        if synthesis_window is None:
            analyse_window=lws.hann(window_length, True, False)
            synthesis_window = analyse_window
        else:
            analyse_window = synthesis_window
    else:
        if not 'synthesis_window' in locals() or not len(synthesis_window):
            synthesis_window = analyse_window

    synthesis_window = lws.synthwin(analyse_window,hop_length,synthesis_window)

    if fft_size >= len(synthesis_window):
        pad = np.zeros(((fft_size - len(synthesis_window)) // 2,))
        synthesis_window = np.hstack((pad, synthesis_window, pad))
    else:
        raise ValueError('Window size must be larger than fft size.')

    signal_length = hop_length * (num_frames-1) + fft_size
    signal=np.zeros(signal_length)

    x_ran=np.arange(fft_size)

    for s in range(num_frames):
        iframe=np.real(scipy.ifft(np.concatenate((stft[s], stft[s][-2:0:-1].conjugate())),n=fft_size))
        iframe= iframe[0:fft_size]
        signal[hop_length*s+x_ran]+= iframe * np.squeeze(synthesis_window)

    if perfectrec is True:
        signal = signal[int(fft_size//2) : -int(fft_size//2)]

    return signal

class lws_mod(object):
    def __init__(self, fft_size=2048, window_length=None, hop_length=None, L = 5, analyse_window = None, synthesis_window = None, look_ahead = 3,
                 nofuture_iterations = 0, nofuture_alpha = 1, nofuture_beta = 0.1, nofuture_gamma = 1,
                 online_iterations = 0, online_alpha = 1, online_beta = 0.1, online_gamma = 1,
                 batch_iterations = 100, batch_alpha = 100, batch_beta = 0.1, batch_gamma = 1,
                 symmetric_win = True, mode= None, perfectrec=True):


        if window_length is None:
            window_length = fft_size

        if hop_length is None:
            hop_length = int(window_length // 4)
        
        if window_length % hop_length is not 0:
            raise ValueError('LWS requires that the window shift divides the window length.')

        if analyse_window is None:
            analyse_window = lws.hann(window_length, symmetric=symmetric_win, use_offset=False)
        
        if fft_size > window_length:
            if (fft_size - window_length) % 2 != 0:
                raise ValueError('The zero-padding should add even length to the original window.')
            pad = np.zeros((fft_size - window_length) // 2)
            analyse_window = np.hstack((pad,analyse_window,pad))

        self.fft_size = fft_size
        self.analyse_window = analyse_window
        self.synthesis_window = lws.synthwin(analyse_window,hop_length,synthesis_window)
        self.hop_length = hop_length
        self.window_length = window_length
        self.perfectrec = perfectrec
        self.L = L
        self.Q = np.int(self.window_length/self.hop_length)
        self.W = lws.create_weights(self.analyse_window,self.synthesis_window,self.hop_length,self.L)
        self.win_ai, self.win_af = lws.build_asymmetric_windows(self.analyse_window * self.synthesis_window, self.hop_length)
        self.W_ai = lws.create_weights(self.win_ai,self.synthesis_window,self.hop_length,self.L)
        self.W_af = lws.create_weights(self.win_af,self.synthesis_window,self.hop_length,self.L)
        self.look_ahead = look_ahead
        
        if mode == 'speech':
            nofuture_iterations = 0
            online_iterations= 0
        elif mode == 'music':
            nofuture_iterations = 1
            online_iterations= 10

        self.batch_iterations = batch_iterations
        self.batch_alpha = batch_alpha
        self.batch_beta  = batch_beta
        self.batch_gamma = batch_gamma
        self.online_iterations = online_iterations
        self.online_alpha = online_alpha
        self.online_beta  = online_beta
        self.online_gamma = online_gamma
        self.nofuture_iterations = nofuture_iterations
        self.nofuture_alpha = nofuture_alpha
        self.nofuture_beta  = nofuture_beta
        self.nofuture_gamma = nofuture_gamma

        
        if (not np.allclose(analyse_window, analyse_window[::-1])):
            print('WARNING: It appears you are using an analysis window that is not symmetric. The current code uses simplifications that rely on such symmetry, so the code may not behave properly.')


    def get_consistency(self,S):
        return lws.get_consistency(S,self.window_length,self.hop_length,self.analyse_window,self.synthesis_window,perfectrec=self.perfectrec)


    def stft(self,S):
        return stft(S, self.fft_size, self.window_length, self.hop_length, self.analyse_window, self.perfectrec)


    def istft(self,S):
        return istft(S, self.hop_length, self.window_length, self.analyse_window, self.synthesis_window, self.perfectrec)

    def nofuture_lws(self,S,iterations=None,thresholds=None):
        if iterations is None:
            iterations = self.nofuture_iterations
        if thresholds is None:
            thresholds = lws.get_thresholds(iterations,self.nofuture_alpha,self.nofuture_beta,self.nofuture_gamma)
        return lws.nofuture_lws(S,self.W_ai,thresholds)


    def online_lws(self,S,iterations=None,thresholds=None):
        if iterations is None:
            iterations = self.online_iterations
        if thresholds is None:
            thresholds = lws.get_thresholds(iterations,self.online_alpha,self.online_beta,self.online_gamma)
        return lws.online_lws(S,self.W,self.W_ai,self.W_af,thresholds,self.look_ahead)


    def batch_lws(self,S,iterations=None,thresholds=None):
        if iterations is None:
            iterations = self.batch_iterations
        if thresholds is None:
            thresholds = lws.get_thresholds(iterations,self.batch_alpha,self.batch_beta,self.batch_gamma)
        return lws.batch_lws(S,self.W,thresholds)
        

    def run_lws(self,S):
        S0 = self.nofuture_lws(S)
        S1 = self.online_lws(S0)
        S2 = self.batch_lws(S1)
        return S2

