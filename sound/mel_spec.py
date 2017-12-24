# -*- coding: utf-8 -*-
# import librosa

import numpy as np
# import soundfile
# from resampy import resample
from sound import core
import warnings


# def load_data(wav_file, sample_rate=16000):
#     """
#     get times series and sampling rate
#     :param wav_file: path of wav file
#     :param sample_rate
#     :return:
#     """
#     audio, samplerate = soundfile.read(wav_file)
#     if samplerate != sample_rate:
#         audio = resample(audio, samplerate, sample_rate)
#     return audio, samplerate


def spectrogram(y=None, frames_size=410, frame_shift=160, power=2):
    """
    :param y:
    :param power:
    :param frames_size:
    :param frame_shift:
    :return:
    """
    frames = core.frame(wav=y, frame_size=frames_size, frame_shift=frame_shift)
    # compute a magnitude spectrogram from input
    spect = np.abs(core.stft(frames=frames))
    spect = spect.T ** power
    # print('spect from spectrogram is', spect)
    # print('hahhahahahh')
    return spect


def fft_frequencies(sr=16000, n_fft=410):
    """
    Alternative implementation of `np.fft.fftfreqs
    :param sr:
    :param n_fft:
    :return:
    """
    return np.linspace(0, float(sr) / 2, int(1 + n_fft // 2), endpoint=True)


def mel_to_hz(mels, htk=False):
    """
    Convert mel bin numbers to frequencies
    :param mels:
    :param htk: use HTK formula instead of Slaney
    :return:
    """
    mels = np.atleast_1d(mels)
    if htk:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # add now the nonliner scale
    min_log_hz = 1000.0                         # beginning of log region (Hz)
    min_log_mel = (min_log_hz-f_min)/f_sp       # same (Mels)
    logstep = np.log(6.4) / 27.0                # step size for log region
    # log_t的值为0或1
    log_t = (mels >= min_log_mel)
    freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    return freqs


def hz_to_mel(frequencies, htk=False):
    """
    Convert Hz to Mels
    :param frequencies:
    :param htk:
    :return:
    mels: np.ndarray [shape=(n,)]
        input frequencies in Mels
    """
    # Convert inputs to arrays with at least one dimension
    frequencies = np.atleast_1d(frequencies)
    if htk:
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)
    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3
    mels = (frequencies - f_min) / f_sp
    # Fill in the log-scale part
    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = np.log(6.4) / 27.0  # step size for log region

    log_t = (frequencies >= min_log_hz)
    mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep
    return mels


def mel_frequencies(n_mels=128, fmin=0.0, fmax=8000.0, htk=False):
    """
    Compute the center frequencies of mel bands.
    :param n_mels:
    :param fmin:
    :param fmax:
    :param htk:
    :return:
    """
    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = hz_to_mel(fmin, htk=htk)
    max_mel = hz_to_mel(fmax, htk=htk)

    mels = np.linspace(min_mel, max_mel, n_mels)
    return mel_to_hz(mels, htk=htk)


def mel(sr, n_fft=410, n_mels=40, fmin=0.0, fmax=None, htk=False, norm=1):
    """
    :param sr
    :param n_fft:
    :param n_mels:
    :param fmin:
    :param fmax:
    :param htk:
    :param norm:
    :return:
    M: np.ndarray [shape=(n_mels, 1 + n_fft/2)]
        Mel transform matrix
    """
    if fmax is None:
        fmax = float(sr) / 2
    # if norm is not None and norm != 1 and norm != np.inf:
    #     raise ParameterError('Unsupported norm: {}'.format(repr(norm)))
    # initialize the weights
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1+n_fft // 2)))

    # Center freqs of each FFT bin
    fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)

    # Calculate the n-th discrete difference along given axis.
    fdiff = np.diff(mel_f)
    # channel normalisation
    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    if norm == 1:
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels])
        weights *= enorm[:, np.newaxis]
        # Only check weights if f_mel[0] is positive

    if not np.all((mel_f[:-2] == 0) | (weights.max(axis=1) > 0)):
        # This means we have an empty channel somewhere
        warnings.warn('Empty filters detected in mel frequency basis. '
                      'Some channels will produce empty responses. '
                      'Try increasing your sampling rate (and fmax) or '
                      'reducing n_mels.')

    return weights


def melspectrogram(y=None, sr=16000, frame_size=410, frame_shift=160, n_mels=40, power=2):
    """
    first, compute its magnitude spectrogram
    then, mapped onto the mel scale
    :param y: time series
    :param sr: smaple rate
    :param frame_size: int > 0 [scalar]
    :param frame_shift:
        length of the FFT window
    :return:
    S : np.ndarray [shape=(n_mels, t)]
        Mel spectrogram
    """
    # compute its magnitude spectrogram
    spect = spectrogram(y=y, frames_size=frame_size, frame_shift=frame_shift, power=power)
    # Build a Mel filter
    # print('spect from melspectrogram is', spect)
    mel_basis = mel(sr, frame_size, n_mels=n_mels)
    # print('mel_basis is', mel_basis)
    return np.dot(mel_basis, spect).T

def power_to_db(S, ref=1.0, amin=1e-10, top_db=80.0):
    """Convert a power spectrogram (amplitude squared) to decibel (dB) units
    See librosa.power_to_db().

    Parameters
    ----------
    S : np.ndarray
        input power

    ref : scalar
        The amplitude `abs(S)` is scaled relative to `ref`:
        `10 * log10(S / ref)`.
        Zeros in the output correspond to positions where `S == ref`.

    amin : float > 0 [scalar]
        minimum threshold for `abs(S)` and `ref`

    top_db : float >= 0 [scalar]
        threshold the output at `top_db` below the peak:
        ``max(10 * log10(S)) - top_db``

    Returns
    -------
    S_db   : np.ndarray
        ``S_db ~= 10 * log10(S) - 10 * log10(ref)``
    """

    if amin <= 0:
        raise ValueError('amin must be strictly positive')

    magnitude = np.abs(S)

    # if six.callable(ref):
    #     # User supplied a function to calculate reference power
    #     ref_value = ref(magnitude)
    # else:
    #     ref_value = np.abs(ref)
    ref_value = np.abs(ref)

    log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))

    if top_db is not None:
        if top_db < 0:
            raise ValueError('top_db must be non-negative')
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)

    return log_spec

# if __name__ == "__main__":
#     import librosa
#     audio, sr =librosa.load('data/TSP/train/FA01_01.wav', 16000)
#     # audio, sr = load_data('data/TSP/train/FA01_01.wav', 16000)
#     mel_spec = melspectrogram(audio, sr)
#     log_mel = power_to_db(mel_spec)
    # print('mel_spec is', mel_spec)
    # # if librosa.feature.melspectrogram(y=audio, sr=sr) == mel_spec:
    # #     print('true')
    # # else:
    # #     print('false')
    # print('spec from librosa is', librosa.feature.melspectrogram(y=audio, sr=sr))
