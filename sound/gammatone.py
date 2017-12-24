# -*- coding: utf-8 -*-
# modified from https://github.com/detly/gammatone
# and Malcolm Slaney's and Dan Ellis' gammatone filterbank MATLAB code
# and https://www.ee.columbia.edu/~dpwe/resources/matlab/gammatonegram/

import numpy as np
import scipy.signal
import scipy.fftpack
from sound import core

_EAR_Q = 9.26449  # the asymptotic filter quality at large frequencies
_MIN_BW = 24.7  # the minimum bandwidth forlow frequencies channels
_ORDER = 1

N_FILTERS_DEFAULT = 64
GFCC_NUM_DEFAULT = 31

def make_erb_filters(sr, channel_num, low_freq):
    '''
    :param sr:
    :param channel_num:
    :param low_freq:
    :return:

    function [fcoefs]=MakeERBFilters(fs,numChannels,lowFreq)
    This function computes the filter coefficients for a bank of
    Gammatone filters.  These filters were defined by Patterson and
    Holdworth for simulating the cochlea.

    The result is returned as an array of filter coefficients.  Each row
    of the filter arrays contains the coefficients for four second order
    filters.  The transfer function for these four filters share the same
    denominator (poles) but have different numerators (zeros).  All of these
    coefficients are assembled into one vector that the ERBFilterBank
    can take apart to implement the filter.

    The filter bank contains "numChannels" channels that extend from
    half the sampling rate (fs) to "lowFreq".  Alternatively, if the numChannels
    input argument is a vector, then the values of this vector are taken to
    be the center frequency of each desired filter.  (The lowFreq argument is
    ignored in this case.)

    Note this implementation fixes a problem in the original code by
    computing four separate second order filters.  This avoids a big
    problem with round off errors in cases of very small cfs (100Hz) and
    large sample rates (44kHz).  The problem is caused by roundoff error
    when a number of poles are combined, all very close to the unit
    circle.  Small errors in the eigth order coefficient, are multiplied
    when the eigth root is taken to give the pole location.  These small
    errors lead to poles outside the unit circle and instability.  Thanks
    to Julius Smith for leading me to the proper explanation.

    Execute the following code to evaluate the frequency
    response of a 10 channel filterbank.
        fcoefs = MakeERBFilters(16000,10,100);
        y = ERBFilterBank([1 zeros(1,511)], fcoefs);
        resp = 20*log10(abs(fft(y')));
        freqScale = (0:511)/512*16000;
        semilogx(freqScale(1:255),resp(1:255,:));
        axis([100 16000 -60 0])
        xlabel('Frequency (Hz)'); ylabel('Filter Response (dB)');

    Rewritten by Malcolm Slaney@Interval.  June 11, 1998.
    (c) 1998 Interval Research Corporation
    '''
    T = 1 / sr
    if not hasattr(channel_num, '__len__'):
        centre_freq = erb_space(low_freq, sr // 2, channel_num)
    else:
        centre_freq = channel_num
        if centre_freq.shape[1] > centre_freq.shape[0]:
            centre_freq = centre_freq.T

    erb = ((centre_freq / _EAR_Q) ** _ORDER + _MIN_BW ** _ORDER) ** (1 / _ORDER)
    B = 1.019 * 2 * np.pi * erb

    arg = 2 * centre_freq * np.pi * T
    vec = np.exp(2j * arg)

    A0 = T
    A2 = 0
    B0 = 1
    B1 = -2 * np.cos(arg) / np.exp(B * T)
    B2 = np.exp(-2 * B * T)

    rt_pos = np.sqrt(3 + 2 ** 1.5)
    rt_neg = np.sqrt(3 - 2 ** 1.5)

    common = -T * np.exp(-(B * T))

    # TODO: This could be simplified to a matrix calculation involving the
    # constant first term and the alternating rt_pos/rt_neg and +/-1 second
    # terms
    k11 = np.cos(arg) + rt_pos * np.sin(arg)
    k12 = np.cos(arg) - rt_pos * np.sin(arg)
    k13 = np.cos(arg) + rt_neg * np.sin(arg)
    k14 = np.cos(arg) - rt_neg * np.sin(arg)

    A11 = common * k11
    A12 = common * k12
    A13 = common * k13
    A14 = common * k14

    gain_arg = np.exp(1j * arg - B * T)

    gain = np.abs(
        (vec - gain_arg * k11)
        * (vec - gain_arg * k12)
        * (vec - gain_arg * k13)
        * (vec - gain_arg * k14)
        * (T * np.exp(B * T)
           / (-1 / np.exp(B * T) + 1 + vec * (1 - np.exp(B * T)))
           ) ** 4
    )

    allfilts = np.ones_like(centre_freq)

    fcoefs = np.column_stack([
        A0 * allfilts, A11, A12, A13, A14, A2 * allfilts,
        B0 * allfilts, B1, B2,
        gain
    ])

    return fcoefs


def pass_erb_filterbank(wave, coefs):
    """
    :param wave: input data (one dimensional sequence)
    :param coefs: gammatone filter coefficients

    Process an input waveform with a gammatone filter bank. This function takes
    a single sound vector, and returns an array of filter outputs, one channel
    per row.

    The fcoefs parameter, which completely specifies the Gammatone filterbank,
    should be designed with the :func:`make_erb_filters` function.

    | Malcolm Slaney @ Interval, June 11, 1998.
    | (c) 1998 Interval Research Corporation
    | Thanks to Alain de Cheveigne' for his suggestions and improvements.
    |
    | (c) 2013 Jason Heeris (Python implementation)
    """
    output = np.zeros((coefs[:, 9].shape[0], wave.shape[0]))

    gain = coefs[:, 9]
    # A0, A11, A2
    As1 = coefs[:, (0, 1, 5)]
    # A0, A12, A2
    As2 = coefs[:, (0, 2, 5)]
    # A0, A13, A2
    As3 = coefs[:, (0, 3, 5)]
    # A0, A14, A2
    As4 = coefs[:, (0, 4, 5)]
    # B0, B1, B2
    Bs = coefs[:, 6:9]

    # Loop over channels
    for idx in range(0, coefs.shape[0]):
        # These seem to be reversed (in the sense of A/B order), but that's what
        # the original code did...
        # Replacing these with polynomial multiplications reduces both accuracy
        # and speed.
        y1 = scipy.signal.lfilter(As1[idx], Bs[idx], wave)
        y2 = scipy.signal.lfilter(As2[idx], Bs[idx], y1)
        y3 = scipy.signal.lfilter(As3[idx], Bs[idx], y2)
        y4 = scipy.signal.lfilter(As4[idx], Bs[idx], y3)
        output[idx, :] = y4 / gain[idx]

    return output


def erb_space(low_freq=100, high_freq=8000, num=64):
    # All of the followFreqing expressions are derived in Apple TR #35,
    # "An Efficient Implementation of the Patterson-Holdsworth Cochlear Filter Bank."

    freq = np.arange(1, num + 1)
    eqbw = _EAR_Q * _MIN_BW
    hfeb = high_freq + eqbw
    # - ear_q * min_bandwidth + np.exp(freq * (-np.log(high_freq + ear_q * min_bandwidth)
    # + np.log(low_freq + ear_q * min_bandwidth)) / num) * (high_freq + ear_q * min_bandwidth)
    return - eqbw + ((low_freq + eqbw) / hfeb) ** (freq / num) * hfeb


    # fcoefs = make_erb_filters(16000,10,100)
    # y = ERBFilterBank([1 zeros(1,511)], fcoefs)


def fft2gammatonemx(nfft, sr=16000, nfilts=N_FILTERS_DEFAULT, minfreq=100, width=1, maxfreq=None, maxlen=None):
    '''
    [wts,cfreqa] = fft2gammatonemx(nfft, sr, nfilts, width, minfreq, maxfreq, maxlen)
     Generate a matrix of weights to combine FFT bins into
     Gammatone bins.  nfft defines the source FFT size at
     sampling rate sr.  Optional nfilts specifies the number of
     output bands required (default 64), and width is the
     constant width of each band in Bark (default 1).
     minfreq, maxfreq specify range covered in Hz (100, sr/2).
     While wts has nfft columns, the second half are all zero.
     Hence, aud spectrum is
     fft2gammatonemx(nfft,sr)*abs(fft(xincols,nfft));
     maxlen truncates the rows to this many bins.
     cfreqs returns the actual center frequencies of each
     gammatone band in Hz.

    2004-09-05  Dan Ellis dpwe@ee.columbia.edu  based on rastamat/audspec.m
    Last updated: $Date: 2009/02/22 02:29:25 $
    '''
    num = nfft // 2 + 1

    if maxfreq is None:
        maxfreq = sr // 2
    if maxlen is None:
        maxlen = num

    wts = np.zeros((nfilts, nfft))

    cfreqs = erb_space(minfreq, maxfreq, nfilts)
    cfreqs = np.flipud(cfreqs)
    GTord = 4
    ucirc = np.exp(2j * np.pi * np.arange(num) / nfft)

    for i in range(nfilts):
        # poles and zeros, following Malcolm's MakeERBFilter
        cf = cfreqs[i]
        ERB = width * ((cf / _EAR_Q) ** _ORDER + _MIN_BW ** _ORDER) ** (1 / _ORDER)
        B = 1.019 * 2 * np.pi * ERB
        r = np.exp(-B / sr)
        theta = 2 * np.pi * cf / sr
        pole = r * np.exp(1j * theta)

        T = 1 / sr
        tmp1 = 2 * T * np.cos(2 * cf * np.pi * T) / np.exp(B * T)
        tmp2 = T * np.sin(2 * cf * np.pi * T) / np.exp(B * T)
        A11 = -(tmp1 + 2 * np.sqrt(3 + 2 ** 1.5) * tmp2) / 2
        A12 = -(tmp1 - 2 * np.sqrt(3 + 2 ** 1.5) * tmp2) / 2
        A13 = -(tmp1 + 2 * np.sqrt(3 - 2 ** 1.5) * tmp2) / 2
        A14 = -(tmp1 - 2 * np.sqrt(3 - 2 ** 1.5) * tmp2) / 2
        zros = -np.array([A11, A12, A13, A14]) / T

        tmp1 = 2 * np.exp(4j * cf * np.pi * T)
        tmp2, tmp3 = np.cos(2 * cf * np.pi * T), np.sin(2 * cf * np.pi * T)
        tmp4 = 2 * np.exp(-(B * T) + 2j * cf * np.pi * T) * T
        gain = np.abs((-tmp1 * T + tmp4 * (tmp2 - np.sqrt(3 - 2 ** 1.5) * tmp3)) *
                      (-tmp1 * T + tmp4 * (tmp2 + np.sqrt(3 - 2 ** 1.5) * tmp3)) *
                      (-tmp1 * T + tmp4 * (tmp2 - np.sqrt(3 + 2 ** 1.5) * tmp3)) *
                      (-tmp1 * T + tmp4 * (tmp2 + np.sqrt(3 + 2 ** 1.5) * tmp3)) /
                      (-2 / np.exp(2 * B * T) - tmp1 + (2 + tmp1) / np.exp(B * T)) ** 4)
        wts[i, :num] = ((T ** 4) / gain) * np.abs(ucirc - zros[0]) * np.abs(ucirc - zros[1]) * \
                       np.abs(ucirc - zros[2]) * np.abs(ucirc - zros[3]) * \
                       (np.abs((pole - ucirc) * (pole.conjugate() - ucirc)) ** (-GTord))

    wts = wts[:, :maxlen]
    return wts


def gammatonegram(X, samplerate=16000, frame_size=410, frame_shift=160, nfilter=64,
                  low_freq=100, high_freq=None, fft_proc=True, width=1):
    if high_freq is None:
        high_freq = samplerate // 2

    if not fft_proc:
        fcoefs = make_erb_filters(samplerate, nfilter, low_freq)
        fcoefs = np.flipud(fcoefs)
        XF = pass_erb_filterbank(X, fcoefs)
        XE = XF ** 2
        num_frame = 1 + (XE.shape[1] - frame_size) // frame_shift
        Y = np.zeros((nfilter, num_frame))
        for i in range(num_frame):
            Y[:, i] = np.sqrt(np.mean(XE[:, i:i+frame_size], axis=1))
    else:
        nfft = frame_size
        gtm = fft2gammatonemx(nfft, samplerate, nfilter, low_freq, width, high_freq, nfft // 2 + 1)

        frame = core.frame(X, frame_size, frame_shift)
        FFTX = core.stft(frame).T
        Y = 1 / nfft * np.dot(gtm, np.abs(FFTX))
    return Y.T


def gtm2gfcc(gtm, dct_start=0, dct_stop=None):
    gfcc = scipy.fftpack.dct(gtm, norm='ortho', axis=1)
    if dct_stop is None:
        return gfcc

    return gfcc[:,dct_start:dct_stop]

