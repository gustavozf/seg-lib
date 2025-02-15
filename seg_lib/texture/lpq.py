import numpy as np
import scipy.signal
import scipy.linalg

def compute_response(img, wr, wc, conv_mode = 'valid'):
    return scipy.signal.convolve2d(
        scipy.signal.convolve2d(img, wr[:, None], mode=conv_mode),
        wc[None, :],
        mode=conv_mode
    )

def lpq(img, winSize=3, decorr=1, freqestim=1, mode='nh'):
    """
    Computes the Local Phase Quantization (LPQ) descriptor for the input image.
    
    Parameters:
        img (numpy.ndarray): Grayscale image.
        winSize (int, optional): Size of the local window (default: 3).
        decorr (int, optional): Decorrelation flag (0: no decorrelation, 1: use decorrelation, default: 1).
        freqestim (int, optional): Frequency estimation method (1: STFT uniform, 2: STFT Gaussian, 3: Gaussian derivative, default: 1).
        mode (str, optional): Output mode ('nh': normalized histogram, 'h': unnormalized histogram, 'im': LPQ codeword image, default: 'nh').

    Returns:
        numpy.ndarray: LPQ descriptor (either a histogram or an LPQ image).
    """
    
    # Check inputs
    if len(img.shape) != 2:
        raise ValueError("Only grayscale images can be used as input.")
    if winSize < 3 or winSize % 2 == 0:
        raise ValueError("Window size must be an odd number and greater than or equal to 3.")
    if decorr not in [0, 1]:
        raise ValueError("decorr parameter must be set to 0 or 1.")
    if freqestim not in [1, 2, 3]:
        raise ValueError("freqestim parameter must be 1, 2, or 3.")
    if mode not in ['nh', 'h', 'im']:
        raise ValueError("mode must be 'nh', 'h', or 'im'.")

    # Convert image to double
    img = img.astype(np.float64)
    
    # Parameters
    rho = 0.90  # Default correlation coefficient
    STFTalpha = 1 / winSize
    sigmaS = (winSize - 1) / 4
    sigmaA = 8 / (winSize - 1)
    
    r = (winSize - 1) // 2
    x = np.arange(-r, r + 1)

    # Form 1-D filters
    if freqestim == 1:  # STFT with uniform window
        w0 = np.ones_like(x)
        w1 = np.exp(-2j * np.pi * x * STFTalpha)
        w2 = np.conj(w1)
    elif freqestim == 2:  # STFT with Gaussian window
        gs = np.exp(-0.5 * (x / sigmaS) ** 2) / (np.sqrt(2 * np.pi) * sigmaS)
        w0 = gs
        w1 = gs * np.exp(-2j * np.pi * x * STFTalpha)
        w2 = np.conj(w1)
        w1 -= np.mean(w1)
        w2 -= np.mean(w2)
    elif freqestim == 3:  # Gaussian derivative quadrature filter pair
        u = np.arange(1, r + 1)
        G0 = np.exp(-x ** 2 * (np.sqrt(2) * sigmaA) ** 2)
        G1 = np.concatenate((np.zeros_like(u), [0], u * np.exp(-u ** 2 * sigmaA ** 2)))
        G0 /= np.max(np.abs(G0))
        G1 /= np.max(np.abs(G1))
        w0 = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(G0))).real
        w1 = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(G1)))
        w2 = np.conj(w1)

    # Compute frequency responses
    freqResp = np.zeros((img.shape[0] - 2 * r, img.shape[1] - 2 * r, 8))

    conv_result = compute_response(img, w0, w1)
    freqResp[..., 0] = conv_result.real
    freqResp[..., 1] = conv_result.imag

    conv_result = compute_response(img, w1, w0)
    freqResp[..., 2] = conv_result.real
    freqResp[..., 3] = conv_result.imag

    conv_result = compute_response(img, w1, w1)
    freqResp[..., 4] = conv_result.real
    freqResp[..., 5] = conv_result.imag

    conv_result = compute_response(img, w1, w2)
    freqResp[..., 6] = conv_result.real
    freqResp[..., 7] = conv_result.imag

    # Perform decorrelation if needed
    if decorr:
        xp, yp = np.meshgrid(np.arange(winSize), np.arange(winSize))
        pp = np.column_stack((xp.ravel(), yp.ravel()))
        dd = scipy.spatial.distance_matrix(pp, pp)
        C = rho ** dd

        q1 = np.outer(w0, w1)
        q2 = np.outer(w1, w0)
        q3 = np.outer(w1, w1)
        q4 = np.outer(w1, w2)

        u = [q1.real, q1.imag, q2.real, q2.imag, q3.real, q3.imag, q4.real, q4.imag]
        M = np.array([u_i.ravel() for u_i in u])
        
        D = M @ C @ M.T
        A = np.diag([1 + i * 1e-6 for i in range(8)])
        U, S, Vt = np.linalg.svd(A @ D @ A)
        
        for i in range(Vt.shape[1]):
            max_idx = np.argmax(np.abs(Vt[i]))
            if Vt[i, max_idx] < 0:
                Vt[i] *= -1

        freqResp = freqResp.reshape(-1, 8)
        freqResp = (Vt @ freqResp.T).T
        freqResp = freqResp.reshape((img.shape[0] - 2 * r, img.shape[1] - 2 * r, 8))

    # Compute LPQ codewords
    LPQdesc = np.zeros_like(freqResp[..., 0], dtype=np.uint8)
    for i in range(8):
        LPQdesc += (freqResp[..., i] > 0).astype(np.uint8) * (2 ** i)

    # Return LPQ image if requested
    if mode == 'im':
        return LPQdesc

    # Compute histogram
    hist, _ = np.histogram(LPQdesc.ravel(), bins=256, range=(0, 255))
    
    # Normalize histogram if needed
    if mode == 'nh':
        hist = hist.astype(np.float64) / hist.sum()

    return hist
