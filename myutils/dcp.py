import numpy as np
import cv2
def get_dark_channel(image, w=15):
    """
    Get the dark channel prior in the (RGB) image data.
    Parameters
    -----------
    image:  an M * N * 3 numpy array containing data ([0, L-1]) in the image where
        M is the height, N is the width, 3 represents R/G/B channels.
    w:  window size
    Return
    -----------
    An M * N array for the dark channel prior ([0, L-1]).
    """
    M, N, _ = image.shape
    padded = np.pad(image, ((w // 2, w // 2), (w // 2, w // 2), (0, 0)), 'edge')
    darkch = np.zeros((M, N))
    for i, j in np.ndindex(darkch.shape):
        darkch[i, j] = np.min(padded[i:i + w, j:j + w, :])  # CVPR09, eq.5
    return darkch


def get_atmosphere(image, p=0.0001, w=15):
    """Get the atmosphere light in the (RGB) image data.
    Parameters
    -----------
    image:      the 3 * M * N RGB image data ([0, L-1]) as numpy array
    w:      window for dark channel
    p:      percentage of pixels for estimating the atmosphere light
    Return
    -----------
    A 3-element array containing atmosphere light ([0, L-1]) for each channel
    """
    #image = image.transpose(1, 2, 0)
    # reference CVPR09, 4.4
    darkch = get_dark_channel(image, w)
    M, N = darkch.shape
    flatI = image.reshape(M * N, 3)
    flatdark = darkch.ravel()
    searchidx = (-flatdark).argsort()[:int(M * N * p)]  # find top M * N * p indexes
    # return the highest intensity for each channel
    return np.max(flatI.take(searchidx, axis=0), axis=0)



def estimation_atmosphere(image,sigmaX = 10,blocksize=61):
    backscattering_light = cv2.GaussianBlur(image,(blocksize,blocksize),sigmaX)
    return backscattering_light



def MutiScaleLuminanceEstimation(img):
    sigma_list  = [15,60,90]
    w,h,c = img.shape
    img = cv2.resize(img,dsize=None,fx=0.3,fy=0.3)
    Luminance = np.ones_like(img).astype(np.float)
    for sigma in sigma_list:
        Luminance1 = np.log10(cv2.GaussianBlur(img, (0,0), sigma))
        Luminance1 = np.clip(Luminance1,0,255)
        Luminance += Luminance1
    Luminance =  Luminance/3
    L = (Luminance - np.min(Luminance)) / (np.max(Luminance) - np.min(Luminance)+0.0001)
    L =  np.uint8(L*255)
    L =  cv2.resize(L,dsize=(h,w))
    return L

