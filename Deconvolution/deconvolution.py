import numpy as np
from scipy.fft import fft2, ifft2

def gaussian_kernel(size, sigma):
    """
    Построение ядра фильтра Гаусса.

    @param  size  int    размер фильтра (нечетный)
    @param  sigma float  параметр размытия
    @return numpy array  фильтр Гаусса размером size x size
    """
    def gaussian(x,y, x0, y0, sigma):
        r = (x - x0)**2 + (y - y0)**2
        return np.exp(-r/2/sigma**2)/2/np.pi/sigma**2
    
    kernel = np.zeros((size,size), dtype=np.float64)
    for i in range(size):
        for j in range(size):
            kernel[i, j] = gaussian(i, j, size // 2, size // 2, sigma)
    kernel = kernel / np.sum(kernel)
    return kernel


def fourier_transform(h, shape):
    """
    Получение Фурье-образа искажающей функции

    @param  h            numpy array  искажающая функция h (ядро свертки)
    @param  shape        list         требуемый размер образа
    @return numpy array  H            Фурье-образ искажающей функции h
    """
    return fft2(h,shape)
    

def inverse_kernel(H, threshold=1e-10):
    """
    Получение H_inv

    @param  H            numpy array    Фурье-образ искажающей функции h
    @param  threshold    float          порог отсечения для избежания деления на 0
    @return numpy array  H_inv
    """
    idx1 = np.abs(H) <= threshold
    idx2 = np.abs(H) > threshold
    H[idx1] = 0
    H[idx2] = 1/H[idx2]
    return H


def inverse_filtering(blurred_img, h, threshold=1e-10):
    """
    Метод инверсной фильтрации

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  threshold      float        параметр получения H_inv
    @return numpy array                 восстановленное изображение
    """
    return np.abs(ifft2(fourier_transform(blurred_img, blurred_img.shape)*
                  inverse_kernel(fourier_transform(h, blurred_img.shape))))


def wiener_filtering(blurred_img, h, K=5e-05):
    """
    Винеровская фильтрация

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  K              float        константа из выражения (8)
    @return numpy array                 восстановленное изображение
    """
    H = fourier_transform(h, blurred_img.shape)
    return np.abs(ifft2(np.conj(H)*fft2(blurred_img)/(np.conj(H)*H + K)))


def compute_psnr(img1, img2):
    """
    PSNR metric

    @param  img1    numpy array   оригинальное изображение
    @param  img2    numpy array   искаженное изображение
    @return float   PSNR(img1, img2)
    """
    mse = np.mean((img1-img2)**2)
    return 20*np.log10(255/np.sqrt(mse))
