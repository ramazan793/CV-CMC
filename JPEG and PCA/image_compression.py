import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.ndimage.filters import gaussian_filter
from skimage.metrics import peak_signal_noise_ratio

def pca_compression(matrix, p):
    """ Сжатие изображения с помощью PCA
    Вход: двумерная матрица (одна цветовая компонента картинки), количество компонент
    Выход: собственные векторы и проекция матрицы на новое пр-во
    """
    M = matrix.copy()

    centered = M - np.mean(M, axis = 1)
    cov = np.cov(centered)
    
    eig_val, eig_vec = np.linalg.eigh(cov)
    
    eig_vec = eig_vec[:, np.argsort(-eig_val)][:,:p]
    
    proj = np.dot(eig_vec.T, centered)
    
    return eig_vec, proj, np.mean(M, axis = 1)

def pca_decompression(compressed):
    """ Разжатие изображения
    Вход: список кортежей из собственных векторов и проекций для каждой цветовой компоненты
    Выход: разжатое изображение
    """
    
    result_img = []
    for comp in compressed:
        result_img.append(np.dot(comp[0], comp[1]) + comp[2])
    
    result_img = np.dstack(result_img)
    return np.clip(result_img, 0, 255).astype(np.uint8)


def pca_visualize():
    plt.clf()
    img = imread('cat.jpg')
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(3, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 5, 10, 20, 50, 100, 150, 200, 256]):
        compressed = []
        for j in range(0, 3):
            evec, proj, means = pca_compression(img[..., j], p)
            compressed.append((evec, proj, means))
        decompressed = pca_decompression(compressed)
            # Your code here
            
        axes[i // 3, i % 3].imshow(decompressed)
        axes[i // 3, i % 3].set_title('#Components: {}'.format(p))

    fig.savefig("pca_visualization.png")


def rgb2ycbcr(img):
    """ Переход из пр-ва RGB в пр-во YCbCr
    Вход: RGB изображение
    Выход: YCbCr изображение
    """
    channels = []
    coeff = np.array([[0.299, 0.587, 0.114],
    [-0.1687, -0.3313, 0.5],
    [0.5, -0.4187, -0.0813]])
    b = np.array([0, 128, 128])
    for i in range(3):
        channel = img[:, :, 0] * coeff[i, 0] + img[:, :, 1] * coeff[i, 1] + img[:, :, 2] * coeff[i, 2]
        channel += b[i]
        channels.append(channel)
    
    return np.dstack(channels).astype(np.uint8)


def ycbcr2rgb(img):
    """ Переход из пр-ва YCbCr в пр-во RGB
    Вход: YCbCr изображение
    Выход: RGB изображение
    """
    channels = []
    coeff = np.array([
        [1, 0, 1.402],
        [1, -0.34414, -0.71414],
        [1, 1.77, 0]
    ])
    b = np.array([0, -128, -128])
    for i in range(3):
        channel = (img[:, :, 0] + b[0]) * coeff[i, 0] + (img[:, :, 1] + b[1]) * coeff[i, 1] + (img[:, :, 2] + b[2]) * coeff[i, 2]
        channels.append(channel)
    
    return np.clip(np.dstack(channels), 0, 255).astype(np.uint8)


def get_gauss_1():
    plt.clf()
    rgb_img = imread('Lenna.png')
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]
        
    ybr = rgb2ycbcr(rgb_img)
    ybr[:,:, 1] = gaussian_filter(ybr[:,:,1], 10)
    ybr[:,:, 2] = gaussian_filter(ybr[:,:,2], 10)
    
    new_rgb = ycbcr2rgb(ybr)
    
    plt.imshow(new_rgb)
    plt.savefig("gauss_1.png")


def get_gauss_2():
    plt.clf()
    rgb_img = imread('Lenna.png')
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]

    ybr = rgb2ycbcr(rgb_img)
    ybr[:,:, 0] = gaussian_filter(ybr[:,:,0], 10)
    
    new_rgb = ycbcr2rgb(ybr)
   
    plt.imshow(new_rgb)
    plt.savefig("gauss_2.png")


def downsampling(component):
    """Уменьшаем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B, 1]
    Выход: цветовая компонента размера [A // 2, B // 2, 1]
    """
    component = gaussian_filter(component, 10)
    
    return component[::2,::2]


def dct(block):
    """Дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после ДКП
    """
    def alpha(x):
        if x == 0:
            return 1/np.sqrt(2)
        return 1
    
    G = np.zeros_like(block, dtype = np.float64)
    
    X, Y = np.indices(block.shape).astype(np.float64)
    
    for i in range(block.shape[0]):
        for j in range(block.shape[1]):
            G[i, j] = alpha(i)*alpha(j)*np.sum(block*np.cos((2*X+1)*i*np.pi/16)*np.cos((2*Y+1)*j*np.pi/16))/4
            
    return G


# # Матрица квантования яркости
y_quantization_matrix = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

# Матрица квантования цвета
color_quantization_matrix = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])


def quantization(block, quantization_matrix):
    """Квантование
    Вход: блок размера 8x8 после применения ДКП; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление осуществляем с помощью np.round
    """
    return np.round(block / quantization_matrix)


def own_quantization_matrix(default_quantization_matrix, q):
    """Генерация матрицы квантования по Quality Factor
    Вход: "стандартная" матрица квантования; Quality Factor
    Выход: новая матрица квантования
    Hint: если после проделанных операций какие-то элементы обнулились, то замените их единицами
    """

    assert 1 <= q <= 100
    
    S = 1
    if 1 <= q < 50:
        S = 5000/q
    elif 50 <= q <= 99:
        S = 200 - 2*q
    
    quantization_matrix = np.floor((50 + S*default_quantization_matrix)/100)
    quantization_matrix[quantization_matrix == 0] = 1
    return quantization_matrix


def zigzag(block):
    """Зигзаг-сканирование
    Вход: блок размера 8x8
    Выход: список из элементов входного блока, получаемый после его обхода зигзаг-сканированием
    """
    return np.concatenate([np.diagonal(block[::-1], k)[::-1 if k % 2 == 0 else 1] for k in range(1 - block.shape[0], block.shape[0])])


def compression(zigzag_list):
    """Сжатие последовательности после зигзаг-сканирования
    Вход: список после зигзаг-сканирования
    Выход: сжатый список в формате, который был приведен в качестве примера в самом начале данного пункта
    """
    lz = -1
    count = 0
    rle = []
    for i in range(len(zigzag_list)):
        if lz == -1:
            if zigzag_list[i] == 0:
                lz = i
                count = 1
                if i == len(zigzag_list) - 1:
                    rle.extend([0, count]) 
            else:
                rle.append(zigzag_list[i])
        else:
            if zigzag_list[i] != 0:
                rle.extend([0, count, zigzag_list[i]])
                lz = -1
            else:
                count += 1
                if i == len(zigzag_list) - 1:
                    rle.extend([0, count])
    
    return rle


def jpeg_compression(img, quantization_matrices):
    """JPEG-сжатие
    Вход: цветная картинка, список из 2-ух матриц квантования
    Выход: список списков со сжатыми векторами: [[compressed_y1,...], [compressed_Cb1,...], [compressed_Cr1,...]]
    """

    # suppose here that img.shape % 8 == (0, 0). otherwise just resize or pad zeros.
    
    # Переходим из RGB в YCbCr
    ycbcr = rgb2ycbcr(img)
    # Уменьшаем цветовые компоненты
    Y, Cb, Cr = ycbcr[:,:,0], downsampling(ycbcr[:,:,1]), downsampling(ycbcr[:,:,2])
    # Делим все компоненты на блоки 8x8 и все элементы блоков переводим из [0, 255] в [-128, 127]
    c_blocks = [[], [], []]
    for k, channel in enumerate([Y, Cb, Cr]):
        for i in range(0, channel.shape[0] - 7, 8):
            for j in range(0, channel.shape[1] - 7, 8):
                block = channel[i:i+8,j:j+8].astype(np.int32) - 128
                q_i = 0 if k == 0 else 1
                block = compression(zigzag(quantization(dct(block), quantization_matrices[q_i])))
                c_blocks[k].append(block)
                
    return c_blocks


def inverse_compression(compressed_list):
    """Разжатие последовательности
    Вход: сжатый список
    Выход: разжатый список
    """
    
    new_list = []
    i = 0
    while i < len(compressed_list):
        if compressed_list[i] == 0:
            count = compressed_list[i + 1]
            new_list.extend([0]*count)
            i += 1
        else:
            new_list.append(compressed_list[i])
        i += 1
    
    return new_list


def inverse_zigzag(input):
    """Обратное зигзаг-сканирование
    Вход: список элементов
    Выход: блок размера 8x8 из элементов входного списка, расставленных в матрице в порядке их следования в зигзаг-сканировании
    """
    
    za = input
    z = np.zeros((8,8))
    c = 0
    for k in range(z.shape[0]):
        if k % 2 == 0:
            i = k
            j = 0
            for p in range(k + 1):
                z[i - p, j + p] = za[c]
                c += 1
        else:
            i = 0
            j = k
            for p in range(k + 1):
                z[i + p, j - p] = za[c]
                c += 1

    for k in reversed(range(0, z.shape[0] - 1)):
        if k % 2 == 1:
            i = z.shape[0] - 1 - k
            j = z.shape[0] - 1
            for p in range(k + 1):
                z[i + p, j - p] = za[c]
                c += 1
        else:
            i = z.shape[0] - 1
            j = z.shape[0] - 1 - k
            for p in range(k + 1):
                z[i - p, j + p] = za[c]
                c += 1
    
    return z


def inverse_quantization(block, quantization_matrix):
    """Обратное квантование
    Вход: блок размера 8x8 после применения обратного зигзаг-сканирования; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление не производится
    """
    return block * quantization_matrix


def inverse_dct(block):
    """Обратное дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после обратного ДКП. Округление осуществляем с помощью np.round
    """
    def alpha(X):
        X = np.copy(X)
        i = (X == 0)
        j = (X != 0)
        X[i] = 1/np.sqrt(2)
        X[j] = 1
        return X
    
    f = np.zeros_like(block, dtype = np.float64)

    X, Y = np.indices(block.shape)
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    block = block.astype(np.float64)
    
    for i in range(block.shape[0]):
        for j in range(block.shape[1]):
            f[i, j] = np.sum(alpha(X)*alpha(Y)*block*np.cos((2*i+1)*X*np.pi/16)*np.cos((2*j+1)*Y*np.pi/16))/4
    return np.round(f)


def upsampling(component):
    """Увеличиваем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B, 1]
    Выход: цветовая компонента размера [2 * A, 2 * B, 1]
    """
    arr = np.ones((*component.shape,2,2), dtype=np.int32) * component[:, :, None, None]
    return np.concatenate(np.concatenate(arr, axis = 2), axis = 0).T


def jpeg_decompression(result, result_shape, quantization_matrices):
    """Разжатие изображения
    Вход: result список сжатых данных, размер ответа, список из 2-ух матриц квантования
    Выход: разжатое изображение
    """
    Y_blocks = []
    Cb_blocks = []
    Cr_blocks = []
    for k, vectors in enumerate(result):
        for vec in vectors:
            q_i = 0 if k == 0 else 1
            block = inverse_dct(inverse_quantization(inverse_zigzag(inverse_compression(vec)), quantization_matrices[q_i])) + 128
            if k == 0:
                Y_blocks.append(block)
            elif k == 1:
                Cb_blocks.append(block)
            else:
                Cr_blocks.append(block)
    # собираем блоки в компоненты    
    Y = []
    c = 0
    for i in range(result_shape[0] // 8):
        Y.append([])
        for j in range(result_shape[1] // 8):
            Y[i].append(Y_blocks[c])
            c += 1
    Y = np.block(Y)
    
    Cb = []
    c = 0
    for i in range(result_shape[0] // 16):
        Cb.append([])
        for j in range(result_shape[1] // 16):
            Cb[i].append(Cb_blocks[c])
            c += 1
    Cb = upsampling(np.block(Cb))
    
    Cr = []
    c = 0
    for i in range(result_shape[0] // 16):
        Cr.append([])
        for j in range(result_shape[1] // 16):
            Cr[i].append(Cr_blocks[c])
            c += 1
    Cr = upsampling(np.block(Cr))
    
    img = ycbcr2rgb(np.dstack([Y, Cb, Cr]))
    
    return img


def jpeg_visualize():
    plt.clf()
    img = imread('cat.jpg')
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(2, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 10, 20, 50, 80, 100]):
        # Your code here
        quantization_matrices = [own_quantization_matrix(y_quantization_matrix, p), own_quantization_matrix(color_quantization_matrix, p)]
        compressed = jpeg_compression(img, quantization_matrices)
        decompressed = jpeg_decompression(compressed, img.shape, quantization_matrices)
        axes[i // 3, i % 3].imshow(decompressed)
        axes[i // 3, i % 3].set_title('Quality Factor: {}'.format(p))

    fig.savefig("jpeg_visualization.png")


def compression_pipeline(img, c_type, param=1):
    """Pipeline для PCA и JPEG
    Вход: исходное изображение; название метода - 'pca', 'jpeg'; 
    param - кол-во компонент в случае PCA, и Quality Factor для JPEG
    Выход: изображение; количество бит на пиксель
    """
    
    assert c_type.lower() == 'jpeg' or c_type.lower() == 'pca'
    
    if c_type.lower() == 'jpeg':
        y_quantization = own_quantization_matrix(y_quantization_matrix, param)
        color_quantization = own_quantization_matrix(color_quantization_matrix, param)
        matrixes = [y_quantization, color_quantization]
        
        compressed = jpeg_compression(img, matrixes)
        img = jpeg_decompression(compressed, img.shape, matrixes)
    elif c_type.lower() == 'pca':
        compressed = []
        for j in range(0, 3):
            compressed.append((pca_compression(img[:, :, j].astype(np.float64).copy(), param)))
            
        img = pca_decompression(compressed)
        compressed.extend([np.mean(img[:, :, 0], axis=1), np.mean(img[:, :, 1], axis=1), np.mean(img[:, :, 2], axis=1)])
        
    if 'tmp' not in os.listdir() or not os.path.isdir('tmp'):
        os.mkdir('tmp')
        
    np.savez_compressed(os.path.join('tmp', 'tmp.npz'), compressed)
    size = os.stat(os.path.join('tmp', 'tmp.npz')).st_size * 8
    os.remove(os.path.join('tmp', 'tmp.npz'))
        
    return img, size / (img.shape[0] * img.shape[1])


def calc_metrics(img_path, c_type, param_list):
    """Подсчет PSNR и Rate-Distortion для PCA и JPEG. Построение графиков
    Вход: пусть до изображения; тип сжатия; список параметров: кол-во компонент в случае PCA, и Quality Factor для JPEG
    """
    
    assert c_type.lower() == 'jpeg' or c_type.lower() == 'pca'
    
    img = imread(img_path)
    if len(img.shape) == 3:
        img = img[..., :3]
    
    outputs = []
    for param in param_list:
        outputs.append(compression_pipeline(img.copy(), c_type, param))
     
    psnr = [peak_signal_noise_ratio(img, output[0]) for output in outputs]
    rate = [output[1] for output in outputs]
    
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    fig.set_figwidth(20)
    fig.set_figheight(5)
    
    ax1.set_title('PSNR for {}'.format(c_type.upper()))
    ax1.plot(param_list, psnr, 'tab:orange')
    ax1.set_xlabel('Quality Factor')
    ax1.set_ylabel('PSNR')
    
    ax2.set_title('Rate-Distortion for {}'.format(c_type.upper()))
    ax2.plot(psnr, rate, 'tab:red')
    ax2.set_xlabel('Distortion')
    ax2.set_ylabel('Rate')
    return fig


def get_pca_metrics_graph():
    plt.clf()
    fig = calc_metrics('Lenna.png', 'pca', [1, 5, 10, 20, 50, 100, 150, 200, 256])
    fig.savefig("pca_metrics_graph.png")


def get_jpeg_metrics_graph():
    plt.clf()
    fig = calc_metrics('Lenna.png', 'jpeg', [1, 10, 20, 50, 80, 100])
    fig.savefig("jpeg_metrics_graph.png")

