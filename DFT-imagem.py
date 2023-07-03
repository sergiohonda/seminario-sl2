import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from PIL import Image

# Carrega a imagem
img = Image.open('./operarios.png').convert('L')
img = np.array(img)

# Aplica a Transformada Discreta de Fourier
img_fft = fftpack.fft2(img)

# Centraliza as imagens
img_fft_shift1 = fftpack.fftshift(img_fft)
img_fft_shift2 = fftpack.fftshift(img_fft)
img_fft_shift3 = fftpack.fftshift(img_fft)

# Cria 3 filtros passa-baixa
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
mask1 = np.zeros((rows, cols))
mask2 = np.zeros((rows, cols))
mask3 = np.zeros((rows, cols))
mask1[crow - 5:crow + 5, ccol - 5:ccol + 5] = 1
mask2[crow - 10:crow + 10, ccol - 10:ccol + 10] = 1
mask3[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1

# Aplica o filtro nas imagens
img_fft_shift1 *= mask1
img_fft_shift2 *= mask2
img_fft_shift3 *= mask3

# Centraliza novamente as imagens
img_fft1 = fftpack.ifftshift(img_fft_shift1)
img_fft2 = fftpack.ifftshift(img_fft_shift2)
img_fft3 = fftpack.ifftshift(img_fft_shift3)

# Aplica a Transformada Inversa de Fourier
img_filtered1 = fftpack.ifft2(img_fft1).real
img_filtered2 = fftpack.ifft2(img_fft2).real
img_filtered3 = fftpack.ifft2(img_fft3).real

# Plota a imagem original e as filtradas
fig, ax = plt.subplots(1, 4, figsize=(16, 8))
ax[0].imshow(img, cmap='gray')
ax[0].set_title('Imagem original')
ax[0].axis('off')
ax[1].imshow(img_filtered1, cmap='gray')
ax[1].set_title('Imagem com filtro forte')
ax[1].axis('off')
ax[2].imshow(img_filtered2, cmap='gray')
ax[2].set_title('Imagem com filtro médio')
ax[2].axis('off')
ax[3].imshow(img_filtered3, cmap='gray')
ax[3].set_title('Imagem com filtro fraco')
ax[3].axis('off')
plt.show()

