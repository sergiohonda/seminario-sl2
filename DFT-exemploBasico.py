import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack

# Gera um sinal senoidal de duração de 
# 20 segundos e 0.1Hz de frequência

t = np.arange(0, 20, 1)
f = 0.1  # Frequência do sinal
s = np.sin(2 * np.pi * f * t)


# Plota o sinal no domínio do tempo
plt.stem(t, s)
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.title('Sinal no domínio do tempo')
plt.show()

# Aplica a Transformada de Fourier
s_fft = fftpack.fft(s)

# Cria um vetor de frequências
freqs = fftpack.fftfreq(len(s)) * 1000

# Plota o espectro de frequência
plt.stem(freqs, np.abs(s_fft))
plt.xlabel('Frequência (Hz)')
plt.ylabel('Amplitude')
plt.title('Espectro de frequência')
plt.show()

# Aplica a Inversa da Transformada de Fourier
s_fft_inv = fftpack.ifft(s_fft)

# Plota a transformada inversa 
# do espectro de frequências
plt.stem(t, s_fft_inv)
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.title('Transformada inversa do espectro de frequências')
plt.show()