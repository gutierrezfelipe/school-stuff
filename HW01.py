import numpy as np
import matplotlib
import matplotlib.pyplot as plt


matplotlib.use('TkAgg')

# 1 - Criar uma imagem 512 x 512 x 3 uint8 com
# (a) retangulo 50 x 50 com origem em(100,100) valor 127, matriz RED
# (b) círculo preenchido blue valor 127 matriz GREEN
# (c) seno com 8 ciclos ao longo da largura 0 a 127 matriz BLUE
# (d) apresentar R, G, B separados como gray scale. Apresentar RGB(junto) como imagem colorida



# 2 - CROP imagem central de 400 x 400 pixels e apresentar


# 3 - Criar matriz 512 x 512 c/ ruído uniformemente distribuído 0-40




N = 512
In = np.random.randint(0, 41, (N, N))

# (a) apresentar imagem
plt.figure(4)
plt.imshow(In, aspect='equal')
plt.show()

# (b) aplicar 20 * log10(In normalizado + 0,001)
In_norm = In/np.linalg.norm(In)
In_db = 20 * np.log10(In_norm+0.001)
# plt.figure()
# plt.imshow(In_norm, aspect='equal')
# plt.show()

# (c) Apresentar AVG e STD_DEV de (A) e (B)
print(f'AVG(A) = {np.average(In)}')
print(f'STD_DEV(A) = {np.std(In)}')
print(f'AVG(B) = {np.average(In_db)}')
print(f'STD_DEV(B) = {np.std(In_db)}')

# 4 - 3 exemplos "SEIS" de começo de histograma


print('END')
