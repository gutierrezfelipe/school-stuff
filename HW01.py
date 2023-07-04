## PDI - HW01 - Felipe Derewlany Gutierrez - UTFPR-CT
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
from skimage import exposure


matplotlib.use('TkAgg')

# 1 - Criar uma imagem 512 x 512 x 3 uint8 com
N = 512
image = np.zeros((N, N, 3), dtype='uint8')

# (a) retangulo 50 x 60 com origem em(100,100) valor 127, matriz RED
rect_size_h = 50
rect_size_w = 60
rect_origin = 100
image[rect_origin:rect_origin+rect_size_h, rect_origin:rect_origin+rect_size_w, 0] = 127

# (b) círculo preenchido blue valor 127 matriz GREEN
circle_radius = 200
circle_center_x = round(N/2)
circle_center_z = round(N/2)

# matemática
for i in range(N):
    for j in range(N):
        if (i - circle_center_x) ** 2 + (j - circle_center_z) ** 2 < circle_radius ** 2:
                image[j, i, 1] = 127

# opencv
# cv2.circle(image, (circle_center_x, circle_center_z), circle_radius, (0, 127, 0), -1, 8)

# (c) seno com 8 ciclos ao longo da largura 0 a 127 matriz BLUE
k = 8
x = np.arange(N)
z = np.arange(N)
X, Z = np.meshgrid(x, z)
I = 1/2 * (1 + np.sin(2*np.pi / N*k*X))
image[:, :, 2] = 255 * I

# (d) apresentar R, G, B separados como gray scale. Apresentar RGB(junto) como imagem colorida
plt.figure(1)
plt.imshow(image[:, :, 0], aspect='equal', cmap='gray', vmin=0, vmax=255)
plt.title('1d - matriz RED')
# plt.show()

plt.figure(2)
plt.imshow(image[:, :, 1], aspect='equal', cmap='gray', vmin=0, vmax=255)
plt.title('1d - matriz GREEN')
# plt.show()

plt.figure(3)
plt.imshow(image[:, :, 2], aspect='equal', cmap='gray', vmin=0, vmax=255)
plt.title('1d - matriz BLUE')
# plt.show()

# junta os canais numa imagem RGB
# merged_rgb_img = cv2.merge([image[:, :, 0], image[:, :, 1], image[:, :, 2]])

plt.figure(4)
plt.imshow(image, aspect='equal')
plt.title('1d - imagem RGB')
# plt.show()

# 2 - CROP imagem central de 400 x 400 pixels e apresentar
crop_size = 400
diff = N - crop_size
cropped_img = image[diff//2:N-diff//2, diff//2:N-diff//2, :]

plt.figure(5)
plt.imshow(cropped_img, aspect='equal')
plt.title('2 - imagem cortada')
# plt.show()

# 3 - Criar matriz 512 x 512 c/ ruído uniformemente distribuído 0-40
N = 512
levels = 40
In = np.random.randint(0, levels+1, (N, N))

# (a) apresentar imagem
plt.figure(6)
plt.imshow(In, aspect='equal', cmap='gray', vmin=0, vmax=255)
plt.title('3 - ruído distribuição uniforme')
# plt.show()

# (b) aplicar 20 * log10(In normalizado + 0,001)
In_norm = In/np.linalg.norm(In)
In_db = 20 * np.log10(In_norm+0.001)

# plt.figure()
# plt.imshow(In_db, aspect='equal', cmap='gray', vmin=0, vmax=255)
# plt.title('3 - ruído distribuição uniforme - normalizado')
# plt.show()

# (c) Apresentar AVG e STD_DEV de (A) e (B)
print(f'AVG(A) = {np.average(In)}')
print(f'STD_DEV(A) = {np.std(In)}')
print(f'AVG(B) = {np.average(In_db)}')
print(f'STD_DEV(B) = {np.std(In_db)}')

# 4 - 3 exemplos "SEUS" de correção de histograma
filename1 = 'simulation_frame_low_brightness.png'
img1 = cv2.imread(filename1)
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

plt.figure()
plt.imshow(img1_gray, aspect='equal', cmap='gray', vmin=0, vmax=255)
plt.suptitle('4 - Exemplo correção/equalização de histograma')
plt.title('Imagem 1 - original')
# plt.show()

plt.figure()
plt.hist(img1_gray)
plt.suptitle('4 - Exemplo correção/equalização de histograma')
plt.title('Histograma Imagem 1 - original')
# plt.show()

# Contrast stretching
p2, p98 = np.percentile(img1_gray, (2, 98))
img1_rescale = 255 * exposure.rescale_intensity(img1_gray, in_range=(p2, p98))

# Equalization
img1_eq_sci = 255 * exposure.equalize_hist(img1_gray)
img1_eq_open = cv2.equalizeHist(img1_gray)

# Adaptive Equalization
img1_adapteq = 255 * exposure.equalize_adapthist(img1_gray, clip_limit=0.03)


plt.figure()
plt.imshow(img1_adapteq, aspect='equal', cmap='gray', vmin=0, vmax=255)
plt.suptitle('4 - Exemplo correção/equalização de histograma')
plt.title('Imagem 1 - corrigida')
# plt.show()

plt.figure()
plt.hist(img1_adapteq)
plt.suptitle('4 - Exemplo correção/equalização de histograma')
plt.title('Histograma Imagem 1 - corrigida')
# plt.show()

filename2 = 'longhair_dude_low.png'

img2 = cv2.imread(filename2)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

plt.figure()
plt.imshow(img2_gray, aspect='equal', cmap='gray', vmin=0, vmax=255)
plt.suptitle('4 - Exemplo correção/equalização de histograma')
plt.title('Imagem 2 - original')
# plt.show()

plt.figure()
plt.hist(img2_gray)
plt.suptitle('4 - Exemplo correção/equalização de histograma')
plt.title('Histograma Imagem 2 - original')
# plt.show()

# Contrast stretching
p2, p98 = np.percentile(img2_gray, (2, 98))
img2_rescale = 255 * exposure.rescale_intensity(img2_gray, in_range=(p2, p98))

# Equalization
img2_eq_sci = 255 * exposure.equalize_hist(img2_gray)
img2_eq_open = cv2.equalizeHist(img2_gray)

# Adaptive Equalization
img2_adapteq = 255 * exposure.equalize_adapthist(img2_gray, clip_limit=0.03)


plt.figure()
plt.imshow(img2_eq_open, aspect='equal', cmap='gray', vmin=0, vmax=255)
plt.suptitle('4 - Exemplo correção/equalização de histograma')
plt.title('Imagem 2 - corrigida')
# plt.show()

plt.figure()
plt.hist(img2_eq_open)
plt.suptitle('4 - Exemplo correção/equalização de histograma')
plt.title('Histograma Imagem 2 - corrigida')
# plt.show()

filename3 = '/home/felipegutierrez/Documents/PDI/bscan_acrilico2_env.jpeg'

img3 = cv2.imread(filename3)
img3_gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

# Contrast stretching
p2, p98 = np.percentile(img3_gray, (2, 98))
img3_rescale = 255 * exposure.rescale_intensity(img3_gray, in_range=(p2, p98))

# Equalization
img3_eq = 255 * exposure.equalize_hist(img3_gray)

# Adaptive Equalization
img3_adapteq = 255 * exposure.equalize_adapthist(img3_gray, clip_limit=0.03)

plt.figure()
plt.imshow(img3_gray, aspect='equal', cmap='gray', vmin=0, vmax=255)
plt.suptitle('4 - Exemplo correção/equalização de histograma')
plt.title('Imagem 3 - original')
# plt.show()

plt.figure()
plt.hist(img3_gray)
plt.suptitle('3 - Exemplo correção/equalização de histograma')
plt.title('Histograma Imagem 3 - original')
# plt.show()

plt.figure()
plt.imshow(img3_adapteq, aspect='equal', cmap='gray', vmin=0, vmax=255)
plt.suptitle('4 - Exemplo correção/equalização de histograma')
plt.title('Imagem 3 - corrigida')
# plt.show()

plt.figure()
plt.hist(img3_adapteq)
plt.suptitle('4 - Exemplo correção/equalização de histograma')
plt.title('Histograma Imagem 3 - corrigida')
plt.show()

print('END')
