import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2


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
circle_radius = 100
circle_center_x = 400
circle_center_z = 400

# matemática
# for i in range(N):
#     for j in range(N):
#         if (i - circle_center_x) ** 2 + (j - circle_center_z) ** 2 < circle_radius ** 2:
#                 image[j, i, 1] = 127

# opencv
cv2.circle(image, (circle_center_x, circle_center_z), circle_radius, (0, 127, 0), -1, 8)

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
merged_rgb_img = cv2.merge([image[:, :, 0], image[:, :, 1], image[:, :, 2]])

plt.figure(4)
plt.imshow(merged_rgb_img, aspect='equal')
plt.title('1d - imagem RGB')
# plt.show()

# 2 - CROP imagem central de 400 x 400 pixels e apresentar
crop_size = 400
diff = N - crop_size
cropped_img = merged_rgb_img[diff//2:N-diff//2, diff//2:N-diff//2, :]

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
# plt.imshow(In_norm, aspect='equal')
# plt.show()

# (c) Apresentar AVG e STD_DEV de (A) e (B)
print(f'AVG(A) = {np.average(In)}')
print(f'STD_DEV(A) = {np.std(In)}')
print(f'AVG(B) = {np.average(In_db)}')
print(f'STD_DEV(B) = {np.std(In_db)}')

# 4 - 3 exemplos "SEUS" de correção de histograma
# filename1 = 'something.png'
# image1 = cv2.imread(filename1)

frame = np.load('frame1501_112x112_liu1997_python.npy')

# plt.figure(10)
# plt.imshow(frame, aspect='equal', cmap='gray', vmin=0, vmax=255)
# # plt.show()
#
# plt.figure(11)
# plt.hist(frame)
# # plt.show()

bug = cv2.imread('stinkbug.png')
bug_gray = cv2.cvtColor(bug, cv2.COLOR_BGR2GRAY)

plt.figure()
plt.imshow(bug_gray, aspect='equal', cmap='gray', vmin=0, vmax=255)
plt.suptitle('4 - Exemplo correção/equalização de histograma')
plt.title('Imagem 1 - original')
# plt.show()

plt.figure()
plt.hist(bug_gray)
plt.suptitle('4 - Exemplo correção/equalização de histograma')
plt.title('Histograma Imagem 1 - original')
# plt.show()

bug_eq = cv2.equalizeHist(bug_gray)

plt.figure()
plt.imshow(bug_eq, aspect='equal', cmap='gray', vmin=0, vmax=255)
plt.suptitle('4 - Exemplo correção/equalização de histograma')
plt.title('Imagem 1 - corrigida')
# plt.show()

plt.figure()
plt.hist(bug_eq)
plt.suptitle('4 - Exemplo correção/equalização de histograma')
plt.title('Histograma Imagem 1 - corrigida')
# plt.show()





print('END')
