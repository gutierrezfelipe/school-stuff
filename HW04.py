## PDI - HW04 - Felipe Derewlany Gutierrez - UTFPR-CT
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import skimage
from scipy import fft


matplotlib.use('TkAgg')

# synthetic
N = 512
synthetic_img = np.zeros((N, N), dtype='uint8')

# rectangles
for i in range(10):
    rect_size_h = np.random.randint(N//2)
    rect_size_w = np.random.randint(N//2)
    rect_origin = np.random.randint(N)
    synthetic_img[rect_origin:rect_origin+rect_size_h, rect_origin:rect_origin+rect_size_w] = 127

# plt.figure()
# plt.imshow(synthetic_img, cmap='gray', vmin=0, vmax=255)

noisy = 255 * skimage.util.random_noise(synthetic_img, mode='gaussian')

# plt.figure()
# plt.imshow(noisy, cmap='gray')

# (c) seno com 8 ciclos ao longo da largura
k = 30
x = np.arange(N)
z = np.arange(N)
X, Z = np.meshgrid(x, z)
I = 1/2 * (1 + np.sin(2*np.pi / N*k*X))
noisy[:, :] += 127 * I

IC = 1/2 * (1 + np.cos(2*np.pi / (2*N) * k * (X +Z)))  # cosseno com 8 ciclos na diagonal
synth_cos = 127 * IC
per_noisy = noisy + synth_cos
# per_noisy /= 2

plt.figure()
plt.imshow(per_noisy, cmap='gray')

PER_NOISYs = fft.fftshift(fft.fft2(per_noisy))

plt.figure()
plt.imshow(np.log(1+np.absolute(PER_NOISYs)), cmap='gray')

# notch filter generation
a1 = 0.064
a2 = 0.064

NF1 = 1 - np.exp(- (a1*(X-226)**2 + a2*(Z-256)**2))  # Gaussian
NF2 = 1 - np.exp(- (a1*(X-241)**2 + a2*(Z-241)**2))  # Gaussian
NF3 = 1 - np.exp(- (a1*(X-271)**2 + a2*(Z-271)**2))  # Gaussian
NF4 = 1 - np.exp(- (a1*(X-286)**2 + a2*(Z-256)**2))  # Gaussian

# filtro
Z = NF1*NF2*NF3*NF4

plt.figure()
plt.imshow(np.log(1+np.absolute(Z)), cmap='gray')

# aplicação do filtro
IMFs = PER_NOISYs * Z

# inverte o fftshift pra calcular a transformada inversa (imagem filtrada)
IMFr = np.fft.ifftshift(IMFs)
imfr = np.fft.ifft2(IMFr)

plt.figure()
plt.imshow(np.log(1+np.absolute(IMFs)), cmap='gray')

plt.figure()
plt.imshow(np.real(imfr), cmap='gray')

### SEGMENTATION and COUNTING
imseg = cv2.imread('./PDI_Final_Project/ode_to_joy.jpg', cv2.IMREAD_GRAYSCALE)
# imseg_g = 255 * skimage.color.rgb2gray(imseg)

# invert grayscale image
inv_img_g = 255 - imseg

# binarization
threshold = skimage.filters.threshold_otsu(imseg)
img_binary = imseg > threshold

inv_img = 1 - img_binary

# staff detection
row_sum = np.sum(img_binary, axis=1)

# plt.figure()
# plt.imshow(imseg, cmap='gray')
# plt.plot(row_sum, np.arange(img_binary.shape[0]))

staffs_line = np.where(row_sum < 50)[0]
max_width = np.min(np.diff(staffs_line)) + 1
real_lines = staffs_line[np.where(np.diff(staffs_line) > 1)]
all_lines = np.append(real_lines, staffs_line[-1])
print(f'A quantidade de linhas de pauta é {all_lines.shape[0]}')

# plt.figure()
# plt.imshow(imseg, cmap='gray')
# for i in range(all_lines.shape[0]):
#     plt.annotate(f'{i+1}', xy=(round(0.95 * image.shape[1]), all_lines[i]))
#
# plt.annotate(f'TOTAL: {i+1}', xy=(round(0.8 * image.shape[1]), round(0.9 * image.shape[0])))
# plt.show()

# separating the staves
n_staff = all_lines.shape[0] // 5
staff = np.zeros((n_staff, 5), dtype='uint')
for i in range(n_staff):
    staff[i, :] = all_lines[5*i:5*(i+1)]


# structuring element vertical bar, size max_width+1
remov_foot = np.ones((max_width+1, 1))
wo_lines = skimage.morphology.opening(inv_img, remov_foot)

# only note heads
wo_beams = skimage.morphology.opening(wo_lines, remov_foot.T)

# open the image with disk structuring element to further isolate note heads
disk_foot = skimage.morphology.disk(3)
black_note_heads = skimage.morphology.opening(wo_beams, disk_foot)  # the g clef ending is still on the image, crop?
note_heads_gray = black_note_heads * inv_img_g

# notes margin
notes_offset_x = imseg.shape[1] // 10  # top-left margin limit x
notes_offset_z = imseg.shape[0] // 7  # top-left margin limit z

# notes mask
note_mask = np.full_like(imseg, True)
note_mask[:notes_offset_z, :] = False
note_mask[:, :notes_offset_x] = False

# only black (painted) notes
only_notes = note_heads_gray * note_mask

# detect blo
blobs_log = skimage.feature.blob_log(only_notes / 255, max_sigma=30, num_sigma=10, threshold=.1)
print(f'A quantidade de notas pretas é {blobs_log.shape[0]}')


# # testing
# image = skimage.color.gray2rgb(imseg/255)
# # image = skimage.color.gray2rgb(inv_img_g/255)
# image_test = np.zeros_like(image)
#
# # draw cirles
# for i in range(blobs_log.shape[0]):
#     circy, circx = skimage.draw.circle_perimeter(int(blobs_log[i, 0]), int(blobs_log[i, 1]), int(blobs_log[i, 2]),
#                                     shape=image.shape)
#     image[circy, circx] = (20, 220, 20)
#     image_test[circy, circx] = (220, 20, 20)

# plt.figure()
# plt.imshow(image)
# for i in range(blobs_log.shape[0]):
#     plt.annotate(f'{i+1}', xy=(int(blobs_log[i, 1] + 3), int(blobs_log[i, 0] - 5)))
#
# plt.annotate(f'TOTAL: {i+1}', xy=(round(0.8 * image.shape[1]), round(0.9 * image.shape[0])))
# plt.show()

print('THE END')
