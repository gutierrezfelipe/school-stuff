## PDI - HW03 - Felipe Derewlany Gutierrez - UTFPR-CT
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import skimage
import scipy


matplotlib.use('TkAgg')

# from file
filename = 'HW03_images/clave-de-sol.png'
img = cv2.imread(filename)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# binarization - otsu threshold
threshold = skimage.filters.threshold_otsu(img_gray)
img_binary = img_gray > threshold

plt.figure()
plt.imshow(img_binary, aspect='equal', cmap='gray', vmin=0, vmax=1)


# # synthetic
# N = 512
# synthetic_img = np.zeros((N, N), dtype='uint8')
#
# # rectangles
# for i in range(10):
#     rect_size_h = np.random.randint(N//2)
#     rect_size_w = np.random.randint(N//2)
#     rect_origin = np.random.randint(N)
#     synthetic_img[rect_origin:rect_origin+rect_size_h, rect_origin:rect_origin+rect_size_w] = 127
#
# plt.figure()
# plt.imshow(synthetic_img, cmap='gray', vmin=0, vmax=255)
#
# noisy = 255 * skimage.util.random_noise(synthetic_img, mode='gaussian')
# noise = np.random.normal(0, 1, (N, N)).astype('uint8')
# synth_noisy = synthetic_img + noise
# plt.figure()
# plt.imshow(synth_noisy, cmap='gray')


# convolution kernels
# 1 - laplacian / edge detection - used for edge detection
lap_kernel = np.array((0, 1, 0, 1, -4, 1, 0, 1, 0))
lap_kernel = lap_kernel.reshape((3, 3))

lap_filtered = scipy.ndimage.convolve(img_gray, lap_kernel, mode='constant')

plt.figure()
plt.imshow(lap_filtered, cmap='gray')
plt.title('Laplaciano [0 1 0, 1 -4 1, 0 1 0]')

# 2 - laplaciano 2 / edge detection - used for edge detection
lap_kernel_2 = np.array((1, 1, 1, 1, -8, 1, 1, 1, 1))
lap_kernel_2 = lap_kernel_2.reshape((3, 3))

lap2_filtered = scipy.ndimage.convolve(img_gray, lap_kernel_2, mode='constant')

plt.figure()
plt.imshow(lap_filtered, cmap='gray')
plt.title('Laplaciano [1 1 1, 1 -8 1, 1 1 1]')

equal = np.equal(lap_filtered, lap2_filtered)
plt.figure()
plt.imshow(equal, cmap='gray')
plt.title('diferença entre lap 1 e 2')

# 3 - box blur filter (mean/average) - smoothing, gaussian denoise
box_kernel = np.array((1, 1, 1, 1, 1, 1, 1, 1, 1))
box_kernel = 1 / 9 * box_kernel.reshape((3, 3))

box_filtered = scipy.ndimage.convolve(img_gray, box_kernel, mode='constant')

plt.figure()
plt.imshow(box_filtered, cmap='gray')
plt.title('Box 1/9 * [1 1 1, 1 1 1, 1 1 1]')

# 4 - pyramid gaussian blur filter (mean/average) - smoothing, gaussian denoise
pyramid = np.array((1, 2, 1, 2, 4, 2, 1, 2, 1))
pyramid = 1 / 16 * pyramid.reshape((3, 3))

pyramid_filtered = scipy.ndimage.convolve(img_gray, pyramid, mode='constant')

plt.figure()
plt.imshow(pyramid_filtered, cmap='gray')
plt.title('Gaussian 1/16 * [1 2 1, 2 4 2, 1 2 1]')

# 5 - box blur 2 filter (mean/average) - smoothing, gaussian denoise
box2_kernel = np.array((1, 1, 1, 1, 2, 1, 1, 1, 1))
box2_kernel = 1 / 10 * box_kernel.reshape((3, 3))

box2_filtered = scipy.ndimage.convolve(img_gray, box2_kernel, mode='constant')

plt.figure()
plt.imshow(box2_filtered, cmap='gray')
plt.title('Box 1/10 * [1 1 1, 1 2 1, 1 1 1]')

# 6 - Gaussian blur 5x5 - smoothing, gaussian denoise
gaussian5 = np.array((1, 4, 6, 4, 1,
                      4, 16, 24, 16, 4,
                      6, 24, 36, 24, 6,
                      4, 16, 24, 16, 4,
                      1, 4, 6, 4, 1))

gaussian5 = 1 / 256 * gaussian5.reshape((5, 5))

gaussian5_filtered = scipy.ndimage.convolve(img_gray, gaussian5, mode='constant')

plt.figure()
plt.imshow(pyramid_filtered, cmap='gray')
plt.title('Gaussian 5 1/256 * [1 4 6 4 1, 4 16 24 16 4, 6 24 36 24 6, 4 16 24 16 4, 1 4 6 4 1]')

# 7 - sharpen - melhoramento de bordas
sharpen_level = 2  # >= 1
sharpen_kernel = np.array((0, -1, 0, -1, sharpen_level+4, -1, 0, -1, 0))
sharpen_kernel = sharpen_kernel.reshape((3, 3))

sharper_img = scipy.ndimage.convolve(img_gray, sharpen_kernel, mode='constant')

plt.figure()
plt.imshow(sharper_img, cmap='gray')
plt.title('Melhoramento [0 -1 0, -1 sharpen_level+4 -1, 0 -1 0]')

# 8 - sharpen 2 - melhoramento de bordas
sharpen2_level = 2  # >= 1
sharpen2_kernel = np.array((-1, -1, -1, -1, sharpen_level+8, -1, -1, -1, -1))
sharpen2_kernel = sharpen2_kernel.reshape((3, 3))

sharper2_img = scipy.ndimage.convolve(img_gray, sharpen2_kernel, mode='constant')

plt.figure()
plt.imshow(sharper_img, cmap='gray')
plt.title('Melhoramento [-1 -1 -1, -1 sharpen_level+8 -1, -1 -1 -1]')

# 9 e 10 - Sobel operator - used for edge detection
sobel_x = np.array((-1, 0, 1, -2, 0, 2, -1, 0, 1))
sobel_x = sobel_x.reshape((3, 3))
sobel_z = sobel_x.T

sobel_img_x = scipy.ndimage.convolve(img_gray, sobel_x, mode='constant')
sobel_img_z = scipy.ndimage.convolve(img_gray, sobel_z, mode='constant')

edge_mag = np.sqrt((sobel_img_x ** 2 + sobel_img_z ** 2))
edge_dir = np.arctan(sobel_img_z / sobel_img_x)

plt.figure()
plt.imshow(sobel_img_x, cmap='gray')
plt.title('Sobel X [-1 0 1, -2 0 2, -1 0 1]')

plt.figure()
plt.imshow(sobel_img_z, cmap='gray')
plt.title('Sobel Z [-1 -2 -1, 0 0 0, 1 2 1]')

plt.figure()
plt.imshow(sobel_img_x+sobel_img_z, cmap='gray')
plt.title('Sobel X+Z')

plt.figure()
plt.imshow(edge_mag, cmap='gray')
plt.title('Sobel edge mag')

# plt.figure()
# plt.imshow(edge_dir, cmap='gray')
# plt.title('Sobel edge dir')

# 11 - Laplacian w/ stressed significance of the central pixel - used for edge detection
str_lap_kernel = np.array((1, -2, 1, -2, 4, -2, 1, -2, 1))
str_lap_kernel = str_lap_kernel.reshape((3, 3))

str_lap_filtered = scipy.ndimage.convolve(img_gray, str_lap_kernel, mode='constant')

plt.figure()
plt.imshow(str_lap_filtered, cmap='gray')
plt.title('Laplaciano stressed [1 -2 1, -2 4 -2, 1 -2 1]')

# 12 - rotated Laplacian w/ stressed central significance
rot_str_lap_kernel = np.array((-2, 1, -2, 1, 4, 1, -2, 1, -2))
rot_str_lap_kernel = rot_str_lap_kernel.reshape((3, 3))

rot_str_lap_filtered = scipy.ndimage.convolve(img_gray, rot_str_lap_kernel, mode='constant')

plt.figure()
plt.imshow(rot_str_lap_filtered, cmap='gray')
plt.title('Laplaciano stressed rot [-2 1 -2, 1 4 1, -2 1 -2]')

# 13 e 14 - Prewitt Operators - used for edge detection
prewitt_kernel_z = np.array((-1, -1, -1, 0, 0, 0, 1, 1, 1))
prewitt_kernel_z = prewitt_kernel_z.reshape((3, 3))
prewitt_kernel_x = prewitt_kernel_z.T

prewitt_img_x = scipy.ndimage.convolve(img_gray, prewitt_kernel_x, mode='constant')
prewitt_img_z = scipy.ndimage.convolve(img_gray, prewitt_kernel_z, mode='constant')

edge_mag = np.sqrt((prewitt_img_x ** 2 + prewitt_img_z ** 2))
edge_dir = np.arctan(prewitt_img_z / prewitt_img_x)

plt.figure()
plt.imshow(prewitt_img_z, cmap='gray')
plt.title('Prewitt Z [-1 -1 -1, 0 0 0, 1 1 1]')

plt.figure()
plt.imshow(prewitt_img_x, cmap='gray')
plt.title('Prewitt X [-1 0 1, -1 0 1, -1 0 1]')

plt.figure()
plt.imshow(prewitt_img_x+prewitt_img_z, cmap='gray')
plt.title('Prewitt X+Z')

plt.figure()
plt.imshow(edge_mag, cmap='gray')
plt.title('Prewitt edge mag')

# plt.figure()
# plt.imshow(edge_dir, cmap='gray')
# plt.title('Prewitt edge dir')

# 15 e 16 - Roberts - diagonal edge detection
h_1 = np.array((1, 0, 0, -1))
h_2 = np.array((0, 1, -1, 0))

h_1 = h_1.reshape((2, 2))
h_2 = h_2.reshape((2, 2))

roberts_h1 = scipy.ndimage.convolve(img_gray, h_1, mode='constant')
roberts_h2 = scipy.ndimage.convolve(img_gray, h_2, mode='constant')

plt.figure()
plt.imshow(roberts_h1, cmap='gray')
plt.title('Roberts h_1')

plt.figure()
plt.imshow(roberts_h2, cmap='gray')
plt.title('Roberts h_2')

# plt.figure()
# plt.imshow(roberts_h1 + roberts_h2, cmap='gray')
# plt.title('Roberts sum')

# 17 e 18 - Isotropic - edge detection
iso_x = np.array((-1, 0, 1, -np.sqrt(2), 0, np.sqrt(2), -1, 0, 1))
iso_x = iso_x.reshape((3, 3))
iso_z = iso_x.T

iso_x_filtered = scipy.ndimage.convolve(img_gray, iso_x, mode='constant')
iso_z_filtered = scipy.ndimage.convolve(img_gray, iso_z, mode='constant')

plt.figure()
plt.imshow(iso_x_filtered, cmap='gray')
plt.title('Isotropic X')

plt.figure()
plt.imshow(iso_z_filtered, cmap='gray')
plt.title('Isotropic Z')

# plt.figure()
# plt.imshow(iso_x_filtered + iso_z_filtered, cmap='gray')
# plt.title('Isotropic sum')


# 19 - Kirsch Compass NE - edge detection in the North-East direction
kirsch_ne = np.array((-3, 5, 5, -3, 0, 5, -3, -3, -3))
kirsch_ne = kirsch_ne.reshape((3, 3))

kirsch_ne_filtered = scipy.ndimage.convolve(img_gray, kirsch_ne, mode='constant')

plt.figure()
plt.imshow(kirsch_ne_filtered, cmap='gray')
plt.title('Kirsch NE')

# 20 Laplacian of Gaussian (LoG)
# Da fórmula na documentação do matlab
# sigma = 1.4  # standard deviation
# x = np.linspace(-4, 4, 9)
# z = np.linspace(-4, 4, 9)
# X, Z = np.meshgrid(x, z)
# # logg = - 1 / (np.pi * sigma ** 4) * (1 - (X ** 2 + Z ** 2)/(2 * sigma ** 2)) * np.e ** (- (X ** 2 + Z ** 2)/(2 * sigma ** 2))
# h_g = np.e ** (- (X ** 2 + Z ** 2)/(2 * sigma ** 2))
# logg = (X ** 2 + Z ** 2 - 2 * sigma ** 2) * h_g / (sigma ** 4 * np.sum(h_g))
# # logg = 40 * logg / np.max(logg)
# central_ratio = logg[4, 4] / logg[0, 1]


# Da apresentação LoG para sigma = 1.4
LoG = np.array((0, 1, 1, 2, 2, 2, 1, 1, 0,
                1, 2, 4, 5, 5, 5, 4, 2, 1,
                1, 4, 5, 3, 0, 3, 5, 4, 1,
                2, 5, 3, -12, -24, -12, 3, 5, 2,
                2, 5, 0, -24, -40, -24, 0, 5, 2,
                2, 5, 3, -12, -24, -12, 3, 5, 2,
                1, 4, 5, 3, 0, 3, 5, 4, 1,
                1, 2, 4, 5, 5, 5, 4, 2, 1,
                0, 1, 1, 2, 2, 2, 1, 1, 0,
                ))

LoG = LoG.reshape((9, 9))

LoG_filtered = scipy.ndimage.convolve(img_gray, LoG, mode='constant')

plt.figure()
plt.imshow(LoG_filtered, cmap='gray')
plt.title('Laplacian of Gaussian')

# Edge detection
# I already tested Laplacian, Sobel, Prewitt and Roberts as convolution Kernels
# But I will now test them as implemented in the skimage library, and along with them the Canny edge detection method

# Laplacian
edge_l = skimage.filters.laplace(img_gray)

plt.figure()
plt.imshow(edge_l, cmap='gray')
plt.title('skimage Laplace')

# Sobel
edge_s = skimage.filters.sobel(img_gray)

plt.figure()
plt.imshow(edge_s, cmap='gray')
plt.title('skimage Sobel')

# Prewitt
edge_p = skimage.filters.prewitt(img_gray)

plt.figure()
plt.imshow(edge_p, cmap='gray')
plt.title('skimage Prewitt')

# Roberts
edges_r = skimage.filters.roberts(img_gray)

plt.figure()
plt.imshow(edges_r, cmap='gray')
plt.title('skimage Roberts')

# Canny
edges = skimage.feature.canny(img_gray)
edges2 = skimage.feature.canny(img_gray, sigma=3)

plt.figure()
plt.imshow(edges, cmap='gray')
plt.title('Canny')

plt.figure()
plt.imshow(edges2, cmap='gray')
plt.title('Canny, sigma = 3')

# There is some difference between the filters from skimage and the kernels convolved in the first part of this HW assignment.
# The Laplacian kernels from the first part performed better detecting the edges of the G clef than the skimage one.
# The Laplacian from skimage gives a smaller magnitude to the edge, as in less constrast, the other methods return a binary image.
# Maybe this is the extra step making them better than only applying the filter kernels.
# All the other ones were better in the skimage library. It seems that all would be suitable to my research.

plt.show()

print('The END!')
