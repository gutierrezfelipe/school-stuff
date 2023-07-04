import imageio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import skimage
import cv2


matplotlib.use('TkAgg')

# opening image
img_gray = cv2.imread('ode_to_joy.jpg', cv2.IMREAD_GRAYSCALE)

# invert grayscale image
inv_img_g = 255 - img_gray

# binarization
threshold = skimage.filters.threshold_otsu(img_gray)
img_binary = img_gray > threshold

# inverse binary
inv_img = 1 - img_binary

# staff detection
row_sum = np.sum(img_binary, axis=1)

# Limiar excolhido analisando gráfico
staffs_line = np.where(row_sum < 50)[0]
real_lines = staffs_line[np.where(np.diff(staffs_line) > 1)]
all_lines = np.append(real_lines, staffs_line[-1])

# separating the staves
n_staff = 4
staff = np.zeros((n_staff, 5), dtype='uint')
for i in range(n_staff):
    staff[i, :] = all_lines[5*i:5*(i+1)]

# staff removal
max_width = np.min(np.diff(staffs_line)) + 1

# structuring element vertical bar, size max_width+1
remov_foot = np.ones((max_width+1, 1))
wo_lines = skimage.morphology.opening(inv_img, remov_foot)

# only note heads
wo_beams = skimage.morphology.opening(wo_lines, remov_foot.T)

# open the image with disk structuring element to further isolate note heads
disk_foot = skimage.morphology.disk(3)
black_note_heads = skimage.morphology.opening(wo_beams, disk_foot)
note_heads_gray = black_note_heads * inv_img_g

# notes margin
notes_offset_x = img_gray.shape[1] // 10  # top-left margin limit
notes_offset_z = img_gray.shape[0] // 7  # top-left margin limit

note_mask = np.full_like(img_gray, True)
note_mask[:notes_offset_z, :] = False
note_mask[:, :notes_offset_x] = False

only_notes = note_heads_gray * note_mask

# detecting the center of note head blobs
blobs_log = skimage.feature.blob_log(only_notes / 255, max_sigma=30, num_sigma=10, threshold=.1)

# checking positions
image = skimage.color.gray2rgb(img_gray)
image_test = np.zeros_like(image)

for i in range(blobs_log.shape[0]):
    circy, circx = skimage.draw.circle_perimeter(int(blobs_log[i, 0]), int(blobs_log[i, 1]), int(blobs_log[i, 2]),
                                    shape=image.shape)
    image[circy, circx] = (20, 220, 20)
    image_test[circy, circx] = (220, 20, 20)


# note position in relation to staff (pitch determination)
notes = []
notes_zx = np.zeros((2, blobs_log.shape[0]), dtype='uint')
notes_zx[0, :] = blobs_log[:, 0].T
notes_zx[1, :] = blobs_log[:, 1].T

# notes on first staff_line
for i in range(notes_zx.shape[1]):
    if (notes_zx[0, i] >= staff[0, 0]) and (notes_zx[0, i] <= staff[1, 0]):
        notes.append(notes_zx[:, i])

# Check Note Pitch
note_names = []
for i in range(len(notes)):
    # notes bellow lines
    if notes[i][0] > staff[0, 4]:
        note_names.append('D4')

    # notes on spaces
    elif (notes[i][0] < staff[0, 1]) and (notes[i][0] > staff[0, 0]):
        note_names.append('E5')
    elif (notes[i][0] < staff[0, 2]) and (notes[i][0] > staff[0, 1]):
        note_names.append('C#5')
    elif (notes[i][0] < staff[0, 3]) and (notes[i][0] > staff[0, 2]):
        note_names.append('A4')
    elif (notes[i][0] < staff[0, 4]) and (notes[i][0] > staff[0, 3]):
        note_names.append('F#4')

    # notes on lines
    elif (notes[i][0] == staff[0, 0]) or (notes[i][0] >= staff[0, 0]):
        note_names.append('F#5')
    elif (notes[i][0] == staff[0, 1]) or (notes[i][0] >= staff[0, 1]):
        note_names.append('D5')
    elif (notes[i][0] == staff[0, 2]) or (notes[i][0] >= staff[0, 2]):
        note_names.append('B4')
    elif (notes[i][0] == staff[0, 3]) or (notes[i][0] >= staff[0, 3]):
        note_names.append('G4')
    elif (notes[i][0] == staff[0, 4]) or (notes[i][0] >= staff[0, 4]):
        note_names.append('E4')

    else:
        note_names.append('missed')

# List Output
for i in range(len(notes)):
    print(f'Note {i} (Pitch): {note_names[i]}')


print('The END!')

