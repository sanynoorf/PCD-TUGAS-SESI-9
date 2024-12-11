import imageio
import numpy as np
import matplotlib.pyplot as plt

def roberts_edge_detection(image):
    # Kernel Roberts
    kernel_x = np.array([[1, 0], [0, -1]])
    kernel_y = np.array([[0, 1], [-1, 0]])

    # Mendapatkan ukuran gambar
    rows, cols = image.shape

    # Membuat gambar kosong untuk hasil
    edge_image = np.zeros((rows, cols))

    # Menerapkan kernel Roberts
    for i in range(rows - 1):
        for j in range(cols - 1):
            gx = np.sum(kernel_x * image[i:i+2, j:j+2])
            gy = np.sum(kernel_y * image[i:i+2, j:j+2])
            edge_image[i, j] = np.sqrt(gx**2 + gy**2)

    # Normalisasi hasil
    edge_image = (edge_image / np.max(edge_image) * 255).astype(np.uint8)

    return edge_image

# Membaca gambar
image_path = 'beruang.jpg'  # Ganti dengan path gambar Anda
image = imageio.imread(image_path)

# Mengonversi gambar ke grayscale
if len(image.shape) == 3:  # Jika gambar berwarna
    image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])  # Konversi ke grayscale

# Menerapkan deteksi tepi
edges = roberts_edge_detection(image)

# Menampilkan gambar asli dan hasil deteksi tepi
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Gambar Asli')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Deteksi Tepi Roberts')
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.show()