# imaging.py

import random
import cv2 # OpenCV for image processing

def show_random_cv2_image(imgs: np.ndarray) -> None:
    cv2_imshow(random.choice(imgs)) # Using cv2_imshow to display the image
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def create_bgr_images(imgs: np.ndarray) -> list:
    bgr_images = []
    for img in imgs:
        bgr_images.append(img)

    bgr_images = np.array(bgr_images)

    # Now, BGR images contains all images...
    print(f'Shape of BGR images: {bgr_images.shape}')

    return bgr_images

def convert_to_rgb(imgs: np.ndarray) -> list:
    rgb_images = []
    for img in imgs:
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb_images.append(rgb_image)

    # Get image RGB
    rgb_images = np.array(rgb_images)

    print(f'Shape of RGB images: {rgb_images.shape}')

    return rgb_images

# Show random image using matplotlib
def show_random_image(imgs: np.ndarray) -> None:
    random_index = random.randint(0, len(imgs))
    plt.imshow(imgs[random_index], cmap='gray')
    plt.title(f'Random Sample Image | Index: {random_index}')
    plt.show()

def visualize_raw_image_data(imgs: np.ndarray, labels: np.ndarray, rows:int=IMAGE_ROWS, cols:int=IMAGE_COLS) -> None:
    """
    Visualizes random image data from the input array in a grid layout.

    Uses plt.tight_layout() to automatically adjust subplot parameters
    for a tight layout, preventing overlap.
    """
    fig = plt.figure(figsize=(10, 8))
    label_cnt = len(labels) # Assuming labels is an array where each element corresponds to an image

    for i in range(cols):
        for j in range(rows):
            # Select a random index from the available labels/images
            # We use label_cnt as the upper bound (exclusive)
            random_index = np.random.randint(0, label_cnt)

            # Subplots are 1-indexed (i * rows + j + 1)
            ax = fig.add_subplot(rows, cols, i * cols + j + 1) # Note: Changed i * rows to i * cols for standard subplot counting

            # Display the image data
            ax.imshow(imgs[random_index, :])

            # Set the title for the subplot
            ax.set_title(labels[random_index])

            # Remove the ticks/labels for a cleaner image visualization
            ax.axis('off')

    # Automatically adjust subplot parameters to give a tight layout.
    plt.tight_layout()

    plt.show()

    # Show images from training data
def visualize_augmented_image_batch(train_generator: NumpyArrayIterator, enc: LabelEncoder) -> None:
    img, img_labels = next(train_generator)
    fig, axes = plt.subplots(4, 4, figsize=(14, 7))
    fig.set_size_inches(12, 12)

    categories = np.unique(img_labels)
    keys = dict(enumerate(label_encoder.classes_))

    for (img, label_index, ax) in zip(img, np.argmax(img_labels, axis=1), axes.flatten()):
        ax.imshow(img)
        ax.set_title(keys[label_index]) # Map numeric label to plant name using keys
        ax.axis('off')

    plt.show()


