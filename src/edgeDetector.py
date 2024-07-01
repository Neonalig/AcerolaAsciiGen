import argparse
import cv2
import numpy as np
from PIL import Image

def difference_of_gaussians(image, ksize1=5, ksize2=9):
    # Apply Gaussian blur with two different kernel sizes
    blur1 = cv2.GaussianBlur(image, (ksize1, ksize1), 0)
    blur2 = cv2.GaussianBlur(image, (ksize2, ksize2), 0)
    # Compute the Difference of Gaussians
    dog = cv2.absdiff(blur1, blur2)
    return dog

def sobel_filter(image):
    # Apply Sobel filter to find edges
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel = np.uint8(sobel)
    return sobel

def main():
    parser = argparse.ArgumentParser(description="Perform Difference of Gaussians and Sobel filter on an image.")
    parser.add_argument('--image', type=str, required=True, help="Path to the input image.")
    parser.add_argument('--dog_output', type=str, default='dog_output.png', help="File to save the DoG output image.")
    parser.add_argument('--sobel_output', type=str, default='sobel_output.png', help="File to save the Sobel filter output image.")
    args = parser.parse_args()

    # Load the image
    image = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found at {args.image}")

    # Perform Difference of Gaussians
    dog_image = difference_of_gaussians(image)

    # Perform Sobel filter on DoG image
    sobel_image = sobel_filter(dog_image)

    # Save the output images
    Image.fromarray(dog_image).save(args.dog_output)
    Image.fromarray(sobel_image).save(args.sobel_output)

    print(f"Difference of Gaussians output saved to {args.dog_output}")
    print(f"Sobel filter output saved to {args.sobel_output}")

if __name__ == "__main__":
    main()
