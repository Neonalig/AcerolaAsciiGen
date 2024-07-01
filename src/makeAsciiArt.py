import argparse
import os

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

metrics = ['sad', 'mse', 'cosine']
color_metrics = ['avg', 'kmeans']

def load_character_data(dat_dir, output_size):
    character_data = {}
    for filename in os.listdir(dat_dir):
        if filename.endswith('.dat'):
            char_code = int(filename.split('.')[0])
            with open(os.path.join(dat_dir, filename), 'rb') as f:
                pixels = np.frombuffer(f.read(), dtype=np.uint8).reshape(output_size)
            character_data[char_code] = pixels
    return character_data

def calculate_sum_of_absolute_differences(char_pixels, image_pixels):
    return np.sum(np.abs(char_pixels - image_pixels))

def calculate_mean_squared_error(char_pixels, image_pixels):
    return np.mean((char_pixels - image_pixels) ** 2)

def calculate_cosine_similarity(char_pixels, image_pixels):
    char_pixels_flat = char_pixels.flatten()
    image_pixels_flat = image_pixels.flatten()
    cosine_similarity = np.dot(char_pixels_flat, image_pixels_flat) / (np.linalg.norm(char_pixels_flat) * np.linalg.norm(image_pixels_flat))
    return 1 - cosine_similarity

def calculate_similarity(char_pixels, image_pixels, metric='sad'):
    if metric == 'sad':
        return calculate_sum_of_absolute_differences(char_pixels, image_pixels)
    elif metric == 'mse':
        return calculate_mean_squared_error(char_pixels, image_pixels)
    elif metric == 'cosine':
        return calculate_cosine_similarity(char_pixels, image_pixels)
    else:
        raise ValueError("Unsupported metric")

def find_best_match(character_data, image_pixels, metric='sad'):
    best_match = None
    lowest_difference = float('inf')

    for char_code, char_pixels in character_data.items():
        difference = calculate_similarity(char_pixels, image_pixels, metric)
        if difference < lowest_difference:
            lowest_difference = difference
            best_match = char_code

    return best_match

def calculate_average_color(quadrant):
    return np.mean(quadrant, axis=(0, 1))

def calculate_kmeans_color(quadrant):
    kmeans = KMeans(n_clusters=1)
    kmeans.fit(quadrant.reshape(-1, 3))
    return kmeans.cluster_centers_[0]

def calculate_dominant_color(quadrant, metric='avg'):
    if metric == 'avg':
        return calculate_average_color(quadrant)
    elif metric == 'kmeans':
        return calculate_kmeans_color(quadrant)
    else:
        raise ValueError("Unsupported metric")

def process_image(image_path, character_data, output_size, invert, metric='sad', color_mode=None):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((image.width // output_size[0] * output_size[0], image.height // output_size[1] * output_size[1]))
    image_data = np.array(image)
    if invert:
        image_data = 255 - image_data

    rows, cols = image_data.shape[0] // output_size[0], image_data.shape[1] // output_size[1]
    output_text = []
    colors = []

    grayscale_image_data = np.array(Image.fromarray(image_data).convert('L'))

    for y in range(rows):
        line = []
        color_row = []
        for x in range(cols):
            quadrant = grayscale_image_data[y*output_size[1]:(y+1)*output_size[1], x*output_size[0]:(x+1)*output_size[0]]
            best_match = find_best_match(character_data, quadrant, metric)
            line.append(chr(best_match))
            if color_mode:
                rgb_quadrant = image_data[y*output_size[1]:(y+1)*output_size[1], x*output_size[0]:(x+1)*output_size[0]]
                dominant_color = calculate_dominant_color(rgb_quadrant, color_mode)
                color_row.append(dominant_color)
        output_text.append(''.join(line))
        if color_mode:
            colors.append(color_row)

    return '\n'.join(output_text), colors

def blend_colors(fg_color, bg_color, alpha):
    return (1 - alpha) * np.array(bg_color) + alpha * np.array(fg_color)

def generate_image(image, character_data, output_size, invert, output_file, metric='sad', output_image=None, color_mode=None, fg_color=None, bg_color=(0, 0, 0)):
    ascii_art, colors = process_image(image, character_data, tuple(output_size), invert, metric, color_mode)
    with open(output_file, 'w') as f:
        f.write(ascii_art)
    print(f"ASCII art saved to {output_file}")

    if output_image:
        img_width = len(ascii_art.split('\n')[0]) * output_size[0]
        img_height = ascii_art.count('\n') * output_size[1]
        output_img = Image.new('RGB', (img_width, img_height), color=(255, 255, 255))

        y_offset = 0
        for i, line in enumerate(ascii_art.split('\n')):
            x_offset = 0
            for j, char in enumerate(line):
                char_code = ord(char)
                if char_code in character_data:
                    char_img_bw = Image.fromarray(character_data[char_code])
                    if color_mode:
                        dominant_color = colors[i][j]
                        char_fg_color = fg_color or dominant_color
                        char_bg_color = bg_color or dominant_color
                        # For each pixel, blend the foreground and background colors based on the pixel intensity
                        char_img = Image.new('RGB', char_img_bw.size)
                        for y in range(char_img_bw.height):
                            for x in range(char_img_bw.width):
                                alpha = char_img_bw.getpixel((x, y)) / 255
                                color = blend_colors(char_bg_color, char_fg_color, alpha)
                                char_img.putpixel((x, y), tuple(color.astype(int)))
                    else:
                        char_img = char_img_bw
                    output_img.paste(char_img, (x_offset, y_offset))
                x_offset += output_size[0]
            y_offset += output_size[1]

        output_img.save(output_image)
        print(f"Image representation saved to {output_image}")

def main():
    parser = argparse.ArgumentParser(description="Convert an image to ASCII art using character rasters.")
    parser.add_argument('--image', type=str, required=True, help="Path to the input image.")
    parser.add_argument('--dat_dir', type=str, required=True, help="Directory containing character .dat files.")
    parser.add_argument('--output_size', type=int, nargs=2, default=(8, 8), help="Size of each character (width, height).")
    parser.add_argument('--output_file', type=str, default='output.txt', help="File to save the ASCII art.")
    parser.add_argument('--output_image', type=str, help="File to save the image representation of the ASCII art.")
    parser.add_argument('--metric', type=str, default='sad', choices=metrics + ['all'], help="Difference metric to use.")
    parser.add_argument('--invert', action='store_true', help="Invert the image before processing.")
    parser.add_argument('--color_mode', type=str, choices=color_metrics, help="Color mode to use for the output image.")
    parser.add_argument('--fg_color', type=str, help="Foreground color in hex format (e.g., #FFFFFF).")
    parser.add_argument('--bg_color', type=str, default='#FFFFFF', help="Background color in hex format (e.g., #FFFFFF).")
    args = parser.parse_args()

    fg_color = tuple(int(args.fg_color[i:i+2], 16) for i in (1, 3, 5)) if args.fg_color else None
    bg_color = tuple(int(args.bg_color[i:i+2], 16) for i in (1, 3, 5))

    character_data = load_character_data(args.dat_dir, tuple(args.output_size))
    if args.metric == 'all':
        output_file_base = args.output_file.split('.')[0]
        for metric in metrics:
            generate_image(args.image, character_data, args.output_size, args.invert, f"{output_file_base}_{metric}.txt", metric, args.output_image, args.color_mode, fg_color, bg_color)
    else:
        generate_image(args.image, character_data, args.output_size, args.invert, args.output_file, args.metric, args.output_image, args.color_mode, fg_color, bg_color)

if __name__ == "__main__":
    main()
