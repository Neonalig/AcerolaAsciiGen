import argparse
import os

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm  # Step 1: Import tqdm

# Define character sets
CHARACTER_SETS = {
    "printable_ascii": range(32, 127),
    "extended_ascii": range(32, 256)
}

def load_charset_from_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return [ord(char) for char in content]

def save_character_image(char, output_size, font, output_dir, antialias, transparent):
    # Create a new image
    if transparent:
        image = Image.new('RGBA', output_size, (255, 255, 255, 0))
    else:
        image = Image.new('L', output_size, 255)

    draw = ImageDraw.Draw(image)

    # Calculate the position to center the character
    bbox = draw.textbbox((0, 0), char, font=font)
    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    position = ((output_size[0] - width) / 2 - bbox[0], (output_size[1] - height) / 2 - bbox[1])

    # Draw the character on the image
    draw.text(position, char, font=font, fill=0)

    if not antialias:
        image = image.convert('1')  # Convert to 1-bit pixels for no antialiasing

    # Save as PNG
    image.save(os.path.join(output_dir, f"{ord(char)}.png"))

    # Save as custom binary .dat file
    pixels = list(image.convert("L").getdata())
    with open(os.path.join(output_dir, f"{ord(char)}.dat"), 'wb') as f:
        f.write(bytearray(pixels))

    return image

def main():
    parser = argparse.ArgumentParser(description="Render characters from a font to PNG and binary .dat files.")
    parser.add_argument('--font', type=str, default="JetBrainsMono.ttf",
                        help="Path to the font file.")
    parser.add_argument('--charset', type=str, default="printable_ascii",
                        help="Character set to render. Either a predefined set name or a path to a text file. Valid set names: " + ", ".join(CHARACTER_SETS.keys()))
    parser.add_argument('--output_size', type=int, nargs=2, default=(8, 8),
                        help="Output size of each character (width height).")
    parser.add_argument('--output_dir', type=str, default='output', help="Directory to save output files.")
    parser.add_argument('--antialias', action='store_true', default=True, help="Enable antialiasing. This is the default behavior.")
    parser.add_argument('--raster', dest='antialias', action='store_false', help="Disable antialiasing, enabling rasterization.")
    parser.add_argument('--transparent', action='store_true', help="Enable transparency.")
    parser.add_argument('--verify', action='store_true', help="Output an image with all characters for verification.")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load font
    font = ImageFont.truetype(args.font, size=args.output_size[1])

    # Determine character set
    if os.path.isfile(args.charset):
        charset = load_charset_from_file(args.charset)
    else:
        charset = CHARACTER_SETS.get(args.charset.lower(), CHARACTER_SETS["printable_ascii"])

    # Process each character in the selected character set with tqdm for progress tracking
    images = {}
    for code in tqdm(charset, desc="Rendering characters"):
        char = chr(code)
        images[char] = save_character_image(char, args.output_size, font, args.output_dir, args.antialias, args.transparent)

    # Create a verification image and text file
    if args.verify:
        if args.transparent:
            verify_image = Image.new('RGBA', (args.output_size[0] * 16, args.output_size[1] * (len(images) // 16 + 1)), (255, 255, 255, 0))
        else:
            verify_image = Image.new('L', (args.output_size[0] * 16, args.output_size[1] * (len(images) // 16 + 1)), 255)

        for i, (char, image) in enumerate(images.items()):
            x = i % 16
            y = i // 16
            verify_image.paste(image, (x * args.output_size[0], y * args.output_size[1]))
        verify_image.save(os.path.join(args.output_dir, "verification.png"))

        with open(os.path.join(args.output_dir, "verification.txt"), 'w') as f:
            for i, char in enumerate(images.keys()):
                f.write(char)
                if (i + 1) % 16 == 0:
                    f.write('\n')

    print("Character images and .dat files have been saved.")

if __name__ == "__main__":
    main()