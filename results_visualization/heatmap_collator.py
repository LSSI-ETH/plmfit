import tqdm
import os
import glob
from PIL import Image, ImageDraw, ImageFont

def main():
    tasks = [
        "gb1_one_vs_rest",
        "gb1_three_vs_rest",
        "aav_one_vs_many",
        "aav_sampled",
        "meltome_mixed",
        "rbd_one_vs_rest",
        "Trastuzumab_one_vs_rest",
        "ss3_sampled",
    ]

    methods_order = [
        'Feature Extraction_linear',
        'Feature Extraction_mlp',
        'LoRA (All Layers)',
        'LoRA- (Last Layer)',
        'Adapters (All Layers)',
        'Adapters- (Last Layer)',
    ]

    # For each task (tqdm) load all png files that start with {task}
    for task in tqdm.tqdm(tasks):
        # Load all png files that start with {task}
        # Sort them by the method name
        # Create a 3x2 grid collage of the images
        # Save the collage as {task}_heatmap_collage.png

        # Load all png files that start with {task}
        image_files = glob.glob(os.path.join("./results/", f"{task}*.png"))

        # Sort them by the method name (based on if methods_order variable appears in the name so Feature Extraction_linear will be first)
        image_files = sorted(image_files, key=lambda x: [method in x for method in methods_order], reverse=True)

        # Load the images
        images = [Image.open(image_file) for image_file in image_files]

        # Create a 3x2 grid collage of the images
        collage_width = max(image.width for image in images) * 2 - 300
        collage_height = max(image.height for image in images) * 3
        collage = Image.new("RGB", (collage_width, collage_height))
        draw = ImageDraw.Draw(collage)

        # Define the font for the text
        font = ImageFont.load_default(size=55)

        for i, image in enumerate(images):
            # Crop all 2nd column images 100 pixels from the left
            if i % 2 == 1:
                image = image.crop((70, 0, image.width, image.height))
            if i % 2 == 0:
                image = image.crop((0, 0, image.width - 230, image.height))
            x_offset = (i % 2) * image.width - (i % 2) * 160
            y_offset = (i // 2) * image.height
            collage.paste(image, (x_offset, y_offset))

            # Add numbers to each cell (i, ii)
            cell_number = "i" if i % 2 == 0 else "ii"
            draw.text(
                (x_offset + 55, y_offset + 31),
                cell_number,
                fill="black",
                font=ImageFont.load_default(size=35),
            )

        # Add letters to the rows (A, B, C)
        for row in range(3):
            row_letter = chr(65 + row)  # Convert 0, 1, 2 to 'A', 'B', 'C'
            draw.text(
                (10, row * images[0].height + 10),
                row_letter,
                fill="black",
                font=font,
                stroke_width=1,
            )

            # Draw horizontal line under each row
            y_line = (row + 1) * images[0].height
            draw.line([(0, y_line), (collage_width, y_line)], fill="black", width=2)

        # Save the collage as {task}_heatmap_collage.png
        collage.save(
            os.path.join("./results/", f"collage_{task}_heatmap.png")
        )

if __name__ == "__main__":
    main()
