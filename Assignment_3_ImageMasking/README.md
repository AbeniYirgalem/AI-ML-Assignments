# Assignment 3: Image Masking

This assignment demonstrates basic image masking techniques using OpenCV and NumPy in Python.

## Files

- `image_masking.py`: Main Python script that loads an image, creates rectangle and circle masks, and applies them to the image.
- `beach.png`: Sample image used for masking operations.

## Features

- Loads an image (`beach.png`) and displays it.
- Creates a black mask and draws a white rectangle in the center, then applies it to the image.
- Creates a circular mask and applies it to the image.
- Displays the original image, masks, and masked results using OpenCV windows.

## Requirements

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy

Install dependencies with:

```powershell
pip install opencv-python numpy
```

## How to Run

Run the script from the `Assignment_3_ImageMasking` folder:

```powershell
python image_masking.py
```

## Output

The script will open several windows showing:

- The original image
- Rectangle mask
- Rectangle masked image
- Circle mask
- Circle masked image

Press any key in an image window to close all windows.

## Date

- August 28, 2025
