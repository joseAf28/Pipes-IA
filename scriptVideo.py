import cv2
import os

import cv2
import os
import re

def extract_number(filename):
    """Extract number from the filename."""
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')

def images_to_video(image_folder, output_video, fps=30):
    images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
    
    # Sort images based on the number in the filename
    images.sort(key=extract_number)
    
    if not images:
        print("No images found in the directory.")
        return
    
    # Read the first image to get the dimensions
    first_image_path = os.path.join(image_folder, images[0])
    first_image = cv2.imread(first_image_path)
    height, width, layers = first_image.shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)
    
    video.release()
    print(f"Video saved as {output_video}")

# Example usage
pathFolder = os.path.dirname(os.path.abspath(__file__))
imageFolder = os.path.join(pathFolder, "plotTableImages")
outputVideo = "output_video.mp4"
images_to_video(imageFolder, outputVideo, fps=1)
