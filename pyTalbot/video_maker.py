from pdf2image import convert_from_path
import os
import cv2

def pdf_to_images(folder_path):
    image_files = []
    for file in os.listdir(folder_path):
        if file.endswith('.pdf'):
            pages = convert_from_path(os.path.join(folder_path, file))
            for i, page in enumerate(pages):
                image_path = os.path.join(folder_path, f"{file[:-4]}_{i}.png")
                page.save(image_path, 'PNG')
                image_files.append(image_path)
    return sorted(image_files)



def create_video_from_images(image_files, output_path, fps=30):
    img = cv2.imread(image_files[0])
    height, width, layers = img.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for image_file in image_files:
        img = cv2.imread(image_file)
        video.write(img)

    video.release()
