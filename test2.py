import cv2
import os
from autoliv_util.autolivutil import ImageProcessing, ImageAnalysis, SqlManagement, VideoProcessing

path = r"temp/images_repositioned"
video_folders = [os.path.join(path, x) for x in os.listdir(path)]

images, data = ImageAnalysis.track_cushion_analysis(images_folder=video_folders[0])
out = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (1920, 1080))
print('')
for i, image in enumerate(images):
    print('\rSaving Output Video: {0}/{1}'.format(i + 1, len(images)), end='')
    out.write(image)
out.release()