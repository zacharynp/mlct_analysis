from autoliv_util.autolivutil import ImageProcessing, ImageAnalysis, SqlManagement, VideoProcessing
from autoliv_tk.autolivtk import TKGeneral, ImageProcessingTK
import pytesseract
import numpy as np
import os
pytesseract.pytesseract.tesseract_cmd = r'autoliv_util\tesseract_x64-windows\tools\tesseract\tesseract.exe'

part_number = '6558541'
num_days = 30
temperature = 'Hot'
facility = 'AOA'
frames = np.arange(0, 40.25, 0.25)
reposition = True
resize = False
fill = None
subtract = True

os.makedirs(part_number, exist_ok=True)
program_path = part_number
orientation_path = os.path.join(program_path, 'front')
training_set_path = os.path.join(orientation_path, 'training')
full_set_path = os.path.join(orientation_path, 'full')
program_templates_path = os.path.join(orientation_path, 'templates')
temp_video_path = os.path.join('temp', 'videos')
temp_images = os.path.join('temp', 'images')
temp_full_path_z = os.path.join('temp', 'images_full', str(0.0))
os.makedirs(temp_video_path, exist_ok=True)

# uts_tbl = SqlManagement.query(part_number=part_number,
#                               time_delta=num_days,
#                               temperature=temperature,
#                               facility=facility)
#
# VideoProcessing.copy_videos_from_uts(uts_tbl, 'temp/videos', ['front'])
# Get training set and full set frames (add 0 ms to have a baseline for template matching)
# frames.insert(0, 0.0)
# print('Extracting Training Set Frames')
# VideoProcessing.get_frames_from_videos('temp/videos', facility=facility, num_samples=5)
print('Extracting Frames at 0ms')
# video_folders = VideoProcessing.get_frames_from_videos('temp/videos', save_folder=temp_images, frames=frames, facility=facility)
# frames.remove(0.0)

# Get templates from 0 ms
video_folders = [os.path.join(temp_images, x) for x in os.listdir(temp_images)]
zero_ms_paths = [os.path.join(x, '0.0ms.jpg') for x in video_folders]
ImageProcessingTK.crop_templates(image_paths=zero_ms_paths, save_folder=program_templates_path, num_templates=1)

# Process 0 ms frames to get the parameters
print('Processing 0 ms images')
zero_ms_images = ImageProcessing.get_images(image_paths=zero_ms_paths)
zero_ms_images, params, translations = ImageProcessing.process_images(program_templates_path,
                                                                      images=zero_ms_images,
                                                                      reposition=reposition,
                                                                      resize=resize,
                                                                      fill=fill)

# # Get processed 0 ms images
# zero_images_training = ImageProcessing.get_images(images_folder=temp_training_path_z)
# zero_images_full = ImageProcessing.get_images(images_folder=temp_full_path_z)

# Loop through additional frames to perform analyses
for i, video_folder in enumerate(video_folders):
    print('\rProcessing Videos: {0}/{1}'.format(i + 1, len(video_folders)), end='')
    this_video_images = ImageProcessing.get_images(images_folder=video_folder)
    # this_video_times = this_video_df['Time'].values
    this_zero_image = zero_ms_images[i]

    # # Set paths for just this video's images
    # training_frames_path = os.path.join(training_set_path, str(float(frame)))
    # full_frames_path = os.path.join(full_set_path, str(float(frame)))
    # temp_training_path = os.path.join('temp', 'images_training', str(float(frame)))
    temp_images_path = os.path.join('temp', 'images_repositioned', os.path.split(video_folder)[1])
    save_paths = [os.path.join(temp_images_path, '{0}ms.jpg'.format(x)) for x in frames]

    # Process training images: reposition, resize, fill, and subtract
    ImageProcessing.process_images(program_templates_path,
                                   images=this_video_images,
                                   save_paths=save_paths,
                                   reposition=reposition,
                                   translations=translations[i],
                                   resize=resize,
                                   fill=fill,
                                   subtract=this_zero_image,
                                   params=params)

    # Perform analysis on images in the temp_training_path and temp_full_path
    # df = ImageAnalysis.average_image_analysis(temp_training_path, temp_full_path)
    # df.to_csv('Scores_{0}ms.csv'.format(str(float(frame))), index=False)