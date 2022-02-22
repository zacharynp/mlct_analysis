from autoliv_util.autolivutil import ImageProcessing, ImageAnalysis, Analysis, VideoProcessing
from autoliv_tk.autolivtk import ImageProcessingTK
import cv2
import pandas as pd
import numpy as np
import os
import pytesseract
import tempfile
print(tempfile.gettempdir())
pytesseract.pytesseract.tesseract_cmd = r'autoliv_util\tesseract_x64-windows\tools\tesseract\tesseract.exe'

save_path = r"C:\Temp\MLCT\tesla_mlct2_test\6558541 ML AMB"
part_number = '6558541'
num_days = 300
temperature = 'AMB'
facility = 'AOA'
frames = [16, 17, 18, 19, 20, 21, 22, 23]
reposition = True
resize = False
fill = None
subtract = True
orientations = ['side']
scale = 1
location = (500, 600)

# all_videos = Analysis.preprocess_from_uts(save_path, part_number=part_number,
#                                           time_delta=num_days, temperature=temperature,
#                                           orientations=orientations, frames=frames,
#                                           reposition=reposition, subtract=True, locations=[location])
videos_path = r"C:\Users\zachary.preator\AppData\Local\Temp\temp_analysis\videos\side"
temp_dir = r"C:\Users\zachary.preator\AppData\Local\Temp\temp_analysis\side"
templates_path = r"C:\Temp\MLCT\tesla_mlct2_test\6558541 ML AMB\templates\side"

_, params = Analysis.preprocess_from_video_folder(videos_path, temp_dir, frames,
                                                  templates_path, 'AOA', reposition_params=location, subtract=True, scale=1)
print(params)
# folder = r"C:\Users\zachary.preator\AppData\Local\Temp\temp_analysis\videos\front"
# temp_dir = r"C:\Users\zachary.preator\AppData\Local\Temp\temp_analysis\front"
# templates = r"C:\Temp\MLCT\tesla_mlct2_test\front\templates"
# videos = Analysis.preprocess_from_video_folder(videos_folder=folder,
#                                                temp_dir=temp_dir,
#                                                frames=frames,
#                                                templates_folder=templates,
#                                                facility=facility,
#                                                reposition=reposition,
#                                                resize=resize,
#                                                fill=fill,
#                                                subtract=subtract)
# for i, orientation in enumerate(orientations):
#     videos = all_videos[i]
#     for video in videos:
#         name = os.path.split(video)[1]
#         new_images, data = ImageAnalysis.track_cushion_analysis(video)
#         processed_videos = os.path.join(save_path, orientation, 'processed_videos')
#         os.makedirs(processed_videos, exist_ok=True)
#         path = os.path.join(processed_videos, '{0}.avi'.format(name))
#         VideoProcessing.create_video(path, images=new_images)
#         this_df = pd.DataFrame(data, columns=['x', 'y', 'w', 'h', 'area'])
#         this_df['Time'] = frames
#         processed_data = os.path.join(save_path, orientation, 'processed_data')
#         os.makedirs(processed_data, exist_ok=True)
#         this_df.to_csv(os.path.join(processed_data, '{0}.csv'.format(name)), index=False)
# processed_path = r"C:\Users\zachary.preator\AppData\Local\Temp\temp_analysis\front\processed_images"
# average_images = r"C:\Temp\MLCT\tesla_mlct2_test\front\average_images"
# save_folder = r"C:\Temp\MLCT\tesla_mlct2_test\front\processed_videos"
# Analysis.batch_track_cushion_analysis(processed_path, save_folder)
# Analysis.batch_mse_analysis(average_images, processed_path, save_folder)

# processed_path = r"C:\Users\zachary.preator\AppData\Local\Temp\temp_analysis\front\processed_images"
# save_folder = r"C:\Temp\MLCT\tesla_mlct2_test\6397522_Hot\processed_videos"
# # Analysis.batch_track_cushion_analysis(processed_path, save_folder)
# videos_path = r"C:\Users\zachary.preator\AppData\Local\Temp\temp_analysis\videos\front"
# temp_dir = r"C:\Users\zachary.preator\AppData\Local\Temp\temp_analysis"
# templates_folder = r"C:\Temp\MLCT\tesla_mlct2_test\6397522_Hot\templates\front"
# frames = np.arange(0, 61, 1)
# videos = Analysis.preprocess_from_video_folder(videos_path, temp_dir, frames, templates_folder, 'AOA')