import cv2
import os
from autoliv_util.autolivutil import ImageProcessing, ImageAnalysis, SqlManagement, VideoProcessing, Analysis

# path = r"temp/images_repositioned"
# video_folders = [os.path.join(path, x) for x in os.listdir(path)]

# images, data = ImageAnalysis.track_cushion_analysis(images_folder=video_folders[0])
# out = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (1920, 1080))
# print('')
# for i, image in enumerate(images):
#     print('\rSaving Output Video: {0}/{1}'.format(i + 1, len(images)), end='')
#     out.write(image)
# out.release()
# templates = r"C:\Temp\MLCT\tesla_mlct2_test\6558541 ML\templates\side"
# images_folder = r"C:\Users\zachary.preator\AppData\Local\Temp\temp_analysis\side\images\6558541_2021Jul29_Hot_B7M78901_side"
# images = ImageProcessing.reposition_images_from_templates(templates, images_folder=images_folder, location=(500, 600))
videos_path = r"C:\Users\zachary.preator\AppData\Local\Temp\raw_videos"
save_path = r"C:\Temp\MLCT\track_cushion\Tesla_ambient"
# results_path = r"C:\Temp\MLCT\track_cushion\Tesla_Side_Results_Preprocessed"
videos = [os.path.join(videos_path, x) for x in os.listdir(videos_path)]
# video = r"C:\Temp\MLCT\track_cushion\temp\processed_images\6558541_2021Jul31_AMB_B7M99001_side"
# Analysis.preprocess_from_video_folder(videos_path, r"C:\Temp\MLCT\track_cushion\temp",
#                                       [27], r"C:\Temp\MLCT\track_cushion\templates", 'AOA', subtract=True)
for video in videos:
    video_name = os.path.split(video)[1]
    sample = video_name.split('_')[3]
    if 'amb' in video_name.lower():
        VideoProcessing.get_frames_from_video(video, frames=[27], save_folder=save_path, frame_rate=4000, mark_frame=20,
                                              save_file=sample)
# Analysis.track_cushion_analysis(video, results_path, shape='rectangle', kind='advanced', iterations=8)
