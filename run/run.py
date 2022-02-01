###############################################################################
# This run.py script goes through all the currently running LAT programs
# with more than 30 tests already completed. It gets any tests that haven't been
# analyzed yet. It can be run any time, but keep in mind that the more time
# between runs means more tests to analyze.
#
# Author: Zachary Preator
# Date: 1/31/2022
###############################################################################
import os
import pandas as pd
import numpy as np
from autoliv_util.autolivutil import SqlManagement, FileManagement, Analysis, ImageProcessing, PlottingUtilities


def run(settings):
    time_delta = int(settings['time_delta'])
    facility = settings['facility']
    save_path = settings['save_path']
    save_folders = [os.path.join(save_path, x) for x in os.listdir(save_path)]
    save_folders = [x for x in save_folders if os.path.isdir(x)]

    # Loop through all existing models in the save_path
    for i, save_folder in enumerate(save_folders):
        print('\nAnalyzing {0} - {1}/{2}'.format(save_folder, i+1, len(save_folders)))
        details_path = os.path.join(save_folder, 'details.ini')
        processed_videos_path = os.path.join(save_folder, 'processed_videos')
        os.makedirs(processed_videos_path, exist_ok=True)
        if not os.path.exists(details_path):
            print('No config file found. Skipping: {0}'.format(os.path.split(save_folder)[1]))
            continue

        details = FileManagement.read_ini(details_path, ['details'])
        part_number = details['part_number']
        test_type = details['test_type']
        temperature = details['temperature']
        tbl = SqlManagement.query(part_number=part_number,
                                  time_delta=time_delta,
                                  facility=facility,
                                  temperature=temperature,
                                  test_type=test_type)
        if tbl.empty:
            print('No tests found')
            continue
        to_test = tbl['sample'].values
        video_folders = os.listdir(processed_videos_path)
        tested = [x.split('_')[1] for x in video_folders if len(x.split('_')) > 1]
        to_test = [x for x in to_test if x not in tested]
        if len(to_test) == 0:
            print('No new tests found')
            continue
        print('Found {0} new tests.'.format(len(to_test)))
        tbl = tbl[tbl['sample'].isin(to_test)]
        resize = details['resize']
        fill = details['fill']
        subtract = details['subtract']
        final_shape = details['final_shape']
        frames_params = [float(x) for x in details['frames'].split(':')]
        frames = np.arange(*frames_params)
        final_shape = tuple([int(x) for x in final_shape.split(':')])
        all_videos = Analysis.preprocess_from_uts(save_path=save_folder, frames=frames,
                                                  uts_tbl=tbl,
                                                  resize=resize,
                                                  fill=fill,
                                                  subtract=subtract,
                                                  final_shape=final_shape)
        # Loop through all the temp video paths from preprocessing
        for videos in all_videos:
            # Get details from first video folder
            name = os.path.split(videos[0])[1]
            orientation = name.split('_')[-1]
            average_images_folder = os.path.join(save_folder, 'average_images', orientation)
            # Create average images if none exist
            if not os.path.exists(average_images_folder):
                videos_folder = os.path.split(videos[0])[0]
                # TODO this only works if there are no other programs in the processed images folder
                ImageProcessing.get_average_video_images(videos_folder, average_images_folder)
            # Loop through each video for this orientation and perform mse analysis
            this_df = pd.DataFrame()
            for i, video in enumerate(videos):
                print('\rMSE Video Analysis: Video {0}/{1}'.format(i + 1, len(videos)), end='')
                name = os.path.split(video)[1]
                date = name.split('_')[1]
                sample = name.split('_')[3]
                orientation = name.split('_')[-1]
                name = '{0}_{1}'.format(date, sample)
                video_save_folder = os.path.join(processed_videos_path, name, orientation)
                df = Analysis.mse_analysis(average_images_folder, video, video_save_folder, verbose=0)
            print('')


if __name__ == '__main__':
    settings = FileManagement.read_ini('config.ini', ['settings', 'paths'])
    run(settings)
