###############################################################################
# This run.py script goes through all the currently running LAT programs
# with more than 30 tests already completed. It gets any tests that haven't been
# analyzed yet. It can be run any time, but keep in mind that the more time
# between runs means more tests to analyze.
#
# Author: Zachary Preator
# Date: 1/31/2022
###############################################################################
import datetime
import os
import pandas as pd
import numpy as np
import tempfile
from scipy.integrate import trapz
from autoliv_util.autolivutil import SqlManagement, FileManagement, Analysis, ImageProcessing, PlottingUtilities, Email


def run():
    # Get settings from config.ini, for global settings
    settings = FileManagement.read_ini('config.ini', ['settings', 'paths'])
    time_delta = int(settings['time_delta'])
    facility = settings['facility']
    save_path = settings['save_path']
    batch_mode = settings['batch_mode']
    save_folders = [os.path.join(save_path, x) for x in os.listdir(save_path)]
    save_folders = [x for x in save_folders if os.path.isdir(x)]
    # Loop through all existing models in the save_path
    for i, save_folder in enumerate(save_folders):
        # Get name from folder name
        name = os.path.split(save_folder)[1]
        print('\nAnalyzing {0} - {1}/{2}'.format(name, i + 1, len(save_folders)))
        print('=============================')
        analyze_model(save_folder, batch_mode, facility, time_delta)


def analyze_model(save_folder, batch_mode, facility, time_delta):
    # Get name from folder name
    name = os.path.split(save_folder)[1]

    # Set paths and create folders if they don't exist
    details_path = os.path.join(save_folder, 'details.ini')
    results_path = os.path.join(save_folder, 'results')
    os.makedirs(results_path, exist_ok=True)
    data_folder = os.path.join(save_folder, 'data')
    os.makedirs(data_folder, exist_ok=True)
    temp_dir = os.path.join(data_folder, 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    # If no config file is found, we exit this
    if not os.path.exists(details_path):
        print('No config file found. Skipping: {0}'.format(name))
        return

    # Get settings from details file (for this model only)
    details = FileManagement.read_ini(details_path, ['details', 'analysis', 'notifications'])
    part_number = details['part_number']
    test_type = details['test_type']
    temperature = details['temperature']
    resize = details['resize']
    fill = details['fill']
    subtract = details['subtract']
    scale = int(details['scale'])
    frames_params = [float(x) for x in details['frames'].split(':')]  # Calculate frames to extract-> beginning:end:step
    frames = np.arange(*frames_params)
    sigma_limit = details['sigma_limit']
    track_cushion = details['track_cushion']
    emails = [x.strip() for x in details['emails'].split(';')]

    # Query UTS database for new tests
    tbl = SqlManagement.query(part_number=part_number,
                              time_delta=time_delta,
                              facility=facility,
                              temperature=temperature,
                              test_type=test_type)
    if tbl.empty:
        print('No tests found')
        return

    # Check for new tests
    new_tests = check_for_new_tests(tbl, results_path)
    if len(new_tests) == 0:
        print('No new tests found')
        return
    print('Found {0} new tests.'.format(len(new_tests)))

    # Get only the new tests
    tbl = tbl[tbl['sample'].isin(new_tests)]
    tbl['Sample'] = tbl['sample']

    # Extract frames, reposition images, etc. and save to temp folder
    all_videos = Analysis.preprocess_from_uts(save_path=save_folder,
                                              frames=frames,
                                              uts_tbl=tbl,
                                              resize=resize,
                                              fill=fill,
                                              subtract=subtract,
                                              scale=scale,
                                              batch_mode=batch_mode)
    if all_videos is None:
        # If in batch mode, the program will run through once, and generate sample images to crop.
        # When this happens, preprocess_from_uts will return None, and it will exit this model
        print('Templates will need to be cropped for: {0}'.format(name))
        return

    # Loop through the front, side and/or rear groups of videos in all_videos
    scores = []
    for videos in all_videos:
        # Get details from first video folder
        name = os.path.split(videos[0])[1]
        orientation = name.split('_')[-1]
        print('Analyzing {0}'.format(orientation))
        scores_temp, data_df = mse_track_analysis(save_folder, results_path, orientation, videos, data_folder, track_cushion)
        scores.extend(scores_temp)
    scores_df = pd.DataFrame(scores)
    scores_df = pd.merge(scores_df, tbl, on='Sample')
    temp_scores_path = os.path.join(data_folder, 'temp', '{0} Scores.csv'.format(datetime.datetime.today().strftime('%Y%b%d')))
    scores_df.to_csv(temp_scores_path, index=False)
    process_results(temp_scores_path, save_folder, sigma_limit, emails)


def mse_track_analysis(save_folder, results_path, orientation, videos, data_folder, track_cushion):
    scores = []
    average_images_folder = os.path.join(save_folder, 'average_images', orientation)
    # Create average images if none exist
    if not os.path.exists(average_images_folder):
        videos_folder = os.path.split(videos[0])[0]
        # TODO this only works if there are no other programs in the processed images folder
        ImageProcessing.get_average_video_images(videos_folder, average_images_folder)
    # Loop through each video for this orientation and perform mse analysis
    data_path = os.path.join(data_folder, 'MSE Data {0}.csv'.format(orientation))
    plot_path_html = os.path.join(data_folder, 'MSE Plot {0}.html'.format(orientation))
    plot_path_png = os.path.join(data_folder, 'MSE Plot {0}.png'.format(orientation))
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        df = pd.DataFrame()
    mse_df = pd.DataFrame()
    for j, video in enumerate(videos):
        try:
            print('\rAnalysis: Video {0}/{1}'.format(j + 1, len(videos)), end='')
            name = os.path.split(video)[1]
            date = name.split('_')[1]
            sample = name.split('_')[3]
            orientation = name.split('_')[-1]
            name = '{0}_{1}'.format(date, sample)
            video_save_folder = os.path.join(results_path, name, orientation)
            mse_df = Analysis.mse_analysis(average_images_folder,
                                           video,
                                           video_save_folder,
                                           save_plot=False,
                                           verbose=0)
            df[name] = mse_df['Score'].values
            integral = trapz(df[name])
            scores.append({'Name': name, 'Sample': sample, 'Date': date, 'Orientation': orientation, 'Score': integral})
            if track_cushion:
                Analysis.track_cushion_analysis(video,
                                                video_save_folder,
                                                'advanced',
                                                shapes=('rectangle', 'contour'),
                                                iterations=2,
                                                verbose=0)
        except:
            pass
    if not mse_df.empty:
        df['Time'] = mse_df.index
    df.to_csv(data_path, index=False)
    df = df.set_index('Time')
    PlottingUtilities.plotly_mse_chart(df, plot_path_html)
    PlottingUtilities.plotly_mse_chart(df, plot_path_png)
    print('')
    return scores, df


def process_results(scores_path, save_folder, sigma_limit, send_list):
    # Compile all the scores for this set of tests, then send alerts if necessary

    data_folder = os.path.join(save_folder, 'data')
    results_folder = os.path.join(save_folder, 'results')
    if not os.path.exists(results_folder):
        raise Exception('The results folder does not exist: {0}'.format(results_folder))

    # Make table with scores
    scores_df = pd.read_csv(scores_path)

    # Check if Scores.csv already exists, read it in if it does
    scores_path = os.path.join(data_folder, 'Scores.csv')
    if os.path.exists(scores_path):
        history_df = pd.read_csv(scores_path)
    else:
        history_df = pd.DataFrame()

    # Concatenate the new test scores to the history table
    history_df = pd.concat([history_df, scores_df])
    history_df.to_csv(scores_path, index=False)
    if not history_df.empty:

        # Calculate control limit, produce SPC charts
        ucls = []
        orientations = history_df['Orientation'].unique()
        for orient in orientations:
            orient_df = history_df[history_df['Orientation'] == orient].copy()
            spc_path = os.path.join(data_folder, 'SPC Chart {0}'.format(orient))
            ucl = create_spcs(orient_df, sigma_limit, orient, spc_path)
            ucls.append(ucl)

        # Loop through the new tests and check if they scored above the upper control limit
        for name in scores_df['Name'].unique():
            try:
                test_df = scores_df[scores_df['Name'] == name]

                # Loop through all ucls (one for each orientation) and send email if either failed
                notify = False
                full_scores = {}
                for i, limit in enumerate(ucls):
                    data_path = os.path.join(data_folder, 'MSE Data {0}.csv'.format(orientations[i]))
                    data_df = pd.read_csv(data_path)
                    full_scores[orientations[i]] = data_df
                    temp_df = test_df[test_df['Orientation'] == orientations[i]]
                    notify = False
                    if temp_df['Score'].iloc[0] > limit:
                        notify = True
                if notify:
                    # Create charts for both front, side (and other orientations if available)
                    for orientation in orientations:
                        mse_standout_html = os.path.join(results_folder, name,
                                                         'MSE Chart {0}'.format(orientation))
                        this_df = full_scores[orientation]
                        PlottingUtilities.plotly_mse_standout(this_df, name,
                                                              title='MSE Chart {0}'.format(orientation.title()),
                                                              save_path=mse_standout_html)
                        orient_df = history_df[history_df['Orientation'] == orientation].copy()
                        spc_path = os.path.join(results_folder, name, 'SPC Chart {0}'.format(orientation))
                        create_spcs(orient_df, sigma_limit, orientation, spc_path, name=name)
                    send_email(send_list, test_df, save_folder, ucls, orientations)
            except Exception as e:
                print(e)
                continue


def create_spcs(df, sigma_limit, orientation, save_path, name=None):
    mean = df['Score'].mean()
    std = df['Score'].std()
    ucl = mean + std * float(sigma_limit)
    lcl = mean - std * float(sigma_limit)
    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%b%d')
    except:
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df = df.sort_values(by=['Date'])
    title = 'SPC Chart {0}<br><sup>Area Under Curve for MSE Scores</sup>'.format(orientation.title())
    PlottingUtilities.plotly_spc(data=df, x='Date', y='Score', names='Sample', ucl=ucl, lcl=lcl,
                                 save_path=save_path, title=title, this_test=name, ylabel='Area Under Curve')
    return ucl


def send_email(send_list, test_df, save_folder, ucls, orientations):
    data_folder = os.path.join(save_folder, 'data')
    results_folder = os.path.join(save_folder, 'results')
    if not os.path.exists(results_folder):
        raise Exception('The results folder does not exist: {0}'.format(results_folder))

    test_part_number = test_df['partnumber'].iloc[0]
    test_name = test_df['Name'].iloc[0]
    test_sample = test_df['Sample'].iloc[0]
    test_date = test_df['Date'].iloc[0]
    test_group = test_df['testingset'].iloc[0]
    links = ''
    attachments = []
    images = []
    for orientation in orientations:
        video_link = test_df['filnm_{0}'.format(orientation)].iloc[0]
        links += '<a href="{0}">{1} Video Link</a><p></p>'.format(video_link, orientation.title())
        # Get all the figures
        spc_path_html = os.path.join(results_folder, test_name, 'SPC Chart {0}.html'.format(orientation))
        spc_path_png = os.path.join(results_folder, test_name, 'SPC Chart {0}.png'.format(orientation))
        mse_standout_html = os.path.join(results_folder, test_name, 'MSE Chart {0}.html'.format(orientation))
        mse_standout_png = os.path.join(results_folder, test_name, 'MSE Chart {0}.png'.format(orientation))
        attachments.extend([spc_path_html, mse_standout_html])
        images.extend([spc_path_png, mse_standout_png])

    message = """
    <h3>Details</h3>
    <p>Part Number: {0}</p>
    <p>Date: {1}</p>
    <p>Sample: {2}</p>
    <h3>Video Links</h3>
    <hr/>
    {3}
    <h3>Charts</h3>
    <hr/>
    """.format(test_part_number, test_date, test_sample, links)
    subject = 'LAT Alert {0}'.format(test_group)
    outfile_name = os.path.join(results_folder, test_name, 'Alert.eml')
    Email.send_email(send_list,
                     subject=subject,
                     attachments=attachments,
                     images=images,
                     body=message,
                     body_type='html',
                     outfile_name=outfile_name)


def check_for_new_tests(tbl, results_path):
    to_test = tbl['sample'].values
    video_folders = os.listdir(results_path)
    tested = [x.split('_')[1] for x in video_folders if len(x.split('_')) > 1]
    to_test = [x for x in to_test if x not in tested]
    return to_test


if __name__ == '__main__':
    # run()
    path = r"C:\Temp\MLCT\NX4-Hyundai\6513457 Hot\data\temp\2022Feb16 Scores.csv"
    data_folder = r"C:\Temp\MLCT\NX4-Hyundai\6513457 Hot"
    process_results(path, data_folder, 1, ['zachary.preator@autoliv.com'])
