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
import shutil

import pandas as pd
import numpy as np
import tempfile
from scipy.integrate import trapz
from autoliv_util.autolivutil import SqlManagement, FileManagement, Analysis, ImageProcessing, PlottingUtilities, Email


def run():
    # Get settings from config.ini, for global settings
    settings = FileManagement.read_ini('config.ini')
    time_delta = int(settings['settings']['time_delta'])
    facility = settings['settings']['facility']
    batch_mode = settings['settings']['batch_mode']
    save_path = settings['paths']['save_path']
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
    details = FileManagement.read_ini(details_path)

    # Details
    part_number = details['details']['part_number']
    temperature = details['details']['temperature']
    test_type = details['details']['test_type']
    bay_code = None
    if 'bay_code' in details['details'].keys():
        bay_code = details['details']['bay_code']

    # Analysis
    frames_params = [float(x) for x in details['analysis']['frames'].split(':')]  # Calculate frames to extract-> beginning:end:step
    frames = np.arange(*frames_params)
    resize = details['analysis']['resize']
    fill = details['analysis']['fill']
    subtract = details['analysis']['subtract']
    scale = int(details['analysis']['scale'])
    track_cushion = details['analysis']['track_cushion']

    # Notifications
    sigma_limit = details['notifications']['sigma_limit']
    emails = [x.strip() for x in details['notifications']['emails'].split(';')]
    if 'key_frames' in details.keys():
        # Calculate key frames to show in alerts -> beginning:end:step
        key_frames_params = [float(x) for x in details['notifications']['key_frames'].split(':')]
        key_frames = frames
    else:
        mid_point = int(len(frames) / 2)
        key_frames = [frames[mid_point]]

    # Query UTS database for new tests
    tbl = SqlManagement.query(part_number=part_number,
                              time_delta=time_delta,
                              facility=facility,
                              temperature=temperature,
                              test_type=test_type)
    if tbl.empty:
        print('No tests found')
        return

    # Filter by bay code if necessary
    if bay_code is not None:
        tbl = tbl[tbl['bay_code'] == bay_code]

    # Check for new tests
    new_tests = check_for_new_tests(tbl, results_path)
    if len(new_tests) == 0:
        print('No new tests found')
        return
    print('Found {0} new tests.'.format(len(new_tests)))

    # Get only the new tests
    tbl = tbl[tbl['sample'].isin(new_tests)]
    tbl['Sample'] = tbl['sample']

    # Get settings for repositioning and resizing if they exist
    reposition_params = []
    resize_params = []
    for section in details.keys():
        if 'parameters' in section:
            if 'reposition_x' in details[section].keys():
                reposition_params.append(details[section])
            if 'baseline_x' in details[section].keys():
                resize_params.append(details[section])
    if len(reposition_params) == 0: reposition_params = None
    if len(resize_params) == 0: resize_params = None

    # Extract frames, reposition images, etc. and save to temp folder
    all_videos, all_params = Analysis.preprocess_from_uts(save_path=save_folder,
                                                          frames=frames,
                                                          uts_tbl=tbl,
                                                          resize=resize,
                                                          fill=fill,
                                                          subtract=subtract,
                                                          scale=scale,
                                                          save_match_templates=True,
                                                          reposition_params=reposition_params,
                                                          resize_params=resize_params,
                                                          batch_mode=batch_mode)
    if all_videos is None:
        # If in batch mode, the program will run through once, and generate sample images to crop.
        # When this happens, preprocess_from_uts will return None, and it will exit this model
        print('Templates will need to be cropped for: {0}'.format(name))
        return

    details_section_titles = ['details', 'analysis', 'notifications']
    details_sections = [details[x] for x in details_section_titles]
    for param in all_params:
        title = 'parameters_{0}'.format(param['orientation'])
        details_sections.append(param)
        details_section_titles.append(title)

    # Loop through the front, side and/or rear groups of videos in all_videos
    scores = []
    data_paths = []
    for videos in all_videos:
        # Get details from first video folder
        name = os.path.split(videos[0])[1]
        orientation = name.split('_')[-1]
        print('Analyzing {0}'.format(orientation))
        scores_temp, data_path = mse_track_analysis(save_folder, results_path, orientation, videos, data_folder,
                                                    key_frames, track_cushion)
        data_paths.append(data_path)
        scores.extend(scores_temp)
    scores_df = pd.DataFrame(scores)
    scores_df = pd.merge(scores_df, tbl, on='Sample')
    temp_scores_path = os.path.join(data_folder, 'temp', '{0} Scores.csv'.format(datetime.datetime.today().strftime('%Y%b%d')))
    scores_df.to_csv(temp_scores_path, index=False)
    FileManagement.write_ini(details_path, details_section_titles, details_sections)
    process_results(temp_scores_path, save_folder, sigma_limit, emails)


def mse_track_analysis(save_folder, results_path, orientation, videos, data_folder, key_frames, track_cushion):
    scores = []
    average_images_folder = os.path.join(save_folder, 'average_images', orientation)
    # Create average images if none exist
    if not os.path.exists(average_images_folder):
        videos_folder = os.path.split(videos[0])[0]
        # TODO this only works if there are no other programs in the processed images folder
        ImageProcessing.get_average_video_images(videos_folder, average_images_folder)
    # Loop through each video for this orientation and perform mse analysis
    data_path = os.path.join(data_folder, 'MSE Data {0}.csv'.format(orientation))
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        df = pd.DataFrame()
    mse_df = pd.DataFrame()
    for j, video in enumerate(videos):
        try:
            print('\rAnalysis: Video {0}/{1}'.format(j + 1, len(videos)), end='')
            full_name = os.path.split(video)[1]
            date = full_name.split('_')[1]
            sample = full_name.split('_')[3]
            orientation = full_name.split('_')[-1]
            name = '{0}_{1}'.format(date, sample)
            folder, test_name = os.path.split(video)
            template_match_path = os.path.join(os.path.split(folder)[0], 'template_matches', test_name, 'Template Match.jpg')
            video_save_folder = os.path.join(results_path, name, orientation)
            template_match_save = os.path.join(video_save_folder, 'Template Match.jpg')
            mse_df = Analysis.mse_analysis(average_images_folder,
                                           video,
                                           video_save_folder,
                                           save_plot=False,
                                           key_frames=key_frames,
                                           verbose=0)
            shutil.copy2(template_match_path, template_match_save)
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
    print('')
    return scores, data_path


def create_html_test_summary(figs, charts_path, title):
    pass


def create_html_model_summary(figs, charts_path, title):
    with open(charts_path, 'w') as f:
        # Title
        f.write('<h1>{0}</h1><br>'.format(title))

        # CSS to create grid
        f.write("""
                <style>
                .grid-container {
                  display: grid;
                  grid-template-columns: auto auto;
                }
                .grid-item {
                  text-align: center;
                }
                </style>
                <body>
                <div class="grid-container">""")

        # Writing each figure to html
        for fig in figs:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write("""</div>""")


def process_results(scores_path, save_folder, sigma_limit, send_list):
    # Compile all the scores for this set of tests, then send alerts if necessary

    data_folder = os.path.join(save_folder, 'data')
    alerts_folder = os.path.join(data_folder, 'alerts')
    os.makedirs(alerts_folder, exist_ok=True)
    results_folder = os.path.join(save_folder, 'results')
    if not os.path.exists(results_folder):
        raise Exception('The results folder does not exist: {0}'.format(results_folder))

    # Make table with scores
    scores_df = pd.read_csv(scores_path)

    # Check if Scores.csv already exists, read it in if it does
    scores_path = os.path.join(data_folder, 'Scores.csv')
    if os.path.exists(scores_path):
        history_df = pd.read_csv(scores_path)
        # Drop rows if they already exist to prevent duplicating entries
        history_df = history_df[history_df['Name'].isin(scores_df['Name'].unique()) == False]
    else:
        history_df = pd.DataFrame()

    # Concatenate the new test scores to the history table and drop duplicates
    history_df = pd.concat([history_df, scores_df])
    history_df = history_df.drop_duplicates(subset=['Name', 'Orientation'])
    FileManagement.check_file_open(scores_path)
    history_df.to_csv(scores_path, index=False)
    if not history_df.empty:

        testingset = history_df['testingset'].iloc[0]

        # Calculate control limit, produce global charts
        ucls = []
        figs = []
        orientations = history_df['Orientation'].unique()
        for orient in orientations:
            orient_df = history_df[history_df['Orientation'] == orient].copy()
            ucl, spc_fig = create_spcs(orient_df, sigma_limit, orient)
            ucls.append(ucl)
            mse_data_path = os.path.join(data_folder, 'MSE Data {0}.csv'.format(orient))
            mse_df = pd.read_csv(mse_data_path)
            mse_df = mse_df.set_index('Time')
            mse_fig = PlottingUtilities.plotly_mse_chart(mse_df, title='MSE Chart {0}'.format(orient.title()))
            figs.extend([spc_fig, mse_fig])

        # Write all plolty charts to a single html file
        charts_path = os.path.join(data_folder, 'Charts.html')
        create_html_model_summary(figs, charts_path, testingset)


        # Loop through the new tests and check if they scored above the upper control limit
        for name in scores_df['Name'].unique():
            try:
                missing = []
                test_df = scores_df[scores_df['Name'] == name]

                # Loop through all ucls (one for each orientation) and send email if either failed
                notify = False
                full_scores = {}
                for i, limit in enumerate(ucls):
                    data_path = os.path.join(data_folder, 'MSE Data {0}.csv'.format(orientations[i]))
                    data_df = pd.read_csv(data_path)
                    full_scores[orientations[i]] = data_df
                    temp_df = test_df[test_df['Orientation'] == orientations[i]]
                    if temp_df.empty:
                        missing.append(orientations[i])
                        continue
                    notify = False
                    if temp_df['Score'].iloc[0] > limit:
                        notify = True
                if notify:
                    # Create charts for both front, side (and other orientations if available)
                    temp_orientations = list(orientations)
                    if len(missing) > 0:
                        for miss in missing:
                            temp_orientations.remove(miss)
                    for orientation in temp_orientations:
                        mse_standout_html = os.path.join(results_folder, name,
                                                         'MSE Chart {0}'.format(orientation))
                        this_df = full_scores[orientation]
                        PlottingUtilities.plotly_mse_standout(this_df, name,
                                                              title='MSE Chart {0}'.format(orientation.title()),
                                                              save_path=mse_standout_html)
                        orient_df = history_df[history_df['Orientation'] == orientation].copy()
                        spc_path = os.path.join(results_folder, name, 'SPC Chart {0}'.format(orientation))
                        create_spcs(orient_df, sigma_limit, orientation, spc_path, name=name)
                    send_email(send_list, test_df, save_folder, ucls, temp_orientations)
            except Exception as e:
                print(e)
                continue


def create_spcs(df, sigma_limit, orientation, spc_path=None, name=None):
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
    fig = PlottingUtilities.plotly_spc(data=df, x='Date', y='Score', names='Sample', ucl=ucl, lcl=lcl,
                                       save_path=spc_path, title=title, this_test=name, ylabel='Area Under Curve')
    return ucl, fig


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
    charts = []
    images = []
    for orientation in orientations:
        test_path = os.path.join(results_folder, test_name)
        video_link = test_df['filnm_{0}'.format(orientation)].iloc[0]
        links += '<a href="{0}">{1} Video Link</a><p></p>'.format(video_link, orientation.title())
        # Get all the figures
        spc_path_html = os.path.join(test_path, 'SPC Chart {0}.html'.format(orientation))
        spc_path_png = os.path.join(test_path, 'SPC Chart {0}.png'.format(orientation))
        mse_standout_html = os.path.join(test_path, 'MSE Chart {0}.html'.format(orientation))
        mse_standout_png = os.path.join(test_path, 'MSE Chart {0}.png'.format(orientation))
        key_frames_path = os.path.join(test_path, orientation, 'key_frames')
        if os.path.exists(key_frames_path):
            if len(os.listdir(key_frames_path)) > 0:
                key_frames_images = ImageProcessing.get_image_paths(key_frames_path)
                images.extend(key_frames_images)
        attachments.extend([spc_path_html, mse_standout_html])
        charts.extend([spc_path_png, mse_standout_png])

    message = """
    <h3>Details</h3>
    <p>Part Number: {0}</p>
    <p>Date: {1}</p>
    <p>Sample: {2}</p>
    <h3>Video Links</h3>
    <hr/>
    {3}
    """.format(test_part_number, test_date, test_sample, links)
    subject = 'LAT Alert {0}'.format(test_group)
    outfile_name = os.path.join(results_folder, test_name, 'Alert.eml')
    Email.send_email(send_list,
                     subject=subject,
                     attachments=attachments,
                     charts=charts,
                     images=images,
                     body=message,
                     body_type='html',
                     outfile_name=outfile_name)

    # Added so that the alerts are all in the same folder (model/data/alerts)
    alerts_folder = os.path.join(data_folder, 'alerts')
    new_path = os.path.join(alerts_folder, '{0}.eml'.format(test_name))
    shutil.copy2(outfile_name, new_path)


def check_for_new_tests(tbl, results_path):
    to_test = tbl['sample'].values
    video_folders = os.listdir(results_path)
    tested = [x.split('_')[1] for x in video_folders if len(x.split('_')) > 1]
    to_test = [x for x in to_test if x not in tested]
    return to_test


if __name__ == '__main__':
    run()
    # path = r"C:\Temp\MLCT\NX4-Hyundai\6513457 Hot\data\temp\2022Feb17 Scores.csv"
    # data_folder = r"C:\Temp\MLCT\NX4-Hyundai\6513457 Hot"
    # process_results(path, data_folder, 2, ['zachary.preator@autoliv.com'])
