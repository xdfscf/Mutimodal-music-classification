# Check full librosa spectrogram tutorial in the following IPython notebook:
# http://nbviewer.jupyter.org/github/bmcfee/librosa/blob/master/examples/LibROSA%20demo.ipynb

import librosa

import numpy as np
import os
import time

from PIL import Image

music_dir = './data/fma_small/'      # directory where you extracted FMA dataset with .mp3s
spectr_dir = './in/mel-specs/'
logs_file = './out/logs/mel-spec.log'
spectro_save_dir='./in/mel-specs/'
regenerate = False

'''
def __unify_img_sizes(min_width, expected_width):
    deleted_cnt = 0
    failed_dlt_cnt = 0
    for subdir, _, files in os.walk(spectr_dir):
        for file in files:
            fpath = os.path.join(subdir, file)
            img = Image.open(fpath)
            width = img.size[0]
            height = img.size[1]
            # no use of problematic spectrograms much shorter than min_width
            if width < min_width:
                try:
                    print('DELETE | {} | {}x{} (width < min_width ({}))'
                          .format(fpath, height, width, min_width))
                    os.remove(fpath)
                    deleted_cnt += 1
                except:
                    print('Error occured while deleting mel-spec')
                    failed_dlt_cnt += 1
                continue
            elif width > expected_width:
                print('CROP | {} | {}x{} -> {}x{} | width > expected_width ({})'
                      .format(fpath, height, width, height, expected_width, expected_width))
                # crop to (height, expected_width) and remove third dimension (channel) to draw grayscale
                img_as_np = np.asarray(img.getdata()).reshape(height, width, -1)[:, :expected_width, :]\
                    .reshape(height, -1)
            elif width < expected_width:
                print('APPEND | {} | {}x{} | min_width ({}) < width < expected_width ({})'
                      .format(fpath, height, width, min_width, expected_width))
                img_as_np = np.asarray(img.getdata()).reshape(height, width, -1)
                # fill in image with black pixels up to (height, expected_width)
                img_as_np = np.hstack((img_as_np.reshape(height, -1), np.zeros((height, expected_width - width))))
            else:
                continue

            # Replace spectrograms
            os.remove(fpath)
            scipy.misc.toimage(img_as_np).save(fpath)

    return deleted_cnt, failed_dlt_cnt
'''




def __extract_melspec(audio_fpath, audio_fname):
    """
    Using librosa to calculate log mel spectrogram values
    and scipy.misc to draw and store them (in grayscale).
    :param audio_fpath:
    :param audio_fname:
    :return:
    """
    # Load sound file
    y, sr = librosa.load(audio_fpath)

    # Let's make and display a mel-scaled power (energy-squared) spectrogram

    S = librosa.feature.melspectrogram(y=y, sr=sr,n_fft=2048, hop_length=1024)

    # Convert to log scale (dB). We'll use the peak power as reference.
    log_S = librosa.amplitude_to_db(S, ref=np.max)

    spectr_fname = audio_fname + '.png'
    spectr_subdir = spectro_save_dir+ spectr_fname[:3] + '/'

    if not os.path.exists(spectr_subdir):
        os.makedirs(spectr_subdir)
    subdir_path = spectr_subdir
    min_ = np.min(log_S)
    max_ = np.max(log_S)
    GI = (255 * (log_S - min_) / (max_ - min_)).astype(np.uint8)
    # Draw log values matrix in grayscale
    width = GI.shape[1]
    height = GI.shape[0]
    GI = np.asarray(GI).reshape(height, width, -1)[:, :646, :] \
        .reshape(height, -1)

    Image.fromarray(GI, 'L').save(subdir_path + spectr_fname)



if __name__ == '__main__':
    regenerate = True

    start_time = time.time()


    nfiles = sum([len(files) for r, d, files in os.walk(music_dir)])
    ok_cnt = 0
    fail_cnt = 0

    for subdir, _, files in os.walk(music_dir):
        for file in files:
            if file.lower().endswith('.mp3'):
                fpath = os.path.join(subdir, file)
                fname, _ = os.path.splitext(file)         # (filename, extension)
                # check if spectrogram is already generated
                if not regenerate and os.path.isfile(spectro_save_dir+ fname[:3] + '/' + fname + '.png'):
                    continue
                op_start_time = time.time()
                __extract_melspec(fpath, fname)


            else:
                continue


    print('Generating spectrogram finished! Generated {}/{} images successfully'.format(ok_cnt, ok_cnt + fail_cnt))

    # aligning spectrograms to the same dimensions to feed convolutional input properly
    # deleted, failed_dlt = __unify_img_sizes(1366, 1366)
    # print('Finished alinging image sizes! Deleted problematic spectrograms: {}/{}'.format(deleted, deleted+failed_dlt))
