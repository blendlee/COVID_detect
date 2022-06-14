import librosa
import numpy
import skimage.io
import os
from tqdm import tqdm

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def spectrogram_image(y, sr, out, hop_length, n_mels):
    # use log-melspectrogram
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                            n_fft=hop_length*2, hop_length=hop_length)
    mels = numpy.log(mels + 1e-9) # add small number to avoid log(0)

    # min-max scale to fit inside 8-bit range
    img = scale_minmax(mels, 0, 255).astype(numpy.uint8)
    img = numpy.flip(img, axis=0) # put low frequencies at the bottom in image
    img = 255-img # invert. make black==more energy

    # save as PNG
    skimage.io.imsave(out, img)


if __name__ == '__main__':
    # settings
    hop_length = 512 # number of samples per time-step in spectrogram
    n_mels = 128 # number of bins in spectrogram. Height of image
    time_steps = 384 # number of time-steps. Width of image
    wrong = ['data/train/03066.wav','data/train/01701.wav','data/train/02258.wav','data/train/02571.wav']
    # load audio. Using example from librosa
    data_types = ['train','test','unlabled']
    for data_type in data_types:
        print(f'start downloading {data_type} data....')
        path = f'data/{data_type}'
        filenames = os.listdir(path)
        filenames.sort()
        for filename in tqdm(filenames):
            img_dir = path+'/'+filename
            if img_dir in wrong:
                continue
            y, sr = librosa.load(img_dir, offset=1.0, duration=10.0, sr=22050)

            out = f'data/{data_type}_image/{filename[:-4]}.png'

            # extract a fixed length window
            start_sample = 0 # starting at beginning
            length_samples = time_steps*hop_length
            window = y[start_sample:start_sample+length_samples]
            

            # convert to PNG
            spectrogram_image(window, sr=sr, out=out, hop_length=hop_length, n_mels=n_mels)