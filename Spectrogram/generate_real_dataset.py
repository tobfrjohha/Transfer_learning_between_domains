# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:00:10 2020

@author: combitech
"""

import read_audio

frame_size = 55000
read_audio.create_spectrograms_from_folder('C:\\Users\\combitech\\Desktop\\Labbdata_6\\Noise',
                                          'C:\\Users\\combitech\\Desktop\\Labbdata_spectrogram_2500ms\\Noise',
                                          name='aa',
                                          frame_size=frame_size,
                                          n_mels=128,
                                          hop_length=256,
                                          win_length=256,
                                          n_fft=1024)


read_audio.create_spectrograms_from_folder('C:\\Users\\combitech\\Desktop\\Labbdata_6\\Racer',
                                          'C:\\Users\\combitech\\Desktop\\Labbdata_spectrogram_2500ms\\Racer',
                                          name='aa',
                                          frame_size=frame_size,
                                          n_mels=128,
                                          hop_length=256,
                                          win_length=256,
                                          n_fft=1024)


read_audio.create_spectrograms_from_folder('C:\\Users\\combitech\\Desktop\\Labbdata_6\\Spy_Cam',
                                          'C:\\Users\\combitech\\Desktop\\Labbdata_spectrogram_2500ms\\Spy_Cam',
                                          name='aa',
                                          frame_size=frame_size,
                                          n_mels=128,
                                          hop_length=256,
                                          win_length=256,
                                          n_fft=1024)


read_audio.create_spectrograms_from_folder('C:\\Users\\combitech\\Desktop\\Labbdata_6\\Sub',
                                          'C:\\Users\\combitech\\Desktop\\Labbdata_spectrogram_2500ms\\Sub',
                                          name='aa',
                                          frame_size=frame_size,
                                          n_mels=128,
                                          hop_length=256,
                                          win_length=256,
                                          n_fft=1024)


read_audio.create_spectrograms_from_folder('C:\\Users\\combitech\\Desktop\\Labbdata_6\\Racer',
                                          'C:\\Users\\combitech\\Desktop\\Labbdata_spectrogram_2500ms\\Racer',
                                          name='aa',
                                          frame_size=frame_size,
                                          n_mels=128,
                                          hop_length=256,
                                          win_length=256,
                                          n_fft=1024)


read_audio.create_spectrograms_from_folder('C:\\Users\\combitech\\Desktop\\Labbdata_6\\Tugboat',
                                          'C:\\Users\\combitech\\Desktop\\Labbdata_spectrogram_2500ms\\Tugboat',
                                          name='aa',
                                          frame_size=frame_size,
                                          n_mels=128,
                                          hop_length=256,
                                          win_length=256,
                                          n_fft=1024)


