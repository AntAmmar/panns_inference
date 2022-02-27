import os
import matplotlib.pyplot as plt
import numpy as np
import librosa
import panns_inference
from panns_inference import AudioTagging, SoundEventDetection, labels
from adverts.models import NeuralNetworkResult

def print_audio_tagging_result(clipwise_output, i):
    """Visualization of audio tagging result.

    Args:
      clipwise_output: (classes_num,)
    """
    sorted_indexes = np.argsort(clipwise_output)[::-1]

    # Print audio tagging top probabilities
    for k in range(10):
        print('{}: {:.3f}'.format(np.array(labels)[sorted_indexes[k]], 
            clipwise_output[sorted_indexes[k]]), i)


def plot_sound_event_detection_result(framewise_output):
    """Visualization of sound event detection result. 

    Args:
      framewise_output: (time_steps, classes_num)
    """
    classwise_output = np.max(framewise_output, axis=0) # (classes_num,)

    idxes = np.argsort(classwise_output)[::-1]
    idxes = idxes[0:5]

    ix_to_lb = {i : label for i, label in enumerate(labels)}
    
    for idx in idxes:
        if 'music' in ix_to_lb[idx].lower():
          started = False
          ranges = []
          new_range = []
          for i in range(len(framewise_output[:, idx])):
            if started and framewise_output[:, idx][i] < 0.3:
              new_range.append(framewise_output[:, idx][i])
              started = False
              if new_range[1] - new_range[0] >= 500:
                ranges.append(new_range)
              new_range = []
            if framewise_output[:, idx][i] >= 0.3 and not started:
              new_range.append(framewise_output[:, idx][i])
              started = True
          if len(ranges) > 0:
            NeuralNetworkResult.objects.create(result=ranges)
            return ranges
          else:
            NeuralNetworkResult.objects.create(result={ 'no_music': True })
            return None


if __name__ == '__main__':
    """Example of using panns_inferece for audio tagging and sound evetn detection.
    """
    device = 'cpu' # 'cuda' | 'cpu'
    audio_path = 'resources/R9_ZSCveAHg_7s.wav'
    (audio, _) = librosa.core.load(audio_path, sr=32000, mono=True)
    audio = audio[None, :]  # (batch_size, segment_samples)

    print('------ Audio tagging ------')
    at = AudioTagging(checkpoint_path=None, device=device)
    (clipwise_output, embedding) = at.inference(audio)
    """clipwise_output: (batch_size, classes_num), embedding: (batch_size, embedding_size)"""

    for i in range(len(clipwise_output)):
      print_audio_tagging_result(clipwise_output[i], i)
    print(embedding, 'e')
    print(len(clipwise_output), 'coutputlen')
    print('------ Sound event detection ------')
    sed = SoundEventDetection(checkpoint_path=None, device=device)
    framewise_output = sed.inference(audio)
    """(batch_size, time_steps, classes_num)"""
    return plot_sound_event_detection_result(framewise_output[0])
