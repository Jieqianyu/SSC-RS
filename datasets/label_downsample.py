from glob import glob
import os
import numpy as np
import time
import argparse
import sys

# Append root directory to system path for imports
repo_path, _ = os.path.split(os.path.realpath(__file__))
repo_path, _ = os.path.split(repo_path)
repo_path, _ = os.path.split(repo_path)
sys.path.append(repo_path)

import datasets.io_data as SemanticKittiIO


def parse_args():
  parser = argparse.ArgumentParser(description='LMSCNet labels lower scales creation')
  parser.add_argument(
    '--dset_root',
    dest='dataset_root',
    default='',
    metavar='DATASET',
    help='path to dataset root folder',
    type=str,
  )
  args = parser.parse_args()
  return args


def majority_pooling(grid, k_size=2):
  result = np.zeros((grid.shape[0] // k_size, grid.shape[1] // k_size, grid.shape[2] // k_size))
  for xx in range(0, int(np.floor(grid.shape[0]/k_size))):
    for yy in range(0, int(np.floor(grid.shape[1]/k_size))):
      for zz in range(0, int(np.floor(grid.shape[2]/k_size))):

        sub_m = grid[(xx*k_size):(xx*k_size)+k_size, (yy*k_size):(yy*k_size)+k_size, (zz*k_size):(zz*k_size)+k_size]
        unique, counts = np.unique(sub_m, return_counts=True)
        if True in ((unique != 0) & (unique != 255)):
          # Remove counts with 0 and 255
          counts = counts[((unique != 0) & (unique != 255))]
          unique = unique[((unique != 0) & (unique != 255))]
        else:
          if True in (unique == 0):
            counts = counts[(unique != 255)]
            unique = unique[(unique != 255)]
        value = unique[np.argmax(counts)]
        result[xx, yy, zz] = value
  return result


def downscale_data(LABEL, downscaling):
    # Majority pooling labels downscaled in 3D
    LABEL = majority_pooling(LABEL, k_size=downscaling)
    # Reshape to 1D
    LABEL = LABEL.reshape(-1)
    # Invalid file downscaled
    INVALID = np.zeros_like(LABEL)
    INVALID[np.isclose(LABEL, 255)] = 1
    return LABEL, INVALID


def main():

  args = parse_args()

  dset_root = args.dataset_root
  remap_lut = SemanticKittiIO.get_remap_lut(os.path.join('data', 'semantic-kitti.yaml'))
  sequences = sorted(glob(os.path.join(dset_root, 'dataset', 'sequences', '*')))
  # Selecting training/validation set sequences only (labels unavailable for test set)
  sequences = sequences[:11]
  grid_dimensions = [256, 256, 32]   # [W, H, D]

  assert len(sequences) > 0, 'Error, no sequences on selected dataset root path'

  for sequence in sequences:
    label_paths  = sorted(glob(os.path.join(sequence, 'voxels', '*.label')))
    invalid_paths = sorted(glob(os.path.join(sequence, 'voxels', '*.invalid')))
    out_dir = os.path.join(sequence, 'voxels')
    downscaling = {'1_2': 2, '1_4': 4, '1_8': 8}

    for i in range(len(label_paths)):
      filename, _ = os.path.splitext(os.path.basename(label_paths[i]))

      LABEL = SemanticKittiIO._read_label_SemKITTI(label_paths[i])
      INVALID = SemanticKittiIO._read_invalid_SemKITTI(invalid_paths[i])
      LABEL = remap_lut[LABEL.astype(np.uint16)].astype(np.float32)  # Remap 20 classes semanticKITTI SSC
      LABEL[np.isclose(INVALID, 1)] = 255  # Setting to unknown all voxels marked on invalid mask...
      LABEL = LABEL.reshape(grid_dimensions)

      for scale in downscaling:
        label_filename = os.path.join(out_dir, filename + '.label_' + scale)
        invalid_filename = os.path.join(out_dir, filename + '.invalid_' + scale)
        # If files have not been created...
        if not (os.path.isfile(label_filename) & os.path.isfile(invalid_filename)):
          LABEL_ds, INVALID_ds = downscale_data(LABEL, downscaling[scale])
          SemanticKittiIO.pack(INVALID_ds.astype(dtype=np.uint8)).tofile(invalid_filename)
          print(time.strftime('%x %X') + ' -- => File {} - Sequence {} saved...'.format(filename + '.label_' + scale, os.path.basename(sequence)))
          LABEL_ds.astype(np.uint16).tofile(label_filename)
          print(time.strftime('%x %X') + ' -- => File {} - Sequence {} saved...'.format(filename + '.invalid_' + scale, os.path.basename(sequence)))

    print(time.strftime('%x %X') + ' -- => All files saved for Sequence {}'.format(os.path.basename(sequence)))

  print(time.strftime('%x %X') + ' -- => All files saved')

  exit()

if __name__ == '__main__':
  main()

