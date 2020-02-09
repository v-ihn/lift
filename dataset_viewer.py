import cv2
import numpy as np


class Point3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def __eq__(self, other):
        return (self.x, self.y, self.z) == (other.x, other.y, other.z)


class KeyPointInfo:
    def __init__(self, file, x, y, angle, size):
        self.file = file
        self.x = x
        self.y = y
        self.angle = angle
        self.size = size


def create_patch_128(files: dict, kp: KeyPointInfo, keypoint_size_multiplier, clamp_crop_size):
    img = files[kp.file]
    x_center = int(kp.x + 0.5)
    y_center = int(kp.y + 0.5)
    if clamp_crop_size:
        crop_size = int(np.clip(keypoint_size_multiplier * kp.size, keypoint_size_multiplier * 2.0, keypoint_size_multiplier * 3.5))
    else:
        crop_size = int(keypoint_size_multiplier * kp.size)

    if x_center - crop_size < 0 or x_center + crop_size >= img.shape[1] or y_center - crop_size < 0 or y_center + crop_size >= img.shape[0]:
        return None

    patch = img[y_center - crop_size:y_center + crop_size, x_center - crop_size:x_center + crop_size].copy()
    patch = cv2.resize(patch, (128, 128))

    patch[63][63] = (255, 0, 255)
    patch[63][64] = (255, 0, 255)
    patch[64][63] = (255, 0, 255)
    patch[64][64] = (255, 0, 255)

    return patch


def load_and_view(data_folder: str, kp_file: str, keypoint_size_multiplier=15, clamp_crop_size=True):
    available_keypoints = {}
    unique_filenames = []

    kp_file = open(data_folder + '/' + kp_file, 'r')
    for line in kp_file.read().splitlines():
        line_arr = line.split(',')
        point = Point3(float(line_arr[0]), float(line_arr[1]), float(line_arr[2]))
        keypoint = KeyPointInfo(data_folder + '/' + line_arr[3], float(line_arr[4]), float(line_arr[5]), float(line_arr[6]), float(line_arr[7]))
        unique_filenames.append(keypoint.file)
        if point not in available_keypoints:
            available_keypoints[point] = [keypoint]
        else:
            available_keypoints[point].append(keypoint)

    unique_filenames = set(unique_filenames)
    files = {}
    for filename in unique_filenames:
        img = cv2.imread(filename, flags=cv2.IMREAD_COLOR)
        files[filename] = img

    for point in available_keypoints:
        i = 0
        for keypoint in available_keypoints[point]:
            patch = create_patch_128(files, keypoint, keypoint_size_multiplier, clamp_crop_size)
            if patch is not None:
                cv2.imshow(str(i), patch)
                cv2.moveWindow(str(i), 10 + 150 * (i % 15), 10 + 160 * int(i / 15))
                i += 1
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# load_and_view('img/', 'train_keypoints.txt')
load_and_view('./', 'test_keypoints_custom.txt', keypoint_size_multiplier=30)
