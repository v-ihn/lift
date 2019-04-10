try:
    import cv2.__init__ as cv2
except ImportError:
    pass
import numpy as np
import random


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


class Dataset:
    def __init__(self, data_folder: str, kp_file: str, keypoint_size_multiplier=15, clamp_crop_size=True, mean=None, std=None):
        available_keypoints = {}
        unique_filenames = []

        kp_file = open(data_folder + '/' + kp_file, 'r')
        for line in kp_file.read().splitlines():
            line_arr = line.split(" ")
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
            img = cv2.imread(filename, flags=cv2.IMREAD_GRAYSCALE)
            files[filename] = img

        self.available_patches = {}
        for point in available_keypoints:
            self.available_patches[point] = []
            for keypoint in available_keypoints[point]:
                patch = self.create_patch_128(files, keypoint, keypoint_size_multiplier, clamp_crop_size)
                if patch is not None:
                    self.available_patches[point].append({'patch': patch, 'angle': keypoint.angle})

        self.mean = None
        self.std = None
        self.normalize(mean, std)

        self.available_indices = {}
        self.init_patch_indices()

        self.fixed_epoch_desc = None
        self.fixed_epoch_desc_index = 0

    def init_patch_indices(self):
        self.available_indices = {}
        for point in self.available_patches:
            num_keypoints = len(self.available_patches[point])
            patch_indices = [i for i in range(num_keypoints)]
            self.available_indices[point] = patch_indices

    def create_fixed_epoch_desc(self, size, rotated=True):
        self.fixed_epoch_desc = []
        for _ in range(size):
            self.fixed_epoch_desc.append(self.get_triplet(rotated))

    def get_batch_desc_triplets_from_fixed_epoch(self, batch_size: int):
        assert len(self.fixed_epoch_desc) % batch_size == 0

        patches_batch = np.array(list(zip(*self.fixed_epoch_desc[self.fixed_epoch_desc_index:self.fixed_epoch_desc_index + batch_size])))
        self.fixed_epoch_desc_index = (self.fixed_epoch_desc_index + batch_size) % len(self.fixed_epoch_desc)
        if self.fixed_epoch_desc_index == 0:
            np.random.shuffle(self.fixed_epoch_desc)

        return patches_batch

    def get_batch_desc_triplets(self, batch_size: int, rotated=True):
        patches = []
        for i in range(batch_size):
            patches_triplet = self.get_triplet2(rotated)
            patches.append(patches_triplet)
        patches_batch = np.array(list(zip(*patches)))
        return patches_batch

    def get_triplet(self, rotated: bool):
        points = random.sample(list(self.available_patches), 2)
        matching_patches = random.sample(self.available_patches[points[0]], 2)
        different_patch = random.choice(self.available_patches[points[1]])

        if rotated:
            matching_patches = [self.crop_patch_64(self.rotate_patch(patch['patch'], patch['angle'])) for patch in
                                matching_patches]
            different_patch = self.crop_patch_64(self.rotate_patch(different_patch['patch'], different_patch['angle']))
        else:
            matching_patches = [self.crop_patch_64(patch['patch']) for patch in matching_patches]
            different_patch = self.crop_patch_64(different_patch['patch'])

        return matching_patches[0], matching_patches[1], different_patch

    def get_triplet2(self, rotated: bool):
        points = random.sample(list(self.available_indices), 2)
        matching_patches_indices = random.sample(self.available_indices[points[0]], 2)
        different_patch_index = random.choice(self.available_indices[points[1]])

        self.available_indices[points[0]] = [index for index in self.available_indices[points[0]] if index not in matching_patches_indices]
        if len(self.available_indices[points[0]]) < 2:
            del self.available_indices[points[0]]

        self.available_indices[points[1]].remove(different_patch_index)
        if len(self.available_indices[points[1]]) < 2:
            del self.available_indices[points[1]]

        if len(self.available_indices) < 2:
            self.init_patch_indices()

        matching_patches = [self.available_patches[points[0]][index] for index in matching_patches_indices]
        different_patch = self.available_patches[points[1]][different_patch_index]

        if rotated:
            matching_patches = [self.crop_patch_64(self.rotate_patch(patch['patch'], patch['angle'])) for patch in
                                matching_patches]
            different_patch = self.crop_patch_64(self.rotate_patch(different_patch['patch'], different_patch['angle']))
        else:
            matching_patches = [self.crop_patch_64(patch['patch']) for patch in matching_patches]
            different_patch = self.crop_patch_64(different_patch['patch'])

        return [matching_patches[0], matching_patches[1], different_patch]

    def create_patch_128(self, files: dict, kp: KeyPointInfo, keypoint_size_multiplier, clamp_crop_size):
        img = files[kp.file]
        x_center = int(kp.x + 0.5)
        y_center = int(kp.y + 0.5)
        if clamp_crop_size:
            crop_size = int(np.clip(keypoint_size_multiplier * kp.size, keypoint_size_multiplier * 2.0, keypoint_size_multiplier * 3.5))
            # crop_size = int(keypoint_size_multiplier * 2.75)
        else:
            crop_size = int(keypoint_size_multiplier * kp.size)

        if x_center - crop_size < 0 or x_center + crop_size >= img.shape[1] or y_center - crop_size < 0 or y_center + crop_size >= img.shape[0]:
            return None

        patch = img[y_center - crop_size:y_center + crop_size, x_center - crop_size:x_center + crop_size].copy()
        patch = cv2.resize(patch, (128, 128))
        # patch = cv2.normalize(patch.astype('float'), None, -1.0, 1.0, cv2.NORM_MINMAX)
        return np.expand_dims(patch, axis=2)

    def crop_patch_64(self, patch_128):
        return patch_128[32:96, 32:96].copy()

    def rotate_patch(self, patch, angle):
        x_center = patch.shape[1] / 2
        y_center = patch.shape[0] / 2
        m = cv2.getRotationMatrix2D((x_center, y_center), 360 - angle, 1)
        rotated_patch = cv2.warpAffine(patch, m, (patch.shape[1], patch.shape[0]))
        return np.expand_dims(rotated_patch, axis=2)

    def normalize(self, mean, std):
        all_patches = []
        for point in self.available_patches:
            for p in self.available_patches[point]:
                all_patches.append(p['patch'])

        all_patches = np.array(all_patches)
        self.mean = np.mean(all_patches) if mean is None else mean
        self.std = np.std(all_patches) if std is None else std

        for point in self.available_patches:
            for i in range(len(self.available_patches[point])):
                self.available_patches[point][i]['patch'] = (self.available_patches[point][i]['patch'] - self.mean) / self.std
