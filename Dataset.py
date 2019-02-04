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
    def __init__(self, data_folder: str, kp_file: str):
        available_keypoints = {}
        unique_labels = []
        unique_filenames = []

        kp_file = open(data_folder + '/' + kp_file, 'r')
        for line in kp_file.read().splitlines():
            line_arr = line.split(" ")
            point = Point3(float(line_arr[0]), float(line_arr[1]), float(line_arr[2]))
            keypoint = KeyPointInfo(data_folder + '/' + line_arr[3], float(line_arr[4]), float(line_arr[5]), float(line_arr[6]), float(line_arr[7]))
            unique_filenames.append(keypoint.file)
            if point not in available_keypoints:
                available_keypoints[point] = [keypoint]
                unique_labels.append(point)
            else:
                available_keypoints[point].append(keypoint)

        unique_labels = set(unique_labels)
        unique_filenames = set(unique_filenames)
        files = {}
        for filename in unique_filenames:
            img = cv2.imread(filename, flags=cv2.IMREAD_GRAYSCALE)
            files[filename] = img

        self.available_patches = {}
        for point in available_keypoints:
            one_hot_point = self.one_hot_label(unique_labels, point)
            self.available_patches[one_hot_point] = []
            for keypoint in available_keypoints[point]:
                patch = self.create_patch_128(files, keypoint)
                if patch is not None:
                    self.available_patches[one_hot_point].append({'patch': patch, 'angle': keypoint.angle})

    def get_batch_desc_pairs(self, batch_size: int, matching: bool):
        patches = []
        labels = []
        for i in range(batch_size):
            if matching:
                label_pair, patch_pair = self.get_matching_pair()
            else:
                label_pair, patch_pair = self.get_non_matching_pair()
            labels.append(label_pair)
            patches.append(patch_pair)
        patches_batch = np.array(list(zip(*patches)))
        labels_batch = np.array(list(zip(*labels)))
        return labels_batch, patches_batch

    def get_batch_desc_triplets(self, batch_size: int):
        patches = []
        labels = []
        for i in range(batch_size):
            labels_triplet, patches_triplet = self.get_triplet(True)
            labels.append(labels_triplet)
            patches.append(patches_triplet)
        patches_batch = np.array(list(zip(*patches)))
        labels_batch = np.array(list(zip(*labels)))
        return labels_batch, patches_batch

    def get_matching_pair(self):
        label = random.choice(list(self.available_patches))
        patches = random.sample(self.available_patches[label], 2)
        # patches = [self.rotate_patch(self.crop_patch_64(patch['patch']), patch['angle']) for patch in patches]
        patches = [self.crop_patch_64(self.rotate_patch(patch['patch'], patch['angle'])) for patch in patches]
        return (label, label), (patches[0], patches[1])

    def get_non_matching_pair(self):
        labels = random.sample(list(self.available_patches), 2)
        patch1 = random.choice(self.available_patches[labels[0]])
        # patch1 = self.rotate_patch(self.crop_patch_64(patch1['patch']), patch1['angle'])
        patch1 = self.crop_patch_64(self.rotate_patch(patch1['patch'], patch1['angle']))
        patch2 = random.choice(self.available_patches[labels[1]])
        # patch2 = self.rotate_patch(self.crop_patch_64(patch2['patch']), patch2['angle'])
        patch2 = self.crop_patch_64(self.rotate_patch(patch2['patch'], patch2['angle']))
        return (labels[0], labels[1]), (patch1, patch2)

    def get_triplet(self, rotated: bool):
        labels = random.sample(list(self.available_patches), 2)
        matching_patches = random.sample(self.available_patches[labels[0]], 2)
        different_patch = random.choice(self.available_patches[labels[1]])

        if rotated:
            # matching_patches = [self.rotate_patch(self.crop_patch_64(patch['patch']), patch['angle']) for patch in
            #                     matching_patches]
            # different_patch = self.rotate_patch(self.crop_patch_64(different_patch['patch']), different_patch['angle'])
            matching_patches = [self.crop_patch_64(self.rotate_patch(patch['patch'], patch['angle'])) for patch in
                                matching_patches]
            different_patch = self.crop_patch_64(self.rotate_patch(different_patch['patch'], different_patch['angle']))
        else:
            matching_patches = [self.crop_patch_64(patch['patch']) for patch in matching_patches]
            different_patch = self.crop_patch_64(different_patch['patch'])

        return (labels[0], labels[0], labels[1]), (matching_patches[0], matching_patches[1], different_patch)

    def create_patch_128(self, files: dict, kp: KeyPointInfo):
        img = files[kp.file]
        x_center = int(kp.x + 0.5)
        y_center = int(kp.y + 0.5)
        crop_size = int(12 * kp.size)

        if x_center - crop_size < 0 or x_center + crop_size >= img.shape[1] or y_center - crop_size < 0 or y_center + crop_size >= img.shape[0]:
            return None

        patch = img[y_center - crop_size:y_center + crop_size, x_center - crop_size:x_center + crop_size].copy()
        patch = cv2.resize(patch, (128, 128))
        patch = cv2.normalize(patch.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        return np.expand_dims(patch, axis=2)

    def crop_patch_64(self, patch_128):
        return patch_128[32:96, 32:96].copy()

    def rotate_patch(self, patch, angle):
        x_center = patch.shape[1] / 2
        y_center = patch.shape[0] / 2
        m = cv2.getRotationMatrix2D((x_center, y_center), 360 - angle, 1)
        rotated_patch = cv2.warpAffine(patch, m, (patch.shape[1], patch.shape[0]))
        # rotated_patch = cv2.normalize(rotated_patch.astype('float'), None, -1.0, 1.0, cv2.NORM_MINMAX)
        return np.expand_dims(rotated_patch, axis=2)

    def one_hot_label(self, unique_labels: set, label: Point3):
        return tuple([1 if label == l else 0 for l in unique_labels])
