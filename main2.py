try:
    import cv2.__init__ as cv2
except ImportError:
    pass
import numpy as np

from Dataset import *
from NetworkDesc import *


folder = 'img'
kp_file = 'grouped_keypoints.txt'

dataset_non_clamped = Dataset(folder, kp_file, clamp_crop_size=False)
dataset_clamped = Dataset(folder, kp_file, clamp_crop_size=True)

network_desc_non_clamped = NetworkDesc(model_file='desc3_keras_non_clamped.h5')
network_desc_clamped = NetworkDesc(model_file='desc3_keras_clamped.h5')

mining_ratio = 1
for i in range(1, 120001):
    patches_non_clamped = dataset_non_clamped.get_batch_desc_triplets(128)
    patches_clamped = dataset_clamped.get_batch_desc_triplets(128)

    loss_non_clamped = network_desc_non_clamped.hardmine_train(patches_non_clamped[0], patches_non_clamped[1], patches_non_clamped[2], mining_ratio)
    loss_clamped = network_desc_clamped.hardmine_train(patches_clamped[0], patches_clamped[1], patches_clamped[2], mining_ratio)

    if i % 10 == 0:
        print('Learning rate Iteration {}, Train losses {} {}'.format(i, loss_non_clamped, loss_clamped))
    if i % 1000 == 0:
        dataset_non_clamped.save_weights()
        network_desc_clamped.save_weights()
    if i % 40000 == 0 and mining_ratio < 4:
        mining_ratio *= 2

patches_non_clamped = dataset_non_clamped.get_batch_desc_triplets(128)
patches_clamped = dataset_clamped.get_batch_desc_triplets(128)

test_nc_nc = network_desc_non_clamped.test_model(patches_non_clamped[0], patches_non_clamped[1], patches_non_clamped[2])
test_nc_c = network_desc_non_clamped.test_model(patches_clamped[0], patches_clamped[1], patches_clamped[2])
test_c_nc = network_desc_clamped.test_model(patches_non_clamped[0], patches_non_clamped[1], patches_non_clamped[2])
test_c_c = network_desc_clamped.test_model(patches_clamped[0], patches_clamped[1], patches_clamped[2])


print('Test non clamped network - NC dataset: {}, C dataset: {}'.format(test_nc_nc, test_nc_c))
print('Test clamped network - NC dataset: {}, C dataset: {}'.format(test_c_nc, test_c_c))

# dataset_12 = Dataset(folder, kp_file, 12)
# dataset_15 = Dataset(folder, kp_file, 15)
# dataset_18 = Dataset(folder, kp_file, 18)

# network_desc_01 = NetworkDesc(0.01, 'desc3_keras_01.h5')
# network_desc_001 = NetworkDesc(0.001, 'desc3_keras_001.h5')
# network_desc_0001 = NetworkDesc(0.0001, 'desc3_keras_0001.h5')
#
# mining_ratio = 1
#
# losses_01 = np.array([])
# losses_001 = np.array([])
# losses_0001 = np.array([])
#
# for i in range(1, 160001):
#     patches = dataset_15.get_batch_desc_triplets(128)
#     loss_01 = network_desc_01.hardmine_train(patches[0], patches[1], patches[2], mining_ratio)
#     loss_001 = network_desc_001.hardmine_train(patches[0], patches[1], patches[2], mining_ratio)
#     loss_0001 = network_desc_0001.hardmine_train(patches[0], patches[1], patches[2], mining_ratio)
#
#     if i % 10 == 0:
#         print('Learning rate Iteration {}, Train losses {} {} {}'.format(i, loss_01, loss_001, loss_0001))
#     if i % 1000 == 0:
#         network_desc_01.save_weights()
#         network_desc_001.save_weights()
#         network_desc_0001.save_weights()
#     if i % 40000 == 0 and mining_ratio < 8:
#         mining_ratio *= 2
#     if i > 155000:
#         np.append(losses_01, loss_01)
#         np.append(losses_001, loss_001)
#         np.append(losses_0001, loss_0001)
#
# avg_loss_01 = np.average(losses_01)
# avg_loss_001 = np.average(losses_001)
# avg_loss_0001 = np.average(losses_0001)
#
# min_loss = min(avg_loss_01, avg_loss_001, avg_loss_0001)
#
# if avg_loss_01 == min_loss:
#     learning_rate = 0.01
#     avg_loss_15 = avg_loss_01
# elif avg_loss_001 == min_loss:
#     learning_rate = 0.001
#     avg_loss_15 = avg_loss_001
# else:
#     learning_rate = 0.0001
#     avg_loss_15 = avg_loss_0001

learning_rate = 0.002

# network_desc_best_lr_12 = NetworkDesc(learning_rate, 'desc3_keras_best_{}_12.h5'.format(learning_rate))
# network_desc_best_lr_15 = NetworkDesc(learning_rate, 'desc3_keras_best_{}_15.h5'.format(learning_rate))
# network_desc_best_lr_18 = NetworkDesc(learning_rate, 'desc3_keras_best_{}_18.h5'.format(learning_rate))
#
# mining_ratio = 1
#
# losses_12 = np.array([])
# losses_15 = np.array([])
# losses_18 = np.array([])
#
# for i in range(1, 160001):
#     patches_12 = dataset_12.get_batch_desc_triplets(128)
#     patches_15 = dataset_15.get_batch_desc_triplets(128)
#     patches_18 = dataset_18.get_batch_desc_triplets(128)
#
#     loss_12 = network_desc_best_lr_12.hardmine_train(patches_12[0], patches_12[1], patches_12[2], mining_ratio)
#     loss_15 = network_desc_best_lr_15.hardmine_train(patches_15[0], patches_15[1], patches_15[2], mining_ratio)
#     loss_18 = network_desc_best_lr_18.hardmine_train(patches_18[0], patches_18[1], patches_18[2], mining_ratio)
#
#     if i % 10 == 0:
#         print('Patch size Iteration {}, Train losses {} {} {} '.format(i, loss_12, loss_15, loss_18))
#     if i % 1000 == 0:
#         network_desc_best_lr_12.save_weights()
#         network_desc_best_lr_15.save_weights()
#         network_desc_best_lr_18.save_weights()
#     if i % 40000 == 0 and mining_ratio < 8:
#         mining_ratio *= 2
#     if i > 155000:
#         losses_12 = np.append(losses_12, loss_12)
#         losses_15 = np.append(losses_15, loss_15)
#         losses_18 = np.append(losses_18, loss_18)
#
# avg_loss_12 = np.average(losses_12)
# avg_loss_15 = np.average(losses_15)
# avg_loss_18 = np.average(losses_18)
#
# min_loss = min(avg_loss_12, avg_loss_15, avg_loss_18)
#
# if avg_loss_12 == min_loss:
#     patch_size = 12
#     avg_loss_rotated = avg_loss_12
# elif avg_loss_15 == min_loss:
#     patch_size = 15
#     avg_loss_rotated = avg_loss_15
# else:
#     patch_size = 18
#     avg_loss_rotated = avg_loss_18
#
#

# patch_size = 15
#
# network_desc_best_lr_size = NetworkDesc(learning_rate, 'desc3_keras_best_{}_{}_final.h5'.format(learning_rate, patch_size))
#
# network_desc_best_lr_size_non_rotated = NetworkDesc(learning_rate, 'desc3_keras_best_{}_{}_non_rotated.h5'.format(learning_rate, patch_size))
# dataset_non_rotated = Dataset(folder, kp_file, patch_size)
#
# network_desc_best_lr_size_wider_range = NetworkDesc(learning_rate, 'desc3_keras_best_{}_{}_wider_range.h5'.format(learning_rate, patch_size))
# dataset_wider_range = Dataset(folder, kp_file, patch_size, (-1.0, 1.0))
#
# mining_ratio = 1
#
# losses_normal = np.array([])
# losses_non_rotated = np.array([])
# losses_wider_range = np.array([])
#
# for i in range(1, 160001):
#     patches_normal = dataset_15.get_batch_desc_triplets(128)
#     patches_non_rotated = dataset_non_rotated.get_batch_desc_triplets(128, False)
#     patches_wider_range = dataset_wider_range.get_batch_desc_triplets(128)
#
#     loss_normal = network_desc_best_lr_size.hardmine_train(patches_normal[0], patches_normal[1], patches_normal[2], mining_ratio)
#     loss_non_rotated = network_desc_best_lr_size_non_rotated.hardmine_train(patches_non_rotated[0], patches_non_rotated[1], patches_non_rotated[2], mining_ratio)
#     loss_wider_range = network_desc_best_lr_size_wider_range.hardmine_train(patches_wider_range[0], patches_wider_range[1], patches_wider_range[2], mining_ratio)
#
#     if i % 10 == 0:
#         print('Rotation, range Iteration {}, Train losses {} {} {}'.format(i, loss_normal, loss_non_rotated, loss_wider_range))
#     if i % 1000 == 0:
#         network_desc_best_lr_size.save_weights()
#         network_desc_best_lr_size_non_rotated.save_weights()
#         network_desc_best_lr_size_wider_range.save_weights()
#     if i % 40000 == 0 and mining_ratio < 8:
#         mining_ratio *= 2
#     if i > 155000:
#         losses_normal = np.append(losses_normal, loss_normal)
#         losses_non_rotated = np.append(losses_non_rotated, loss_non_rotated)
#         losses_wider_range = np.append(losses_wider_range, loss_wider_range)
#
# avg_loss_normal = np.average(losses_normal)
# avg_loss_non_rotated = np.average(losses_non_rotated)
# avg_loss_wider_range = np.average(losses_wider_range)


# print('Different learning rates, best {}:'.format(learning_rate))
# print([0.01, 0.001, 0.0001])
# print([avg_loss_01, avg_loss_001, avg_loss_0001])

# print('Different patch sizes, best {}:'.format(patch_size))
# print([12, 15, 18])
# print([avg_loss_12, avg_loss_15, avg_loss_18])

# print('With/without rotation:')
# print([avg_loss_normal, avg_loss_non_rotated])
#
# print('Normal/wider range:')
# print([avg_loss_normal, avg_loss_wider_range])
