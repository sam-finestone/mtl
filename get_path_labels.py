import os
import numpy as np
import pickle

root_dir = './dataset/cholec80'
img_dir = os.path.join(root_dir, 'frame_resize')
tool_dir = os.path.join(root_dir, 'tool_annotations')
phase_dir = os.path.join(root_dir, 'phase_annotations')

print(root_dir)
print(img_dir)
print(phase_dir)

def get_dirs(root_dir):
    file_paths = []
    file_names = []
    for lists in os.listdir(root_dir):
        path = os.path.join(root_dir, lists)
        if os.path.isdir(path):
            file_paths.append(path)
            file_names.append(os.path.basename(path))
    file_names.sort()
    file_paths.sort()
    return file_names, file_paths

def get_files(root_dir):
    file_paths = []
    file_names = []
    for lists in os.listdir(root_dir):
        path = os.path.join(root_dir, lists)
        if not os.path.isdir(path):
            file_paths.append(path)
            file_names.append(os.path.basename(path))
    file_names.sort()
    file_paths.sort()
    return file_names, file_paths


img_dir_names, img_dir_paths = get_dirs(img_dir)
print(img_dir_names)
print(img_dir_paths)
phase_file_names, phase_file_paths = get_files(phase_dir)
print(phase_file_names)
print(phase_file_paths)
phase_dict = {}
phase_dict_key = ['Preparation', 'CalotTriangleDissection', 'ClippingCutting', 'GallbladderDissection',
                  'GallbladderPackaging', 'CleaningCoagulation', 'GallbladderRetraction']
for i in range(len(phase_dict_key)):
    phase_dict[phase_dict_key[i]] = i
print(phase_dict)


all_info_all = []

print(len(phase_file_names))
for j in range(1, len(phase_file_names)):
    if phase_file_paths[j].endswith('.txt'):
        phase_file = open(phase_file_paths[j], 'rb')
        phase_count = 0
        file_count = 0
        frame_num = len(os.listdir(img_dir_paths[j-1]))
        info_all = []
        for phase_line in phase_file:
            phase_count += 1
            if phase_count > 1:
                if phase_count % 25 == 2 and file_count <= frame_num:
                    file_count += 1
                    phase_split = phase_line.split()
                    info_each = []
                    name = 'frame%d' % file_count
                    img_file_each_path = os.path.join(img_dir_paths[j-1], img_dir_names[j-1] + '-' + name + '.jpg')
                    info_each.append(img_file_each_path)
                    info_each.append(phase_dict[phase_split[1].decode()])
                    info_all.append(info_each)


        # print(len(info_all))
        all_info_all.append(info_all)

# for k in range(10):
# print(all_info_all[0][k])
with open('cholec80_.pkl', 'wb') as f:
    pickle.dump(all_info_all, f)

import pickle

with open('cholec80_.pkl', 'rb') as f:
    all_info = pickle.load(f)

train_file_paths = []
test_file_paths = []
val_file_paths = []
val_labels = []
train_labels = []
test_labels = []

train_num_each = []
val_num_each = []
test_num_each = []
print(len(all_info))
for i in range(0, 1):
    train_num_each.append(len(all_info[i]))
    for j in range(len(all_info[i])):
        train_file_paths.append(all_info[i][j][0])
        train_labels.append(all_info[i][j][1:])

print(len(train_file_paths))
print(len(train_labels))

for i in range(1, 2):
    val_num_each.append(len(all_info[i]))
    for j in range(len(all_info[i])):
        val_file_paths.append(all_info[i][j][0])
        val_labels.append(all_info[i][j][1:])

print(len(val_file_paths))
print(len(val_labels))

for i in range(2, 3):
    test_num_each.append(len(all_info[i]))
    for j in range(len(all_info[i])):
        test_file_paths.append(all_info[i][j][0])
        test_labels.append(all_info[i][j][1:])

print(len(test_file_paths))
print(len(test_labels))

# for i in range(10):
#     print(train_file_paths[i], train_labels[i])
#     print(test_file_paths[i], test_labels[i])

train_val_test_paths_labels = {}
train_val_test_paths_labels['train_paths'] = train_file_paths
train_val_test_paths_labels['val_paths'] = val_file_paths
train_val_test_paths_labels['test_paths'] = test_file_paths

train_val_test_paths_labels['train_labels'] = train_labels
train_val_test_paths_labels['val_labels'] = val_labels
train_val_test_paths_labels['test_labels'] = test_labels

train_val_test_paths_labels['train_num_each'] = train_num_each
train_val_test_paths_labels['val_num_each'] = val_num_each
train_val_test_paths_labels['test_num_each'] = test_num_each

with open('train_val_test_paths_labels.pkl', 'wb') as f:
    pickle.dump(train_val_test_paths_labels, f)


print('Done')