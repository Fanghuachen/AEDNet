from __future__ import print_function
import os
import os.path
import torch
import torch.utils.data as data
import numpy as np



def load_train_shape(point_filename, labels):
    data = np.load(point_filename)
    num = data.shape[0]
    t = data[:, 3]
    t_min = min(t)
    t_max = max(t)
    t_lim = (t_max - t_min) / round(num / 5000)
    t_permutation = np.argsort(t)
    data = data[t_permutation]
    pts = data[:,1:4]
    pol = data[:, 4]

    if labels != None:
        labels = data[:,0]
    else:
        labels = None

    return Shape(pts=pts, pol=pol, labels=labels, t_lim=t_lim)

def load_test_shape(n, shape_num, point_filename, labels):
    data = np.load(point_filename)
    num = data.shape[0]
    if labels:
        t = data[:, 3]
    else:
        t = data[:, 2]
    t_min = min(t)
    t_max = max(t)
    t_lim = (t_max - t_min) / round(num / 5000)
    t_permutation = np.argsort(t)
    data = data[t_permutation]
    num -= n * shape_num
    if num <= 1.7*shape_num:
        data = data[n*shape_num:num+n*shape_num, :]
    else:
        data = data[n * shape_num:(n + 1) * shape_num, :]

    if labels:
        labels = data[:,0]
        pts = data[:, 1:4]
        pol = data[:, 4]
    else:
        labels = None
        pts = data[:, 0:3]
        pol = data[:, 3]

    return Shape(pts=pts, pol = pol, labels = labels, t_lim=t_lim)

class SequentialAEDPatchSampler(data.sampler.Sampler):

    def __init__(self, data_source):
        self.data_source = data_source
        self.total_patch_count = None

        self.total_patch_count = 0
        for shape_ind, _ in enumerate(self.data_source.shape_patch_count):
            self.total_patch_count = self.total_patch_count + self.data_source.shape_patch_count[shape_ind]

    def __iter__(self):
        return iter(range(self.total_patch_count))

    def __len__(self):
        return self.total_patch_count


class SequentialShapeRandomAEDPatchSampler(data.sampler.Sampler):

    def __init__(self, data_source, patches_per_shape, seed=None, sequential_shapes=False, identical_epochs=False):
        self.data_source = data_source
        self.patches_per_shape = patches_per_shape
        self.sequential_shapes = sequential_shapes
        self.seed = seed
        self.identical_epochs = identical_epochs
        self.total_patch_count = None
        self.shape_patch_inds = None

        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32-1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        self.total_patch_count = 0
        for shape_ind, _ in enumerate(self.data_source.shape_names):
            self.total_patch_count = self.total_patch_count + min(self.patches_per_shape, self.data_source.shape_patch_count[shape_ind])

    def __iter__(self):

        # optionally always pick the same permutation (mainly for debugging)
        if self.identical_epochs:
            self.rng.seed(self.seed)

        # global point index offset for each shape
        shape_patch_offset = list(np.cumsum(self.data_source.shape_patch_count))
        shape_patch_offset.insert(0, 0)
        shape_patch_offset.pop()

        shape_inds = range(len(self.data_source.shape_names))
        prob = self.rng.uniform(low=0.4, high=0.6, size=len(self.data_source.shape_names))

        if not self.sequential_shapes:
            shape_inds = self.rng.permutation(shape_inds)  # 随机排序

        # return a permutation of the events in the dataset where all events in the same shape are adjacent (for performance reasons):
        # first permute shapes, then concatenate a list of permuted events in each shape
        self.shape_patch_inds = [[]]*len(self.data_source.shape_names)
        point_permutation = []
        for shape_ind in shape_inds:
            shape_permutation = []
            shape_prob = prob[shape_ind]
            events_num = int(self.patches_per_shape*shape_prob)
            noise_num = self.patches_per_shape - events_num
            shape = self.data_source.shape_cache.get(shape_ind)
            labels = shape.labels
            events_ind = np.argwhere(labels == 0).reshape(-1)
            noise_ind = np.argwhere(labels == 1).reshape(-1)
            events_start = shape_patch_offset[shape_ind]+events_ind[0]
            events_end = shape_patch_offset[shape_ind]+events_ind[-1]
            noise_start = shape_patch_offset[shape_ind]+noise_ind[0]
            noise_end = shape_patch_offset[shape_ind]+noise_ind[-1]
            events_ind = events_ind+shape_patch_offset[shape_ind]
            noise_ind = noise_ind+shape_patch_offset[shape_ind]
            events_global_patch_inds = self.rng.choice(events_ind, size=min(events_num, events_end-events_start), replace=False)
            noise_global_patch_inds = self.rng.choice(noise_ind, size=min(noise_num, noise_end-noise_start), replace=False)
            shape_permutation.extend(events_global_patch_inds)
            shape_permutation.extend(noise_global_patch_inds)
            self.rng.shuffle(shape_permutation)
            point_permutation.extend(shape_permutation)

            # save indices of shape point subset
            self.shape_patch_inds[shape_ind] = np.array(shape_permutation) - shape_patch_offset[shape_ind]

        return iter(point_permutation)

    def __len__(self):
        return self.total_patch_count

class RandomAEDPatchSampler(data.sampler.Sampler):

    def __init__(self, data_source, patches_per_shape, seed=None, identical_epochs=False):
        self.data_source = data_source
        self.patches_per_shape = patches_per_shape
        self.seed = seed
        self.identical_epochs = identical_epochs
        self.total_patch_count = None

        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**31-1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        self.total_patch_count = 0
        for shape_ind, _ in enumerate(self.data_source.shape_names):
            self.total_patch_count = self.total_patch_count + min(self.patches_per_shape, self.data_source.shape_patch_count[shape_ind])

    def __iter__(self):

        # optionally always pick the same permutation (mainly for debugging)
        if self.identical_epochs:
            self.rng.seed(self.seed)

        return iter(self.rng.choice(sum(self.data_source.shape_patch_count), size=self.total_patch_count, replace=False))

    def __len__(self):
        return self.total_patch_count


class Shape():
    def __init__(self, pts, pol, labels, t_lim):
        self.pts = pts
        self.pol = pol
        self.labels = labels
        self.t_lim = t_lim

class Cache():
    def __init__(self, capacity, loader, loadfunc):
        self.elements = {}
        self.used_at = {}
        self.capacity = capacity
        self.loader = loader
        self.loadfunc = loadfunc
        self.counter = 0

    def get(self, element_id):
        if element_id not in self.elements:
            # cache miss

            # if at capacity, throw out least recently used item
            if len(self.elements) >= self.capacity:
                remove_id = min(self.used_at, key=self.used_at.get)
                del self.elements[remove_id]
                del self.used_at[remove_id]

            # load element
            self.elements[element_id] = self.loadfunc(self.loader, element_id)

        self.used_at[element_id] = self.counter
        self.counter += 1

        return self.elements[element_id]



class AEDPatchDataset(data.Dataset):

    # different sampling rules along temporal line and spatial surface
    def __init__(self, root, shapes_list_file, shape_num, points_per_patch, x_limitation, y_limitation, x_frame, y_frame, seed=None, identical_epochs=False, center='point',
                 cache_capacity=1, shape_names = None, train = True, label=True):

        # initialize parameters
        self.root = root
        self.shapes_list_file = shapes_list_file
        self.points_per_patch = points_per_patch
        self.identical_epochs = identical_epochs
        self.center = center
        self.seed = seed
        self.x_lim = x_limitation
        self.y_lim = y_limitation
        self.x_frame = x_frame
        self.y_frame = y_frame
        if shape_num != 0:
            self.num = shape_num

        self.train = train
        self.include_labels = label
        if train == False:
            self.include_original = True
        else:
            self.include_original = False

        self.load_iteration = 0
        if self.train:
            self.shape_cache = Cache(cache_capacity, self, AEDPatchDataset.load_train_shape_by_index)
        else:
            self.shape_cache = Cache(cache_capacity, self, AEDPatchDataset.load_test_shape_by_index)

        # get all shape names in the dataset
        if shape_names is not None:
            self.shape_names = shape_names
        else:
            self.shape_names = []
            with open(os.path.join(root, self.shapes_list_file)) as f:
                self.shape_names = f.readlines()
            self.shape_names = [x.strip() for x in self.shape_names]
            self.shape_names = list(filter(None, self.shape_names))
        # initialize rng for picking points in a patch
        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**31-1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        # get basic information for each shape in the dataset
        self.shape_patch_count = []
        if self.train:
            for shape_ind, shape_name in enumerate(self.shape_names):
                print('getting information for shape %s' % (shape_name))

                shape = self.shape_cache.get(shape_ind)
                self.shape_patch_count.append(shape.pts.shape[0])
        else:
            n = 0
            for shape_ind, shape_name in enumerate(self.shape_names):
                print('getting information for shape %s' % (shape_name))
                point_filename = os.path.join(self.root, self.shape_names[shape_ind] + '.npy')
                data = np.load(point_filename)
                num = data.shape[0]

                while num > int(1.7*self.num):
                    num = num - self.num
                    shape = self.shape_cache.get(n)
                    n += 1
                    self.shape_patch_count.append(shape.pts.shape[0])
                shape = self.shape_cache.get(n)
                self.shape_patch_count.append(shape.pts.shape[0])



    def select_patch_points(self, global_point_index, center_point_ind, shape,
    patch_pts_valid, patch_pts, labels=False):

        if labels:
            patch_pts[0] = torch.from_numpy(np.array(shape.labels[center_point_ind]))
        else:
            pts = shape.pts
            t_lim =shape.t_lim
            data_num = pts.shape[0]
            local_index = np.arange(data_num)
            data = np.zeros((data_num, 4))
            data[:, 0:3] = pts[:, 0:3]
            data[:, 3] = local_index
            x = data[:, 0]
            center_point = data[center_point_ind]
            x_center = center_point[0]
            if x_center <= self.x_lim / 2:
                x_indice = np.argwhere(x <= self.x_lim).reshape(-1)
            elif x_center >= self.x_frame - self.x_lim / 2:
                x_indice = np.argwhere(x >= self.x_frame - self.x_lim).reshape(-1)
            else:
                x_indice = np.argwhere(x >= x_center - self.x_lim / 2).reshape(-1)
                data = data[x_indice]
                x = data[:, 0]
                x_indice = np.argwhere(x <= x_center + self.x_lim / 2).reshape(-1)
            data = data[x_indice]
            y = data[:, 1]
            y_center = center_point[1]
            if y_center <= self.y_lim / 2:
                y_indice = np.argwhere(y <= self.y_lim).reshape(-1)
            elif y_center >= self.y_frame - self.y_lim / 2:
                y_indice = np.argwhere(y >= self.y_frame - self.y_lim).reshape(-1)
            else:
                y_indice = np.argwhere(y >= y_center - self.y_lim / 2).reshape(-1)
                data = data[y_indice]
                y = data[:, 1]
                y_indice = np.argwhere(y <= y_center + self.y_lim / 2).reshape(-1)
            data = data[y_indice]
            t_center = center_point[2]
            t_min = t_center - t_lim
            t_max = t_center + t_lim
            t = data[:, 2]
            t_indice = np.argwhere(t>=t_min).reshape(-1)
            data = data[t_indice]
            t = data[:, 2]
            t_indice = np.argwhere(t<=t_max).reshape(-1)
            data = data[t_indice]
            ind = data[:, 3]
            center_ind = np.argwhere(ind == center_point_ind).reshape(-1)[0]

            num = data.shape[0]
            if num < self.points_per_patch:
                patch_point_inds = center_ind * np.ones(self.points_per_patch).astype(np.int)
                patch_point_inds[0:num] = np.array(range(num))
            elif center_ind <= int(0.5 * self.points_per_patch):
                patch_point_inds = np.array(range(self.points_per_patch))
            elif center_ind >= num - int(0.5 * self.points_per_patch):
                patch_point_inds = np.array(range(num - self.points_per_patch, num))
            else:
                patch_point_inds = np.array(range(center_ind - int(0.5 * self.points_per_patch), center_ind + int(0.5 * self.points_per_patch)))

            patch_pts[0:self.points_per_patch, :] = torch.from_numpy(data[patch_point_inds, 0:2])

            if self.center == 'mean':
                patch_pts[0:self.points_per_patch, :] = patch_pts[0:self.points_per_patch, :] - patch_pts.mean(0)
            elif self.center == 'point':
                patch_pts[0:self.points_per_patch, :] = patch_pts[0:self.points_per_patch, :] - torch.from_numpy(shape.pts[center_point_ind, 0:2])
            elif self.center == 'none':
                pass
            else:
                raise ValueError('Unknown patch centering option: %s' % (self.center))

        # optionally always pick the same points for a given patch index (mainly for debugging)
        if self.identical_epochs:
            self.rng.seed((self.seed + global_point_index) % (2**31))

        patch_pts_valid += list(range(self.points_per_patch))


        return patch_pts, patch_pts_valid


    def get_gt_point(self, index):
        shape_ind, patch_ind = self.shape_index(index)
        shape = self.shape_cache.get(shape_ind)
        center_point_ind = patch_ind

        return shape.pts[center_point_ind]


    # returns a patch centered at the event with the given global index
    def __getitem__(self, index):

        shape_ind, patch_ind = self.shape_index(index)

        shape = self.shape_cache.get(shape_ind)
        center_point_ind = patch_ind

        patch_pts = torch.FloatTensor(self.points_per_patch, 2).zero_()
        patch_pts_valid = []
        patch_pts, patch_pts_valid = self.select_patch_points(index, center_point_ind, shape, patch_pts_valid, patch_pts)

        if self.include_original:
            original = shape.pts[center_point_ind]
        if self.include_labels:
            tmp = []
            patch_labels = torch.LongTensor(1).zero_()
            patch_labels, _ = self.select_patch_points(index, center_point_ind, shape, tmp, patch_labels, labels=True)

        pol = torch.from_numpy(np.array(shape.pol[center_point_ind]))

        patch_feats = ()
        if self.include_labels == True:
            patch_feats = patch_feats + (patch_labels,)
        if self.include_original == True:
            patch_feats = patch_feats + (original, pol)

        return (patch_pts,) + patch_feats



    def __len__(self):
        return sum(self.shape_patch_count)


    def shape_index(self, index):
        shape_patch_offset = 0
        shape_ind = None
        for shape_ind, shape_patch_count in enumerate(self.shape_patch_count):
            if index >= shape_patch_offset and index < shape_patch_offset + shape_patch_count:
                shape_patch_ind = index - shape_patch_offset
                break
            shape_patch_offset = shape_patch_offset + shape_patch_count

        return shape_ind, shape_patch_ind

    def load_train_shape_by_index(self, shape_ind):
        point_filename = os.path.join(self.root, self.shape_names[shape_ind]+'.npy')
        labels = self.include_labels
        return load_train_shape(point_filename, labels)

    def load_test_shape_by_index(self, n):
        point_filename = os.path.join(self.root, self.shape_names[0]+'.npy')
        labels = self.include_labels
        shape_num = self.num
        return load_test_shape(n, shape_num, point_filename, labels)