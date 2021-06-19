from __future__ import division
import random
import pprint
import sys
import time
import numpy as np
import pickle
import re
import os
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
from keras_frcnn import config, data_generators
import keras_frcnn.roi_helpers as roi_helpers
from keras.utils import generic_utils
from typing import List, Set, Any
from keras_frcnn.losses import class_loss_cls, class_loss_regr, rpn_loss_cls, rpn_loss_regr

from keras_frcnn.simple_parser import get_data

sys.setrecursionlimit(40000)


class FasterRCNN:

    @staticmethod
    def train_frcnn(annotation_path: str, model_output_path: str, base_net_weights_path: str,
                    resume_weights_model_path: str or None, config_output_path: str, num_rois: int, is_vgg: bool,
                    num_epochs: int):

        if not os.path.exists(os.path.dirname(annotation_path)):
            os.makedirs(os.path.dirname(annotation_path))

        if not os.path.exists(os.path.dirname(model_output_path)):
            os.makedirs(os.path.dirname(model_output_path))

        if not os.path.exists(os.path.dirname(base_net_weights_path)):
            os.makedirs(os.path.dirname(base_net_weights_path))

        if resume_weights_model_path is not None and not os.path.exists(os.path.dirname(resume_weights_model_path)):
            os.makedirs(os.path.dirname(resume_weights_model_path))

        if not os.path.exists(os.path.dirname(config_output_path)):
            os.makedirs(os.path.dirname(config_output_path))

        C = config.Config()

        C.base_net_weights = base_net_weights_path
        C.use_horizontal_flips = False
        C.use_vertical_flips = False
        C.rot_90 = False
        C.model_path = model_output_path
        C.num_rois = num_rois
        if is_vgg:
            C.network = 'vgg'
            from keras_frcnn import vgg as nn
        else:
            from keras_frcnn import resnet as nn
            C.network = 'resnet50'

        model_path_regex = re.match("^(.+)(.hdf5)$", C.model_path)
        if model_path_regex.group(2) != '.hdf5':
            print('Output weights must have .hdf5 filetype')
            exit(1)

        # # set the path to weights based on backend and model
        # C.base_net_weights = nn.get_weight_path()

        train_imgs, classes_count, class_mapping = get_data(annotation_path)
        val_imgs, _, _ = get_data(annotation_path)

        if 'bg' not in classes_count:
            classes_count['bg'] = 0
            class_mapping['bg'] = len(class_mapping)

        C.class_mapping = class_mapping

        inv_map = {v: k for k, v in class_mapping.items()}

        print('Training images per class:')
        pprint.pprint(classes_count)
        print(f'Num classes (including bg) = {len(classes_count)}')

        with open(config_output_path, 'wb') as config_f:
            pickle.dump(C, config_f)
            print(
                f'Config has been written to {config_output_path}, and can be loaded when testing to ensure correct results')

        random.shuffle(train_imgs)

        num_imgs = len(train_imgs)

        # train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
        # val_imgs = [s for s in all_imgs if s['imageset'] == 'test']

        print('Num train samples (images) {}'.format(len(train_imgs)))
        print('Num val samples {}'.format(len(val_imgs)))

        data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, nn.get_img_output_length,
                                                       mode='train')
        data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, C, nn.get_img_output_length, mode='val')

        input_shape_img = (None, None, 3)

        img_input = Input(shape=input_shape_img)
        roi_input = Input(shape=(None, 4))

        # define the base network (resnet here, can be VGG, Inception, etc)
        shared_layers = nn.nn_base(img_input, trainable=True)

        # define the RPN, built on the base layers
        num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
        rpn = nn.rpn(shared_layers, num_anchors)

        classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count), trainable=True)

        model_rpn = Model(img_input, rpn[:2])
        model_classifier = Model([img_input, roi_input], classifier)

        # this is a model that holds both the RPN and the classifier, used to load/save weights for the models
        model_all = Model([img_input, roi_input], rpn[:2] + classifier)

        try:
            print(f'loading weights from {C.base_net_weights}')
            model_rpn.load_weights(C.base_net_weights, by_name=True)
            model_classifier.load_weights(C.base_net_weights, by_name=True)
        except:
            print('Could not load pretrained model weights. Weights can be found in the keras application folder '
                  'https://github.com/fchollet/deep-learning-models/releases/tag/v0.1')

        try:
            if resume_weights_model_path is not None and len(resume_weights_model_path) > 0:
                print(f'resuming weights from {resume_weights_model_path}')
                model_all.load_weights(resume_weights_model_path)
        except:
            print("Resuming weights failed")

        optimizer = Adam(lr=1e-5)
        optimizer_classifier = Adam(lr=1e-5)
        model_rpn.compile(optimizer=optimizer,
                          loss=[rpn_loss_cls(num_anchors), rpn_loss_regr(num_anchors)])
        model_classifier.compile(optimizer=optimizer_classifier,
                                 loss=[class_loss_cls, class_loss_regr(len(classes_count) - 1)],
                                 metrics={f'dense_class_{len(classes_count)}': 'accuracy'})
        model_all.compile(optimizer='sgd', loss='mae')

        epoch_length = 10
        num_epochs = int(num_epochs)
        iter_num = 0

        losses = np.zeros((epoch_length, 5))
        rpn_accuracy_rpn_monitor = []
        rpn_accuracy_for_epoch = []
        start_time = time.time()

        best_loss = np.Inf

        class_mapping_inv = {v: k for k, v in class_mapping.items()}
        print('Starting training')

        vis = True

        for epoch_num in range(num_epochs):

            progbar = generic_utils.Progbar(epoch_length)
            print(f'Epoch {epoch_num + 1}/{num_epochs}')

            while True:
                try:

                    if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
                        mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor)) / len(rpn_accuracy_rpn_monitor)
                        rpn_accuracy_rpn_monitor = []
                        print(
                            f'Average number of overlapping bounding boxes from RPN = {mean_overlapping_bboxes} for {epoch_length} previous iterations')
                        if mean_overlapping_bboxes == 0:
                            print(
                                'RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

                    X, Y, img_data = next(data_gen_train)

                    loss_rpn = model_rpn.train_on_batch(X, Y)

                    P_rpn = model_rpn.predict_on_batch(X)

                    R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, use_regr=True, overlap_thresh=0.7, max_boxes=300)
                    # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
                    X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping)

                    if X2 is None:
                        rpn_accuracy_rpn_monitor.append(0)
                        rpn_accuracy_for_epoch.append(0)
                        continue

                    neg_samples = np.where(Y1[0, :, -1] == 1)
                    pos_samples = np.where(Y1[0, :, -1] == 0)

                    if len(neg_samples) > 0:
                        neg_samples = neg_samples[0]
                    else:
                        neg_samples = []

                    if len(pos_samples) > 0:
                        pos_samples = pos_samples[0]
                    else:
                        pos_samples = []

                    rpn_accuracy_rpn_monitor.append(len(pos_samples))
                    rpn_accuracy_for_epoch.append((len(pos_samples)))

                    if C.num_rois > 1:
                        if len(pos_samples) < C.num_rois // 2:
                            selected_pos_samples = pos_samples.tolist()
                        else:
                            selected_pos_samples = np.random.choice(pos_samples, C.num_rois // 2,
                                                                    replace=False).tolist()
                        try:
                            selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples),
                                                                    replace=False).tolist()
                        except:
                            selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples),
                                                                    replace=True).tolist()

                        sel_samples = selected_pos_samples + selected_neg_samples
                    else:
                        # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                        selected_pos_samples = pos_samples.tolist()
                        selected_neg_samples = neg_samples.tolist()
                        if np.random.randint(0, 2):
                            sel_samples = random.choice(neg_samples)
                        else:
                            sel_samples = random.choice(pos_samples)

                    loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]],
                                                                 [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

                    losses[iter_num, 0] = loss_rpn[1]
                    losses[iter_num, 1] = loss_rpn[2]

                    losses[iter_num, 2] = loss_class[1]
                    losses[iter_num, 3] = loss_class[2]
                    losses[iter_num, 4] = loss_class[3]

                    progbar.update(iter_num + 1, [('rpn_cls', losses[iter_num, 0]), ('rpn_regr', losses[iter_num, 1]),
                                                  ('detector_cls', losses[iter_num, 2]),
                                                  ('detector_regr', losses[iter_num, 3])])

                    iter_num += 1

                    if iter_num == epoch_length:
                        loss_rpn_cls = np.mean(losses[:, 0])
                        loss_rpn_regr = np.mean(losses[:, 1])
                        loss_class_cls = np.mean(losses[:, 2])
                        loss_class_regr = np.mean(losses[:, 3])
                        class_acc = np.mean(losses[:, 4])

                        mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                        rpn_accuracy_for_epoch = []

                        if C.verbose:
                            print(
                                f'Mean number of bounding boxes from RPN overlapping ground truth boxes: {mean_overlapping_bboxes}')
                            print(f'Classifier accuracy for bounding boxes from RPN: {class_acc}')
                            print(f'Loss RPN classifier: {loss_rpn_cls}')
                            print(f'Loss RPN regression: {loss_rpn_regr}')
                            print(f'Loss Detector classifier: {loss_class_cls}')
                            print(f'Loss Detector regression: {loss_class_regr}')
                            print(f'Elapsed time: {time.time() - start_time}')

                        curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                        iter_num = 0
                        start_time = time.time()

                        if curr_loss < best_loss:
                            model_all.save_weights(
                                model_path_regex.group(1) + "_" + '{:04d}'.format(
                                    epoch_num + 1) + model_path_regex.group(
                                    2))
                            if C.verbose:
                                print(f'Total loss decreased from {best_loss} to {curr_loss}, saved weights')
                            best_loss = curr_loss

                        break

                except Exception as e:
                    print(f'Exception: {e}')
                    continue

        print('Training complete, exiting.')
