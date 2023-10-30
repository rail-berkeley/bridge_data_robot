from tqdm import tqdm
import numpy as np
import torch
import clip
from easydict import EasyDict

import collections
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import patches

from PIL import Image
from pprint import pprint
from scipy.special import softmax
# import yaml

import tensorflow as tf

import cv2
import math

import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
class ViLD:

    def __init__(self):

        FLAGS = {
        'prompt_engineering': True,
        'this_is': True,

        'temperature': 100.0,
        'use_softmax': False,
        }

        self.FLAGS = EasyDict(FLAGS)


        # Global matplotlib settings
        self.SMALL_SIZE = 16#10
        self.MEDIUM_SIZE = 18#12
        self.BIGGER_SIZE = 20#14

        plt.rc('font', size=self.MEDIUM_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=self.BIGGER_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=self.MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=self.SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=self.SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=self.MEDIUM_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=self.BIGGER_SIZE)  # fontsize of the figure title

        # Parameters for drawing figure.
        self.display_input_size = (10, 10)
        self.overall_fig_size = (18, 24)

        self.line_thickness = 2
        self.fig_size_w = 35
        # fig_size_h = min(max(5, int(len(category_names) / 2.5) ), 10)
        self.mask_color =   'red'
        self.alpha = 0.5

        self.single_template = [
            'a photo of {article} {}.'
        ]

        self.multiple_templates = [
            'There is {article} {} in the scene.',
            'There is the {} in the scene.',
            'a photo of {article} {} in the scene.',
            'a photo of the {} in the scene.',
            'a photo of one {} in the scene.',


            'itap of {article} {}.',
            'itap of my {}.',  # itap: I took a picture of
            'itap of the {}.',
            'a photo of {article} {}.',
            'a photo of my {}.',
            'a photo of the {}.',
            'a photo of one {}.',
            'a photo of many {}.',

            'a good photo of {article} {}.',
            'a good photo of the {}.',
            'a bad photo of {article} {}.',
            'a bad photo of the {}.',
            'a photo of a nice {}.',
            'a photo of the nice {}.',
            'a photo of a cool {}.',
            'a photo of the cool {}.',
            'a photo of a weird {}.',
            'a photo of the weird {}.',

            'a photo of a small {}.',
            'a photo of the small {}.',
            'a photo of a large {}.',
            'a photo of the large {}.',

            'a photo of a clean {}.',
            'a photo of the clean {}.',
            'a photo of a dirty {}.',
            'a photo of the dirty {}.',

            'a bright photo of {article} {}.',
            'a bright photo of the {}.',
            'a dark photo of {article} {}.',
            'a dark photo of the {}.',

            'a photo of a hard to see {}.',
            'a photo of the hard to see {}.',
            'a low resolution photo of {article} {}.',
            'a low resolution photo of the {}.',
            'a cropped photo of {article} {}.',
            'a cropped photo of the {}.',
            'a close-up photo of {article} {}.',
            'a close-up photo of the {}.',
            'a jpeg corrupted photo of {article} {}.',
            'a jpeg corrupted photo of the {}.',
            'a blurry photo of {article} {}.',
            'a blurry photo of the {}.',
            'a pixelated photo of {article} {}.',
            'a pixelated photo of the {}.',

            'a black and white photo of the {}.',
            'a black and white photo of {article} {}.',

            'a plastic {}.',
            'the plastic {}.',

            'a toy {}.',
            'the toy {}.',
            'a plushie {}.',
            'the plushie {}.',
            'a cartoon {}.',
            'the cartoon {}.',

            'an embroidered {}.',
            'the embroidered {}.',

            'a painting of the {}.',
            'a painting of a {}.',
        ]

        clip.available_models()
        self.model, self.preprocess = clip.load("ViT-B/32")


        self.session = tf.compat.v1.Session(graph=tf.Graph())

        saved_model_dir = './image_path_v2' #@param {type:"string"}

        _ = tf.compat.v1.saved_model.loader.load(self.session, ['serve'], saved_model_dir)

        self.numbered_categories = [{'name': str(idx), 'id': idx,} for idx in range(50)]
        self.numbered_category_indices = {cat['id']: cat for cat in self.numbered_categories}

        self.STANDARD_COLORS = [
            'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
            'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
            'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Cyan',
            'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
            'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
            'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FloralWhite',
            'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
            'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'Ivory', 'Khaki',
            'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
            'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
            'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
            'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
            'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
            'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
            'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
            'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange',
            'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
            'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
            'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
            'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
            'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
            'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
            'WhiteSmoke', 'Yellow', 'YellowGreen'
        ]

        #self.game_board_str = "square game board"
        #self.white_piece_str = "white circle"
        #self.black_piece_str = "black circle"
        #self.calibration_str = "green block"
        #self.category_name_string = ';'.join([self.calibration_str, self.white_piece_str, self.black_piece_str, self.game_board_str])
        #self.category_name_string = ';'.join([self.white_piece_str, self.black_piece_str, self.calibration_str, self.game_board_str])
        #self.category_names = ['background'] + [x.strip() for x in self.category_name_string.split(';')]
        self.max_boxes_to_draw = 25 #@param {type:"integer"}

        self.nms_threshold = 0.6 #@param {type:"slider", min:0, max:0.9, step:0.05}
        self.min_rpn_score_thresh = 0.9  #@param {type:"slider", min:0, max:1, step:0.01}
        self.min_box_area = 220 #@param {type:"slider", min:0, max:10000, step:1.0}

    def article(self, name):
        return 'an' if name[0] in 'aeiou' else 'a'

    def processed_name(self, name, rm_dot=False):
        # _ for lvis
        # / for obj365
        res = name.replace('_', ' ').replace('/', ' or ').lower()
        if rm_dot:
            res = res.rstrip('.')
        return res

    def build_text_embedding(self, categories):
        if self.FLAGS.prompt_engineering:
            templates = self.multiple_templates
        else:
            templates = self.single_template

        run_on_gpu = torch.cuda.is_available()

        with torch.no_grad():
            all_text_embeddings = []
            print('Building text embeddings...')
            # tqdm categories
            for category in tqdm(categories):
                texts = [
                    template.format(self.processed_name(category['name'], rm_dot=True),
                                    article=self.article(category['name']))
                    for template in templates]
                if self.FLAGS.this_is:
                    texts = [
                            'This is ' + text if text.startswith('a') or text.startswith('the') else text
                            for text in texts
                            ]
                texts = clip.tokenize(texts) #tokenize
                if run_on_gpu:
                    texts = texts.cuda()
                text_embeddings = self.model.encode_text(texts) #embed with text encoder
                text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
                text_embedding = text_embeddings.mean(dim=0)
                text_embedding /= text_embedding.norm()
                all_text_embeddings.append(text_embedding)
            all_text_embeddings = torch.stack(all_text_embeddings, dim=1)
            if run_on_gpu:
                all_text_embeddings = all_text_embeddings.cuda()

        return all_text_embeddings.cpu().numpy().T

    def nms(self, dets, scores, thresh, max_dets=1000):
        """Non-maximum suppression.
        Args:
            dets: [N, 4]
            scores: [N,]
            thresh: iou threshold. Float
            max_dets: int.
        """
        y1 = dets[:, 0]
        x1 = dets[:, 1]
        y2 = dets[:, 2]
        x2 = dets[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0 and len(keep) < max_dets:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h
            overlap = intersection / (areas[i] + areas[order[1:]] - intersection + 1e-12)

            inds = np.where(overlap <= thresh)[0]
            order = order[inds + 1]
        return keep

    #@title Plot instance masks
    def preprocess_categories(self, category_names):
        categories = [{'name': item, 'id': idx+1,} for idx, item in enumerate(category_names)]
        category_indices = {cat['id']: cat for cat in categories}
        fig_size_h = min(max(5, int(len(category_names) / 2.5) ), 10)
        return categories, category_indices, fig_size_h

    def main_fxn(self, image_path, category_names, params):
  #################################################################
  # Preprocessing categories and get params
        categories, category_indices, fig_size_h = self.preprocess_categories(category_names)
        max_boxes_to_draw, nms_threshold, min_rpn_score_thresh, min_box_area = params

        #################################################################
        # Obtain results and read image
        roi_boxes, roi_scores, detection_boxes, scores_unused, box_outputs, detection_masks, visual_features, image_info = self.session.run(
                ['RoiBoxes:0', 'RoiScores:0', '2ndStageBoxes:0', '2ndStageScoresUnused:0', 'BoxOutputs:0', 'MaskOutputs:0', 'VisualFeatOutputs:0', 'ImageInfo:0'],
                feed_dict={'Placeholder:0': [image_path,]})

        roi_boxes = np.squeeze(roi_boxes, axis=0)  # squeeze
        # no need to clip the boxes, already done
        roi_scores = np.squeeze(roi_scores, axis=0)

        detection_boxes = np.squeeze(detection_boxes, axis=(0, 2))
        scores_unused = np.squeeze(scores_unused, axis=0)
        box_outputs = np.squeeze(box_outputs, axis=0)
        detection_masks = np.squeeze(detection_masks, axis=0)
        visual_features = np.squeeze(visual_features, axis=0)

        image_info = np.squeeze(image_info, axis=0)  # obtain image info
        image_scale = np.tile(image_info[2:3, :], (1, 2))
        image_height = int(image_info[0, 0])
        image_width = int(image_info[0, 1])

        rescaled_detection_boxes = detection_boxes / image_scale # rescale

        # Read image
        image = np.asarray(Image.open(open(image_path, 'rb')).convert("RGB"))
        assert image_height == image.shape[0]
        assert image_width == image.shape[1]

        #################################################################
        # Filter boxes

        # Apply non-maximum suppression to detected boxes with nms threshold.
        nmsed_indices = self.nms(
            detection_boxes,
            roi_scores,
            thresh=nms_threshold
            )

        # Compute RPN box size.
        box_sizes = (rescaled_detection_boxes[:, 2] - rescaled_detection_boxes[:, 0]) * (rescaled_detection_boxes[:, 3] - rescaled_detection_boxes[:, 1])

        # Filter out invalid rois (nmsed rois)
        #dtype = int
        valid_indices = np.where(
            np.logical_and(
                np.isin(np.arange(len(roi_scores), dtype=int), nmsed_indices),
                np.logical_and(
                    np.logical_not(np.all(roi_boxes == 0., axis=-1)),
                    np.logical_and(
                    roi_scores >= min_rpn_score_thresh,
                    box_sizes > min_box_area
                    )
                )
            )
        )[0]
        #print('number of valid indices', len(valid_indices))

        detection_roi_scores = roi_scores[valid_indices][:max_boxes_to_draw, ...]
        detection_boxes = detection_boxes[valid_indices][:max_boxes_to_draw, ...]
        detection_masks = detection_masks[valid_indices][:max_boxes_to_draw, ...]
        detection_visual_feat = visual_features[valid_indices][:max_boxes_to_draw, ...]
        rescaled_detection_boxes = rescaled_detection_boxes[valid_indices][:max_boxes_to_draw, ...]

        #################################################################
        # Compute text embeddings and detection scores, and rank results
        text_features = self.build_text_embedding(categories)

        raw_scores = detection_visual_feat.dot(text_features.T)
        if self.FLAGS.use_softmax:
            scores_all = softmax(self.FLAGS.temperature * raw_scores, axis=-1)
        else:
            scores_all = raw_scores

        indices = np.argsort(-np.max(scores_all, axis=1))  # Results are ranked by scores
        indices_fg = np.array([i for i in indices if np.argmax(scores_all[i]) != 0])

        #################################################################
        # Get bounding boxes and segmentations
        ymin, xmin, ymax, xmax = np.split(rescaled_detection_boxes, 4, axis=-1)
        processed_boxes = np.concatenate([xmin, ymin, xmax - xmin, ymax - ymin], axis=-1)
        #segmentations = self.paste_instance_masks(detection_masks, processed_boxes, image_height, image_width)

        #################################################################
        # Plot detected boxes on the input image
        #self.plot_boxes_on_image(image, segmentations, rescaled_detection_boxes, detection_masks, image_height,
        #                    image_width, indices_fg, valid_indices, detection_roi_scores)

        #################################################################
        # Plot individual detections
        raw_image = np.array(image)
        n_boxes = rescaled_detection_boxes.shape[0]
        #cnt = self.plot_indiv_detections(raw_image, indices, segmentations, category_names, n_boxes, scores_all,
        #                        detection_roi_scores, rescaled_detection_boxes, fig_size_h)

        #print('Detection counts:', cnt)
        return processed_boxes, indices, scores_all, n_boxes, detection_roi_scores
        #, detection_roi_scores, rescaled_detection_boxes, segmentations, raw_image, fig_size_h

    def get_best_bbox(self, n_boxes, processed_boxes, indices, scores_all, detection_roi_scores, category, category_names):
        max_rpn_score = 0
        max_idx = -1
        for anno_idx in indices[0:int(n_boxes)]:
            scores = scores_all[anno_idx]
            top_category = category_names[np.argmax(scores)]
            if top_category == category:
                rpn_score = detection_roi_scores[anno_idx]
                if rpn_score > max_rpn_score:
                    max_rpn_score, max_idx = rpn_score, anno_idx
        
        if max_idx != -1: 
            return processed_boxes[max_idx]
        return np.array([])
    
    # Given a bounding box representing the edges of the board, return
# the "letter-number" notation of the piece centered at piece_x, piece_x
    def get_square_num(self, board_bbox, piece_x, piece_y):
        top_left_x, top_left_y, width, height = board_bbox[0], board_bbox[1], board_bbox[2], board_bbox[3]

        sq_map = {1 : "A", 2 : "B", 3 : "C"}

        col = sq_map[int(math.ceil(((piece_x - top_left_x) / (width / 3))))]
        row = int(math.ceil(((piece_y - top_left_y) / (height / 3))))

        return col + str(row)

    # Checks to see whether or not the center of the piece is within the bounds of the board's bounding box
    def is_on_board(self, board_bbox, piece_x, piece_y):
        top_left_x, top_left_y, width, height = board_bbox[0], board_bbox[1], board_bbox[2], board_bbox[3]
        return piece_x > top_left_x and piece_x < top_left_x + width and piece_y > top_left_y and piece_y < top_left_y + height

    def print_board(self, board_state):
        print("  A B C")
        for row in range(len(board_state)):
            print(row, end=" ")
            for col in range(len(board_state[row])):
                print(board_state[row][col], end=" ")
            print()

    # converts from leter-number representation to [row, col] (ex. A1 -> [0, 0])
    def letter_number_to_row_col(self, letter_number_str):
        letter = letter_number_str[0]
        number = int(letter_number_str[1])
        col = ord(letter) - ord('A')
        row = number - 1
        return row, col

    def print_board_dict(self, board_state_dict):
        board_state = [['-' for _ in range(3)] for _ in range(3)]
        for letter_number in board_state_dict.keys():
            row, col = self.letter_number_to_row_col(letter_number)
            board_state[row][col] = board_state_dict[letter_number]
        self.print_board(board_state)

    def get_board_state(self, board_state_dict):
        if not board_state_dict:
            return None 
        board_state = [[' ' for _ in range(3)] for _ in range(3)]
        for letter_number in board_state_dict.keys():
            row, col = self.letter_number_to_row_col(letter_number)
            board_state[row][col] = board_state_dict[letter_number]
        return board_state

    # gets the centroids of calibration piece
    """def get_centroids(self, image_path, category_names, game_board_str):
        params = self.max_boxes_to_draw, self.nms_threshold, self.min_rpn_score_thresh, self.min_box_area
        processed_boxes, indices, scores_all, n_boxes, detection_roi_scores = self.main_fxn(image_path, category_names, params)
        centroids = []

        for anno_idx in indices[0:int(n_boxes)]:
            scores = scores_all[anno_idx]
            top_category = category_names[np.argmax(scores)]
            print(top_category)
            bbox = processed_boxes[anno_idx]
            center_x, center_y = (2 * bbox[0] + bbox[2]) / 2, (2 * bbox[1] + bbox[3]) / 2
            if top_category == self.calibration_str:
                centroids.append([center_x, center_y])

        return centroids"""
    
    """def get_results2(self, image_path, category_names, game_board_str): 
        params = self.max_boxes_to_draw, self.nms_threshold, self.min_rpn_score_thresh, self.min_box_area
        processed_boxes, indices, scores_all, n_boxes, detection_roi_scores = self.main_fxn(image_path, category_names, params)

        num_x_off_board = 0
        num_o_off_board = 0
        centroids = dict() 

        for anno_idx in indices[0:int(n_boxes)]:
            scores = scores_all[anno_idx]
            top_category = category_names[np.argmax(scores)]
            bbox = processed_boxes[anno_idx]
            center_x, center_y = (2 * bbox[0] + bbox[2]) / 2, (2 * bbox[1] + bbox[3]) / 2
            if top_category == self.white_piece_str:
                num_x_off_board += 1
                centroids["white_circle" + str(num_x_off_board)] = np.array([center_x, center_y])
            elif top_category == self.black_piece_str:
                num_o_off_board += 1
                centroids["black_circle" + str(num_x_off_board)] = np.array([center_x, center_y])
            elif top_category == self.calibration_str:
                num_o_off_board += 1
                centroids["green_block" + str(num_x_off_board)] = np.array([center_x, center_y])

        return centroids"""


    
    def get_results(self, image_path, piece_mappings, game_board_str):
        piece_names = list(piece_mappings.keys())        
        category_names = ['background'] + piece_names + [game_board_str]
        params = self.max_boxes_to_draw, self.nms_threshold, self.min_rpn_score_thresh, self.min_box_area
        processed_boxes, indices, scores_all, n_boxes, detection_roi_scores = self.main_fxn(image_path, category_names, params)
        off_piece_counts = {key: 0 for key in piece_names}

        board_bbox = self.get_best_bbox(n_boxes, processed_boxes, indices, scores_all, detection_roi_scores, game_board_str, category_names)

        if not board_bbox.any():
            print("No board found!")

        board_state_dict = {}

        valid_centroids = dict() 

        for anno_idx in indices[0:int(n_boxes)]:
            scores = scores_all[anno_idx]
            top_category = category_names[np.argmax(scores)]
            if top_category != game_board_str and top_category != 'background':
                bbox = processed_boxes[anno_idx]
                if board_bbox.any() and (bbox[2] * bbox[3] > ((board_bbox[2])/3) * ((board_bbox[3])/3)):
                      continue
                center_x, center_y = (2 * bbox[0] + bbox[2]) / 2, (2 * bbox[1] + bbox[3]) / 2
                if board_bbox.any() and self.is_on_board(board_bbox, center_x, center_y): 
                    board_state_dict[self.get_square_num(board_bbox, center_x, center_y)] = piece_mappings[top_category]
                else: 
                    off_piece_counts[top_category] = off_piece_counts[top_category] + 1
                    valid_centroids[top_category + str(anno_idx)] = np.array([center_x, center_y])

        for x in off_piece_counts:
            print("There are " + str(off_piece_counts[x]) + " " + x + "s not on the board.")

        return self.get_board_state(board_state_dict), board_bbox, valid_centroids