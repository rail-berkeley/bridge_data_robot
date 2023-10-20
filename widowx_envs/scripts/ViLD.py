#@title Import libraries


from tqdm import tqdm
import numpy as np
import torch
import clip
from easydict import EasyDict


from matplotlib import pyplot as plt
from matplotlib import patches

import collections
import numpy as np

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

        self.game_board_str = "square game board"
        self.category_name_string = ';'.join(['black checkers piece', 'white checkers piece', self.game_board_str])
        self.category_names = ['background'] + [x.strip() for x in self.category_name_string.split(';')]
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
    

    def draw_bounding_box_on_image(self, image, ymin, xmin, ymax, xmax, color='red', thickness=4, display_str_list=(), use_normalized_coordinates=True):
        draw = ImageDraw.Draw(image)
        im_width, im_height = image.size
        if use_normalized_coordinates:
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                        ymin * im_height, ymax * im_height)
        else:
            (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
        draw.line([(left, top), (left, bottom), (right, bottom),
                    (right, top), (left, top)], width=thickness, fill=color)
        try:
            font = ImageFont.truetype('arial.ttf', 24)
        except IOError:
            font = ImageFont.load_default()

        # If the total height of the display strings added to the top of the bounding
        # box exceeds the top of the image, stack the strings below the bounding box
        # instead of above.
        # getbbox instead of getsize
        display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
        # Each display_str has a top and bottom margin of 0.05x.
        total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

        if top > total_display_str_height:
            text_bottom = top
        else:
            text_bottom = bottom + total_display_str_height
        # Reverse list and print from bottom to top.
        for display_str in display_str_list[::-1]:
            text_left = min(5, left)
            text_width, text_height = font.getsize(display_str)
            margin = np.ceil(0.05 * text_height)
            draw.rectangle(
                [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                                text_bottom)],
                fill=color)
            draw.text(
                (left + margin, text_bottom - text_height - margin),
                display_str,
                fill='black',
                font=font)
            text_bottom -= text_height - 2 * margin

    def draw_bounding_box_on_image_array(self, image, ymin, xmin, ymax, xmax, color='red', thickness=4, display_str_list=(), use_normalized_coordinates=True):
        image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
        self.draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color,
                                    thickness, display_str_list,
                                    use_normalized_coordinates)
        np.copyto(image, np.array(image_pil))
            
    
    def draw_mask_on_image_array(self, image, mask, color='red', alpha=0.4):
        if image.dtype != np.uint8:
            raise ValueError('`image` not of type np.uint8')
        if mask.dtype != np.uint8:
            raise ValueError('`mask` not of type np.uint8')
        if np.any(np.logical_and(mask != 1, mask != 0)):
            raise ValueError('`mask` elements should be in [0, 1]')
        if image.shape[:2] != mask.shape:
            raise ValueError('The image has spatial dimensions %s but the mask has '
                            'dimensions %s' % (image.shape[:2], mask.shape))
        rgb = ImageColor.getrgb(color)
        pil_image = Image.fromarray(image)

        solid_color = np.expand_dims(
            np.ones_like(mask), axis=2) * np.reshape(list(rgb), [1, 1, 3])
        pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert('RGBA')
        pil_mask = Image.fromarray(np.uint8(255.0*alpha*mask)).convert('L')
        pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
        np.copyto(image, np.array(pil_image.convert('RGB')))



    def visualize_boxes_and_labels_on_image_array(self, 
        image,
        boxes,
        classes,
        scores,
        category_index,
        instance_masks=None,
        instance_boundaries=None,
        use_normalized_coordinates=False,
        max_boxes_to_draw=20,
        min_score_thresh=.5,
        agnostic_mode=False,
        line_thickness=4,
        groundtruth_box_visualization_color='black',
        skip_scores=False,
        skip_labels=False,
        mask_alpha=0.4,
        plot_color=None,
    ):
    

        box_to_display_str_map = collections.defaultdict(list)
        box_to_color_map = collections.defaultdict(str)
        box_to_instance_masks_map = {}
        box_to_score_map = {}
        box_to_instance_boundaries_map = {}

        if not max_boxes_to_draw:
            max_boxes_to_draw = boxes.shape[0]
        for i in range(min(max_boxes_to_draw, boxes.shape[0])):
            if scores is None or scores[i] > min_score_thresh:
                box = tuple(boxes[i].tolist())
                if instance_masks is not None:
                    box_to_instance_masks_map[box] = instance_masks[i]
                if instance_boundaries is not None:
                    box_to_instance_boundaries_map[box] = instance_boundaries[i]
                if scores is None:
                    box_to_color_map[box] = groundtruth_box_visualization_color
                else:
                    display_str = ''
                    if not skip_labels:
                        if not agnostic_mode:
                            if classes[i] in list(category_index.keys()):
                                class_name = category_index[classes[i]]['name']
                            else:
                                class_name = 'N/A'
                            display_str = str(class_name)
                    if not skip_scores:
                        if not display_str:
                            display_str = '{}%'.format(int(100*scores[i]))
                        else:
                            float_score = ("%.2f" % scores[i]).lstrip('0')
                            display_str = '{}: {}'.format(display_str, float_score)
                        box_to_score_map[box] = int(100*scores[i])

                    box_to_display_str_map[box].append(display_str)
                    if plot_color is not None:
                        box_to_color_map[box] = plot_color
                    elif agnostic_mode:
                        box_to_color_map[box] = 'DarkOrange'
                    else:
                        box_to_color_map[box] = self.STANDARD_COLORS[
                        classes[i] % len(self.STANDARD_COLORS)]

        # Handle the case when box_to_score_map is empty.
        if box_to_score_map:
            box_color_iter = sorted(
                box_to_color_map.items(), key=lambda kv: box_to_score_map[kv[0]])
        else:
            box_color_iter = box_to_color_map.items()

        # Draw all boxes onto image.
        for box, color in box_color_iter:
            ymin, xmin, ymax, xmax = box
            if instance_masks is not None:
                self.draw_mask_on_image_array(
                    image,
                    box_to_instance_masks_map[box],
                    color=color,
                    alpha=mask_alpha
                )
            if instance_boundaries is not None:
                self.draw_mask_on_image_array(
                    image,
                    box_to_instance_boundaries_map[box],
                    color='red',
                    alpha=1.0
                )
            self.draw_bounding_box_on_image_array(
                image,
                ymin,
                xmin,
                ymax,
                xmax,
                color=color,
                thickness=line_thickness,
                display_str_list=box_to_display_str_map[box],
                use_normalized_coordinates=use_normalized_coordinates)

        return image


    def paste_instance_masks(self, masks,
                            detected_boxes,
                            image_height,
                            image_width):

        def expand_boxes(boxes, scale):
            """Expands an array of boxes by a given scale."""
            # Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/boxes.py#L227  # pylint: disable=line-too-long
            # The `boxes` in the reference implementation is in [x1, y1, x2, y2] form,
            # whereas `boxes` here is in [x1, y1, w, h] form
            w_half = boxes[:, 2] * .5
            h_half = boxes[:, 3] * .5
            x_c = boxes[:, 0] + w_half
            y_c = boxes[:, 1] + h_half

            w_half *= scale
            h_half *= scale

            boxes_exp = np.zeros(boxes.shape)
            boxes_exp[:, 0] = x_c - w_half
            boxes_exp[:, 2] = x_c + w_half
            boxes_exp[:, 1] = y_c - h_half
            boxes_exp[:, 3] = y_c + h_half

            return boxes_exp

    # Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/core/test.py#L812  # pylint: disable=line-too-long
    # To work around an issue with cv2.resize (it seems to automatically pad
    # with repeated border values), we manually zero-pad the masks by 1 pixel
    # prior to resizing back to the original image resolution. This prevents
    # "top hat" artifacts. We therefore need to expand the reference boxes by an
    # appropriate factor.
        _, mask_height, mask_width = masks.shape
        scale = max((mask_width + 2.0) / mask_width,
                    (mask_height + 2.0) / mask_height)

        ref_boxes = expand_boxes(detected_boxes, scale)
        ref_boxes = ref_boxes.astype(np.int32)
        padded_mask = np.zeros((mask_height + 2, mask_width + 2), dtype=np.float32)
        segms = []
        for mask_ind, mask in enumerate(masks):
            im_mask = np.zeros((image_height, image_width), dtype=np.uint8)
            # Process mask inside bounding boxes.
            padded_mask[1:-1, 1:-1] = mask[:, :]

            ref_box = ref_boxes[mask_ind, :]
            w = ref_box[2] - ref_box[0] + 1
            h = ref_box[3] - ref_box[1] + 1
            w = np.maximum(w, 1)
            h = np.maximum(h, 1)

            mask = cv2.resize(padded_mask, (w, h))
            mask = np.array(mask > 0.5, dtype=np.uint8)

            x_0 = min(max(ref_box[0], 0), image_width)
            x_1 = min(max(ref_box[2] + 1, 0), image_width)
            y_0 = min(max(ref_box[1], 0), image_height)
            y_1 = min(max(ref_box[3] + 1, 0), image_height)

            im_mask[y_0:y_1, x_0:x_1] = mask[
                (y_0 - ref_box[1]):(y_1 - ref_box[1]),
                (x_0 - ref_box[0]):(x_1 - ref_box[0])
            ]
            segms.append(im_mask)

        segms = np.array(segms)
        assert masks.shape[0] == segms.shape[0]
        return segms
    

    def show_points(self, coords, labels, ax, marker_size=375):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

    def show_points_on_image(self, raw_image, input_points, input_labels=None):
        plt.imshow(raw_image)
        input_points = np.array(input_points)
        if input_labels is None:
            labels = np.ones_like(input_points[:, 0])
        else:
            labels = np.array(input_labels)
        self.show_points(input_points, labels, plt.gca())
        plt.axis('on')
        plt.show()

    #@title Plot instance masks
    def plot_mask(self, color, alpha, original_image, mask):
        rgb = ImageColor.getrgb(color)
        pil_image = Image.fromarray(original_image)

        solid_color = np.expand_dims(
            np.ones_like(mask), axis=2) * np.reshape(list(rgb), [1, 1, 3])
        pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert('RGBA')
        pil_mask = Image.fromarray(np.uint8(255.0*alpha*mask)).convert('L')
        pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
        img_w_mask = np.array(pil_image.convert('RGB'))
        return img_w_mask

    def display_image(self, path_or_array, size=(10, 10)):
        if isinstance(path_or_array, str):
            image = np.asarray(Image.open(open(path_or_array, 'rb')).convert("RGB"))
        else:
            image = path_or_array

        plt.figure(figsize=size)
        plt.imshow(image)
        plt.axis('off')
        plt.show()

    def preprocess_categories(self, category_names):
        categories = [{'name': item, 'id': idx+1,} for idx, item in enumerate(category_names)]
        category_indices = {cat['id']: cat for cat in categories}
        fig_size_h = min(max(5, int(len(category_names) / 2.5) ), 10)
        return categories, category_indices, fig_size_h
    

    # Plot detected boxes on the input image
    def plot_boxes_on_image(self, image, segmentations, rescaled_detection_boxes, detection_masks, image_height,
                            image_width, indices_fg, valid_indices, detection_roi_scores):

        if len(indices_fg) == 0:
            self.display_image(np.array(image), size=self.overall_fig_size)
            print('ViLD does not detect anything belong to the given category')

        else:
            image_with_detections = self.visualize_boxes_and_labels_on_image_array(
                np.array(image),
                rescaled_detection_boxes[indices_fg],
                valid_indices[:self.max_boxes_to_draw][indices_fg],
                detection_roi_scores[indices_fg],
                self.numbered_category_indices,
                instance_masks=segmentations[indices_fg],
                use_normalized_coordinates=False,
                max_boxes_to_draw=self.max_boxes_to_draw,
                min_score_thresh=self.min_rpn_score_thresh,
                skip_scores=False,
                skip_labels=True)

            plt.figure(figsize=self.overall_fig_size)
            plt.imshow(image_with_detections)
            plt.axis('off')
            plt.title('Detected objects and RPN scores')
            plt.show()


    def plot_indiv_detections(self, raw_image, indices, segmentations, category_names, n_boxes, scores_all,
                          detection_roi_scores, rescaled_detection_boxes, fig_size_h):
        cnt = 0
        for anno_idx in indices[0:int(n_boxes)]:
            rpn_score = detection_roi_scores[anno_idx]
            bbox = rescaled_detection_boxes[anno_idx]
            scores = scores_all[anno_idx]
            if np.argmax(scores) == 0:
                continue

            y1, x1, y2, x2 = int(np.floor(bbox[0])), int(np.floor(bbox[1])), int(np.ceil(bbox[2])), int(np.ceil(bbox[3]))
            img_w_mask = self.plot_mask(self.mask_color, self.alpha, raw_image, segmentations[anno_idx])
            crop_w_mask = img_w_mask[y1:y2, x1:x2, :]


            fig, axs = plt.subplots(1, 4, figsize=(self.fig_size_w, fig_size_h), gridspec_kw={'width_ratios': [3, 1, 1, 2]}, constrained_layout=True)

            # Draw bounding box.
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=self.line_thickness, edgecolor='r', facecolor='none')
            axs[0].add_patch(rect)

            axs[0].set_xticks([])
            axs[0].set_yticks([])
            axs[0].set_title(f'bbox: {y1, x1, y2, x2} area: {(y2 - y1) * (x2 - x1)} rpn score: {rpn_score:.4f}')
            axs[0].imshow(raw_image)

            # Draw image in a cropped region.
            crop = np.copy(raw_image[y1:y2, x1:x2, :])
            axs[1].set_xticks([])
            axs[1].set_yticks([])

            axs[1].set_title(f'predicted: {category_names[np.argmax(scores)]}')
            axs[1].imshow(crop)

            # Draw segmentation inside a cropped region.
            axs[2].set_xticks([])
            axs[2].set_yticks([])
            axs[2].set_title('mask')
            axs[2].imshow(crop_w_mask)

            # Draw category scores.
            fontsize = max(min(fig_size_h / float(len(category_names)) * 45, 20), 8)
            for cat_idx in range(len(category_names)):
                axs[3].barh(cat_idx, scores[cat_idx],
                        color='orange' if scores[cat_idx] == max(scores) else 'blue')
            axs[3].invert_yaxis()
            axs[3].set_axisbelow(True)
            axs[3].set_xlim(0, 1)
            plt.xlabel("confidence score")
            axs[3].set_yticks(range(len(category_names)))
            axs[3].set_yticklabels(category_names, fontdict={
                'fontsize': fontsize})

            cnt += 1
            # fig.tight_layout()
        return cnt
    

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
        print('number of valid indices', len(valid_indices))

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
        segmentations = self.paste_instance_masks(detection_masks, processed_boxes, image_height, image_width)

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
        return processed_boxes, indices, scores_all, n_boxes
        #, detection_roi_scores, rescaled_detection_boxes, segmentations, raw_image, fig_size_h
    


    def get_largest_bbox(self, n_boxes, processed_boxes, indices, scores_all, category):
        max_area = 0
        max_idx = -1
        for anno_idx in indices[0:int(n_boxes)]:
            scores = scores_all[anno_idx]
            top_category = self.category_names[np.argmax(scores)]
            if top_category == category:
                bbox = processed_boxes[anno_idx]
                area = bbox[2] * bbox[3]
                if area > max_area:
                    max_area, max_idx = area, anno_idx
        return processed_boxes[max_idx]
    

    # Given a bounding box representing the edges of the board, return
# the "letter-number" notation of the piece centered at piece_x, piece_x
    def get_square_num(self, board_bbox, piece_x, piece_y):
        top_left_x, top_left_y, width, height = board_bbox[0], board_bbox[1], board_bbox[2], board_bbox[3]

        sq_map = {1 : "A", 2 : "B", 3 : "C"}

        col = sq_map[int(math.ceil(((piece_x - top_left_x) / (width / 3))))]
        row = int(math.ceil(((piece_y - top_left_y) / (height / 3))))

        return col + str(row)
    
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

    
    def get_centroids(self, image_path, category_names, game_board_str): 
        # let's assume that white pieces are Xs and black pieces are Os
        params = self.max_boxes_to_draw, self.nms_threshold, self.min_rpn_score_thresh, self.min_box_area
        processed_boxes, indices, scores_all, n_boxes = self.main_fxn(image_path, category_names, params)

        board_state = [['-' for _ in range(3)] for _ in range(3)]
        raw_image = np.asarray(Image.open(open(image_path, 'rb')).convert("RGB"))
        num_x_off_board = 0
        num_o_off_board = 0
        board_bbox = self.get_largest_bbox(n_boxes, processed_boxes, indices, scores_all, game_board_str)
        board_state_dict = {}

        centroids = dict()  

        for anno_idx in indices[0:int(n_boxes)]:
            scores = scores_all[anno_idx]
            top_category = category_names[np.argmax(scores)]
            if top_category != game_board_str:
                bbox = processed_boxes[anno_idx]
                center_x, center_y = (2 * bbox[0] + bbox[2]) / 2, (2 * bbox[1] + bbox[3]) / 2
                if top_category == "white checkers piece":
                    if self.is_on_board(board_bbox, center_x, center_y):
                        board_state_dict[self.get_square_num(board_bbox, center_x, center_y)] = 'X'
                    # show_points_on_image(raw_image, [[center_x, center_y]])
                        #centroids[top_category + str(anno_idx)] = [center_x, center_y] 
                    else:
                        centroids[top_category + str(num_x_off_board)] = [center_x, center_y] 
                        num_x_off_board += 1
                elif top_category == "black checkers piece":
                    if self.is_on_board(board_bbox, center_x, center_y):
                        board_state_dict[self.get_square_num(board_bbox, center_x, center_y)] = 'O'
                        #centroids[top_category + str(anno_idx)] = [center_x, center_y] 
                    else:
                        centroids[top_category + str(num_o_off_board)] = [center_x, center_y] 
                        num_o_off_board += 1

        print(board_state_dict)
        self.print_board_dict(board_state_dict)
        print(f"There are {num_x_off_board} white pieces (Xs) and {num_o_off_board} black pieces (Os) not on the board.")

        print("Centroids:", centroids)

        return board_bbox, centroids 