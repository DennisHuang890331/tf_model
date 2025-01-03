
import keras
import numpy as np
import tensorflow as tf


@keras.utils.register_keras_serializable()
class DecodePredictions(keras.layers.Layer):
    """
    The most simple version decoding prediction and NMS:

    >>> from keras_cv_attention_models import efficientdet, test_images
    >>> model = efficientdet.EfficientDetD0()
    >>> preds = model(model.preprocess_input(test_images.dog()))

    # Decode and NMS
    >>> from keras_cv_attention_models import coco
    >>> input_shape = model.input_shape[1:-1]
    >>> anchors = coco.get_anchors(input_shape=input_shape, pyramid_levels=[3, 7], anchor_scale=4)
    >>> dd = coco.decode_bboxes(preds[0], anchors).numpy()
    >>> rr = tf.image.non_max_suppression(dd[:, :4], dd[:, 4:].max(-1), score_threshold=0.3, max_output_size=15, iou_threshold=0.5)
    >>> dd_nms = tf.gather(dd, rr).numpy()
    >>> bboxes, labels, scores = dd_nms[:, :4], dd_nms[:, 4:].argmax(-1), dd_nms[:, 4:].max(-1)
    >>> print(f"{bboxes = }, {labels = }, {scores = }")
    >>> # bboxes = array([[0.433231  , 0.54432285, 0.8778939 , 0.8187578 ]], dtype=float32), labels = array([17]), scores = array([0.85373735], dtype=float32)
    """

    def __init__(
        self,
        input_shape=512,
        pyramid_levels=[3, 7],
        anchors_mode=None,
        use_object_scores="auto",
        anchor_scale="auto",
        aspect_ratios=(1, 2, 0.5),
        num_scales=3,
        regression_len=4,  # bbox output len, typical value is 4, for yolov8 reg_max=16 -> regression_len = 16 * 4 == 64
        score_threshold=0.3,  # decode parameter, can be set new value in `self.call`
        iou_or_sigma=0.5,  # decode parameter, can be set new value in `self.call`
        max_output_size=100,  # decode parameter, can be set new value in `self.call`
        method="hard",  # decode parameter, can be set new value in `self.call`
        mode="global",  # decode parameter, can be set new value in `self.call`
        topk=0,  # decode parameter, can be set new value in `self.call`
        use_static_output=False,  # Set to True if using this as an actual layer, especially for converting tflite
        use_sigmoid_on_score=False,  # wether applying sigmoid on score outputs. Set True if model is built using `classifier_activation=None`
        num_masks=0,  # Set > 0 value for segmentation model with masks output
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.pyramid_levels = list(range(min(pyramid_levels), max(pyramid_levels) + 1))
        self.regression_len, self.aspect_ratios, self.num_scales, self.num_masks = regression_len, aspect_ratios, num_scales, num_masks
        self.anchors_mode, self.use_object_scores, self.anchor_scale = anchors_mode, use_object_scores, anchor_scale  # num_anchors not using
        if input_shape is not None and (isinstance(input_shape, (list, tuple)) and input_shape[1] is not None):
            self.__init_anchor__(input_shape)
        else:
            self.anchors = None
        self.__input_shape__ = input_shape
        self.use_static_output, self.use_sigmoid_on_score = use_static_output, use_sigmoid_on_score
        self.nms_kwargs = {
            "score_threshold": score_threshold,
            "iou_or_sigma": iou_or_sigma,
            "max_output_size": max_output_size,
            "method": method,
            "mode": mode,
            "topk": topk,
        }
        super().build(input_shape)

    def __init_anchor__(self, input_shape):
        if isinstance(input_shape, (list, tuple)) and len(input_shape) > 2:
            # input_shape = input_shape[:2] if backend.image_data_format() == "channels_last" else input_shape[-2:]
            channel_axis, channel_dim = min(enumerate(input_shape), key=lambda xx: xx[1])  # Assume the smallest value is the channel dimension
            input_shape = [dim for axis, dim in enumerate(input_shape) if axis != channel_axis]
        elif isinstance(input_shape, int):
            input_shape = (input_shape, input_shape)

        if self.anchors_mode == "anchor_free":
            self.anchors = get_anchor_free_anchors(input_shape, self.pyramid_levels)
        elif self.anchors_mode == "yolor":
            self.anchors = get_yolor_anchors(input_shape, self.pyramid_levels)
        elif self.anchors_mode == "yolov8":
            self.anchors = get_anchor_free_anchors(input_shape, self.pyramid_levels, grid_zero_start=False)
        else:
            grid_zero_start = False
            self.anchors = get_anchors(input_shape, self.pyramid_levels, self.aspect_ratios, self.num_scales, self.anchor_scale, grid_zero_start)
        self.__input_shape__ = input_shape
        return self.anchors

    def __topk_class_boxes_single__(self, pred, topk=5000):
        # https://github.com/google/automl/tree/master/efficientdet/tf2/postprocess.py#L82
        bbox_outputs, class_outputs = pred[:, : self.regression_len], pred[:, self.regression_len :]
        num_classes = class_outputs.shape[-1]
        class_outputs_flatten = tf.reshape(class_outputs, -1)
        topk = class_outputs_flatten.shape[0] if topk == -1 else min(topk, class_outputs_flatten.shape[0])  # select all if -1
        _, class_topk_indices = tf.math.top_k(class_outputs_flatten, k=topk, sorted=False)
        # get original indices for class_outputs, original_indices_hh -> picking indices, original_indices_ww -> picked labels
        original_indices_hh, original_indices_ww = class_topk_indices // num_classes, class_topk_indices % num_classes
        class_indices = tf.stack([original_indices_hh, original_indices_ww], axis=-1)
        scores_topk = tf.gather_nd(class_outputs, class_indices)
        bboxes_topk = tf.gather(bbox_outputs, original_indices_hh)
        return bboxes_topk, scores_topk, original_indices_ww, original_indices_hh


    @staticmethod
    def nms_per_class(bbs, ccs, labels, score_threshold=0.3, iou_threshold=0.5, soft_nms_sigma=0.5, max_output_size=100):
        # From torchvision.ops.batched_nms strategy: in order to perform NMS independently per class. we add an offset to all the boxes.
        # The offset is dependent only on the class idx, and is large enough so that boxes from different classes do not overlap
        # Same result with per_class method: https://github.com/google/automl/tree/master/efficientdet/tf2/postprocess.py#L409
        cls_offset = tf.cast(labels, bbs.dtype) * (tf.reduce_max(bbs) + 1)
        bbs_per_class = bbs + tf.expand_dims(cls_offset, -1)
        indices, nms_scores = tf.image.non_max_suppression_with_scores(bbs_per_class, ccs, max_output_size, iou_threshold, score_threshold, soft_nms_sigma)
        return tf.gather(bbs, indices), tf.gather(labels, indices), nms_scores, indices

    @staticmethod
    def nms_global(bbs, ccs, labels, score_threshold=0.3, iou_threshold=0.5, soft_nms_sigma=0.5, max_output_size=100):
        indices, nms_scores = tf.image.non_max_suppression_with_scores(bbs, ccs, max_output_size, iou_threshold, score_threshold, soft_nms_sigma)
        return tf.gather(bbs, indices), tf.gather(labels, indices), nms_scores, indices

    def __object_score_split__(self, pred):
        return pred[:, :-1], pred[:, -1]  # May overwrite

    def __to_static__(self, bboxs, lables, confidences, masks=None, max_output_size=100):
        indices = tf.expand_dims(tf.range(tf.shape(bboxs)[0]), -1)
        lables = tf.cast(lables, bboxs.dtype)
        if masks is None:
            concated = tf.concat([bboxs, tf.expand_dims(lables, -1), tf.expand_dims(confidences, -1)], axis=-1)
        else:
            masks = tf.reshape(tf.cast(masks, bboxs.dtype), [-1, masks.shape[1] * masks.shape[2]])
            concated = tf.concat([bboxs, tf.expand_dims(lables, -1), tf.expand_dims(confidences, -1), masks], axis=-1)
        concated = tf.tensor_scatter_nd_update(tf.zeros([max_output_size, concated.shape[-1]], dtype=bboxs.dtype), indices, concated)
        return concated

    @staticmethod
    def process_mask_proto_single(mask_proto, masks, bboxs):
        # mask_proto: [input_height // 4, input_width // 4, 32], masks: [num, 32], bboxs: [num, 4]
        protos_height, protos_width = mask_proto.shape[:2]
        mask_proto = tf.transpose(tf.reshape(mask_proto, [-1, mask_proto.shape[-1]]), [1, 0])
        masks = tf.sigmoid(masks @ mask_proto)  # [num, protos_height * protos_width]
        masks = tf.reshape(masks, [-1, protos_height, protos_width])  # [num, protos_height, protos_width]

        """ Filter by bbox area """
        top, left, bottom, right = tf.split(bboxs[:, :, None], [1, 1, 1, 1], axis=1)  # [num, 1_pos, 1]
        height_range = tf.range(protos_height, dtype=top.dtype)[None, :, None] / protos_height  # [1, protos_height, 1]
        width_range = tf.range(protos_width, dtype=top.dtype)[None, None] / protos_width  # [1, 1, protos_width]
        height_cond = tf.logical_and(height_range >= top, height_range < bottom)  # [num, protos_height, 1]
        width_cond = tf.logical_and(width_range >= left, width_range < right)  # [num, 1, protos_width]
        masks *= tf.cast(tf.logical_and(height_cond, width_cond), masks.dtype)  # [num, protos_height, protos_width]
        return masks

    def __decode_single__(
        self, pred, mask_proto=None, score_threshold=0.3, iou_or_sigma=0.5, max_output_size=100, method="hard", mode="global", topk=0, input_shape=None
    ):
        # https://github.com/google/automl/tree/master/efficientdet/tf2/postprocess.py#L159
        pred = keras.backend.cast(pred.detach() if hasattr(pred, "detach") else pred, "float32")
        if input_shape is not None:
            self.__init_anchor__(input_shape)

        if self.num_masks > 0:  # Segmentation masks
            pred, masks = pred[:, : -self.num_masks], pred[:, -self.num_masks :]
        else:
            masks = None

        if self.use_object_scores:  # YOLO outputs: [bboxes, classses_score, object_score]
            pred, object_scores = self.__object_score_split__(pred)

        if topk != 0:
            bbs, ccs, labels, picking_indices = self.__topk_class_boxes_single__(pred, topk)
            anchors = tf.gather(self.anchors, picking_indices)
            if self.use_object_scores:
                ccs = ccs * keras.backend.gather(object_scores, picking_indices)
        else:
            bbs, scores = pred[:, : self.regression_len], pred[:, self.regression_len :]
            ccs, labels = tf.reduce_max(scores, axis=-1), tf.argmax(scores, axis=-1)
            anchors = self.anchors
            if self.use_object_scores:
                ccs = ccs * object_scores
        ccs = tf.sigmoid(ccs) if self.use_sigmoid_on_score else ccs

        # print(f"{bbs.shape = }, {anchors.shape = }")
        bbs_decoded = self.decode_bboxes(bbs, anchors, regression_len=self.regression_len)
        iou_threshold, soft_nms_sigma = (1.0, iou_or_sigma / 2) if method.lower() == "gaussian" else (iou_or_sigma, 0.0)

        if mode == "per_class":
            bboxs, lables, confidences, indices = self.nms_per_class(bbs_decoded, ccs, labels, score_threshold, iou_threshold, soft_nms_sigma, max_output_size)
        elif mode == "global":
            bboxs, lables, confidences, indices = self.nms_global(bbs_decoded, ccs, labels, score_threshold, iou_threshold, soft_nms_sigma, max_output_size)
        else:
            bboxs, lables, confidences, indices = bbs_decoded, labels, ccs, None  # Return raw decoded data for testing

        if self.num_masks > 0 and indices is not None:  # Segmentation masks
            masks = tf.gather(masks, indices)
            masks = self.process_mask_proto_single(mask_proto, masks, bboxs)

        if self.use_static_output:
            return self.__to_static__(bboxs, lables, confidences, masks, max_output_size)
        elif self.num_masks > 0:
            return bboxs, lables, confidences, masks
        else:
            return bboxs, lables, confidences
        
    def decode_bboxes(self, preds, anchors, regression_len=4, return_centers=False):
        if anchors.shape[-1] == 6:  # Currently, it's yolor / yolov7 anchors
            bboxes_center, bboxes_hw, preds_others = _yolor_decode_bboxes(preds, anchors)
        elif regression_len > 4:  # YOLOV8
            bboxes_center, bboxes_hw, preds_others = _yolov8_decode_bboxes(preds, anchors, regression_len)
        else:  # Currently, it's yolox / efficientdet anchors
            bboxes_center, bboxes_hw, preds_others = _efficientdet_decode_bboxes(preds, anchors)

        if return_centers:
            return tf.concat([bboxes_center, bboxes_hw, preds_others], axis=-1)
        else:
            preds_top_left = bboxes_center - 0.5 * bboxes_hw
            pred_bottom_right = preds_top_left + bboxes_hw
            return tf.concat([preds_top_left, pred_bottom_right, preds_others], axis=-1)
        
    

    def call(self, preds, mask_protos=None, input_shape=None, training=False, **nms_kwargs):
        """
        https://github.com/google/automl/tree/master/efficientdet/tf2/postprocess.py#L159

        mask_protos: mask output from segmentation model.
        input_shape: actual input shape if model using dynamic input shape `[None, None, 3]`.
        nms_kwargs:
          score_threshold: float value in (0, 1), min score threshold, lower output score will be excluded. Default 0.3.
          iou_or_sigma: means `soft_nms_sigma` if method is "gaussian", else `iou_threshold`. Default 0.5.
          max_output_size: max output size for `tf.image.non_max_suppression_with_scores`. Default 100.
              If use_static_output=True, fixed output shape will be `[batch, max_output_size, 6]`.
          method: "gaussian" or "hard".  Default "hard".
          mode: "global" or "per_class". "per_class" is strategy from `torchvision.ops.batched_nms`. Default "global".
          topk: Using topk highest scores, each bbox may have multi labels. Set `0` to disable, `-1` using all. Default 0.
        """
        self.nms_kwargs.update(nms_kwargs)
        if self.num_masks > 0:  # Segmentation model
            assert mask_protos is not None, "self.num_masks={} > 0, but mask_protos not provided".format(self.num_masks)

        if self.use_static_output and self.num_masks > 0:  # Segmentation model
            return tf.map_fn(lambda xx: self.__decode_single__(xx[0], xx[1], **nms_kwargs), [preds, mask_protos], fn_output_signature=preds.dtype)
        elif self.use_static_output:
            return tf.map_fn(lambda xx: self.__decode_single__(xx, **nms_kwargs), preds)
        elif len(preds.shape) == 3 and self.num_masks > 0:  # Segmentation model
            return [self.__decode_single__(pred, mask_proto, **self.nms_kwargs, input_shape=input_shape) for pred, mask_proto in zip(preds, mask_protos)]
        elif len(preds.shape) == 3:
            return [self.__decode_single__(pred, **self.nms_kwargs, input_shape=input_shape) for pred in preds]
        else:
            return self.__decode_single__(preds, mask_protos, **self.nms_kwargs, input_shape=input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_shape": self.__input_shape__,
                "pyramid_levels": self.pyramid_levels,
                "anchors_mode": self.anchors_mode,
                "use_object_scores": self.use_object_scores,
                "anchor_scale": self.anchor_scale,
                "aspect_ratios": self.aspect_ratios,
                "num_scales": self.num_scales,
                "use_static_output": self.use_static_output,
                "use_sigmoid_on_score": self.use_sigmoid_on_score,
                "num_masks": self.num_masks,
            }
        )
        config.update(self.nms_kwargs)
        return config

def _efficientdet_decode_bboxes(preds, anchors):
    preds_center, preds_hw, preds_others = tf.split(preds, [2, 2, -1], axis=-1)

    anchors_hw = anchors[:, 2:] - anchors[:, :2]
    anchors_center = (anchors[:, :2] + anchors[:, 2:]) * 0.5

    bboxes_center = preds_center * anchors_hw + anchors_center
    bboxes_hw = tf.exp(preds_hw) * anchors_hw
    return bboxes_center, bboxes_hw, preds_others


def _yolor_decode_bboxes(preds, anchors):
    preds_center, preds_hw, preds_others = tf.split(preds, [2, 2, -1], axis=-1)

    # anchors: [grid_y, grid_x, base_anchor_y, base_anchor_x, stride_y, stride_x]
    bboxes_center = preds_center * 2 * anchors[:, 4:] + anchors[:, :2]
    bboxes_hw = (preds_hw * 2) ** 2 * anchors[:, 2:4]
    return bboxes_center, bboxes_hw, preds_others


def _yolov8_decode_bboxes(preds, anchors, regression_len=64):
    preds_bbox, preds_others = tf.split(preds, [regression_len, -1], axis=-1)
    preds_bbox = tf.reshape(preds_bbox, [*preds_bbox.shape[:-1], 4, regression_len // 4])
    preds_bbox = tf.nn.softmax(preds_bbox, axis=-1) * tf.range(preds_bbox.shape[-1], dtype="float32")
    preds_bbox = tf.reduce_sum(preds_bbox, axis=-1)
    preds_top_left, preds_bottom_right = tf.split(preds_bbox, [2, 2], axis=-1)

    anchors_hw = anchors[:, 2:] - anchors[:, :2]
    anchors_center = (anchors[:, :2] + anchors[:, 2:]) * 0.5

    bboxes_center = (preds_bottom_right - preds_top_left) / 2 * anchors_hw + anchors_center
    bboxes_hw = (preds_bottom_right + preds_top_left) * anchors_hw
    return bboxes_center, bboxes_hw, preds_others

def get_anchors(input_shape=(512, 512, 3), pyramid_levels=[3, 7], aspect_ratios=[1, 2, 0.5], num_scales=3, anchor_scale=4, grid_zero_start=False):
    """
    >>> from keras_cv_attention_models.coco import anchors_func
    >>> input_shape = [512, 128]
    >>> anchors = anchors_func.get_anchors([512, 128], pyramid_levels=[7])
    >>> anchors.draw_bboxes(anchors * [512, 128, 512, 128])

    grid_zero_start: grid starts from 0, else from strides // 2. False for efficientdet anchors, True for yolo anchors.
    """
    # base anchors
    scales = [2 ** (ii / num_scales) * anchor_scale for ii in range(num_scales)]
    aspect_ratios_tensor = np.array(aspect_ratios, dtype="float32")
    if len(aspect_ratios_tensor.shape) == 1:
        # aspect_ratios = [0.5, 1, 2]
        sqrt_ratios = np.sqrt(aspect_ratios_tensor)
        ww_ratios, hh_ratios = sqrt_ratios, 1 / sqrt_ratios
    else:
        # aspect_ratios = [(1, 1), (1.4, 0.7), (0.7, 1.4)]
        ww_ratios, hh_ratios = aspect_ratios_tensor[:, 0], aspect_ratios_tensor[:, 1]
    base_anchors_hh = np.reshape(np.expand_dims(scales, 1) * np.expand_dims(hh_ratios, 0), [-1])
    base_anchors_ww = np.reshape(np.expand_dims(scales, 1) * np.expand_dims(ww_ratios, 0), [-1])
    base_anchors_hh_half, base_anchors_ww_half = base_anchors_hh / 2, base_anchors_ww / 2
    base_anchors = np.stack([base_anchors_hh_half * -1, base_anchors_ww_half * -1, base_anchors_hh_half, base_anchors_ww_half], axis=1)
    # base_anchors = tf.gather(base_anchors, [3, 6, 0, 4, 7, 1, 5, 8, 2])  # re-order according to official generated anchors
    # For anchor_free, base_anchors = np.array([[-0.5, -0.5, 0.5, 0.5]])

    # make grid
    pyramid_levels = list(range(min(pyramid_levels), max(pyramid_levels) + 1))
    feature_sizes = get_feature_sizes(input_shape, pyramid_levels)

    all_anchors = []
    for level in pyramid_levels:
        stride_hh, stride_ww = feature_sizes[0][0] / feature_sizes[level][0], feature_sizes[0][1] / feature_sizes[level][1]
        top, left = (0, 0) if grid_zero_start else (stride_hh / 2, stride_ww / 2)
        hh_centers = np.arange(top, input_shape[0], stride_hh)
        ww_centers = np.arange(left, input_shape[1], stride_ww)
        ww_grid, hh_grid = np.meshgrid(ww_centers, hh_centers)
        grid = np.reshape(np.stack([hh_grid, ww_grid, hh_grid, ww_grid], 2), [-1, 1, 4])
        anchors = np.expand_dims(base_anchors * [stride_hh, stride_ww, stride_hh, stride_ww], 0) + grid.astype(base_anchors.dtype)
        anchors = np.reshape(anchors, [-1, 4])
        all_anchors.append(anchors)
    all_anchors = np.concatenate(all_anchors, axis=0) / [input_shape[0], input_shape[1], input_shape[0], input_shape[1]]
    # if width_first:
    #      all_anchors = tf.gather(all_anchors, [1, 0, 3, 2], axis=-1)

    return tf.convert_to_tensor(all_anchors.astype("float32"))

def get_feature_sizes(input_shape, pyramid_levels=[3, 7]):
    # https://github.com/google/automl/tree/master/efficientdet/utils.py#L509
    feature_sizes = [input_shape[:2]]
    for _ in range(max(pyramid_levels)):
        pre_feat_size = feature_sizes[-1]
        feature_sizes.append(((pre_feat_size[0] - 1) // 2 + 1, (pre_feat_size[1] - 1) // 2 + 1))  # ceil mode, like padding="same" downsampling
    return feature_sizes

def get_anchors(input_shape=(512, 512, 3), pyramid_levels=[3, 7], aspect_ratios=[1, 2, 0.5], num_scales=3, anchor_scale=4, grid_zero_start=False):
    """
    >>> from keras_cv_attention_models.coco import anchors_func
    >>> input_shape = [512, 128]
    >>> anchors = anchors_func.get_anchors([512, 128], pyramid_levels=[7])
    >>> anchors.draw_bboxes(anchors * [512, 128, 512, 128])

    grid_zero_start: grid starts from 0, else from strides // 2. False for efficientdet anchors, True for yolo anchors.
    """
    # base anchors
    scales = [2 ** (ii / num_scales) * anchor_scale for ii in range(num_scales)]
    aspect_ratios_tensor = np.array(aspect_ratios, dtype="float32")
    if len(aspect_ratios_tensor.shape) == 1:
        # aspect_ratios = [0.5, 1, 2]
        sqrt_ratios = np.sqrt(aspect_ratios_tensor)
        ww_ratios, hh_ratios = sqrt_ratios, 1 / sqrt_ratios
    else:
        # aspect_ratios = [(1, 1), (1.4, 0.7), (0.7, 1.4)]
        ww_ratios, hh_ratios = aspect_ratios_tensor[:, 0], aspect_ratios_tensor[:, 1]
    base_anchors_hh = np.reshape(np.expand_dims(scales, 1) * np.expand_dims(hh_ratios, 0), [-1])
    base_anchors_ww = np.reshape(np.expand_dims(scales, 1) * np.expand_dims(ww_ratios, 0), [-1])
    base_anchors_hh_half, base_anchors_ww_half = base_anchors_hh / 2, base_anchors_ww / 2
    base_anchors = np.stack([base_anchors_hh_half * -1, base_anchors_ww_half * -1, base_anchors_hh_half, base_anchors_ww_half], axis=1)
    # base_anchors = tf.gather(base_anchors, [3, 6, 0, 4, 7, 1, 5, 8, 2])  # re-order according to official generated anchors
    # For anchor_free, base_anchors = np.array([[-0.5, -0.5, 0.5, 0.5]])

    # make grid
    pyramid_levels = list(range(min(pyramid_levels), max(pyramid_levels) + 1))
    feature_sizes = get_feature_sizes(input_shape, pyramid_levels)

    all_anchors = []
    for level in pyramid_levels:
        stride_hh, stride_ww = feature_sizes[0][0] / feature_sizes[level][0], feature_sizes[0][1] / feature_sizes[level][1]
        top, left = (0, 0) if grid_zero_start else (stride_hh / 2, stride_ww / 2)
        hh_centers = np.arange(top, input_shape[0], stride_hh)
        ww_centers = np.arange(left, input_shape[1], stride_ww)
        ww_grid, hh_grid = np.meshgrid(ww_centers, hh_centers)
        grid = np.reshape(np.stack([hh_grid, ww_grid, hh_grid, ww_grid], 2), [-1, 1, 4])
        anchors = np.expand_dims(base_anchors * [stride_hh, stride_ww, stride_hh, stride_ww], 0) + grid.astype(base_anchors.dtype)
        anchors = np.reshape(anchors, [-1, 4])
        all_anchors.append(anchors)
    all_anchors = np.concatenate(all_anchors, axis=0) / [input_shape[0], input_shape[1], input_shape[0], input_shape[1]]
    # if width_first:
    #      all_anchors = tf.gather(all_anchors, [1, 0, 3, 2], axis=-1)

    return tf.convert_to_tensor(all_anchors.astype("float32"))

def get_anchor_free_anchors(input_shape=(512, 512, 3), pyramid_levels=[3, 5], grid_zero_start=True):
    return get_anchors(input_shape, pyramid_levels, aspect_ratios=[1], num_scales=1, anchor_scale=1, grid_zero_start=grid_zero_start)


def get_yolor_anchors(input_shape=(512, 512), pyramid_levels=[3, 5], offset=0.5, is_for_training=False):
    assert max(pyramid_levels) - min(pyramid_levels) < 5
    # Original yolor using width first, height first here
    if max(pyramid_levels) - min(pyramid_levels) < 3:  # [3, 5], YOLOR_CSP / YOLOR_CSPX
        anchor_ratios = np.array([[[16.0, 12], [36, 19], [28, 40]], [[75, 36], [55, 76], [146, 72]], [[110, 142], [243, 192], [401, 459]]])
        # anchor_ratios = tf.convert_to_tensor([[[13.0, 10], [30, 16], [23, 33]], [[61, 30], [45, 62], [119, 59]], [[90, 116], [198, 156], [326, 373]]])
    elif max(pyramid_levels) - min(pyramid_levels) < 4:  # [3, 6], YOLOR_*6
        anchor_ratios = np.array(
            [[[27.0, 19], [40, 44], [94, 38]], [[68, 96], [152, 86], [137, 180]], [[301, 140], [264, 303], [542, 238]], [[615, 436], [380, 739], [792, 925]]]
        )
    else:  # [3, 7] from YOLOV4_P7, using first 3 for each level
        anchor_ratios = np.array(
            [
                [[17.0, 13], [25, 22], [66, 27]],
                [[88, 57], [69, 112], [177, 69]],
                [[138, 136], [114, 287], [275, 134]],
                [[248, 268], [504, 232], [416, 445]],
                [[393, 812], [808, 477], [908, 1070]],
            ]
        )

    pyramid_levels = list(range(min(pyramid_levels), max(pyramid_levels) + 1))
    feature_sizes = get_feature_sizes(input_shape, pyramid_levels)
    # print(f"{pyramid_levels = }, {feature_sizes = }, {anchor_ratios = }")
    if is_for_training:
        # YOLOLayer https://github.com/WongKinYiu/yolor/blob/main/models/models.py#L351
        anchor_ratios = anchor_ratios[: len(pyramid_levels)] / [[[2**ii]] for ii in pyramid_levels]
        feature_sizes = np.array(feature_sizes[min(pyramid_levels) : max(pyramid_levels) + 1], "int32")
        return tf.convert_to_tensor(anchor_ratios.astype("float32")), feature_sizes

    all_anchors = []
    for level, anchor_ratio in zip(pyramid_levels, anchor_ratios):
        stride_hh, stride_ww = feature_sizes[0][0] / feature_sizes[level][0], feature_sizes[0][1] / feature_sizes[level][1]
        # hh_grid, ww_grid = tf.meshgrid(tf.range(feature_sizes[level][0]), tf.range(feature_sizes[level][1]))
        ww_grid, hh_grid = np.meshgrid(np.arange(feature_sizes[level][1]), np.arange(feature_sizes[level][0]))
        grid = np.stack([hh_grid, ww_grid], 2).astype("float32") - offset
        grid = np.reshape(grid, [-1, 1, 2])  # [1, level_feature_sizes, 2]
        cur_base_anchors = anchor_ratio[np.newaxis, :, :]  # [num_anchors, 1, 2]

        grid_nd = np.repeat(grid, cur_base_anchors.shape[1], axis=1) * [stride_hh, stride_ww]
        cur_base_anchors_nd = np.repeat(cur_base_anchors, grid.shape[0], axis=0)
        stride_nd = np.zeros_like(grid_nd) + [stride_hh, stride_ww]
        # yield grid_nd, cur_base_anchors_nd, stride_nd
        anchors = np.concatenate([grid_nd, cur_base_anchors_nd, stride_nd], axis=-1)
        all_anchors.append(np.reshape(anchors, [-1, 6]))
    all_anchors = np.concatenate(all_anchors, axis=0) / ([input_shape[0], input_shape[1]] * 3)
    return tf.convert_to_tensor(all_anchors.astype("float32"))  # [center_h, center_w, anchor_h, anchor_w, stride_h, stride_w]