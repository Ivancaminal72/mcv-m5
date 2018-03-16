import numpy as np
from keras import backend as K
dim_ordering = K.image_dim_ordering()
import tensorflow as tf
if dim_ordering == 'th':
    import theano
    from theano import tensor as T
else:
    from tensorflow.python.framework import ops

def cce_flatt(void_class, weights_class):
    def categorical_crossentropy_flatt(y_true, y_pred):
        '''Expects a binary class matrix instead of a vector of scalar classes.
        '''
        if dim_ordering == 'th':
            y_pred = K.permute_dimensions(y_pred, (0, 2, 3, 1))
        shp_y_pred = K.shape(y_pred)
        y_pred = K.reshape(y_pred, (shp_y_pred[0]*shp_y_pred[1]*shp_y_pred[2],
                           shp_y_pred[3]))  # go back to b01,c
        # shp_y_true = K.shape(y_true)

        if dim_ordering == 'th':
            y_true = K.cast(K.flatten(y_true), 'int32')  # b,01 -> b01
        else:
            y_true = K.cast(K.flatten(y_true), 'int32')  # b,01 -> b01

        # remove void classes from cross_entropy
        if len(void_class):
            for i in range(len(void_class)):
                # get idx of non void classes and remove void classes
                # from y_true and y_pred
                idxs = K.not_equal(y_true, void_class[i])
                if dim_ordering == 'th':
                    idxs = idxs.nonzero()
                    y_pred = y_pred[idxs]
                    y_true = y_true[idxs]
                else:
                    y_pred = tf.boolean_mask(y_pred, idxs)
                    y_true = tf.boolean_mask(y_true, idxs)

        if dim_ordering == 'th':
            y_true = T.extra_ops.to_one_hot(y_true, nb_class=y_pred.shape[-1])
        else:
            y_true = tf.one_hot(y_true, K.shape(y_pred)[-1], on_value=1, off_value=0, axis=None, dtype=None, name=None)
            y_true = K.cast(y_true, 'float32')  # b,01 -> b01
        out = K.categorical_crossentropy(y_pred, y_true)

        # Class balancing
        if weights_class is not None:
            weights_class_var = K.variable(value=weights_class)
            class_balance_w = weights_class_var[y_true].astype(K.floatx())
            out = out * class_balance_w

        return K.mean(out)  # b01 -> b,01
    return categorical_crossentropy_flatt


def IoU(n_classes, void_labels):
    def IoU_flatt(y_true, y_pred):
        '''Expects a binary class matrix instead of a vector of scalar classes.
        '''
        if dim_ordering == 'th':
            y_pred = K.permute_dimensions(y_pred, (0, 2, 3, 1))
        shp_y_pred = K.shape(y_pred)
        y_pred = K.reshape(y_pred, (shp_y_pred[0]*shp_y_pred[1]*shp_y_pred[2],
                           shp_y_pred[3]))  # go back to b01,c
        # shp_y_true = K.shape(y_true)
        y_true = K.cast(K.flatten(y_true), 'int32')  # b,01 -> b01
        y_pred = K.argmax(y_pred, axis=-1)

        # We use not_void in case the prediction falls in the void class of
        # the groundtruth
        for i in range(len(void_labels)):
            if i == 0:
                not_void = K.not_equal(y_true, void_labels[i])
            else:
                not_void = not_void * K.not_equal(y_true, void_labels[i])

        sum_I = K.zeros((1,), dtype='float32')

        out = {}
        for i in range(n_classes):
            y_true_i = K.equal(y_true, i)
            y_pred_i = K.equal(y_pred, i)

            if dim_ordering == 'th':
                I_i = K.sum(y_true_i * y_pred_i)
                U_i = K.sum(T.or_(y_true_i, y_pred_i) * not_void)
                # I = T.set_subtensor(I[i], I_i)
                # U = T.set_subtensor(U[i], U_i)
                sum_I = sum_I + I_i
            else:
                U_i = K.sum(K.cast(tf.logical_and(tf.logical_or(y_true_i, y_pred_i), not_void), 'float32'))
                y_true_i = K.cast(y_true_i, 'float32')
                y_pred_i = K.cast(y_pred_i, 'float32')
                I_i = K.sum(y_true_i * y_pred_i)
                sum_I = sum_I + I_i
            out['I'+str(i)] = I_i
            out['U'+str(i)] = U_i

        if dim_ordering == 'th':
            accuracy = K.sum(sum_I) / K.sum(not_void)
        else:
            accuracy = K.sum(sum_I) / tf.reduce_sum(tf.cast(not_void, 'float32'))
        out['acc'] = accuracy
        return out
    return IoU_flatt



"""
    YOLO loss function
    code adapted from https://github.com/thtrieu/darkflow/
"""

def YOLOLoss(input_shape=(3,640,640),num_classes=45,priors=[[0.25,0.25], [0.5,0.5], [1.0,1.0], [1.7,1.7], [2.5,2.5]],max_truth_boxes=30,thresh=0.6,object_scale=5.0,noobject_scale=1.0,coord_scale=1.0,class_scale=1.0):

  # Def custom loss function using numpy
  def _YOLOLoss(y_true, y_pred, name=None, priors=priors):

      net_out = tf.transpose(y_pred, perm=[0, 2, 3, 1])

      _,h,w,c = net_out.get_shape().as_list()
      b = len(priors)
      anchors = np.array(priors)

      _probs, _confs, _coord, _areas, _upleft, _botright = tf.split(y_true, [num_classes,1,4,1,2,2], axis=3)
      _confs = tf.squeeze(_confs,3)
      _areas = tf.squeeze(_areas,3)

      net_out_reshape = tf.reshape(net_out, [-1, h, w, b, (4 + 1 + num_classes)])
      # Extract the coordinate prediction from net.out
      coords = net_out_reshape[:, :, :, :, :4]
      coords = tf.reshape(coords, [-1, h*w, b, 4])
      adjusted_coords_xy = tf.sigmoid(coords[:,:,:,0:2])
      adjusted_coords_wh = tf.exp(tf.clip_by_value(coords[:,:,:,2:4], -1e3, 10)) * np.reshape(anchors, [1, 1, b, 2]) / np.reshape([w, h], [1, 1, 1, 2])
      adjusted_coords_wh = tf.sqrt(tf.clip_by_value(adjusted_coords_wh, 1e-10, 1e5))
      coords = tf.concat([adjusted_coords_xy, adjusted_coords_wh], 3)

      adjusted_c = tf.sigmoid(net_out_reshape[:, :, :, :, 4])
      adjusted_c = tf.reshape(adjusted_c, [-1, h*w, b, 1])

      adjusted_prob = tf.nn.softmax(net_out_reshape[:, :, :, :, 5:])
      adjusted_prob = tf.reshape(adjusted_prob, [-1, h*w, b, num_classes])

      adjusted_net_out = tf.concat([adjusted_coords_xy, adjusted_coords_wh, adjusted_c, adjusted_prob], 3)

      wh = tf.square(coords[:,:,:,2:4]) *  np.reshape([w, h], [1, 1, 1, 2])
      area_pred = wh[:,:,:,0] * wh[:,:,:,1]
      centers = coords[:,:,:,0:2]
      floor = centers - (wh * .5)
      ceil  = centers + (wh * .5)

      # calculate the intersection areas
      intersect_upleft   = tf.maximum(floor, _upleft)
      intersect_botright = tf.minimum(ceil, _botright)
      intersect_wh = intersect_botright - intersect_upleft
      intersect_wh = tf.maximum(intersect_wh, 0.0)
      intersect = tf.multiply(intersect_wh[:,:,:,0], intersect_wh[:,:,:,1])

      # calculate the best IOU, set 0.0 confidence for worse boxes
      iou = tf.truediv(intersect, _areas + area_pred - intersect)
      best_box = tf.equal(iou, tf.reduce_max(iou, [2], True))
      #best_box = tf.grater_than(iou, 0.5) # LLUIS ???
      best_box = tf.to_float(best_box)
      confs = tf.multiply(best_box, _confs)

      # take care of the weight terms
      conid = noobject_scale * (1. - confs) + object_scale * confs
      weight_coo = tf.concat(4 * [tf.expand_dims(confs, -1)], 3)
      cooid = coord_scale * weight_coo
      weight_pro = tf.concat(num_classes * [tf.expand_dims(confs, -1)], 3)
      proid = class_scale * weight_pro

      true = tf.concat([_coord, tf.expand_dims(confs, 3), _probs ], 3)
      wght = tf.concat([cooid, tf.expand_dims(conid, 3), proid ], 3)

      loss = tf.square(adjusted_net_out - true)
      loss = tf.multiply(loss, wght)
      loss = tf.reshape(loss, [-1, h*w*b*(4 + 1 + num_classes)])
      loss = tf.reduce_sum(loss, 1)

      return .5*tf.reduce_mean(loss)

  return _YOLOLoss


"""
    YOLO detection metrics
    code adapted from https://github.com/thtrieu/darkflow/
"""
def YOLOMetrics(input_shape=(3,640,640),num_classes=45,priors=[[0.25,0.25], [0.5,0.5], [1.0,1.0], [1.7,1.7], [2.5,2.5]],max_truth_boxes=30,thresh=0.6,nms_thresh=0.3,name='avg_iou'):

  def avg_recall(y_true, y_pred, name=None):
      net_out = tf.transpose(y_pred, perm=[0, 2, 3, 1])

      _,h,w,c = net_out.get_shape().as_list()
      b = len(priors)
      anchors = np.array(priors)

      _probs, _confs, _coord, _areas, _upleft, _botright = tf.split(y_true, [num_classes,1,4,1,2,2], axis=3)
      _confs = tf.squeeze(_confs,3)
      _areas = tf.squeeze(_areas,3)

      net_out_reshape = tf.reshape(net_out, [-1, h, w, b, (4 + 1 + num_classes)])
      # Extract the coordinate prediction from net.out
      coords = net_out_reshape[:, :, :, :, :4]
      coords = tf.reshape(coords, [-1, h*w, b, 4])
      adjusted_coords_xy = tf.sigmoid(coords[:,:,:,0:2])
      adjusted_coords_wh = tf.exp(tf.clip_by_value(coords[:,:,:,2:4], -1e3, 10)) * np.reshape(anchors, [1, 1, b, 2]) / np.reshape([w, h], [1, 1, 1, 2])
      adjusted_coords_wh = tf.sqrt(tf.clip_by_value(adjusted_coords_wh, 1e-10, 1e5))
      coords = tf.concat([adjusted_coords_xy, adjusted_coords_wh], 3)

      adjusted_c = tf.sigmoid(net_out_reshape[:, :, :, :, 4])
      adjusted_c = tf.reshape(adjusted_c, [-1, h*w, b, 1])

      adjusted_prob = tf.nn.softmax(net_out_reshape[:, :, :, :, 5:])
      adjusted_prob = tf.reshape(adjusted_prob, [-1, h*w, b, num_classes])

      adjusted_net_out = tf.concat([adjusted_coords_xy, adjusted_coords_wh, adjusted_c, adjusted_prob], 3)

      wh = tf.square(coords[:,:,:,2:4]) *  np.reshape([w, h], [1, 1, 1, 2])
      area_pred = wh[:,:,:,0] * wh[:,:,:,1]
      centers = coords[:,:,:,0:2]
      floor = centers - (wh * .5)
      ceil  = centers + (wh * .5)

      # calculate the intersection areas
      intersect_upleft   = tf.maximum(floor, _upleft)
      intersect_botright = tf.minimum(ceil, _botright)
      intersect_wh = intersect_botright - intersect_upleft
      intersect_wh = tf.maximum(intersect_wh, 0.0)
      intersect = tf.multiply(intersect_wh[:,:,:,0], intersect_wh[:,:,:,1])

      # calculate the best IOU and metrics 
      iou = tf.truediv(intersect, _areas + area_pred - intersect)
      best_ious     = tf.reduce_max(iou, [2], True)
      recall        = tf.reduce_sum(tf.to_float(tf.greater(best_ious,0.5)), [1])
      gt_obj_areas  = tf.reduce_mean(_areas, [2], True)
      num_gt_obj    = tf.reduce_sum(tf.to_float(tf.greater(gt_obj_areas,tf.zeros_like(gt_obj_areas))), [1])
      avg_recall    = tf.truediv(recall, num_gt_obj)
 
      return tf.reduce_mean(avg_recall)

  def avg_iou(y_true, y_pred, name=None):
      net_out = tf.transpose(y_pred, perm=[0, 2, 3, 1])

      _,h,w,c = net_out.get_shape().as_list()
      b = len(priors)
      anchors = np.array(priors)

      _probs, _confs, _coord, _areas, _upleft, _botright = tf.split(y_true, [num_classes,1,4,1,2,2], axis=3)
      _confs = tf.squeeze(_confs,3)
      _areas = tf.squeeze(_areas,3)

      net_out_reshape = tf.reshape(net_out, [-1, h, w, b, (4 + 1 + num_classes)])
      # Extract the coordinate prediction from net.out
      coords = net_out_reshape[:, :, :, :, :4]
      coords = tf.reshape(coords, [-1, h*w, b, 4])
      adjusted_coords_xy = tf.sigmoid(coords[:,:,:,0:2])
      adjusted_coords_wh = tf.exp(tf.clip_by_value(coords[:,:,:,2:4], -1e3, 10)) * np.reshape(anchors, [1, 1, b, 2]) / np.reshape([w, h], [1, 1, 1, 2])
      adjusted_coords_wh = tf.sqrt(tf.clip_by_value(adjusted_coords_wh, 1e-10, 1e5))
      coords = tf.concat([adjusted_coords_xy, adjusted_coords_wh], 3)

      adjusted_c = tf.sigmoid(net_out_reshape[:, :, :, :, 4])
      adjusted_c = tf.reshape(adjusted_c, [-1, h*w, b, 1])

      adjusted_prob = tf.nn.softmax(net_out_reshape[:, :, :, :, 5:])
      adjusted_prob = tf.reshape(adjusted_prob, [-1, h*w, b, num_classes])

      adjusted_net_out = tf.concat([adjusted_coords_xy, adjusted_coords_wh, adjusted_c, adjusted_prob], 3)

      wh = tf.square(coords[:,:,:,2:4]) *  np.reshape([w, h], [1, 1, 1, 2])
      area_pred = wh[:,:,:,0] * wh[:,:,:,1]
      centers = coords[:,:,:,0:2]
      floor = centers - (wh * .5)
      ceil  = centers + (wh * .5)

      # calculate the intersection areas
      intersect_upleft   = tf.maximum(floor, _upleft)
      intersect_botright = tf.minimum(ceil, _botright)
      intersect_wh = intersect_botright - intersect_upleft
      intersect_wh = tf.maximum(intersect_wh, 0.0)
      intersect = tf.multiply(intersect_wh[:,:,:,0], intersect_wh[:,:,:,1])

      # calculate the best IOU and metrics 
      iou = tf.truediv(intersect, _areas + area_pred - intersect)
      best_ious     = tf.reduce_max(iou, [2], True)
      sum_best_ious = tf.reduce_sum(best_ious, [1])
      gt_obj_areas  = tf.reduce_mean(_areas, [2], True)
      num_gt_obj    = tf.reduce_sum(tf.to_float(tf.greater(gt_obj_areas,tf.zeros_like(gt_obj_areas))), [1])
      avg_iou       = tf.truediv(sum_best_ious, num_gt_obj)
 
      return tf.reduce_mean(avg_iou)

  if name=='avg_iou':
    return avg_iou
  else:
    return avg_recall

"""
  SSD loss function
  code adapted from https://github.com/thtrieu/darkflow/
"""

class MultiboxLoss(object):
    """Multibox loss with some helper functions.

    # Arguments
        num_classes: Number of classes including background.
        alpha: Weight of L1-smooth loss.
        neg_pos_ratio: Max ratio of negative to positive boxes in loss.
        background_label_id: Id of background label.
        negatives_for_hard: Number of negative boxes to consider
            it there is no positive boxes in batch.

    # References
        https://arxiv.org/abs/1512.02325

    # TODO
        Add possibility for background label id be not zero
    """
    def __init__(self, num_classes, alpha=1.0, neg_pos_ratio=3.0,
                 background_label_id=0, negatives_for_hard=100.0):
        self.num_classes = num_classes
        self.alpha = alpha
        self.neg_pos_ratio = neg_pos_ratio
        if background_label_id != 0:
            raise Exception('Only 0 as background label id is supported')
        self.background_label_id = background_label_id
        self.negatives_for_hard = negatives_for_hard

    def _l1_smooth_loss(self, y_true, y_pred):
        """Compute L1-smooth loss.

        # Arguments
            y_true: Ground truth bounding boxes,
                tensor of shape (?, num_boxes, 4).
            y_pred: Predicted bounding boxes,
                tensor of shape (?, num_boxes, 4).

        # Returns
            l1_loss: L1-smooth loss, tensor of shape (?, num_boxes).

        # References
            https://arxiv.org/abs/1504.08083
        """
        abs_loss = tf.abs(y_true - y_pred)
        sq_loss = 0.5 * (y_true - y_pred)**2
        l1_loss = tf.where(tf.less(abs_loss, 1.0), sq_loss, abs_loss - 0.5)
        return tf.reduce_sum(l1_loss, -1)

    def _softmax_loss(self, y_true, y_pred):
        """Compute softmax loss.

        # Arguments
            y_true: Ground truth targets,
                tensor of shape (?, num_boxes, num_classes).
            y_pred: Predicted logits,
                tensor of shape (?, num_boxes, num_classes).

        # Returns
            softmax_loss: Softmax loss, tensor of shape (?, num_boxes).
        """
        y_pred = tf.maximum(tf.minimum(y_pred, 1 - 1e-15), 1e-15)
        softmax_loss = -tf.reduce_sum(y_true * tf.log(y_pred),
                                      axis=-1)
        return softmax_loss

    def compute_loss(self, y_true, y_pred):
        """Compute mutlibox loss.

        # Arguments
            y_true: Ground truth targets,
                tensor of shape (?, num_boxes, 4 + num_classes + 8),
                priors in ground truth are fictitious,
                y_true[:, :, -8] has 1 if prior should be penalized
                    or in other words is assigned to some ground truth box,
                y_true[:, :, -7:] are all 0.
            y_pred: Predicted logits,
                tensor of shape (?, num_boxes, 4 + num_classes + 8).

        # Returns
            loss: Loss for prediction, tensor of shape (?,).
        """
        batch_size = tf.shape(y_true)[0]
        num_boxes = tf.to_float(tf.shape(y_true)[1])

        # loss for all priors
        conf_loss = self._softmax_loss(y_true[:, :, 4:-8],
                                       y_pred[:, :, 4:-8])
        loc_loss = self._l1_smooth_loss(y_true[:, :, :4],
                                        y_pred[:, :, :4])

        # get positives loss
        num_pos = tf.reduce_sum(y_true[:, :, -8], axis=-1)
        pos_loc_loss = tf.reduce_sum(loc_loss * y_true[:, :, -8],
                                     axis=1)
        pos_conf_loss = tf.reduce_sum(conf_loss * y_true[:, :, -8],
                                      axis=1)

        # get negatives loss, we penalize only confidence here
        num_neg = tf.minimum(self.neg_pos_ratio * num_pos,
                             num_boxes - num_pos)
        pos_num_neg_mask = tf.greater(num_neg, 0)
        has_min = tf.to_float(tf.reduce_any(pos_num_neg_mask))
        num_neg = tf.concat(axis=0, values=[num_neg,
                                [(1 - has_min) * self.negatives_for_hard]])
        num_neg_batch = tf.reduce_min(tf.boolean_mask(num_neg,
                                                      tf.greater(num_neg, 0)))
        num_neg_batch = tf.to_int32(num_neg_batch)
        confs_start = 4 + self.background_label_id + 1
        confs_end = confs_start + self.num_classes - 1
        max_confs = tf.reduce_max(y_pred[:, :, confs_start:confs_end],
                                  axis=2)
        _, indices = tf.nn.top_k(max_confs * (1 - y_true[:, :, -8]),
                                 k=num_neg_batch)
        batch_idx = tf.expand_dims(tf.range(0, batch_size), 1)
        batch_idx = tf.tile(batch_idx, (1, num_neg_batch))
        full_indices = (tf.reshape(batch_idx, [-1]) * tf.to_int32(num_boxes) +
                        tf.reshape(indices, [-1]))
        # full_indices = tf.concat(2, [tf.expand_dims(batch_idx, 2),
        #                              tf.expand_dims(indices, 2)])
        # neg_conf_loss = tf.gather_nd(conf_loss, full_indices)
        neg_conf_loss = tf.gather(tf.reshape(conf_loss, [-1]),
                                  full_indices)
        neg_conf_loss = tf.reshape(neg_conf_loss,
                                   [batch_size, num_neg_batch])
        neg_conf_loss = tf.reduce_sum(neg_conf_loss, axis=1)

        # loss is sum of positives and negatives
        total_loss = pos_conf_loss + neg_conf_loss
        total_loss /= (num_pos + tf.to_float(num_neg_batch))
        num_pos = tf.where(tf.not_equal(num_pos, 0), num_pos,
                            tf.ones_like(num_pos))
        total_loss += (self.alpha * pos_loc_loss) / num_pos
        return total_loss


"""Some utils for SSD."""

class BBoxUtility(object):
    """Utility class to do some stuff with bounding boxes and priors.

    # Arguments
        num_classes: Number of classes including background.
        priors: Priors and variances, numpy tensor of shape (num_priors, 8),
            priors[i] = [xmin, ymin, xmax, ymax, varxc, varyc, varw, varh].
        overlap_threshold: Threshold to assign box to a prior.
        nms_thresh: Nms threshold.
        top_k: Number of total bboxes to be kept per image after nms step.

    # References
        https://arxiv.org/abs/1512.02325
    """
    # TODO add setter methods for nms_thresh and top_K
    def __init__(self, num_classes, priors=None, overlap_threshold=0.5,
                 nms_thresh=0.45, top_k=400):
        self.num_classes = num_classes
        self.priors = priors
        self.num_priors = 0 if priors is None else len(priors)
        self.overlap_threshold = overlap_threshold
        self._nms_thresh = nms_thresh
        self._top_k = top_k
        self.boxes = tf.placeholder(dtype='float32', shape=(None, 4))
        self.scores = tf.placeholder(dtype='float32', shape=(None,))
        self.nms = tf.image.non_max_suppression(self.boxes, self.scores,
                                                self._top_k,
                                                iou_threshold=self._nms_thresh)
        self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))

    @property
    def nms_thresh(self):
        return self._nms_thresh

    @nms_thresh.setter
    def nms_thresh(self, value):
        self._nms_thresh = value
        self.nms = tf.image.non_max_suppression(self.boxes, self.scores,
                                                self._top_k,
                                                iou_threshold=self._nms_thresh)

    @property
    def top_k(self):
        return self._top_k

    @top_k.setter
    def top_k(self, value):
        self._top_k = value
        self.nms = tf.image.non_max_suppression(self.boxes, self.scores,
                                                self._top_k,
                                                iou_threshold=self._nms_thresh)

    def iou(self, box):
        """Compute intersection over union for the box with all priors.

        # Arguments
            box: Box, numpy tensor of shape (4,).

        # Return
            iou: Intersection over union,
                numpy tensor of shape (num_priors).
        """
        # compute intersection
        inter_upleft = np.maximum(self.priors[:, :2], box[:2])
        inter_botright = np.minimum(self.priors[:, 2:4], box[2:])
        inter_wh = inter_botright - inter_upleft
        inter_wh = np.maximum(inter_wh, 0)
        inter = inter_wh[:, 0] * inter_wh[:, 1]
        # compute union
        area_pred = (box[2] - box[0]) * (box[3] - box[1])
        area_gt = (self.priors[:, 2] - self.priors[:, 0])
        area_gt *= (self.priors[:, 3] - self.priors[:, 1])
        union = area_pred + area_gt - inter
        # compute iou
        iou = inter / union
        return iou

    def encode_box(self, box, return_iou=True):
        """Encode box for training, do it only for assigned priors.

        # Arguments
            box: Box, numpy tensor of shape (4,).
            return_iou: Whether to concat iou to encoded values.

        # Return
            encoded_box: Tensor with encoded box
                numpy tensor of shape (num_priors, 4 + int(return_iou)).
        """
        iou = self.iou(box)
        encoded_box = np.zeros((self.num_priors, 4 + return_iou))
        assign_mask = iou > self.overlap_threshold
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True
        if return_iou:
            encoded_box[:, -1][assign_mask] = iou[assign_mask]
        assigned_priors = self.priors[assign_mask]
        box_center = 0.5 * (box[:2] + box[2:])
        box_wh = box[2:] - box[:2]
        assigned_priors_center = 0.5 * (assigned_priors[:, :2] +
                                        assigned_priors[:, 2:4])
        assigned_priors_wh = (assigned_priors[:, 2:4] -
                              assigned_priors[:, :2])
        # we encode variance
        encoded_box[:, :2][assign_mask] = box_center - assigned_priors_center
        encoded_box[:, :2][assign_mask] /= assigned_priors_wh
        encoded_box[:, :2][assign_mask] /= assigned_priors[:, -4:-2]
        encoded_box[:, 2:4][assign_mask] = np.log(box_wh /
                                                  assigned_priors_wh)
        encoded_box[:, 2:4][assign_mask] /= assigned_priors[:, -2:]
        return encoded_box.ravel()

    def assign_boxes(self, boxes):
        """Assign boxes to priors for training.

        # Arguments
            boxes: Box, numpy tensor of shape (num_boxes, 4 + num_classes),
                num_classes without background.

        # Return
            assignment: Tensor with assigned boxes,
                numpy tensor of shape (num_boxes, 4 + num_classes + 8),
                priors in ground truth are fictitious,
                assignment[:, -8] has 1 if prior should be penalized
                    or in other words is assigned to some ground truth box,
                assignment[:, -7:] are all 0. See loss for more details.
        """
        assignment = np.zeros((self.num_priors, 4 + self.num_classes + 8))
        assignment[:, 4] = 1.0
        if len(boxes) == 0:
            return assignment
        encoded_boxes = np.apply_along_axis(self.encode_box, 1, boxes[:, :4])
        encoded_boxes = encoded_boxes.reshape(-1, self.num_priors, 5)
        best_iou = encoded_boxes[:, :, -1].max(axis=0)
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
        best_iou_mask = best_iou > 0
        best_iou_idx = best_iou_idx[best_iou_mask]
        assign_num = len(best_iou_idx)
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]
        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx,
                                                         np.arange(assign_num),
                                                         :4]
        assignment[:, 4][best_iou_mask] = 0
        assignment[:, 5:-8][best_iou_mask] = boxes[best_iou_idx, 4:]
        assignment[:, -8][best_iou_mask] = 1
        return assignment

    def decode_boxes(self, mbox_loc, mbox_priorbox, variances):
        """Convert bboxes from local predictions to shifted priors.

        # Arguments
            mbox_loc: Numpy array of predicted locations.
            mbox_priorbox: Numpy array of prior boxes.
            variances: Numpy array of variances.

        # Return
            decode_bbox: Shifted priors.
        """
        prior_width = mbox_priorbox[:, 2] - mbox_priorbox[:, 0]
        prior_height = mbox_priorbox[:, 3] - mbox_priorbox[:, 1]
        prior_center_x = 0.5 * (mbox_priorbox[:, 2] + mbox_priorbox[:, 0])
        prior_center_y = 0.5 * (mbox_priorbox[:, 3] + mbox_priorbox[:, 1])
        decode_bbox_center_x = mbox_loc[:, 0] * prior_width * variances[:, 0]
        decode_bbox_center_x += prior_center_x
        decode_bbox_center_y = mbox_loc[:, 1] * prior_width * variances[:, 1]
        decode_bbox_center_y += prior_center_y
        decode_bbox_width = np.exp(mbox_loc[:, 2] * variances[:, 2])
        decode_bbox_width *= prior_width
        decode_bbox_height = np.exp(mbox_loc[:, 3] * variances[:, 3])
        decode_bbox_height *= prior_height
        decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
        decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
        decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
        decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height
        decode_bbox = np.concatenate((decode_bbox_xmin[:, None],
                                      decode_bbox_ymin[:, None],
                                      decode_bbox_xmax[:, None],
                                      decode_bbox_ymax[:, None]), axis=-1)
        decode_bbox = np.minimum(np.maximum(decode_bbox, 0.0), 1.0)
        return decode_bbox

    def detection_out(self, predictions, background_label_id=0, keep_top_k=200,
                      confidence_threshold=0.01):
        """Do non maximum suppression (nms) on prediction results.

        # Arguments
            predictions: Numpy array of predicted values.
            num_classes: Number of classes for prediction.
            background_label_id: Label of background class.
            keep_top_k: Number of total bboxes to be kept per image
                after nms step.
            confidence_threshold: Only consider detections,
                whose confidences are larger than a threshold.

        # Return
            results: List of predictions for every picture. Each prediction is:
                [label, confidence, xmin, ymin, xmax, ymax]
        """
        mbox_loc = predictions[:, :, :4]
        variances = predictions[:, :, -4:]
        mbox_priorbox = predictions[:, :, -8:-4]
        mbox_conf = predictions[:, :, 4:-8]
        results = []
        for i in range(len(mbox_loc)):
            results.append([])
            decode_bbox = self.decode_boxes(mbox_loc[i],
                                            mbox_priorbox[i], variances[i])
            for c in range(self.num_classes):
                if c == background_label_id:
                    continue
                c_confs = mbox_conf[i, :, c]
                c_confs_m = c_confs > confidence_threshold
                if len(c_confs[c_confs_m]) > 0:
                    boxes_to_process = decode_bbox[c_confs_m]
                    confs_to_process = c_confs[c_confs_m]
                    feed_dict = {self.boxes: boxes_to_process,
                                 self.scores: confs_to_process}
                    idx = self.sess.run(self.nms, feed_dict=feed_dict)
                    good_boxes = boxes_to_process[idx]
                    confs = confs_to_process[idx][:, None]
                    labels = c * np.ones((len(idx), 1))
                    c_pred = np.concatenate((labels, confs, good_boxes),
                                            axis=1)
                    results[-1].extend(c_pred)
            if len(results[-1]) > 0:
                results[-1] = np.array(results[-1])
                argsort = np.argsort(results[-1][:, 1])[::-1]
                results[-1] = results[-1][argsort]
                results[-1] = results[-1][:keep_top_k]
        return results