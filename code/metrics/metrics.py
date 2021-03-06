import numpy as np
from keras import backend as K
dim_ordering = K.image_dim_ordering()
import tensorflow as tf
if dim_ordering == 'th':
    import theano
    from theano import tensor as T
else:
    from tensorflow.python.framework import ops

def jaccard_coef(y_true, y_pred, eps=1e-37):
    y_pred = tf.one_hot(tf.argmax(y_pred,axis=2), tf.shape(y_pred)[2])
    intersection = K.sum(y_true * y_pred, axis=(1))
    union = K.sum(y_true, axis=(1)) + K.sum(y_pred, axis=(1)) - intersection
    mean  = K.mean( (intersection ) / (union + eps), axis=(1))
    mean  = tf.clip_by_value(mean, 0.0, 1.0)
    return K.mean( mean, axis=0)

def jaccard_coef_smooth(y_true, y_pred, smooth=0.0):
    '''Average jaccard coefficient per batch.'''
    intersection = K.sum(y_true * y_pred, axis=(1))
    union = K.sum(y_true, axis=(1)) + K.sum(y_pred, axis=(1)) - intersection
    mean  = K.mean( (intersection + smooth) / (union + smooth), axis=(1))
    return K.mean( mean, axis=0)

def cce_flatt(void_class, weights_class):
    def categorical_crossentropy_flatt(y_true, y_pred):
        '''Expects a binary class matrix instead of a vector of scalar classes.
        '''
        print(np.shape(y_true))
        print(np.shape(y_pred))
        #print('categorical_crossentropy_flatt')
        if dim_ordering == 'th':
            y_pred = K.permute_dimensions(y_pred, (0, 2, 1))
            #y_pred = K.permute_dimensions(y_pred, (0, 2, 3, 1))
        shp_y_pred = K.shape(y_pred)
        #y_pred = K.reshape(y_pred, (shp_y_pred[0]*shp_y_pred[1]*shp_y_pred[2],
                           #shp_y_pred[3]))  # go back to b01,c
        y_pred = K.reshape(y_pred, (shp_y_pred[0]*shp_y_pred[1],shp_y_pred[2]))  # go back to b01,c
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
            #y_true = K.cast(y_true, 'float32')  # b,01 -> b01
        out = K.categorical_crossentropy(y_pred, K.cast(y_true, 'float32'))

        # Class balancing
        if weights_class is not None:
            weights_class_var = K.variable(value=weights_class)
            class_balance_w = weights_class_var[y_true].astype(K.floatx())
            out = out * class_balance_w

        return K.mean(out)  # b01 -> b,01
    return categorical_crossentropy_flatt


def pix_weight_loss(void_class, weights_class):
    def categorical_crossentropy_weighted(y_true, y_pred):
        '''Expects a binary class matrix instead of a vector of scalar classes.
        '''
        out = K.categorical_crossentropy(y_pred, y_true)

        y_true_hot = tf.one_hot(K.cast(y_true,'int32'), K.shape(y_pred)[-1], on_value=1, off_value=0, axis=None, dtype='int32', name=None)
        y_true_hot = K.flatten(y_true_hot)
        out1 = K.flatten(out)

        #print(weights_class)
        #print('categorical_crossentropy_flatt')
        #if dim_ordering == 'th':

        # remove void classes from cross_entropy


        if weights_class is None:
            return(K.mean(out))
        weight_map = K.zeros_like(out)
        #tmp = out
        #print(weight_map)
        b = tf.constant(0,dtype='float32')

        for i in range(len(weights_class)):

            Idx = tf.squeeze(tf.where(K.tf.equal(y_true_hot,tf.multiply(K.ones_like(y_true_hot,dtype='int32'),K.variable(i,dtype='int32')))),-1)
            #c = tf.count
            n_b = tf.reduce_sum(tf.multiply(tf.gather(out1,Idx,name=None),K.variable(weights_class[i],dtype='float32')))
            #n_b = tf.div(n_b,tf.clip_by_value(tf.cast(tf.count_nonzero(Idx),'float32'),tf.cast(1,'float32'),tf.cast(1e+20,'float32')))

            b = tf.reduce_sum(tf.stack([b,n_b]))
            #b = tf.stack([b,n_b])
            #b = K.sum([n_b,b])


            #a[Idx] = out1[Idx]
            #weight_map = weight_map[Idx].assign(tmp[Idx])
        #print('\n\n\n\n?????????????????????????????????\n\n\n\n')
        #print(weight_map)
        #out =K.tf.multiply(out, weight_map)
        #print(out.shape)
        # Class balancing
        # if weights_class is not None:
        #     weights_class_var = K.variable(value=weights_class)
        #     class_balance_w = weights_class_var[y_true].astype(K.floatx())
        #     out = out * class_balance_w
        clip_v = tf.constant(1,dtype='int32')
        return tf.div(b,tf.cast(tf.count_nonzero(tf.clip_by_value(y_true_hot,clip_v,clip_v)),'float32')) # b01 -> b,01
    return categorical_crossentropy_weighted


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

"""
    SSD detection metrics
    code adapted from our own
"""
def SSDMetrics():

    def avg_iou(y_truet, y_predt, name=None):
        # avg = []
        # y_true = y_truet[:, :, :4]
        # y_pred = y_predt[:, :, :4]
        # for i in y_true.shape(0):
        #     for j in y_pred.shape(1):
        #         # compute intersection
        #         inter_upleft = np.maximum(y_true[i,:, :2], y_pred[i,j, :2])
        #         inter_botright = np.minimum(y_true[i,:, 2:4], y_pred[i,j, 2:])
        #         inter_wh = inter_botright - inter_upleft
        #         inter_wh = np.maximum(inter_wh, 0)
        #         inter = inter_wh[:, 0] * inter_wh[:, 1]
        #         # compute union
        #         area_pred = (y_pred[i,j,2] - y_pred[i,j,0]) * (y_pred[i,j,3] - y_pred[i,j,1])
        #         area_gt = (y_true[i,:, 2] - y_true[i,:, 0])
        #         area_gt *= (y_true[i,:, 3] - y_true[i,:, 1])
        #         union = area_pred + area_gt - inter
        #         # compute iou
        #         avg.append(inter / union)
        #
        # return np.array(avg).mean()
        return tf.constant(1)

    return avg_iou
