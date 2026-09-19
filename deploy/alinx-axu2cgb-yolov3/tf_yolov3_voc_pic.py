import time
import os
import cv2
import colorsys
import random
import numpy as np
from dnndk import n2cube

# test sample folder list
TEST_SET = ['t1']

INPUT_IMAGE_PATH = 'test_sample/'

# creat output result floder 
for folder_name in TEST_SET:
    is_exist = os.path.exists('output_result/' + folder_name)
    if not is_exist:
        os.makedirs('output_result/' + folder_name)

OBJECT_TYPE = 'office'

CLASSES_PATH = './model_data/classes_' + OBJECT_TYPE + '.txt'
ANCHORS_PATH = './model_data/yolo_anchors_' + OBJECT_TYPE + '.txt'
SCORE_THRESH = 0.4
NMS_THRESH = 0.9

# WATER_MARK_PATH='alinx_logo.png'

# def watermark(src_img, logo_path, alpha = 1):
#    logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
#    (h, w, d) = src_img.shape
#    
#    if d == 3:
#        # 构建一个和输入图像完全一致的叠加层，
#        # 此Alpha通道具有与原始图像相同的空间尺寸，
#        # 并且Alpha通道中的所有值均设置为255，指示像素完全不透明。
#        src_img = np.dstack([src_img, np.ones((h, w), dtype="uint8") * 255])
#  
#    # 构建水印图片的叠加层（使得其具有与输入图像完全相同的宽度和高度），透明度全部设为0，即完全透明
#    overlay = np.zeros((h, w, 4), dtype="uint8")
#    # logo = cv2.resize(logo,(int(logo.shape[1] / 5), int(logo.shape[0] / 5)))
#    (lh, lw, ld) = logo.shape
#    overlay[380:lh + 380, w - lw - 20:w - 20] = logo
#    # 应用cv2.addWeighted构造水印的图像
#    # cv2.addWeighted(src1=overlay, alpha=alpha, src2=src_img, beta=1.0, gamma=0, dst=src_img)
#    src_img = cv2.add(overlay,src_img)
#    
#    return src_img
 

'''resize image with unchanged aspect ratio using padding'''
def letterbox_image(image, size):
    ih, iw, _ = image.shape
    w, h = size
    scale = min(w/iw, h/ih)
    #print("image scale:",scale)
    
    nw = int(iw*scale)
    nh = int(ih*scale)

    #print("image wide:",nw)
    #print("image height:",nh)

    image = cv2.resize(image, (nw,nh), interpolation=cv2.INTER_LINEAR)
    new_image = np.ones((h,w,3), np.uint8) * 128
    h_start = (h-nh)//2
    w_start = (w-nw)//2
    new_image[h_start:h_start+nh, w_start:w_start+nw, :] = image
    return new_image

'''image preprocessing'''
def pre_process(image, model_image_size):
    image = image[...,::-1]
    image_h, image_w, _ = image.shape
 
    if model_image_size != (None, None):
        assert model_image_size[0]%32 == 0, 'Multiples of 32 required'
        assert model_image_size[1]%32 == 0, 'Multiples of 32 required'
        boxed_image = letterbox_image(image, tuple(reversed(model_image_size)))
    else:
        new_image_size = (image_w - (image_w % 32), image_h - (image_h % 32))
        boxed_image = letterbox_image(image, new_image_size)
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0) 	
    return image_data

'''Get model classification information'''	
def get_class(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

'''Get model anchors value'''
def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

def output_fix(feats, num_classes):
    num_anchors = 3
    nu = num_classes + 5
    grid_size = np.shape(feats)[1 : 3]
    predictions = np.reshape(feats, [-1, grid_size[0], grid_size[1], num_anchors, nu])

	# replace 0 with broken data in some output channel
    predictions[..., 0:1, :18] = np.zeros(shape=predictions[..., 0:1, :18].shape)
    feats_fixed = np.reshape(predictions, feats.shape)

    # box_confidence = predictions[..., 4:5]
    # single_layer_box_confidence = box_confidence[..., 2:3,:]
    # print('\nOriginal output:{}'.format(feats.shape))
    # print('reshaped output:{}'.format(predictions.shape))
    # print('score output:{}'.format(box_confidence.shape))
    # print('single layer score output:{}\n'.format(single_layer_box_confidence.shape))

    return feats_fixed
    
def _get_feats(feats, anchors, num_classes, input_shape):
    num_anchors = len(anchors)
    anchors_tensor = np.reshape(np.array(anchors, dtype=np.float32), [1, 1, 1, num_anchors, 2])
    grid_size = np.shape(feats)[1:3]
    nu = num_classes + 5
    predictions = np.reshape(feats, [-1, grid_size[0], grid_size[1], num_anchors, nu])
    grid_y = np.tile(np.reshape(np.arange(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
    grid_x = np.tile(np.reshape(np.arange(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
    grid = np.concatenate([grid_x, grid_y], axis = -1)
    grid = np.array(grid, dtype=np.float32)

    box_xy = (1/(1+np.exp(-predictions[..., :2])) + grid) / np.array(grid_size[::-1], dtype=np.float32)
    box_wh = np.exp(predictions[..., 2:4]) * anchors_tensor / np.array(input_shape[::-1], dtype=np.float32)
    box_confidence = 1/(1+np.exp(-predictions[..., 4:5]))
    box_class_probs = 1/(1+np.exp(-predictions[..., 5:]))
    return box_xy, box_wh, box_confidence, box_class_probs

def correct_boxes(box_xy, box_wh, input_shape, image_shape):
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = np.array(input_shape, dtype = np.float32)
    image_shape = np.array(image_shape, dtype = np.float32)
    new_shape = np.around(image_shape * np.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = np.concatenate([
        box_mins[..., 0:1],
        box_mins[..., 1:2],
        box_maxes[..., 0:1],
        box_maxes[..., 1:2]
    ], axis = -1)
    boxes *= np.concatenate([image_shape, image_shape], axis = -1)
    return boxes

def boxes_and_scores(feats, anchors, classes_num, input_shape, image_shape):
    box_xy, box_wh, box_confidence, box_class_probs = _get_feats(feats, anchors, classes_num, input_shape)
    boxes = correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = np.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = np.reshape(box_scores, [-1, classes_num])
    return boxes, box_scores	

'''Draw detection frame'''
def draw_bbox(image, bboxes, classes):
    """
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    """
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 0.5
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = colors[class_ind]
        bbox_thick = int((image_h + image_w) / 550)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)
    return image

def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.

    # Arguments
        boxes: ndarray, boxes of objects.
        scores: ndarray, scores of objects.

    # Returns
        keep: ndarray, index of effective boxes.
    """

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2-x1+1)*(y2-y1+1)
    # sort the scores from minimun to maximun and invert order =>  ' maximun -> minimun '
    # output sub-number
    order = scores.argsort()[::-1] 

    keep = []
    while order.size > 0:
        i = order[0] # maximun score sub-number and send it to 'keep'
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 1)
        h1 = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # gets the index of the array within the threshold range
        inds = np.where(ovr <= (1-NMS_THRESH))[0]
        order = order[inds + 1]

    return keep	
  
'''Model post-processing'''
def eval(yolo_outputs, image_shape, class_names, anchors):

    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    boxes = []
    box_scores = []
    
    input_shape = np.shape(yolo_outputs[0])[1 : 3]
    input_shape = np.array(input_shape)*32

    # repare broken output data
    # yolo_outputs[2] = output_fix(yolo_outputs[2], len(class_names))

    for i in range(len(yolo_outputs)):
        _boxes, _box_scores = boxes_and_scores(yolo_outputs[i], anchors[anchor_mask[i]], len(class_names), input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = np.concatenate(boxes, axis = 0)
    box_scores = np.concatenate(box_scores, axis = 0)
    
    mask = box_scores >= SCORE_THRESH
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(len(class_names)):
        class_boxes_np = boxes[mask[:, c]]
        class_box_scores_np = box_scores[:, c]
        class_box_scores_np = class_box_scores_np[mask[:, c]]
        nms_index_np = nms_boxes(class_boxes_np, class_box_scores_np) 
        class_boxes_np = class_boxes_np[nms_index_np]
        class_box_scores_np = class_box_scores_np[nms_index_np]
        classes_np = np.ones_like(class_box_scores_np, dtype = np.int32) * c
        boxes_.append(class_boxes_np)
        scores_.append(class_box_scores_np)
        classes_.append(classes_np)
    boxes_ = np.concatenate(boxes_, axis = 0)
    scores_ = np.concatenate(scores_, axis = 0)
    classes_ = np.concatenate(classes_, axis = 0)
    
    return boxes_, scores_, classes_

"""DPU Kernel Name for tf_yolov3_xxx"""
KERNEL_CONV='tf_yolov3_'+ OBJECT_TYPE

"""DPU IN/OUT Name for tf_yolov3_xxx"""
CONV_INPUT_NODE="conv2d_1_convolution"
CONV_OUTPUT_NODE1="conv2d_59_convolution"
CONV_OUTPUT_NODE2="conv2d_67_convolution"
CONV_OUTPUT_NODE3="conv2d_75_convolution"

if __name__ == "__main__":
       
    """ Attach to DPU driver and prepare for running """
    n2cube.dpuOpen()

    """ Create DPU Kernels for tf_yolov3_xxx """
    kernel = n2cube.dpuLoadKernel(KERNEL_CONV)

    """ Create DPU Tasks for tf_yolov3_xxx """
    task = n2cube.dpuCreateTask(kernel, 0) # 1 = T_MODE_PROF; 0 = normal;
    # task = n2cube.dpuEnableTaskProfile(task)    

    """Get model classification information"""
    class_names = get_class(CLASSES_PATH)
    NN_obj_info_dim = 3*(5+len(class_names))

    """Get model anchor value"""
    anchors = get_anchors(ANCHORS_PATH)    

    cnt=0
    list_idx=0

    while True:

        total_time_start = time.process_time() # <-------- Start total Running Time Recording (1)

        """Load image to DPU"""
        cnt = cnt + 1

        image_name = TEST_SET[list_idx] + '/img{:0>5d}.jpg'.format(cnt)
        image = cv2.imread(INPUT_IMAGE_PATH + image_name)
        
        if image is None:
            list_idx = list_idx + 1
            cnt=0
            if list_idx>len(TEST_SET)-1:
                break
            continue

        preprocess_time = time.process_time() # <------ Start preprocess time-----------
        image_ho, image_wo, _ = image.shape
        image_size = image.shape[:2]
        image_data = pre_process(image, (416, 416))

        image_data = np.array(image_data,dtype=np.float32)
        input_len = n2cube.dpuGetInputTensorSize(task, CONV_INPUT_NODE)

        preprocess_time = time.process_time() - preprocess_time # <---------- Stop preprocess time----------

        conv_time_start = time.process_time() # <------------- Start Convolution Time Recording (2)

        """Get input Tesor"""
        n2cube.dpuSetInputTensorInHWCFP32(task,CONV_INPUT_NODE,image_data,input_len)

        """Model run on DPU"""
        n2cube.dpuRunTask(task)

        conv_sbbox_size = n2cube.dpuGetOutputTensorSize(task, CONV_OUTPUT_NODE1)
        conv_out1 = n2cube.dpuGetOutputTensorInHWCFP32(task, CONV_OUTPUT_NODE1, conv_sbbox_size)
        conv_out1 = np.reshape(conv_out1, (1, 13, 13, NN_obj_info_dim))

        conv_mbbox_size = n2cube.dpuGetOutputTensorSize(task, CONV_OUTPUT_NODE2)
        conv_out2 = n2cube.dpuGetOutputTensorInHWCFP32(task, CONV_OUTPUT_NODE2, conv_mbbox_size)
        conv_out2 = np.reshape(conv_out2, (1, 26, 26, NN_obj_info_dim))
        
        conv_lbbox_size = n2cube.dpuGetOutputTensorSize(task, CONV_OUTPUT_NODE3)
        conv_out3 = n2cube.dpuGetOutputTensorInHWCFP32(task, CONV_OUTPUT_NODE3, conv_lbbox_size)
        conv_out3 = np.reshape(conv_out3, (1, 52, 52, NN_obj_info_dim))

        conv_time = time.process_time()-conv_time_start # <------------ Stop Convolution Time Recording (2)
        #print('Convolution Speed: %s FPS'%(1/(time.process_time()-conv_time_start))) 

        yolo_outputs = [conv_out1, conv_out2, conv_out3]    

        """Post-processing"""   
        out_boxes, out_scores, out_classes = eval(yolo_outputs, image_size, class_names, anchors)

        items = []
        draws = []
        texted_image = image

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image_ho, np.floor(bottom + 0.5).astype('int32'))
            right = min(image_wo, np.floor(right + 0.5).astype('int32'))
            draw  = [left, top, right, bottom, score, c]
            item  = [predicted_class, score, left, top, right, bottom]
            draws.append(draw)
            items.append(item)

            texted_image = cv2.putText(image, predicted_class+':'+str(round(score,2)), (left,top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (92,51,255), 1)

        """ Running time calculate(over) and result print """
        run_time = time.process_time() - total_time_start # <------------- Stop total Running Time Recording (1)
        
        texted_image = cv2.putText(texted_image, 'DPU Convolution Time' + ': '+ str(round(1/conv_time,2)) + 'FPS (' + str(round(conv_time*1000,2)) +' ms)', (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (148,230,26), 1)
        texted_image = cv2.putText(texted_image, 'Pre-processing time' + ': '+ str(round(preprocess_time * 1000,2)) + 'ms', (0,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (148,230,26), 1)
        texted_image = cv2.putText(texted_image, 'Post-processing time' + ': '+ str(round((run_time-preprocess_time-conv_time) * 1000,2)) + 'ms', (0,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (148,230,26), 1)
        texted_image = cv2.putText(texted_image, 'Total Time' + ': '+ str(round(1/run_time,2)) + 'FPS (' + str(round(run_time*1000,2)) +' ms)', (0,80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (148,230,26), 1)

        image_result = draw_bbox(texted_image, draws, class_names)
        #watermarked_image = watermark(image_result,WATER_MARK_PATH)
        
        cv2.namedWindow(' YOLO-V3 ALINX '+ OBJECT_TYPE + ' Detection ',cv2.WINDOW_FREERATIO)
        cv2.imshow(' YOLO-V3 ALINX '+ OBJECT_TYPE + ' Detection ',image_result)

        k=cv2.waitKey(1)
        cv2.imwrite('output_result/' + image_name, image_result)
        if k == 27:
            cv2.destroyAllWindows()
            break
            
    cv2.destroyAllWindows()



