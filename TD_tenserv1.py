
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import cv2
import math
import numpy as np
import os
import sys
import tensorflow as tf
#import tensorflow_io as tfio
import time
import yaml
from scipy import ndimage
# import imutils
import lanms
# import locality_aware_nms as nms_locality
import time

# with open(c,'w',newline='') as file:
#   writer = csv.writer(file)
#   writer.writerow(['Image_name','scores'])


def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.compat.v1.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_file(file_name,
                                input_height,
                                input_width,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.compat.v1.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
#  elif file_name.endswith(".tif"):
#    image_reader = tfio.experimental.image.decode_tiff(file_reader, index=0, name=None)
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.compat.v1.image.resize_bilinear(dims_expander, [input_height, input_width])
  # normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.compat.v1.Session()

  result = sess.run(resized)

  return result

def restore_rectangle(origin, geometry):
  return restore_rectangle_rbox(origin, geometry)

def restore_rectangle_rbox(origin, geometry):
  d = geometry[:, :4]
  angle = geometry[:, 4]
  # for angle > 0
  origin_0 = origin[angle >= 0]
  d_0 = d[angle >= 0]
  angle_0 = angle[angle >= 0]
  if origin_0.shape[0] > 0:
      p = np.array([np.zeros(d_0.shape[0]), -d_0[:, 0] - d_0[:, 2],
                    d_0[:, 1] + d_0[:, 3], -d_0[:, 0] - d_0[:, 2],
                    d_0[:, 1] + d_0[:, 3], np.zeros(d_0.shape[0]),
                    np.zeros(d_0.shape[0]), np.zeros(d_0.shape[0]),
                    d_0[:, 3], -d_0[:, 2]])
      p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

      rotate_matrix_x = np.array([np.cos(angle_0), np.sin(angle_0)]).transpose((1, 0))
      rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

      rotate_matrix_y = np.array([-np.sin(angle_0), np.cos(angle_0)]).transpose((1, 0))
      rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

      p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
      p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

      p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

      p3_in_origin = origin_0 - p_rotate[:, 4, :]
      new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
      new_p1 = p_rotate[:, 1, :] + p3_in_origin
      new_p2 = p_rotate[:, 2, :] + p3_in_origin
      new_p3 = p_rotate[:, 3, :] + p3_in_origin

      new_p_0 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
  else:
      new_p_0 = np.zeros((0, 4, 2))
  # for angle < 0
  origin_1 = origin[angle < 0]
  d_1 = d[angle < 0]
  angle_1 = angle[angle < 0]
  if origin_1.shape[0] > 0:
      p = np.array([-d_1[:, 1] - d_1[:, 3], -d_1[:, 0] - d_1[:, 2],
                    np.zeros(d_1.shape[0]), -d_1[:, 0] - d_1[:, 2],
                    np.zeros(d_1.shape[0]), np.zeros(d_1.shape[0]),
                    -d_1[:, 1] - d_1[:, 3], np.zeros(d_1.shape[0]),
                    -d_1[:, 1], -d_1[:, 2]])
      p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

      rotate_matrix_x = np.array([np.cos(-angle_1), -np.sin(-angle_1)]).transpose((1, 0))
      rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

      rotate_matrix_y = np.array([np.sin(-angle_1), np.cos(-angle_1)]).transpose((1, 0))
      rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

      p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
      p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

      p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

      p3_in_origin = origin_1 - p_rotate[:, 4, :]
      new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
      new_p1 = p_rotate[:, 1, :] + p3_in_origin
      new_p2 = p_rotate[:, 2, :] + p3_in_origin
      new_p3 = p_rotate[:, 3, :] + p3_in_origin

      new_p_1 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
  else:
      new_p_1 = np.zeros((0, 4, 2))
  return np.concatenate([new_p_0, new_p_1])


def resize_image(im, max_side_len=1024):
  '''
  resize image to a size multiple of 32 which is required by the network
  :param im: the resized image
  :param max_side_len: limit of max image size to avoid out of memory in gpu
  :return: the resized image and the resize ratio
  '''
  h, w, _ = im.shape

  resize_w = w
  resize_h = h

  # limit the max side
  if max(resize_h, resize_w) > max_side_len:
      ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
  else:
      ratio = 1.
  resize_h = int(resize_h * ratio)
  resize_w = int(resize_w * ratio)

  resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
  resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
  resize_h = max(32, resize_h)
  resize_w = max(32, resize_w)
  im = cv2.resize(im, (int(resize_w), int(resize_h)))

  ratio_h = resize_h / float(h)
  ratio_w = resize_w / float(w)
  print(resize_h, resize_w)
  return im, resize_w, resize_h, (ratio_h, ratio_w)

# def detect(score_map, geo_map, timer, score_map_thresh=0.1, box_thresh=0.0001, nms_thres=0.009):
# def detect(score_map, geo_map, timer, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):

def detect(score_map, geo_map, timer, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
  '''
  restore text boxes from score map and geo map
  :param score_map:
  :param geo_map:
  :param timer:
  :param score_map_thresh: threshhold for score map
  :param box_thresh: threshhold for boxes
  :param nms_thres: threshold for nms
  :return:
  '''
  # print(score_map.shape)
  # print(geo_map.shape)
  if len(score_map.shape) == 4:
      # print(score_map)
      # print(geo_map)
      print("before slicing")
      score_map = score_map[0, :, :, 0]
      # print("scoremap----------------------------------",score_map)
      # print(score_map.shape)
      geo_map = geo_map[0, :, :, ]
      # print(geo_map.shape)
      # print(geo_map)

  # filter the score map
  xy_text = np.argwhere(score_map > score_map_thresh)

  # sort the text boxes via the y axis
  xy_text = xy_text[np.argsort(xy_text[:, 0])]

  # restore
  start = time.time()
  text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
  print('{} text boxes before nms'.format(text_box_restored.shape[0]))
  boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
  # print("boxes ------------2", boxes)

  boxes[:, :8] = text_box_restored.reshape((-1, 8))
  boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
  timer['restore'] = time.time() - start
  # nms part
  start = time.time()
  # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
  boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
  # print("boxes ------------3", boxes)
  timer['nms'] = time.time() - start

  if boxes.shape[0] == 0:
      return None, timer

  # here we filter some low score boxes by the average score map, this is different from the orginal paper
  for i, box in enumerate(boxes):
      mask = np.zeros_like(score_map, dtype=np.uint8)
      cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
      boxes[i, 8] = cv2.mean(score_map, mask)[0]
  boxes = boxes[boxes[:, 8] > box_thresh]

  return boxes, timer

def check_col(bin_img, x,  height):
    res = False
    for i in range(0, height-1, 1):
        if bin_img[i, x]<129:
            res = True
            break
    return res

def check_full(bin_img, x,  height):
    res = True
    for i in range(1, height-1, 1):
        if bin_img[i, x]>129:
            res = False
            break
    return res

def before_find(bin_img, start_x, height):
    space_x = int(height/4)
    i=0
    while(i < space_x):
        if check_col(bin_img, start_x + i, height):
            break
        i=i+1

    if i>= space_x:
        return start_x + i

    prev_x= start_x
    orig_x = prev_x
    cnt = 0
    while(start_x > 1):
        start_x=start_x -1
        if check_col(bin_img, start_x, height):
            if prev_x - start_x +1 >= space_x:
                break
            else:
                if check_full(bin_img, start_x, height):
                    cnt = cnt+1
                    if cnt >= space_x:
                        prev_x = orig_x
                        break
                prev_x = start_x
        else:
            cnt = 0
            if prev_x == start_x + 1:
                orig_x = prev_x

    return prev_x

def after_find(bin_img, max_x,  end_x, height):
    prev_x = end_x
    while(end_x < max_x -1):
        end_x = end_x +1
        if check_col(bin_img, end_x, height):
            if end_x - prev_x > height/3:
                break
            else:
                prev_x = end_x

    return prev_x

# rotate image and crop, return rotated rect
def crop_rect(img, rect):
    # get the parameter of the small rectangle
    center, size, angle = rect[0], rect[1], rect[2]
    center, size = tuple(map(int, center)), list(map(int, size))

    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    if angle < -45:
        angle = 90 + angle
        tmp = size[0]
        size[0]=size[1]
        size[1]=tmp
        # print("angle",angle)
    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
        angle = angle
        # print("angle",angle)


    size = tuple(size)
    # get row and col num in img
    height, width = img.shape[0], img.shape[1]

    # calculate the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1)
    # rotate the original image
    img_rot = cv2.warpAffine(img, M, (width, height))

    rect = list(rect)
    rect[2] = 0
    rect[1] = size
    rect = tuple(rect)

    # now rotated rectangle becomes vertical and we crop it
    img_crop = cv2.getRectSubPix(img_rot, size, center)

    return img_crop, img_rot, rect,angle

def get_box(rect, box):
    box[0,0] = int(rect[1,0])
    box[0,1] = int(rect[1,1])
    box[1, 0] = int(rect[2, 0])
    box[1, 1] = int(rect[2, 1])
    box[2, 0] = int(rect[3, 0])
    box[2, 1] = int(rect[3, 1])
    box[3, 0] = int(rect[0, 0])
    box[3, 1] = int(rect[0, 1])
    return box
if __name__ == "__main__":
  input_height = 512
  input_width = 512
  input_mean = 0
  input_std = 255
  # input_layer = "model_2_input"
  # output_layer = "dense_4/Sigmoid"
  input_layer = "input_images"
  output_layer = "feature_fusion/concat_3"
  output_layer_2 = "feature_fusion/Conv_7/Sigmoid"

  file_name = sys.argv[1]  # prints python_script.py
  # csvs = sys.argv[2]  # prints var1
  model_file = sys.argv[2]
  outfile = sys.argv[3]

#   print (file_name)
#   # print(csvs)
#   print(model_file)
  full_start_time = time.time()
  start_time = time.time()
  graph = load_graph(model_file)
  end_time = time.time()
  model_time = end_time - start_time
  print("printing the session run time-------------------------------------------------main", model_time)
  # print(graph.get_operations())

  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  output_name_2 = "import/" + output_layer_2

  start_time = time.time()
  input_operation = graph.get_operation_by_name(input_name)
  end_time = time.time()
  total_time = end_time - start_time
  print("printing the session run time-------------------------------------------------input", total_time)
      # print(input_operation)
  start_time = time.time()
  output_operation = graph.get_operation_by_name(output_name)
  end_time = time.time()
  total_time = end_time - start_time
  print("printing the session run time-------------------------------------------------output1", total_time)

  start_time = time.time()
  output_2_operation = graph.get_operation_by_name(output_name_2)
  end_time = time.time()
  total_time = end_time - start_time
  print("printing the session run time-------------------------------------------------output2", total_time)
  count = 0

  # images = args["image"]
  with tf.compat.v1.Session(graph=graph) as sess:
    for fl in os.listdir(file_name):
      # try:
      if fl == ".DS_Store" or fl == "_DS_Store":
        print ("sorry")
        print(fl)
      else:
        images2 = os.path.join(file_name,fl)


        #degree = check_rotation(images2)
        im = cv2.imread(images2)
        full_image = im
        #if abs(degree) > 2.5:
        #    im = rotateImage(im, degree)
        print("the file name",fl)

        try :
            im = im[:, :, ::-1]
        # try:
            height, width, _ = im.shape
        except:
            continue
        # print("so what",width,height)

      if width<900 and height<620:
        im_up = cv2.resize(full_image, dsize=(900,620), interpolation = cv2.INTER_CUBIC)
        height, width, _ = im_up.shape
        im1 = im
        cv_interpolation = cv2.INTER_LANCZOS4
        im = cv2.resize(im, dsize=(600,414), interpolation=cv_interpolation)
        height, width, _ = im.shape
        # print(width,height)
        q=(900/600)
        p=(620/414)
        print("Q and P valures are")
        print(q,p)
        im_resized,  resize_w, resize_h, (ratio_h, ratio_w) = resize_image(im)
        t = read_tensor_from_image_file(
            images2,
            input_height=resize_h,
            input_width=resize_w,
            input_mean=input_mean,
            input_std=input_std)
        # print(input_name)
        
        start_time = time.time()
        start = time.time()
        results_1, results_2 = sess.run([output_2_operation.outputs[0], output_operation.outputs[0]], feed_dict={input_operation.outputs[0]: t})
        # print("results_1 and 222",results_1,results_2)
        end_time = time.time()
        total_time = end_time - start_time
        print("print time for each image--------------", total_time)
     
        end_time = time.time()
        print("ending time",end_time)
        total_time = end_time - start_time
        # print("ifklfjkalsjfkajflkfj")
        print("printing the session run time____@____", total_time)
        timer = {'net': 0, 'restore': 0, 'nms': 0}
        # start = time.time()
        # score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})
        timer['net'] = time.time() - start
        # tf.keras.backend.clear_session()

        boxes, timer = detect(score_map=results_1, geo_map=results_2, timer=timer)
        print(boxes)
        print('{} : net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
            images2, timer['net']*1000, timer['restore']*1000, timer['nms']*1000))

        if boxes is not None:
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h
            boxes[:, :, 0] *= q
            boxes[:, :, 1] *= p


        duration = time.time() - start_time
        print('[timing] {}'.format(duration))

        fname = os.path.basename(images2)
        cnt = 0

        if boxes is None:
            continue

        total_angle = []
        for boxxx in boxes:
            # print("after model coordinates",boxxx)
            box = np.copy(boxxx)
            # print(box)
            # rotation and crop
            # find the exact rectangle enclosing the text area
            # rect is a tuple consisting of 3 elements: the first element is the center
            # of the rectangle, the second element is the width, height, and the
            # third element is the detected rotation angle.
            # Example output: ((227.5, 187.50003051757812),
            # (94.57575225830078, 417.98736572265625), -36.982906341552734)
            rect = cv2.minAreaRect(box)
            print("rect: {}".format(rect))
            img_crop, img_rot, rect, angle = crop_rect(im_up, rect)

            # cv2.imwrite('imgrot'+str(angle)+'.jpg',img_rot)
            # cv2.imwrite("the_image.jpg",img_rot)
            # print("the angle is angle",angle)

            angle = angle
      
            rr = cv2.boxPoints(tuple(rect))
            # print("rr",rr)
            box = get_box(rr, box)
            # print("the oput_put coordinates to crop the images",box)
            long_line_x = box[1,0]-box[0,0]
            long_line_y = box[3,1]-box[0,1]
            div = long_line_x/long_line_y
            print("the value of x and y and d", long_line_x, long_line_y, div)

            cnt = cnt + 1
            # if box[1, 0] - box[0, 0] > 620:
            #     print("long_came")
            if box[1, 0] - box[0, 0] > 14 * (box[3, 1] - box[0, 1]):
                # expand region
                start_x = int(min(box[0, 0], box[3, 0]))

                # gray = cv2.cvtColor(img_rot[int(box[0,1]):int(box[3,1]), 0:int(box[1,0]),:], cv2.COLOR_BGR2GRAY)
                gray = img_rot[int(box[0, 1]):int(box[3, 1]), 0:int(box[1, 0]), 1]
                blur = cv2.GaussianBlur(gray, (3, 3), 0)
                ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                new_x = before_find(th3, start_x + int((box[3, 1] - box[0, 1]) / 5), int(box[3, 1] - box[0, 1]))
                box[0, 0] = new_x
                box[3, 0] = new_x
                end_x = int(max(box[1, 0], box[2, 0]))
                gray = cv2.cvtColor(img_rot[int(box[1, 1]):int(box[2, 1]), :, :], cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                new_x = after_find(th3, 900, end_x, th3.shape[0])

                box[1, 0] = new_x
                box[2, 0] = new_x

            top_x = int(box[0, 0])
            top_y = int(min(box[1, 1], box[2, 1]))
            bottom_x = int(box[1, 0])
            bottom_y = int(max(box[0, 1], box[3, 1]))
            print(box)
            print()
            # print("xmim, ymin, xmax, ymax",top_x,top_y,bottom_x,bottom_y)
            # cv2.imwrite("imge"+str(angle)+".jpg",img_crop)
            try:
              cv2.imwrite(os.path.join(outfile,'{}_{}.jpg'.format(fname[:-4], cnt)),
                          img_rot[int(box[0, 1]-2):int(box[3, 1]+2), int(box[0, 0]-4):int(box[1, 0]+4), :])
            except:
              continue
            full_end_time = time.time()
            full_total_time = full_end_time - full_start_time
#            print("fst : ", full_start_time)
 #           print("fet : " , full_end_time)
            # with open(c,'a',newline='') as f:
            #     writer = csv.writer(f)
            #     im1 = os.path.join(outfile,'{}_{}.jpg'.format(fname[:-4], cnt))
            #     final2 = os.path.basename(im1)
            #     writer.writerow([final2,box])
            


            # cv2.imwrite(os.path.join(outfile,'{}_{}.jpg'.format(fname[:-4], cnt)),
            #             img_rot[top_y:bottom_y, top_x:bottom_x])
            # cv2.imwrite(outfile,'{}_{}.jpg',img_rot)


      else:
        cv_interpolation = cv2.INTER_LANCZOS4
        im = cv2.resize(im, dsize=(600,414), interpolation=cv_interpolation)
        # print(width,height)
        q=(width/600)
        p=(height/414)
        print("Q and P valures are")
        print(q,p)
        im_resized,  resize_w, resize_h, (ratio_h, ratio_w) = resize_image(im)

        t = read_tensor_from_image_file(
            images2,
            input_height=resize_h,
            input_width=resize_w,
            input_mean=input_mean,
            input_std=input_std)
        # print(input_name)
        
        start_time = time.time()
        start = time.time()     
        results_1, results_2 = sess.run([output_2_operation.outputs[0], output_operation.outputs[0]], feed_dict={input_operation.outputs[0]: t})
        # print("resultttttt",results_1)
        # print("results_22222222222222222222222222222222222222",results_2)
        end_time = time.time()
        model_time = end_time - start_time
        print("print time for each image--------------", model_time)
     
        end_time = time.time()
        print("ending time",end_time)
        total_time = end_time - start_time

        print("printing the session run time____@____", total_time)
        timer = {'net': 0, 'restore': 0, 'nms': 0}
        # start = time.time()
        # score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})
        timer['net'] = time.time() - start

        boxes, timer = detect(score_map=results_1, geo_map=results_2, timer=timer)
        # print(boxes)
        # print(boxes)
        print('{} : net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
            images2, timer['net']*1000, timer['restore']*1000, timer['nms']*1000))

        if boxes is not None:
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h
            boxes[:, :, 0] *= q
            boxes[:, :, 1] *= p
        duration = time.time() - start_time
        print('[timing] {}'.format(duration))

        fname = os.path.basename(images2)
        cnt = 0

        if boxes is None:
            continue
        for boxxx in boxes:
            # print("after model coordinates",boxxx)
            box = np.copy(boxxx)

            rect = cv2.minAreaRect(box)
            print("rect: {}".format(rect))
            img_crop, img_rot, rect, angle = crop_rect(full_image, rect)
            # cv2.imwrite("image"+str(angle)+"",img_crop
            # print("the angle is angle",angle)

            angle = angle
            rr = cv2.boxPoints(tuple(rect))
            # print("rr",rr)
            box = get_box(rr, box)
            # print("the oput_put coordinates to crop the images",box)

            cnt = cnt + 1
            if box[1, 0] - box[0, 0] > 14 * (box[3, 1] - box[0, 1]):
                # expand region
                start_x = int(min(box[0, 0], box[3, 0]))

                # gray = cv2.cvtColor(img_rot[int(box[0,1]):int(box[3,1]), 0:int(box[1,0]),:], cv2.COLOR_BGR2GRAY)
                gray = img_rot[int(box[0, 1]):int(box[3, 1]), 0:int(box[1, 0]), 1]
                blur = cv2.GaussianBlur(gray, (3, 3), 0)
                ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                new_x = before_find(th3, start_x + int((box[3, 1] - box[0, 1]) / 5), int(box[3, 1] - box[0, 1]))
                box[0, 0] = new_x
                box[3, 0] = new_x
                end_x = int(max(box[1, 0], box[2, 0]))
                gray = cv2.cvtColor(img_rot[int(box[1, 1]):int(box[2, 1]), :, :], cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                new_x = after_find(th3, width, end_x, th3.shape[0])

                box[1, 0] = new_x
                box[2, 0] = new_x

            top_x = int(box[0, 0])
            top_y = int(min(box[1, 1], box[2, 1]))
            bottom_x = int(box[1, 0])
            bottom_y = int(max(box[0, 1], box[3, 1]))
            print(box)
            # cv2.imwrite("imge"+str(angle)+".jpg",img_crop)

            # print("the tx,ty,bx,by",top_x,top_y,bottom_x,bottom_y)
            # cv2.imwrite(os.path.join(outfile,'{}_{}.jpg'.format(fname[:-4], cnt)),img_crop)
            try:
              cv2.imwrite(os.path.join(outfile,'{}_{}.jpg'.format(fname[:-4], cnt)),
                        img_rot[top_y-2:bottom_y+2, top_x-4:bottom_x+4])
            except:
              continue
            full_end_time = time.time()
            full_total_time = full_end_time - full_start_time
  #          print("fst : ", full_start_time)
   #         print("fet : " , full_end_time)
            # with open(c,'a',newline='') as f:
            #     writer = csv.writer(f)
            #     im1 = os.path.join(outfile,'{}_{}.jpg'.format(fname[:-4], cnt))
            #     final2 = os.path.basename(im1)
            #     writer.writerow([final2, box])

      
