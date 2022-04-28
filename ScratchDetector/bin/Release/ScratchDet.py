import numpy as np
import cv2
import random as rand
import datetime
from numba import njit, prange
import argparse
import sys
import os
import time

def createParser():
    r = argparse.ArgumentParser()
    r.add_argument('-i', '--image')
    return r


def path_leaf(path):
    import ntpath
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def otsu_canny(image, lowrate=0.1):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, _ = cv2.threshold(image, thresh=0, maxval=255,
                           type=(cv2.THRESH_BINARY + cv2.THRESH_OTSU))
    edged = cv2.Canny(image, threshold1=(ret * lowrate), threshold2=ret)
    return edged


def single_filt(img):
    median = np.median(img)
    sigma = 0.3
    low = int(max(0, (1 - sigma) * median))
    up = int(min(255, (1 + sigma) * median))
    lower = np.array(low)
    upper = np.array(up)
    img = cv2.inRange(img, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    img = cv2.dilate(img, kernel)
    return img


def three_filt(img):
    median = np.median(img)
    sigma = 0.3
    low = int(max(0, (1 - sigma) * median))
    up = int(min(255, (1 + sigma) * median))
    lower = np.array([low, low, low])
    upper = np.array([up, up, up])
    img = cv2.inRange(img, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    img = cv2.dilate(img, kernel)
    return img


def Hough(edge_image, max_line_gap=7, min_line_length=50):
    rho_res = .1  # [pixels]
    theta_res = np.pi / 180.  # [radians]
    threshold = 0  # [# votes]
    lines = cv2.HoughLinesP(edge_image, rho_res, theta_res, threshold, np.array([]),
                            minLineLength=min_line_length, maxLineGap=max_line_gap)
    return lines


def get_large_rec(contours):
    maxS = 0
    rec = None
    if len(contours) != 0:
        for (i, c) in enumerate(contours):
            area = cv2.contourArea(c)
            if area > 300:
                r = cv2.boundingRect(c)
                S = r[2] * r[3]
                if S > maxS:
                    maxS = S
                    rec = r
    return rec

@njit
def light_compare(edges, found):
    for i in prange(edges.shape[0]):
        for j in prange(edges.shape[1]):
            if edges[i][j] != 0:
                found[i][j][0] = 0
                found[i][j][1] = 255
                found[i][j][2] = 0
                found[0][0][0] = 0


def light_canny(found):
    found_copy = found.copy()
    median = np.median(found)
    sigma = 0.3
    low = int(max(0, (1 - sigma) * median))
    up = int(min(255, (1 + sigma) * median))
    edges = cv2.Canny(found, low, up)
    lines = Hough(edges, 20, 80)
    count = 0
    if lines is not None:
        count = drawRects(lines, found_copy, 200)
    light_compare(edges, found)
    return found_copy, count


@njit
def lower(img, delta=0.1):
    for i in prange(img.shape[0]):
        for j in prange(img.shape[1]):
            img[i][j] = img[i][j] * delta

@njit
def compare(edges, copy, ed, f):
    for i in prange(edges.shape[0]):
        for j in prange(edges.shape[1]):
            if edges[i][j] != 0 and copy[i][j][0] == 0 and copy[i][j][1] == 0 and copy[i][j][2] == 255:
                f[i][j][0] = 0
                f[i][j][1] = 255
                f[i][j][2] = 0
                ed[i][j] = 255
                if i + 1 < edges.shape[0]:
                    ed[i + 1][j] = 255
                if j + 1 < edges.shape[1]:
                    ed[i][j + 1] = 255
                    if i + 1 < edges.shape[0]:
                        ed[i + 1][j + 1] = 255

            else:
                ed[i][j] = 0

def detection(found, edges, gap, leng):
    lines = Hough(edges, max_line_gap=gap, min_line_length=leng)
    copy = found.copy()
    f = found.copy()
    if lines is not None:
        for i in range(0, len(lines)):
            coords = lines[i][0]
            cv2.line(copy, (coords[0], coords[1]), (coords[2],
                     coords[3]), (0, 0, 255), 8, cv2.LINE_AA)
    ed = edges.copy()
    
    if edges is not None:
        compare(edges, copy, ed, f)
    max_dimension = float(max(copy.shape))
    scale = 900 / max_dimension
    copy = cv2.resize(copy, None, fx=scale, fy=scale)
    return f, ed, lines


#@njit
def getLine(lines, maxGap, line):
    buf = [line]
    lines.remove(line)
    linesStack = [line]
    while len(linesStack) != 0:
        line = linesStack.pop()
        lines_copy = lines.copy()
        for line_copy in lines_copy:
            if ((np.sqrt((line_copy[0][0]-line[0][0])**2+(line_copy[0][1]-line[0][1])**2) < maxGap) or
            (np.sqrt((line_copy[0][2]-line[0][2])**2+(line_copy[0][3]-line[0][3])**2) < maxGap) or
            (np.sqrt((line_copy[0][0]-line[0][0])**2+(line_copy[0][3]-line[0][3])**2) < maxGap) or
            (np.sqrt((line_copy[0][2]-line[0][2])**2+(line_copy[0][1]-line[0][1])**2) < maxGap)):
                linesStack.append(line_copy)
                lines.remove(line_copy)
                buf.append(line_copy)
    return buf

#@njit
def drawRects(lines, img, maxGap):
    lines_copy = lines.tolist()
    clasterCount = 0
    while len(lines_copy) > 0:
        clasterCount = clasterCount + 1
        line = lines_copy[0]
        buf = getLine(lines_copy, maxGap, line)
        minX = img.shape[1]
        minY = img.shape[0]
        maxX = 0
        maxY = 0
        count = 1
        for Line in buf:
            count = count + 1
            cv2.line(img, (Line[0][0], Line[0][1]),
                          (Line[0][2], Line[0][3]), (255, 0, 0), 10)
            minX = min(minX, Line[0][0], Line[0][2])
            minY = min(minY, Line[0][1], Line[0][3])
            maxX = max(maxX, Line[0][0], Line[0][2])
            maxY = max(maxY, Line[0][1], Line[0][3])
        cv2.rectangle(img, (minX, minY), (maxX, maxY), (rand.randint(100, 255), rand.randint(100, 255), rand.randint(100, 255)), 10)
    return clasterCount

def cont(img, result):  # dark pics
    contours, hierarchy = cv2.findContours(
        result.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    im_copy = img.copy()
    segmCount = 0
    if len(contours) != 0:
        for (i, c) in enumerate(contours):
            area = cv2.contourArea(c)
            if area > 10000:
                r = cv2.boundingRect(c)
                found = img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]]
                x, y, w, h = r[0:]
                edges = TH(found)
                gaps = [20, 4, 40]
                leng = [25, 40, 130]
                save = found.copy()
                #found = (255 - found)
                found_copy = im_copy[r[1]:r[1] + r[3], r[0]:r[0] + r[2]]
                for i in range(3):
                    found, edges, lines = detection(save, edges, gaps[i], leng[i])
                    cp = found.copy()
                    max_dimension = float(max(cp.shape))
                    scale = 900 / max_dimension
                    cp = cv2.resize(cp, None, fx=scale, fy=scale)
                if lines is not None:
                    segmCount += drawRects(lines, found_copy, 500)
                img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = found
                im_copy[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = found_copy
                cv2.drawContours(im_copy, c, -1, (255, 255, 0), 12)
    #print("SEGMENTS - ", segmCount)
    #max_dimension = float(max(im_copy.shape))
    #scale = 900 / max_dimension
    #im_copy = cv2.resize(im_copy, None, fx=scale, fy=scale)
    return img, im_copy, segmCount


def light_cont(img, result): # light pics
    contours, hierarchy = cv2.findContours(
        result.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    im_copy = img.copy()
    if len(contours) != 0:
        segmCount = 0
        for (i, c) in enumerate(contours):
            area = cv2.contourArea(c)
            if area > 10000:
                r = cv2.boundingRect(c)
                found = img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]]
                found_copy = im_copy[r[1]:r[1] + r[3], r[0]:r[0] + r[2]]
                x, y, w, h = r[0:]
                #cv2.rectangle(im_copy, (x, y), (x + w, y + h),(255, 0, 255), 12)
                found_copy, count = light_canny(found)
                segmCount += count
                img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = found
                im_copy[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = found_copy
                cv2.drawContours(im_copy, c, -1, (255, 255, 0), 12)
    #max_dimension = float(max(im_copy.shape))
    #scale = 900 / max_dimension
    #im_copy = cv2.resize(im_copy, None, fx=scale, fy=scale)
    return img, im_copy, segmCount


def edge_demo(image):
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_16SC1, 1, 0)
    grad_y = cv2.Sobel(gray, cv2.CV_16SC1, 0, 1)
    x_grad = cv2.convertScaleAbs(grad_x)
    y_grad = cv2.convertScaleAbs(grad_y)
    src1 = cv2.addWeighted(x_grad, 0.5, y_grad, 0.5, 0)

    edge = cv2.Canny(src1, 50, 100)
    edge1 = cv2.Canny(grad_x, grad_y, 10, 100)
    edge2 = cv2.bitwise_and(image, image, mask=edge1)
    return edge


def morph(im):
    morph = im.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    # take morphological gradient
    gradient_image = cv2.morphologyEx(morph, cv2.MORPH_GRADIENT, kernel)

    # split the gradient image into channels
    image_channels = np.split(np.asarray(gradient_image), 3, axis=2)

    channel_height, channel_width, _ = image_channels[0].shape

    # apply Otsu threshold to each channel
    for i in range(0, 3):
        _, image_channels[i] = cv2.threshold(~image_channels[i], 254, 255,
                                             cv2.THRESH_BINARY_INV | cv2.THRESH_TOZERO_INV)
        image_channels[i] = np.reshape(
            image_channels[i], newshape=(channel_height, channel_width, 1))

    # merge the channels
    image_channels = np.concatenate(
        (image_channels[0], image_channels[1], image_channels[2]), axis=2)
    return image_channels


def TH(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(gray, (71, 71))
    diff = cv2.subtract(blur, gray)
    ret, th = cv2.threshold(diff, 13, 255, cv2.THRESH_BINARY)
    return th


def main():
    arg_parser = createParser()
    namespace = arg_parser.parse_args(sys.argv[1:])
    imgName = namespace.image
    try:
        start_time = time.time()
        img = cv2.imdecode(np.fromfile(
            imgName, dtype=np.uint8), cv2.IMREAD_COLOR)
        median = np.median(img)
        im_copy = None
        segmCount = 0
        if median < 60:
            ret, thresh5 = cv2.threshold(img, 255, 13, cv2.THRESH_TOZERO_INV)
            result = thresh5
            result = cv2.GaussianBlur(result, (75, 75), 0)
            result = three_filt(result)
            img, im_copy, segmCount = cont(img, result)
            path = os.getcwd()
            if not os.path.isdir(path + '/countours_img'):
                os.mkdir('countours_img')
            cv2.imwrite(os.path.join(path + '/countours_img',
                                     path_leaf(imgName)), im_copy)
        else:
            count = 0
            lower(img, delta=1.0)
            result = edge_demo(img)
            result = cv2.GaussianBlur(result, (75, 75), 0)
            result = single_filt(result)
            img, im_copy, segmCount = light_cont(img, result)
            path = os.getcwd()
            if not os.path.isdir(path + '/countours_img'):
                os.mkdir('countours_img')
            cv2.imwrite(os.path.join(path + '/countours_img',
                                     path_leaf(imgName)), im_copy)
        path = os.getcwd()
        if not os.path.isdir(path+'/out_img'):
            os.mkdir('out_img')
        cv2.imwrite(os.path.join(path + '/out_img', path_leaf(imgName)), img)
        end_time = time.time()
        with open('info.log', 'a') as log_file:
            log_file.write(
                f'image:{path_leaf(imgName)} time:{datetime.datetime.now().time()} date:{datetime.datetime.now().date()} found:{segmCount > 0} scrathes:{segmCount} process-time:{round(end_time - start_time, 2)}s' )
            log_file.write('\n')
    except Exception as e:
        sys.stderr.write(str(e))


if __name__ == "__main__":
    main()
