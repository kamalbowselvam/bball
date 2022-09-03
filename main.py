# -*- coding: utf-8 -*-
# @Time    : 9/1/2022 4:31 PM
# @Author  : Kamal SELVAM
# @Email   : kamal.selvam@orange.com
# @File    : main.py.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion, iterate_structure
plt.rcParams['figure.figsize'] = [10, 10]
from skimage.feature import peak_local_max
import math

class HoughBundler:
    def __init__(self,min_distance=5,min_angle=2):
        self.min_distance = min_distance
        self.min_angle = min_angle

    def get_orientation(self, line):
        orientation = math.atan2(abs((line[3] - line[1])), abs((line[2] - line[0])))
        return math.degrees(orientation)

    def check_is_line_different(self, line_1, groups, min_distance_to_merge, min_angle_to_merge):
        for group in groups:
            for line_2 in group:
                if self.get_distance(line_2, line_1) < min_distance_to_merge:
                    orientation_1 = self.get_orientation(line_1)
                    orientation_2 = self.get_orientation(line_2)
                    if abs(orientation_1 - orientation_2) < min_angle_to_merge:
                        group.append(line_1)
                        return False
        return True

    def distance_point_to_line(self, point, line):
        px, py = point
        x1, y1, x2, y2 = line

        def line_magnitude(x1, y1, x2, y2):
            line_magnitude = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
            return line_magnitude

        lmag = line_magnitude(x1, y1, x2, y2)
        if lmag < 0.00000001:
            distance_point_to_line = 9999
            return distance_point_to_line

        u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
        u = u1 / (lmag * lmag)

        if (u < 0.00001) or (u > 1):
            #// closest point does not fall within the line segment, take the shorter distance
            #// to an endpoint
            ix = line_magnitude(px, py, x1, y1)
            iy = line_magnitude(px, py, x2, y2)
            if ix > iy:
                distance_point_to_line = iy
            else:
                distance_point_to_line = ix
        else:
            # Intersecting point is on the line, use the formula
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            distance_point_to_line = line_magnitude(px, py, ix, iy)

        return distance_point_to_line

    def get_distance(self, a_line, b_line):
        dist1 = self.distance_point_to_line(a_line[:2], b_line)
        dist2 = self.distance_point_to_line(a_line[2:], b_line)
        dist3 = self.distance_point_to_line(b_line[:2], a_line)
        dist4 = self.distance_point_to_line(b_line[2:], a_line)

        return min(dist1, dist2, dist3, dist4)

    def merge_lines_into_groups(self, lines):
        groups = []  # all lines groups are here
        # first line will create new group every time
        groups.append([lines[0]])
        # if line is different from existing gropus, create a new group
        for line_new in lines[1:]:
            if self.check_is_line_different(line_new, groups, self.min_distance, self.min_angle):
                groups.append([line_new])

        return groups

    def merge_line_segments(self, lines):
        orientation = self.get_orientation(lines[0])

        if(len(lines) == 1):
            return np.block([[lines[0][:2], lines[0][2:]]])

        points = []
        for line in lines:
            points.append(line[:2])
            points.append(line[2:])
        if 45 < orientation <= 90:
            #sort by y
            points = sorted(points, key=lambda point: point[1])
        else:
            #sort by x
            points = sorted(points, key=lambda point: point[0])

        return np.block([[points[0],points[-1]]])

    def process_lines(self, lines):
        lines_horizontal  = []
        lines_vertical  = []

        for line_i in [l[0] for l in lines]:
            orientation = self.get_orientation(line_i)
            # if vertical
            if 45 < orientation <= 90:
                lines_vertical.append(line_i)
            else:
                lines_horizontal.append(line_i)

        lines_vertical  = sorted(lines_vertical , key=lambda line: line[1])
        lines_horizontal  = sorted(lines_horizontal , key=lambda line: line[0])
        merged_lines_all = []

        # for each cluster in vertical and horizantal lines leave only one line
        for i in [lines_horizontal, lines_vertical]:
            if len(i) > 0:
                groups = self.merge_lines_into_groups(i)
                merged_lines = []
                for group in groups:
                    merged_lines.append(self.merge_line_segments(group))
                merged_lines_all.extend(merged_lines)

        return np.asarray(merged_lines_all)

def frame_extraction(input_file,start_frame,end_frame,verbose):

    cap = cv2.VideoCapture(input_file)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if verbose:
        print ('Filename: ' + input_file)
        print ('Open success?: ' + str(cap.isOpened()))
        print ('Frame width: %d' % cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print ('Frame height: %d' % cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print ('FPS: ' + str(cap.get(cv2.CAP_PROP_FPS)))
        print ('Frame count: %d' % frame_count)

    cap.read()
    # Get length in msec of movie, then reset
    cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
    len_in_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
    cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)

    # Compute frames to capture
    if verbose:
        print('Extracting frames %d to %d.' % (start_frame, end_frame))

    # Extract
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_num in range(start_frame, end_frame + 1):
        success, image = cap.read()
        frames.append(image)
        if verbose:
            cv2.imwrite('./' + '%d.jpg' % frame_num, image)

    return frames


def calculate_historgram(img,verbose=True):
    CROWD_TOP_HEIGHT_FRACTION = .375
    CROWD_BOTTOM_HEIGHT_FRACTION = .2
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    ignore_crowd = False
    if ignore_crowd:
        img = img[CROWD_TOP_HEIGHT_FRACTION*img.shape[0] : -CROWD_BOTTOM_HEIGHT_FRACTION*img.shape[0]]

    hist = cv2.calcHist([img],[1,2],None,[256,256],[0,256,0,256])

    if verbose:
        from matplotlib import cm
        xx, yy = np.meshgrid(np.linspace(0,1,256), np.linspace(0,1,256))

        # create vertices for a rotated mesh (3D rotation matrix)
        X =  xx
        Y =  yy
        hist = hist/np.max(hist)
        data = hist
        # create the figure


        # show the reference image
        #ax1 = fig.add_subplot(121)
        #ax1.imshow(data, interpolation='nearest')


        # show the 3D rotated projection
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        cset = ax.contourf(X, Y, data, 100)
        plt.savefig("histogram_3D.png")


        fig = plt.figure()
        plt.imshow(data, interpolation='nearest')
        plt.savefig("histogram_2D.png")

    return hist



def detect_peaks(image):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    image_max = maximum_filter(image, size=3, mode='constant')
    coordinates = peak_local_max(image, min_distance=20)
    print(coordinates)


    if verbose:
        fig = plt.figure()
        plt.imshow(image)
        plt.plot(coordinates[0, 1], coordinates[0, 0], 'r.')
        plt.plot(coordinates[1, 1], coordinates[1, 0], 'r.')
        plt.savefig("histogram_peaks.png")

    return image_max, coordinates


def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def ycbcr_to_bgr(ycbcr_img):
    img = ycbcr_img.copy()
    return cv2.cvtColor(img, cv2.COLOR_YCR_CB2BGR)

def ycbcr_to_gray(ycbcr_img):
    print('called ycbcr')
    img = ycbcr_img.copy()
    img = ycbcr_to_bgr(img)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def find_dominant_color(img):

    data = np.reshape(img, (-1,3))
    print(data.shape)
    data = np.float32(data)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness,labels,centers = cv2.kmeans(data,2,None,criteria,10,flags)
    print('Dominant color is: bgr({})'.format(centers[0].astype(np.int32)))
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    res = res.reshape(img.shape)
    layer = res.copy()
    mask = cv2.inRange(layer, centers[1], centers[1])

    # apply mask to layer
    layer[mask == 0] = [0,0,0]
    layer[mask != 0] = [255,255,255]
    #cv2.imshow('layer', layer)
    # save kmeans clustered image and layer
    cv2.imwrite("jellyfish_layer.png", layer)

    return layer


def get_double_flooded_mask(gray_mask):
    gray_flooded = fill_holes_with_contour_filling(gray_mask)
    gray_flooded2 = fill_holes_with_contour_filling(gray_flooded, inverse=True)
    return gray_flooded2

def fill_holes_with_contour_filling(gray_mask, inverse=False):
    filled = gray_mask.copy()
    if inverse:
        filled = cv2.bitwise_not(filled)
    contour, _ = cv2.findContours(filled,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        print(cnt)
        cv2.drawContours(filled, [cnt], 0, 255, -1)
    if inverse:
        filled = cv2.bitwise_not(filled)
    return filled



def create_court_mask(_bgr_img, dominant_colorset, binary_gray=False):
    img = cv2.cvtColor(_bgr_img, cv2.COLOR_BGR2YCR_CB)
    YCBCR_BLACK = (0,128,128)
    YCBCR_WHITE = (255,128,128)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            print(row,col)
            idx = (row, col)
            _, cr, cb = img[idx]
            if (cr, cb) not in dominant_colorset:
                img[idx] = YCBCR_BLACK
            elif binary_gray:
                img[idx] = YCBCR_WHITE

    return ycbcr_to_gray(img) if binary_gray else img

def get_top_pixels(court_mask):
    top_pixels = np.copy(court_mask)
    # print 'Num columns is %s' %(court_mask.shape[1])
    # print 'Num rows is %s' %(court_mask.shape[0])
    for col in range(court_mask.shape[1]):
        top_found = False
        for row in range(court_mask.shape[0]):
            if top_found:
                top_pixels[row][col] = 0
            else:
                if top_pixels[row][col]:
                    # print "Row is %s, column is %s, binary value is %s" %(row, col, top_line_only[row][col])
                    top_found = True
    return top_pixels


def hough_find_top_line(top_line_only):
    top_line_copy = np.copy(top_line_only)
    lines = cv2.HoughLines(top_line_copy,5,np.pi/180 * 3,75)[0]
    # print 'The number of lines with threshold at %d is %d' %(75, len(lines))

    theta_0 = lines[0][1]
    rho_0 = lines[0][0]
    theta_1 = None
    rho_1 = 0

    for rho,theta in lines[1:]:
        if abs(theta_0 - theta) > 0.4:
            theta_1 = theta
            rho_1 = rho
            break

    # # To print lines
    # a = np.cos(theta_0)
    # b = np.sin(theta_0)
    # x0 = a*rho_0
    # y0 = b*rho_0
    # x1 = int(x0 + 1000*(-b))
    # y1 = int(y0 + 1000*(a))
    # x2 = int(x0 - 1000*(-b))
    # y2 = int(y0 - 1000*(a))
    # cv2.line(top_line_copy,(x1,y1),(x2,y2),(82,240,90),2)

    # a = np.cos(theta_1)
    # b = np.sin(theta_1)
    # x0 = a*rho_1
    # y0 = b*rho_1
    # x1 = int(x0 + 1000*(-b))
    # y1 = int(y0 + 1000*(a))
    # x2 = int(x0 - 1000*(-b))
    # y2 = int(y0 - 1000*(a))
    # cv2.line(top_line_copy,(x1,y1),(x2,y2),(82,240,90),2)

    # print 'The first theta is %s, the second theta is %s' %(theta_0, theta_1)

    if theta_0 < 1.6:
        theta_sideline = theta_0
        theta_baseline = theta_1
        rho_sideline = rho_0
        rho_baseline = rho_1
    else:
        theta_sideline = theta_1
        theta_baseline = theta_0
        rho_sideline = rho_1
        rho_baseline = rho_0

    # Find intersection point
    if theta_baseline:
        return ((rho_sideline, theta_sideline), (rho_baseline, theta_baseline))
    else:
        return ((rho_sideline, theta_sideline))

def put_lines_on_img(bgr_img, lines_rho_theta):
    redness = np.linspace(0, 255, len(lines_rho_theta))
    redness = np.floor(redness)
    blueness = 255 - redness
    for i, (rho, theta) in enumerate(lines_rho_theta):
        # print 'The parameters of the line: rho = %s, theta = %s' %(rho, theta)
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        red = redness[i]
        blue = blueness[i]
        cv2.line(bgr_img,(x1,y1),(x2,y2),(blue,0,red),2)

def FLD(image):
    # Create default Fast Line Detector class
    fld = cv2.createLineSegmentDetector(0)
    # Get line vectors from the image
    lines = fld.detect(image)[0]
    # Draw lines on the image
    line_on_image = fld.drawSegments(image, lines)
    #for line in lines:
    #    x1, y1, x2, y2 = line[0]
    #    print(x1,y1,x2,y2)
    #    cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 20)
    # Plot
    plt.figure()
    plt.imshow(line_on_image, interpolation='nearest')
    plt.savefig("court_fld.png")
    return line_on_image

if __name__ == "__main__":
    input_file = "ATLANTA HAWKS AT CLEVELAND CAVALIERS_102321_3.mp4"
    verbose = True
    start_frame = 16500
    end_frame = 16501
    BGR_BLACK = (0,0,0)
    BGR_RED = (0, 0, 255)
    BGR_BLUE = (255, 0, 0)

    bgr_img = frame_extraction(input_file,start_frame,end_frame,verbose=True)
    img_histogram = calculate_historgram(bgr_img[0].copy(),verbose=True)
    blur = cv2.blur(img_histogram,(5,5))
    img_max, peak_location = detect_peaks(blur)

    cluster = find_dominant_color(bgr_img[0].copy())
    gray = cv2.cvtColor(cluster, cv2.COLOR_BGR2GRAY)
    fig = plt.figure()
    plt.imshow(gray)
    print(gray.shape)
    plt.savefig("court_gray.png")

    kernel = np.ones((5, 5), np.uint8)
    e1 = cv2.erode(gray,kernel)
    #e2 = cv2.erode(e1,kernel)
    d1 = cv2.dilate(e1,kernel)
    #d2 = cv2.dilate(d1,kernel)


    contour, _ = cv2.findContours(d1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    k = 0
    areas = []
    for i in range(len(contour)):
        area = cv2.contourArea(contour[i])
        areas.append(area)

    idx = np.argmax(areas)
    cnt = contour[idx]

    image_binary = np.zeros((bgr_img[0].shape[0],
                             bgr_img[0].shape[1], 1),
                            np.uint8)

    cv2.drawContours(image_binary, [contour[idx]], 0, 255, -1)

    #FLD(image_binary)

    #filled =  get_double_flooded_mask(d1)
    perimeter = cv2.arcLength(cnt,True)
    epsilon = 0.01*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)

    #hull = cv2.convexHull(cnt)

    # cv2.isContourConvex(cnt)
    #ib = np.zeros((bgr_img[0].shape[0],bgr_img[0].shape[1], 1),np.uint8)
    #cv2.drawContours(image_binary, cnt, -1, (255, 255, 255), 3)
    #print(approx)
    #cv2.drawContours(image_binary, approx,  -1, (255, 255, 255), 10)
    #print(cnt.shape)



    #plt.figure()
    #plt.imshow(image_binary)
    #plt.savefig("contour_lines.png")

    tp = get_top_pixels(image_binary)
    print(tp)

    low_threshold = 100
    high_threshold = 150
    edges = cv2.Canny(image_binary, low_threshold, high_threshold)

    #top_line_copy = np.copy(tp)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180,10,50,5)
    #bundler = HoughBundler(min_distance=30,min_angle=5)
    #lines = bundler.process_lines(lines)
    #print(lines)
    ib = np.zeros((bgr_img[0].shape[0],
                             bgr_img[0].shape[1], 1),
                            np.uint8)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.atan2(y1 - y2, x1 - x2) * (180/np.pi)
        if abs(angle) != 90.0:
            cv2.line(ib, (x1, y1), (x2, y2), (255, 0, 0), 3)



    #put_lines_on_img(tp,[out])
    plt.imshow(ib)
    plt.savefig("contour_lines.png")

    '''
    mask1 = create_circular_mask(256,256,(peak_location[0, 1],peak_location[0, 0]),radius=10)
    nx,ny = mask1.nonzero()
    dominantset = []
    for x,y in zip(nx,ny):
        dominantset.append((x,y))


    court_mask = create_court_mask(bgr_img[0],dominantset,binary_gray=True)
    
    #masked_img = img_max.copy()
    #masked_img[mask] = 1
    fig = plt.figure()
    plt.imshow(court_mask)
    plt.savefig("court_mask.png")
    '''




