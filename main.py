import cv2 as cv
import numpy as np
import time
import os


# Counts how many pixels of each grey value 
def build_histogram(gray_image):

    hist_array = np.zeros(256, dtype=np.int64)  # Empty histogram

    height, width = gray_image.shape  # The image size/shape

    # Goes through every pixel
    for y in range(height):
        for x in range(width):

            pixel_value = gray_image[y, x]  # gets the pixel value

            hist_array[pixel_value] += 1  # Adds to histogram

    return hist_array



# Find automatic threshold value
def compute_threshold(gray_image):

    threshold = int(np.mean(gray_image))  # start with average brightness

    while True:

        bright_group = gray_image[gray_image > threshold]  # bright pixels
        dark_group = gray_image[gray_image <= threshold]   # dark pixels

        if len(bright_group) == 0 or len(dark_group) == 0:
            break

        mean_bright = np.mean(bright_group)  # mean of bright
        mean_dark = np.mean(dark_group)      # mean of dark

        new_threshold = int((mean_bright + mean_dark) / 2)  # new threshold

        if new_threshold == threshold:  # stop if no change
            break

        threshold = new_threshold

    return threshold



# Converts any of the grey image to black and white
def create_binary_image(gray_image, threshold):

    height, width = gray_image.shape

    binary_image = np.zeros((height, width), dtype=np.uint8)  # empty binary images

    for y in range(height):
        for x in range(width):

            if gray_image[y, x] > threshold:  # If brighter than threshold
                binary_image[y, x] = 255      # Make it white

    return binary_image



# Dilation grows the white areas
def apply_dilation(binary_image, kernel):

    k_size = kernel.shape[0]
    offset = k_size // 2

    height, width = binary_image.shape

    result = np.zeros((height, width), dtype=np.uint8)

    for y in range(offset, height - offset):
        for x in range(offset, width - offset):

            found_white = False

            for ky in range(k_size):
                for kx in range(k_size):

                    if kernel[ky, kx] == 1:

                        if binary_image[y - offset + ky, x - offset + kx] == 255:
                            found_white = True  # found white pixel

            if found_white:
                result[y, x] = 255  # make the output white

    return result



# Erosion shrinks the white areas
def apply_erosion(binary_image, kernel):

    k_size = kernel.shape[0]
    offset = k_size // 2

    height, width = binary_image.shape

    result = np.zeros((height, width), dtype=np.uint8)

    for y in range(offset, height - offset):
        for x in range(offset, width - offset):

            all_white = True

            for ky in range(k_size):
                for kx in range(k_size):

                    if kernel[ky, kx] == 1:

                        if binary_image[y - offset + ky, x - offset + kx] == 0:
                            all_white = False  # found black pixel

            if all_white:
                result[y, x] = 255  # keep white

    return result


# Closing = dilation then erosion
def perform_closing(binary_image, kernel):

    step1 = apply_dilation(binary_image, kernel)  # expands the white areas
    step2 = apply_erosion(step1, kernel)          # shrink back to original size but small holes are filled in

    return step2


# Find separate white regions and label them with different numbers (1,2,3 etc)
def find_regions(binary_image):

    height, width = binary_image.shape

    label_map = np.zeros((height, width), dtype=np.int32)

    current_label = 1

     # region growing algorithm using a queue
    for y in range(height):
        for x in range(width):

            if binary_image[y, x] == 255 and label_map[y, x] == 0:

                label_map[y, x] = current_label

                queue = [(y, x)]  # list for region growing    

                while len(queue) > 0:

                    cy, cx = queue.pop(0)

                    neighbours = [
                        (cy-1, cx-1),(cy-1, cx),(cy-1, cx+1),
                        (cy, cx-1),(cy, cx+1),
                        (cy+1, cx-1),(cy+1, cx),(cy+1, cx+1)
                    ]

                    for ny, nx in neighbours:

                        if 0 <= ny < height and 0 <= nx < width:

                            if binary_image[ny, nx] == 255 and label_map[ny, nx] == 0:

                                label_map[ny, nx] = current_label
                                queue.append((ny, nx))

                current_label += 1  # next label for next region    

    return label_map


# Find the biggest region (the O-ring) and return its label number
def largest_region(label_map):

    best_label = -1
    best_size = 0

    max_label = label_map.max()

    for label in range(1, max_label + 1):

        size = np.sum(label_map == label)  # count pixels in this region

        if size > best_size:
            best_size = size
            best_label = label

    return best_label



# Measure properties of the region with the given label number area, centroid, bounding box, circularity, fill ratio
def region_measurements(label_map, label_id):

    area = 0
    sum_y = 0
    sum_x = 0

    min_y = label_map.shape[0]
    max_y = 0
    min_x = label_map.shape[1]
    max_x = 0

    height, width = label_map.shape

    # find area and bounding box
    for y in range(height):
        for x in range(width):
              
            if label_map[y, x] == label_id:

                area += 1
                sum_y += y
                sum_x += x

                min_y = min(min_y, y)
                max_y = max(max_y, y)
                min_x = min(min_x, x)
                max_x = max(max_x, x)

    if area == 0:
        return 0,0,0,0,0,0,0,0.0,0.0

    centroid_y = sum_y / area  # center y
    centroid_x = sum_x / area  # center x

    distances = []

    # check edge pixels
    for y in range(1, height-1):
        for x in range(1, width-1):

            if label_map[y, x] == label_id:

                if (label_map[y-1,x]!=label_id or
                    label_map[y+1,x]!=label_id or
                    label_map[y,x-1]!=label_id or
                    label_map[y,x+1]!=label_id):

                    dist = np.sqrt((y-centroid_y)**2 + (x-centroid_x)**2)
                    distances.append(dist)

    if len(distances) > 0:
        mean_d = np.mean(distances)
        std_d = np.std(distances)
        circularity = mean_d/(std_d+1)
    else:
        circularity = 0

    box_area = (max_y-min_y)*(max_x-min_x) # area of bounding box

    fill_ratio = area/box_area if box_area>0 else 0 # how much of the bounding box is filled by the regionn

    return area,centroid_y,centroid_x,min_y,max_y,min_x,max_x,circularity,fill_ratio


# Find the region with the biggest area and return its properties
# Decide PASS or FAIL based on the measurements and return the result
def pass_fail(area, circularity, fill_ratio):

    result = "PASS"

    if fill_ratio < 0.38:
        result = "FAIL"

    if area < 4000:
        result = "FAIL"

    if area > 13000:
        result = "FAIL"

    return result


# MAIN PROGRAM
def main():

    folder_path = r"C:\Users\dhavi\Desktop\oring_assignment\Orings"

    image_list = sorted(os.listdir(folder_path))  # get list of files in folder

    for filename in image_list:

        if not filename.lower().endswith((".jpg",".png",".jpeg")):
            continue

        path = os.path.join(folder_path, filename)

        colour_image = cv.imread(path)  # colour image for display
        gray_image = cv.imread(path,0)  # grayscale image for processing

        if gray_image is None:
            print("Image could not load")
            continue

        print("Processing:",filename)
        

        start = time.time()  # start timer

        histogram = build_histogram(gray_image)
    
        threshold = compute_threshold(gray_image)

        print("Threshold:", threshold)
         
        binary = create_binary_image(gray_image,threshold)

        # if more than 50% of the pixels are white, invert the binary image (some images have white background and black O-ring)
        if np.sum(binary==255) > (binary.size*0.5):
            binary = 255-binary

         
         # create a 5x5 kernel of ones for morphological operations
        kernel = np.ones((5,5),dtype=np.uint8)

        cleaned = perform_closing(binary,kernel)

        labels = find_regions(cleaned) 
   
        biggest = largest_region(labels) # find the biggest region (the O-ring) and return its label number

        if biggest == -1:
            print("No O-ring found\n")
            continue

        (area,cy,cx,miny,maxy,minx,maxx,circ,fill) = region_measurements(labels,biggest)

        result = pass_fail(area,circ,fill)

        end = time.time()

        print("Area:",area)
        print("Circularity:",round(circ,3))
        print("Fill:",round(fill,3))
        print("Result:",result)
        print("Time:",round(end-start,3),"s\n")

        display = colour_image.copy()

        # draw bounding box and put text result on the image
        if result=="PASS":
            colour=(0,220,0)
        else:
            colour=(0,0,220)
          
        cv.rectangle(display,(minx,miny),(maxx,maxy),colour,2)
           
        cv.putText(display,result,(10,40),
                   cv.FONT_HERSHEY_SIMPLEX,1.2,colour,3)

        cv.imshow("Result "+filename,display)

        cv.waitKey(0)
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()