import os
import time
import sys
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def get_plateau_new(image, axis, Th_h=50, Th_v=15):
    sum_vals_thresh = image.sum(axis=axis)
    #print(sum_vals_thresh.shape[0])
    #sys.exit()
    #print(sum_vals_thresh.shape)
    Th = Th_h if axis == 0 else Th_v
    count, i = 0, 0
    counts, peaks = [], []
    while i < sum_vals_thresh.shape[0]:
        l = 0
        while (i < sum_vals_thresh.shape[0]) and not sum_vals_thresh[i]:
            #print(sum_vals_thresh[i])
            #sys.exit()
            count += 1
            l = 1
            i += 1
        if l == 1:
            counts.append(count)
            if count % 2 == 0:
                plat = count // 2
            else:
                plat = count // 2 + 1
            peaks.append(i - plat)
            count = 0
            # i -= 1
        i += 1
    if not sum_vals_thresh[0] and counts:
        counts.pop(0)
        peaks.pop(0)
    if not sum_vals_thresh[-1] and counts:
        counts.pop(-1)
        peaks.pop(-1)

    final_peaks = [peaks[ind] for ind, val in enumerate(counts) if val > Th]
    return final_peaks


def vertical_split(peaks_list, section):
    splitted_regions_v = []
    for im in range(len(peaks_list)):
        if im == 0:
            splitted_regions_v.append(section[:, : peaks_list[0]])
        else:
            splitted_regions_v.append(section[:, peaks_list[im - 1]: peaks_list[im]])

        if im == (len(peaks_list) - 1):
            splitted_regions_v.append(section[:, peaks_list[len(peaks_list) - 1]:])
    return splitted_regions_v


def horizontal_split(peaks_list, section):
    splitted_regions_h = []

    for im in range(len(peaks_list)):
        if im == 0:
            splitted_regions_h.append(section[: peaks_list[0], :])
        else:
            splitted_regions_h.append(section[peaks_list[im - 1]: peaks_list[im], :])
        if im == (len(peaks_list) - 1):
            splitted_regions_h.append(section[peaks_list[len(peaks_list) - 1]:, :])
    return splitted_regions_h


def get_relative_coordinates(peaks_list, origin_coordinates, axis):
    x, y, w, h = origin_coordinates
    new_coordinates = []

    for im in range(len(peaks_list)):
        if im == 0:
            if axis == 0:
                new_coordinates.append([x, y, peaks_list[im], h])
            else:
                new_coordinates.append([x, y, w, peaks_list[im]])
        else:
            if axis == 0:
                new_coordinates.append(
                    [x + peaks_list[im - 1], y, peaks_list[im] - peaks_list[im - 1], h]
                )
            else:
                new_coordinates.append(
                    [x, y + peaks_list[im - 1], w, peaks_list[im] - peaks_list[im - 1]]
                )

        if im == (len(peaks_list) - 1):
            if axis == 0:
                new_coordinates.append([x + peaks_list[im], y, w - peaks_list[im], h])
            else:
                new_coordinates.append([x, y + peaks_list[im], w, h - peaks_list[im]])
    return new_coordinates


def crop_new(image):
    if np.all(image == 0):
        return 0, int(image.shape[1]), 0, int(image.shape[0])
    (y, x) = np.where(image == 255)
    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    return topx, (bottomx - topx) + 1, topy, (bottomy - topy) + 1


def get_density(image):
    total_pixels = image.shape[0] * image.shape[1]
    return np.sum((image).astype(int)) / total_pixels


def get_text_cells(only_text_img, debug_img, pageno, width_, height_, Th_h=10, Th_v=10, ratio=0.25, debug=False):
    #print('inside')
    gray_image = cv2.cvtColor(only_text_img, cv2.COLOR_BGR2GRAY)
    height, width = gray_image.shape
    new_height = int(height * ratio)
    new_width = int(ratio * width)
    Th_v, Th_h = (Th_v * ratio), (Th_h * ratio)
    scaled_image = cv2.resize(gray_image, (new_width, new_height))
    scaled_image = cv2.threshold(
        scaled_image, 200, 255, cv2.THRESH_BINARY_INV
    )[1]

    h, w = scaled_image.shape
    # Initialize initial input
    sections = [scaled_image]
    section_coordinates = [[0, 0, w, h]]
    extracted_coordinates = []

    # While loop till there are sections to split
    while len(sections) > 0:
        # Initialize list of sections to process
        remaining_sections = []
        remaining_coordinates = []
        for i, section in enumerate(sections):
            peaks_list_v = get_plateau_new(section, 0, Th_h=Th_h, Th_v=Th_v)
            peaks_list_h = get_plateau_new(section, 1, Th_h=Th_h, Th_v=Th_v)
            if len(peaks_list_h):
                horizontal_splits = horizontal_split(peaks_list_h, section)
                horizontal_coordinates = get_relative_coordinates(
                    peaks_list_h, section_coordinates[i], 1
                )
                remaining_sections.extend(horizontal_splits)
                remaining_coordinates.extend(horizontal_coordinates)
            elif len(peaks_list_v):
                vertical_splits = vertical_split(peaks_list_v, section)
                vertical_coordinates = get_relative_coordinates(
                    peaks_list_v, section_coordinates[i], 0
                )
                remaining_sections.extend(vertical_splits)
                remaining_coordinates.extend(vertical_coordinates)
            else:
                extracted_coordinates.append(section_coordinates[i])

        sections = remaining_sections.copy()
        section_coordinates = remaining_coordinates.copy()

    extracted_coordinates = np.array(extracted_coordinates) / ratio
    image = cv2.resize(scaled_image, (width, height))
    image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)[1]
    extracted_cropped = []

    for x, y, w, h in extracted_coordinates:
        x, y, w, h = int(x), int(y), int(w), int(h)
        # if not cropped and (w >= 0.9 * width) and (h >= 0.9 * height):
        #     continue
        temp = image[y: y + h, x: x + w]
        x_, w_, y_, h_ = crop_new(temp)
        if w_ and h_:
            if (get_density(temp[y_: y_ + h_, x_: x_ + w_]) < 0.009) or (
                    (min(h_, w_) <= min(height, width) / 150)
                    and max(h_, w_) / min(h_, w_) >= 20
            ):
                continue
            extracted_cropped.append(
                [
                    max(x + x_ - 5, 0),
                    max(y + y_ - 5, 0),
                    min(x + x_ + w_ + 5, width - 1),
                    int(min(y + y_ + h_ + 6, height - 1)),
                    pageno,
                    width_,
                    height_,
                    "text",
                ]
            )
    headers = ['x1', 'y1', 'x2', 'y2', 'pageno', 'width', 'height', 'type']
    df_out = pd.DataFrame(extracted_cropped, columns=headers)

    if debug:
        for x, y, w, h in zip(df_out.x1, df_out.y1, df_out.x2, df_out.y2):
            cv2.rectangle(debug_img, (x, y), (w, h), (0, 0, 255), 2)
    return df_out, debug_img 


if __name__ == "__main__":
    input_dir = "./test_sample"
    debug = True
    ratio = 6
    output_dir = input_dir + str(ratio)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # output = pd.DataFrame()
    for name in os.listdir(input_dir):
        if not name.endswith(".png"):
            continue
        image = cv2.imread(os.path.join(input_dir, name))
        #print(image.shape)
        debug_img = image.copy()
        height,width = image.shape[0], image.shape[1]
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        df_out, debug_img = get_text_cells(image, debug_img, pageno=1, height_=height, width_=width, ratio=0.25,
                                           debug=True)

        out_path = os.path.join(output_dir, name)
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(out_path, debug_img)
    df_out.to_csv(out_path + ".csv")
        # output = pd.concat([output, df_out], ignore_index=True)
    # output.to_csv("output.csv")
