import pytest 
import page_segmentation as pg
import cv2
image = cv2.imread("test_image.png")

def test_get_text_cells():
	debug_img = image.copy()
	height, width = image.shape[0], image.shape[1]
	df_out, debug_img = pg.get_text_cells(image, debug_img, pageno=1, height_=height,
									   width_=width, ratio=0.25,debug=True)
	assert len(df_out) > 0, "df_out function error"
	assert len(debug_img) > 0, "df_out debug_img error"

def test_get_density():
	val=pg.get_density(image)
	assert val>=0, "get_density fucntion undefined value error"


def test_get_relative_coordinates():
	Th_h = 10
	Th_v = 10
	ratio = 0.25
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	height, width = gray_image.shape
	new_height = int(height * ratio)
	new_width = int(ratio * width)
	Th_v, Th_h = (Th_v * ratio), (Th_h * ratio)
	scaled_image = cv2.resize(gray_image, (new_width, new_height))
	scaled_image = cv2.threshold(scaled_image, 200, 255, cv2.THRESH_BINARY_INV)[1]

	sections = [scaled_image]
	h, w = scaled_image.shape
	section_coordinates = [[0, 0, w, h]]

	for i, section in enumerate(sections):
		peaks_list_v = pg.get_plateau_new(section, 0, Th_h=Th_h, Th_v=Th_v)
		horizontal_coordinates = pg.get_relative_coordinates(peaks_list_v, section_coordinates[i], 1)
		assert len(horizontal_coordinates) >= 0, "function get_relative_coordinates error"