import os
import numpy as np
import cv2
import time
from ExifData import getExif, restoreOrientation
from EoData import readEO, convertCoordinateSystem, Rot3D
from Boundary import boundary
from BackprojectionResample import projectedCoord, backProjection, resample, createGeoTiff

if __name__ == '__main__':
    ground_height = 65  # unit: m
    sensor_width = 6.3  # unit: mm

    img_fname = 'DJI_0386.JPG'
    eo_fname = 'DJI_0386.txt'
    img_path = os.path.join('./Data', img_fname)
    eo_path = os.path.join('./Data', eo_fname)

    start_time = time.time()

    print('Read the image - ' + img_fname)
    image = cv2.imread(img_path)

    # 0. Extract EXIF data from a image
    focal_length, orientation = getExif(img_path) # unit: m

    # 1. Restore the image based on orientation information
    restored_image = restoreOrientation(image, orientation)

    image_rows = restored_image.shape[0]
    image_cols = restored_image.shape[1]

    pixel_size = sensor_width / image_cols  # unit: mm/px
    pixel_size = pixel_size / 1000  # unit: m/px

    end_time = time.time()
    print("--- %s seconds ---" % (time.time() - start_time))

    read_time = end_time - start_time

    print('Read EOP - ' + img_fname)
    print('Latitude | Longitude | Height | Omega | Phi | Kappa')
    eo = readEO(eo_path)
    eo = convertCoordinateSystem(eo)
    R = Rot3D(eo)

    # 2. Extract a projected boundary of the image
    bbox = boundary(restored_image, eo, R, ground_height, pixel_size, focal_length)
    print("--- %s seconds ---" % (time.time() - start_time))

    gsd = (pixel_size * (eo[2] - ground_height)) / focal_length  # unit: m/px

    # Boundary size
    boundary_cols = int((bbox[1, 0] - bbox[0, 0]) / gsd)
    boundary_rows = int((bbox[3, 0] - bbox[2, 0]) / gsd)

    print('projectedCoord')
    start_time = time.time()
    proj_coords = projectedCoord(bbox, boundary_rows, boundary_cols, gsd, eo, ground_height)
    print("--- %s seconds ---" % (time.time() - start_time))

    # Image size
    image_size = np.reshape(restored_image.shape[0:2], (2, 1))

    print('backProjection')
    start_time = time.time()
    backProj_coords = backProjection(proj_coords, R, focal_length, pixel_size, image_size)
    print("--- %s seconds ---" % (time.time() - start_time))

    print('resample')
    start_time = time.time()
    b, g, r, a = resample(backProj_coords, boundary_rows, boundary_cols, image)
    print("--- %s seconds ---" % (time.time() - start_time))

    print('Save the image in GeoTiff')
    start_time = time.time()
    dst = './' + img_fname
    createGeoTiff(b, g, r, a, bbox, gsd, boundary_rows, boundary_cols, dst)
    print("--- %s seconds ---" % (time.time() - start_time))

    print('*** Processing time per each image')
    print("--- %s seconds ---" % (time.time() - start_time + read_time))
