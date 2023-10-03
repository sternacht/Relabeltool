from typing import Tuple

import SimpleITK as sitk
import numpy as np
import scipy.ndimage as nd
import cv2
import os
import math
from typing import Union
from os.path import join
class CTScanResampler:
    def __init__(self, mask_threshold: int = 50, mask_resample_threshold: int = 100) -> None:
        # for CT scan resampling
        self.mask_threshold = mask_threshold
        self.mask_resample_threshold = mask_resample_threshold

    def load_dicom_image(self, dicom_series_folder: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
            Load the dicom images in the folder.
            
            If the images are loaded successfully, this function returns 1 np.ndarray for images
            and 1 np.ndarray for pixel spacing.
            Else, this function returns 2 NoneType objects.
        """
        if os.path.isdir(dicom_series_folder) == False: # if the path doesn't exist
            return None, None, None
        
        # initialize dicom images reader
        reader = sitk.ImageSeriesReader()
        # get fileneames in the given folder and sort them based on dicom instance number
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_series_folder)
        reader.SetFileNames(dicom_names)
        # reader.MetaDataDictionaryArrayUpdateOn() # read metadata
        
        # read dicom images
        sitk_image = reader.Execute() 

        # # ignore the series with missing slices
        # series_is_consecutive = True
        # previous_instance_number = 0
        # if reader.HasMetaDataKey(0, '0020|0013'):
        #     previous_instance_number = int(reader.GetMetaData(0, '0020|0013'))
        # for i in range(1, len(reader.GetFileNames()), 1):
        #     if reader.HasMetaDataKey(i, '0020|0013'):
        #         if abs(previous_instance_number - int(reader.GetMetaData(i, '0020|0013'))) == 1:
        #             previous_instance_number = int(reader.GetMetaData(i, '0020|0013'))
        #         else:
        #             return None, None, None
        #     else:
        #         return None, None, None

        # get read dicom images
        image = sitk.GetArrayFromImage(sitk_image) # z, y, x
        image = np.transpose(image, (1, 2, 0)) # convert to y, x, z

        # get pixel spacing of dicom images
        origin_world = np.array(list(reversed(sitk_image.GetOrigin()))) # z, y, x
        origin_world = np.roll(origin_world, -1) # convert to y, x, z
        spacing = np.array(list(reversed(sitk_image.GetSpacing()))) # z, y, x
        spacing = np.roll(spacing, -1) # convert to y, x, z

        if spacing[2] > 3: # ignore the series if the z-spacing is bigger than the acceptable spacing
            return None, None, None

        return image, origin_world, spacing
    
    def load_mask(self, mask_folder: str, image: np.ndarray) -> np.ndarray:
        """
            Load the masks in the folder.
            
            If the masks are loaded successfully, this function returns 1 np.ndarray for masks.
            Else, this function returns 1 NoneType object.
        """
        if os.path.isdir(mask_folder) == False: # if the path doesn't exist
            return None
        
        if image is None: # don't load the related masks if the images don't exist
            return None
        
        # for current data format (jpg images)
        # get the filenames in the given folder
        mask_list = os.listdir(mask_folder)
        
        # create empty masks (all black)
        mask = np.zeros(image.shape, dtype=np.uint8)

        # fill in non-empty mask
        for m in mask_list:
            slice_number = int(m.split('-')[-1].replace('.jpg', '').replace('S', ''))
            # the first slice of the CT scan is the last slice of JPG images, and vice versa
            # thus, we fill in masks from the end
            # read the mask in the given folder
            mask[..., -slice_number] = cv2.imread(mask_folder + '/' + m, cv2.IMREAD_GRAYSCALE)

        # binarize
        mask = np.where(mask < self.mask_threshold, 0, 255)
        # mask[mask<self.mask_threshold] = 0
        # mask[mask>=self.mask_threshold] = 255
        
        return mask
    
    def resample(self, 
                 target_to_resample: np.ndarray, 
                 spacing: np.ndarray, 
                 target_spacing: np.ndarray, 
                 target_is_mask: bool=False) -> Tuple[np.ndarray, np.ndarray]:
        """
            For resampling dicom images and masks.

            If the target to resample is not a NoneType object, this function returns 1 np.ndarray for resampled images or masks
            and 1 np.ndarray for the spacing after resampling.
            Else, this function returns 2 NoneType objects.
        """
        if target_to_resample is None: # if there isn't a target to resample
            return None, None

        shape_resample = np.round(target_to_resample.shape * (spacing / target_spacing)) # shape after resampling
        # calculate the resizing scale based on shapes
        resize_factor = shape_resample / target_to_resample.shape
        # calculate the actual resampling spacing
        spacing_resample = spacing / resize_factor
        
        # resize the target to target spacing
        order = 1 if target_is_mask else 3
        target_resample = nd.zoom(target_to_resample, resize_factor, mode='nearest', order = order)
        if target_is_mask == False:
            target_resample = target_resample.astype(np.int16, copy=False) # for the range of HU values
        else:
            target_resample = target_resample.astype(np.uint8, copy=False) # for 0 ~ 255

            # binarize the mask
            target_resample = np.where(target_resample < self.mask_resample_threshold, 0, 255)
            # target_resample[target_resample<self.mask_resample_threshold] = 0
            # target_resample[target_resample>=self.mask_resample_threshold] = 255

        return target_resample, spacing_resample
    
    def reshape_yx(self, 
                   target_to_reshape: np.ndarray, 
                   target_yx_shape: list=[512, 512], 
                   target_is_mask: bool=False) -> np.ndarray:
        """
            For reshaping dicom images and masks

            If the target to reshape is not a NoneType object, this function returns 1 np.ndarray for reshaped images or masks.
            Else, this function returns 1 NoneType object.
        """
        if target_to_reshape is None: # if there isn't a target to resample
            return None

        # handle x-axis and y-axis separately
        offset_y = abs(target_yx_shape[0] - target_to_reshape.shape[0])
        offset_y_first = round(offset_y/2)
        offset_y_second = offset_y - offset_y_first

        offset_x = abs(target_yx_shape[1] - target_to_reshape.shape[1])
        offset_x_first = round(offset_x/2)
        offset_x_second = offset_x - offset_x_first

        # prepare an initial numpy array
        def get_target_reshape():
            output_shape = [target_yx_shape[0], target_yx_shape[1], target_to_reshape.shape[-1]]
            if target_is_mask == False:
                return np.full(output_shape, -1024, dtype = np.int16)
            else:
                return np.zeros(output_shape, dtype = np.uint8)
    
        # fill in the pixel values of the reshaped target
        if target_yx_shape[0] > target_to_reshape.shape[0]: # y-axis
            if target_yx_shape[1] > target_to_reshape.shape[1]: # x-axis
                target_reshape = get_target_reshape()
                target_reshape[offset_y_first: -offset_y_second, offset_x_first:-offset_x_second, :] = target_to_reshape

            elif target_yx_shape[1] < target_to_reshape.shape[1]: # x-axis
                target_reshape = get_target_reshape()
                target_reshape[offset_y_first:-offset_y_second, :, :] = target_to_reshape[:, offset_x_first:-offset_x_second, :]

            else: # same size in x-axis
                target_reshape = get_target_reshape()
                target_reshape[offset_y_first:-offset_y_second, :, :] = target_to_reshape

        elif target_yx_shape[0] < target_to_reshape.shape[0]: # y-axis
            if target_yx_shape[1] > target_to_reshape.shape[1]: # x-axis
                target_reshape = get_target_reshape()
                target_reshape[:, offset_x_first:-offset_x_second, :] = target_to_reshape[offset_y_first:-offset_y_second, :, :]

            elif target_yx_shape[1] < target_to_reshape.shape[1]: # x-axis
                target_reshape = target_to_reshape[offset_y_first:-offset_y_second, offset_x_first:-offset_x_second, :]

            else: # same size in x-axis
                target_reshape = target_to_reshape[offset_y_first:-offset_y_second, :, :]
        else: # same size in y-axis
            if target_yx_shape[1] > target_to_reshape.shape[1]: # x-axis
                target_reshape = get_target_reshape()
                target_reshape[:, offset_x_first:-offset_x_second, :] = target_to_reshape

            elif target_yx_shape[1] < target_to_reshape.shape[1]: # x-axis
                target_reshape = target_to_reshape[:, offset_x_first:-offset_x_second, :]

            else: # same size in x-axis
                target_reshape = target_to_reshape

        if target_is_mask == False:
            return target_reshape.astype(np.int16, copy = False)
        else:
            return target_reshape.astype(np.uint8, copy = False)
    
    def reorder(self, target_to_reorder: np.ndarray, axis: int) -> np.ndarray:
        """
            Flip the order of dicom images and masks

            If the target to reorder is not a NoneType object, this function returns 1 np.ndarray for reordered images or masks.
            Else, this function returns 1 NoneType object.
        """
        if target_to_reorder is None:
            return None
        
        return np.flip(target_to_reorder, axis=axis)
    
    def calculate_distance(self, bbox1: list, bbox2: list) -> float:
        if abs(bbox1[0] - bbox2[0]) > 1: # 2 bboxes are not consecutive
            return 1000
        
        x1 = (int(bbox1[1])+int(bbox1[3])) / 2
        y1 = (int(bbox1[2])+int(bbox1[4])) / 2
        x2 = (int(bbox2[1])+int(bbox2[3])) / 2
        y2 = (int(bbox2[2])+int(bbox2[4])) / 2

        return math.sqrt((x2-x1)**2 + (y2-y1)**2)

    def annotate_with_contour(self, mask: np.ndarray) -> Union[list, None]:
        """
            Generate bounding box annotations from masks

            If the input mask is not a NoneType object, this function returns 1 list containing groups of bounding boxes.
            Each group is like a 3D nodule, consisting of 2D bounding boxes on each slice.
            Else, this function returns 1 NoneType object.
        """
        if mask is None: # if there isn't a mask to find bounding boxes
            return None
        
        annotation = [] # list of list, each list contains the bounding boxes of the 3D nodule

        for i in range(mask.shape[2]):
            if np.any(mask[..., i]):
                contours, _ = cv2.findContours(mask[..., i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                for c in contours:
                    # handle each bounding box
                    x, y, w, h = cv2.boundingRect(c)
                    bbox = [i, x, y, x+w, y+h] # [slice_number, x1, y1, x2, y2], Update (2023/02/06): slice number start from 0 now

                    if len(annotation) == 0: # if the list is empty
                        annotation.append([bbox])
                        continue
                    
                    # calculate the distances from the current bounding box to the last bounding box of each list
                    distances = []
                    for j in range(len(annotation)):
                        distances.append(self.calculate_distance(annotation[j][-1], bbox))
                    
                    if min(distances) <= 25:
                        # if the minimum distance is smaller than 25, append it to the list of with minimum distance
                        annotation[distances.index(min(distances))].append(bbox)
                    else:
                        # else, create a new list (a new 3D nodule)
                        annotation.append([bbox])
        
        return annotation

def resample_mask(series_folder: str, 
                  mask: np.ndarray,
                  spacing,
                  target_spacing = np.array([0.8, 0.8, 1])):
    """
    Args:
        series_folder: str
            A folder to store data of series, e.g /hdd/openfl_system/dicom/ID-000001/Std-0001/Series-001
        mask: np.ndarray
            the mask of series image
    """    
    # series_metadata_path = os.path.join(series_folder, 'npy', 'series_metadata.txt')
    # with open(series_metadata_path, 'r') as f:
    #     lines = f.readlines()
    #     # lines[0] is: 'Image shape (h,w,d)'
    #     # lines[1] is: 'Pixel spacing(y,x,z)'
    #     # lines[2] is: 'World origin (y,x,z)'
    #     # lines[3] is: '-----'
    #     spacing = np.array([float(number) for number in lines[5].split(',')], dtype=np.float32)
    if not isinstance(spacing, np.ndarray):
        spacing = np.array(spacing, dtype=np.float64)
    resampler = CTScanResampler()
    resampled_mask, spacing_resample = resampler.resample(mask, spacing, target_spacing, target_is_mask=True) # Resampling
    reshaped_mask = resampler.reshape_yx(resampled_mask, target_yx_shape=[512, 512], target_is_mask=True) # reshape to 512*512 (w*h)
    reshaped_mask = resampler.reorder(reshaped_mask, axis=-1) # Adjust the order of image and mask to be same as JPG images

    if reshaped_mask is not None:
        return [True, reshaped_mask]
    return [False, None]

def resample_dicom_and_save_npy(dicoms_folder: str, 
                                saving_path: str, 
                                target_spacing: np.ndarray,
                                mininum_num_slice: int) -> bool:
    """
    Return: bool
        if success, then return True, 
        Else return False.
    """
    resampler = CTScanResampler()
    # Give a folder path to load the dicom images as a 3D numpy array (y, x, z)
    image, origin_world, spacing = resampler.load_dicom_image(dicoms_folder)
    image_shape = image.shape

    # Resampling
    resampled_image, spacing_resample = resampler.resample(image, spacing, target_spacing, target_is_mask=False)
    # reshape to 512*512 (w*h)
    reshaped_image = resampler.reshape_yx(resampled_image, target_yx_shape=[512, 512], target_is_mask=False)
    # Adjust the order of image and mask to be same as JPG images
    reshaped_image = resampler.reorder(reshaped_image, axis=-1)
    
    # Saving images
    if reshaped_image is not None and reshaped_image.shape[-1] >= mininum_num_slice:
        saving_folder = os.path.dirname(saving_path)
        os.makedirs(saving_folder, exist_ok=True)
        np.save(saving_path, reshaped_image)
        # Update (2023/02/06): save some metadata of the CT scan that will be used later
        with open(join(saving_folder, 'series_metadata.txt'), 'w') as f:
            f.write('Image shape (h,w,d)\nPixel spacing (y,x,z)\nWorld origin (y,x,z)\n-----\n' + 
                    '{},{},{}\n{},{},{}\n{},{},{}'.format(*image_shape, *spacing, *origin_world))
        return True
    return False

if __name__ == '__main__':
    dicom_folder = 'C:/Users/Ben-ThinkPad/Documents/Postgraduate/FL/dicom'
    to_resample = [['ID-000001', 'Std-0001', 'Ser-001']]
    dicom_save_folder = 'C:/Users/Ben-ThinkPad/Documents/Postgraduate/FL/dicom'

    target_spacing = np.array([0.8, 0.8, 1])

    resampler = CTScanResampler()

    for i in range(len(to_resample)):
        # give a folder path to load the dicom images as a 3D numpy array (y, x, z)
        image, origin_world, spacing = resampler.load_dicom_image('{}/{}/{}/{}/dicom_file'.format(dicom_folder, to_resample[i][0], to_resample[i][1], to_resample[i][2]))
        # give a folder path to load the masks of the CT scan as a 3D numpy array (y, x, z)                                                   
        mask = resampler.load_mask('{}/{}/{}/{}/relabel'.format(dicom_folder, to_resample[i][0], to_resample[i][1], to_resample[i][2]), image)
        # the order of image and mask are reversed (index 0 is the last slice of JPG images)

        # resampling
        image_resample, spacing_resample = resampler.resample(image, spacing, target_spacing, target_is_mask=False)
        mask_resample, _ = resampler.resample(mask, spacing, target_spacing, target_is_mask=True)

        # reshape to 512*512 (w*h)
        image_reshape = resampler.reshape_yx(image_resample, target_yx_shape=[512, 512], target_is_mask=False)
        mask_reshape = resampler.reshape_yx(mask_resample, target_yx_shape=[512, 512], target_is_mask=True)

        # adjust the order of image and mask to be same as JPG images
        image_reshape = resampler.reorder(image_reshape, axis=-1)
        mask_reshape = resampler.reorder(mask_reshape, axis=-1)

        annotation_reshape = resampler.annotate_with_contour(mask_reshape)
        # the order is just like JPG images

        # saving images
        if image_reshape is not None:
            np.save('{}/{}/{}/{}/npz/image.npy'.format(dicom_folder, to_resample[i][0], to_resample[i][1], to_resample[i][2]), image_reshape)

            # Update (2023/02/06): save some metadata of the CT scan that will be used later
            with open('{}/{}/{}/{}/npz/series_metadata.txt'.format(dicom_folder, to_resample[i][0], to_resample[i][1], to_resample[i][2]), 'w') as f:
                f.write('Image shape (h,w,d)\nPixel spacing (y,x,z)\nWorld origin (y,x,z)\n-----\n' + 
                        '{},{},{}\n{},{},{}\n{},{},{}'.format(*image.shape, *spacing, *origin_world))
            

        # saving masks
        if mask_reshape is not None:
            np.save('{}/{}/{}/{}/npz/mask.npy'.format(dicom_folder, to_resample[i][0], to_resample[i][1], to_resample[i][2]), mask_reshape)

        # saving annotations
        if annotation_reshape is not None:
            for j in range(len(annotation_reshape)):
                annotation_filename = 'N' + str(j+1).zfill(2) + '.txt'

                with open('{}/{}/{}/{}/npz/annotation_filename'.format(dicom_folder, to_resample[i][0], to_resample[i][1], to_resample[i][2]), 'a') as f:
                    f.write('slice_number x1 y1 x2 y2\n')

                    for bbox in annotation_reshape[j]:
                        f.write('{} {} {} {} {}\n'.format(*bbox))

        print(str(i+1) + ' / ' + str(len(to_resample)), end='\r')
