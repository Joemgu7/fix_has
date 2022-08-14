"""
    Author: Jonas Briguet
    Date: 2022-08-14
    Description: This is a class to detect Hasan's face and fix its size

"""

import face_recognition
import cv2
from PIL import Image
from rembg import remove
from matplotlib import pyplot as plt
import numpy as np
import os
import argparse

class HasanDetector:
    def __init__(self,
            detection_model='cnn',
            encoding_model='large',
            resample_factor=1,
            eye_eye_mult=5,
            chin_nose_mult=2.5,
            smaller_factor=0.8,
            chin_margin=0.05, 
            mask_threshold = 5,
            big_mask_threshold = 100,
            off_center_modifier = 0.5,
            save_masks=False,
            reference_image_path='reference_images',
            max_size=4000):
        
        self.detection_model = detection_model
        self.encoding_model = encoding_model
        self.resample_factor = resample_factor
        self.eye_eye_mult = eye_eye_mult
        self.chin_nose_mult = chin_nose_mult
        self.smaller_factor = smaller_factor
        self.chin_margin = chin_margin
        self.mask_threshold = mask_threshold
        self.big_mask_threshold = big_mask_threshold
        self.save_masks = save_masks
        self.off_center_modifier = off_center_modifier
        self.max_size = max_size

        # compute the reference encodings
        self.known_images = [face_recognition.load_image_file(os.path.join(reference_image_path, ref)) for ref in os.listdir(reference_image_path)]
        self.REFERENCE_FACE_LOCATIONS = [face_recognition.face_locations(img, model=self.detection_model) for img in self.known_images]
        self.REFERENCE_ENCODING_LISTS = [face_recognition.face_encodings(img, known_face_locations=loc, num_jitters=self.resample_factor, model=self.encoding_model) for img, loc in zip(self.known_images, self.REFERENCE_FACE_LOCATIONS)]
        self.REFERENCE_ENCODINGS = [enc for enc_list in self.REFERENCE_ENCODING_LISTS for enc in enc_list]

    def detect_hasan(self, image):
        face_locations = face_recognition.face_locations(image, model=self.detection_model)
        all_unknown_encoding = face_recognition.face_encodings(image, known_face_locations=face_locations, num_jitters=self.resample_factor, model=self.encoding_model)

        results = []
        for hasan_encoding in self.REFERENCE_ENCODINGS:
            results.append(face_recognition.compare_faces(all_unknown_encoding, hasan_encoding))

        # convert boolean array to int array and compute average across columns
        results = [sum(col)/len(col) for col in zip(*results)]
        
        results = [result > 0.5 for result in results]

        face_locations_hasan = [face_locations[i] for i in range(len(results)) if results[i]]

        return face_locations_hasan

    def get_landmarks(self, image, face_locations=None, model='large'):
        return face_recognition.face_landmarks(image, face_locations, model)

    def compute_head(self, face_locations, chin_nose_distances, eye_to_eye_distance, face_midpoints, image_size, mult=1):
        # resize each face in face_locations by self.region from the center
        new_face_locations = []
        for i in range(len(face_locations)):
            face_location = face_locations[i]
            top, right, bottom, left = face_location

            new_height = self.chin_nose_mult*chin_nose_distances[i]*mult
            new_width = self.eye_eye_mult*eye_to_eye_distance[i]*mult

            # compute the new face location from the midpoint of the face
            new_top = round(face_midpoints[i][1] - new_height/2)
            new_left = round(face_midpoints[i][0] - new_width/2)
            new_bottom = round(new_top + new_height)
            new_right = round(new_left + new_width)
            
            # round and append to new_face_locations
            new_face_locations.append((max(new_top, 0), min(new_right, image_size[1]), min(image_size[0],new_bottom), max(0, new_left)))
        
        
        return new_face_locations

    def compute_landmark_metrics(self, landmarks):
        # for every landmark, compute the distance between chin and nose
        chin_nose_distances = []
        eye_to_eye_distances = []
        face_midpoints = []
        chin_middle_points = []
        chins = []
        top_lip_middle_points = []
        bottom_lip_middle_points = []
        for landmark in landmarks:
            # compute the middlepoint of each eye
            eye_middle_points = [np.mean(landmark['left_eye'], axis=0, dtype=np.int32), np.mean(landmark['right_eye'], axis=0, dtype=np.int32)]
            
            # compute the middle point between both eyebrows
            eye_middle_point = np.mean([np.mean(landmark['left_eyebrow'], axis=0, dtype=np.int32), np.mean(landmark['right_eyebrow'], axis=0, dtype=np.int32)], axis=0, dtype=np.int32)
            #eye_middle_point = np.mean(eye_middle_points, axis=0, dtype=np.int32)
            

            # select the point that is in the middle of the chin
            chin_middle_point = landmark['chin'][len(landmark['chin'])//2]
            chins.append(landmark['chin'])
            # compute middlepoint of top_lip
            top_lip_middle_point = np.mean(landmark['top_lip'], axis=0, dtype=np.int32)
            # compute middlepoint of bottom_lip
            bottom_lip_middle_point = np.mean(landmark['bottom_lip'], axis=0, dtype=np.int32)

            eye_to_eye_distances.append(np.linalg.norm(eye_middle_points[0]-eye_middle_points[1]))
            face_midpoints.append(eye_middle_point)
            chin_middle_points.append(chin_middle_point)
            chin_nose_distances.append(np.linalg.norm(chin_middle_point-eye_middle_point))
            top_lip_middle_points.append(top_lip_middle_point)
            bottom_lip_middle_points.append(bottom_lip_middle_point)

        # transform to numpy array
        chin_nose_distances = np.array(chin_nose_distances)
        eye_to_eye_distances = np.array(eye_to_eye_distances)
        face_midpoints = np.array(face_midpoints)
        chin_middle_points = np.array(chin_middle_points)
        chins = np.array(chins)
        top_lip_middle_points = np.array(top_lip_middle_points)
        bottom_lip_middle_points = np.array(bottom_lip_middle_points)
        return chin_nose_distances, eye_to_eye_distances, face_midpoints, chin_middle_points, chins, top_lip_middle_points, bottom_lip_middle_points

    def mix_alpha(self, image, overlay):
        # Extract the RGB channels
        srcRGB = image[...,:3]
        dstRGB = overlay[...,:3]

        # Extract the alpha channels and normalise to range 0..1
        srcA = image[...,3]/255.0
        dstA = overlay[...,3]/255.0

        # Work out resultant alpha channel
        outA = srcA + dstA*(1-srcA)

        # Work out resultant RGB
        outRGB = (srcRGB*srcA[...,np.newaxis] + dstRGB*dstA[...,np.newaxis]*(1-srcA[...,np.newaxis])) / outA[...,np.newaxis]

        # Merge RGB and alpha (scaled back up to 0..255) back into single image
        outRGBA = np.dstack((outRGB,outA*255)).astype(np.uint8)

        return outRGBA

    def smol_face(self, img_path, verbose=1):
        result = {}

        image = face_recognition.load_image_file(img_path)
        # if the longer side of the image is greater than max_size, resize the image
        max_size = self.max_size
        if image.shape[0] > image.shape[1] and image.shape[0] > max_size:
            image = cv2.resize(image, (int(max_size*(image.shape[1]/image.shape[0])), max_size))
        elif image.shape[1] > image.shape[0] and image.shape[1] > max_size:
            # use cv2
            image = cv2.resize(image, (max_size, int(max_size*(image.shape[0]/image.shape[1]))))
        
        new_height = 2*(image.shape[1]//2+1 if (image.shape[1]//2)%2 != 0 else image.shape[1]//2)
        new_width = 2*(image.shape[0]//2+1 if (image.shape[0]//2)%2 != 0 else image.shape[0]//2)
        # make sure the dimensions are divisible by 2
        image = cv2.resize(image, (new_height, new_width))
        img_size = image.shape[:2]
        
        face_locations_hasan = self.detect_hasan(image)
        landmarks = self.get_landmarks(image, face_locations_hasan)

        # compute landmark metrics
        chin_nose_distances, eye_to_eye_distances, face_midpoints, chin_middlepoints, chins, top_lip_middlepoint, bottom_lip_middlepoint = self.compute_landmark_metrics(landmarks)

        head_locations = self.compute_head(
            face_locations_hasan, chin_nose_distances, eye_to_eye_distances, face_midpoints, img_size)

        large_locations = self.compute_head(
            face_locations_hasan, chin_nose_distances, eye_to_eye_distances, chin_middlepoints, img_size, mult=2)
        
        # debug: show the face locations        
        if verbose > 1:
            tmp_img = image.copy()
            #plot rectangles around the faces
            for i in range(len(head_locations)):
                top, right, bottom, left = head_locations[i]
                print(head_locations[i])
                cv2.rectangle(tmp_img, (left, top), (right, bottom), (0, 0, 255), 2)
            plt.imshow(tmp_img)
            plt.show()


        # cut out faces from original_image and perform background removal
        head_regions = []
        large_regions = []
        faces_no_bg = []
        masks = []
        for i in range(len(head_locations)):
            head_region = image[head_locations[i][0]:head_locations[i][2], head_locations[i][3]:head_locations[i][1]]
            head_regions.append(head_region)

            large_region = image[large_locations[i][0]:large_locations[i][2], large_locations[i][3]:large_locations[i][1]]
            large_regions.append(large_region)

            region_no_bg = remove(large_region)
                
            # compute the location of head in the original in the large region
            new_top = head_locations[i][0] - large_locations[i][0]
            new_left = head_locations[i][3] - large_locations[i][3]
            new_bottom = new_top + head_locations[i][2] - head_locations[i][0]
            new_right = new_left + head_locations[i][1] - head_locations[i][3]
            # cut out the head from the large region
            face_no_bg = region_no_bg[new_top:new_bottom, new_left:new_right]
            faces_no_bg.append(face_no_bg)

            # compute the location of the whoole chin in the original in the face region
            chin = chins[i]
            new_chin = []
            for point in chin:
                new_chin.append((point[0] - head_locations[i][3], point[1] - head_locations[i][0]))

            # compute the location of the top and bottom lips in the original in the face region
            top_lip = top_lip_middlepoint[i]
            new_top_lip = (top_lip[0] - head_locations[i][3], top_lip[1] - head_locations[i][0])
            bottom_lip = bottom_lip_middlepoint[i]
            new_bottom_lip = (bottom_lip[0] - head_locations[i][3], bottom_lip[1] - head_locations[i][0])

            orig_chin_middle_point = chin_middlepoints[i]
            # compute the location of chin in the original in the face region
            new_chin_middle_point = (orig_chin_middle_point[0] - head_locations[i][3], orig_chin_middle_point[1] - head_locations[i][0])
            new_face_midpoint = (face_midpoints[i][0] - head_locations[i][3], face_midpoints[i][1] - head_locations[i][0])
            chin_location_y = round(new_chin_middle_point[1])
            # set all pixels in the mask to from 1 to 0 linearly that are below the chin
            step_size = max(round(255/new_bottom_lip[1]), 1)
            # create a mask array by setting all pixels in the mask to 1 if alpha is not 0 and 0 otherwise
            mask = np.zeros(head_region.shape[:2], dtype=np.uint8)
            mask[:,:] = 255
            mask[face_no_bg[:, :, 3] <= self.mask_threshold] = 0
            mask_value = 0

            furthest_left_point = None
            furthest_right_point = None
            for j in range(2, len(new_chin)-3):
                segment = new_chin[j:j+2]
                # compute the middle point of the segment
                segment_middle_point = ((segment[0][0]+segment[1][0])//2, (segment[0][1]+segment[1][1])//2)
                # if this middle point is below the bottom lip, set the points left of the segment to 0
                #
                # set all pixels in the mask to 0 that are > ever
                for k in range(segment[0][0], segment[1][0]):
                    # k enumerates the x-coordinate of the mask
                    # get the y coordinate by interpolating linearly that has its x coordinate closest to k
                    y = int(round(segment[0][1] + (segment[1][1]-segment[0][1])/(segment[1][0]-segment[0][0])*(k-segment[0][0])))
                    # set all pixels in the mask to 0 that are below the y coordinate 
                    val = 255
                    if y > new_bottom_lip[1]:
                        if furthest_left_point is None or k < furthest_left_point[0]:
                            furthest_left_point = (k, y)
                        if furthest_right_point is None or k > furthest_right_point[0]:
                            furthest_right_point = (k, y)
                        for l in range(y, mask.shape[0]):
                            mask[l, k] = min(max(val, 0), mask[l, k])
                            val -= step_size

            # set all pixels that are below the chin to 0 from left ro right to furthest left point
            # the further of center of the chine, the greater the step size
            for k in range(0, furthest_left_point[0]):
                val = 255
                for l in range(new_bottom_lip[1], mask.shape[0]):
                    mask[l, k] = min(val, mask[l, k])
                    val = max(0, val-step_size)

            # set all pixels that are below the chin to 0 from right to furthest right point
            for k in range(furthest_right_point[0], mask.shape[1]):
                val = 255
                for l in range(new_bottom_lip[1], mask.shape[0]):
                    mask[l, k] = min(val, mask[l, k])
                    val = max(0, val-step_size)

            # iterate left to right
            self.off_center_steps = 10
            for k in range(0, mask.shape[1]):
                if new_chin_middle_point[0] > k:
                    center_mult = k*(self.off_center_steps/new_chin_middle_point[0])
                else:
                    center_mult = (mask.shape[1]-k)*(self.off_center_steps/(mask.shape[1]-new_chin_middle_point[0]))
                
                # iterate from top to bottom
                val = 0
                for l in range(mask.shape[0]-1, new_face_midpoint[1], -1):
                        mask[l, k] = min(val, mask[l, k])
                        val = min(255, val + center_mult)


            if verbose > 0:
                tmp_img = mask.copy()
                cv2.circle(tmp_img, new_chin_middle_point, 10, (0, 0, 0), -1)
                for point in new_chin:
                    cv2.circle(tmp_img, point, 5, (0, 0, 0), -1)
                cv2.circle(tmp_img, new_top_lip, 5, (0, 0, 0), -1)
                plt.imshow(tmp_img)
                plt.show()
            
            masks.append(mask)

        if verbose > 0:
            print('Length of masks: ', len(masks))

        # create a new big mask containing all the masks with the same size as the original image
        big_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for mask in masks:
            # create a new mask which thresholds the mask
            mask_thresholded = mask.copy()
            mask_thresholded[mask < self.big_mask_threshold] = 0
            # put the small mask back to its original location in the big mask
            big_mask[head_locations[masks.index(mask)][0]:head_locations[masks.index(mask)][2], head_locations[masks.index(mask)][3]:head_locations[masks.index(mask)][1]] = mask_thresholded

        # select the dilation kernel size based on the size of the image
        if image.shape[0] > 800:
            dilation_kernel_size = 7
        elif image.shape[0] > 500:
            dilation_kernel_size = 5
        else:
            dilation_kernel_size = 3

        big_mask = cv2.dilate(big_mask, np.ones((dilation_kernel_size, dilation_kernel_size)), iterations=1)
        inpainted_image = cv2.inpaint(image, big_mask, 3, cv2.INPAINT_NS)

        inpainted_image = cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2BGRA)

        if self.save_masks:
            result['inpainted_image'] = inpainted_image

        # make all faces without background 30% smaller and put them onto a single new image
        face_mask = np.zeros(image.shape[:2]+(4,), dtype=np.uint8)
        alpha = np.zeros(image.shape[:2], dtype=np.uint8)
        for i in range(len(faces_no_bg)):
            face_no_bg = cv2.resize(faces_no_bg[i], (0, 0), fx=self.smaller_factor, fy=self.smaller_factor)
            small_alpha = cv2.resize(masks[i], (0, 0), fx=self.smaller_factor, fy=self.smaller_factor)
            # adjust position and merge the face into the face_mask
            # center it horizontally
            new_x_start = head_locations[i][3] + (head_locations[i][1] - head_locations[i][3]) * (1-self.smaller_factor)/2
            new_x_end = new_x_start + face_no_bg.shape[1]
            new_y_start = head_locations[i][0] + (head_locations[i][2] - head_locations[i][0]) * (1-self.smaller_factor)
            new_y_end = new_y_start + face_no_bg.shape[0]

            # select the erosion kernel size based on the size of the face
            if face_no_bg.shape[0] > 800:
                erosion_kernel_size = 7
                gaussian_kernel_size = 23
            elif face_no_bg.shape[0] > 500:
                erosion_kernel_size = 5
                gaussian_kernel_size = 15
            else:
                erosion_kernel_size = 3
                gaussian_kernel_size = 7

            alpha_channel = cv2.erode(small_alpha, np.ones((erosion_kernel_size, erosion_kernel_size)), iterations=4)
            alpha_channel = cv2.GaussianBlur(alpha_channel, (gaussian_kernel_size,gaussian_kernel_size), 0)

            face_mask[round(new_y_start):round(new_y_end), round(new_x_start):round(new_x_end)] = face_no_bg
            alpha[round(new_y_start):round(new_y_end), round(new_x_start):round(new_x_end)] = alpha_channel

        # add alpha channel to face_mask
        face_mask[:, :, 3] = alpha

        if self.save_masks:
            result['face_mask'] = face_mask

        negative_alpha = 255 - face_mask[:, :, 3]
        inpainted_image[:, :, 3] = negative_alpha
        
        # merge the face_mask with the inpainted_image
        # use their alpha channels to blend the two images
        mixed = self.mix_alpha(inpainted_image, face_mask)
        # set the alpha channel to 255
        mixed[:, :, 3] = 255

        if verbose > 0:
            # plot face_mask, inpainted_image, mixed, image, big_mask, and label them
            fig, axs = plt.subplots(3, 2, figsize=(20, 20))
            plt.subplot(3, 2, 1)
            plt.imshow(face_mask)
            plt.title('face_mask')
            plt.subplot(3, 2, 2)
            plt.imshow(inpainted_image)
            plt.title('inpainted_image')
            plt.subplot(3, 2, 3)
            plt.imshow(mixed)
            plt.title('mixed')
            plt.subplot(3, 2, 4)
            plt.imshow(image)
            plt.title('image')
            plt.subplot(3, 2, 5)
            plt.imshow(big_mask)
            plt.title('big_mask')           
            plt.show()

        result['processed'] = mixed

        return result

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Tool to detect Hasan's face and fix its size")
    # parse an image "mandatory"
    parser.add_argument("-i", "--image", help="path to input image", required=True)
    # parse a verbose level "optional"
    parser.add_argument("-v", "--verbose", help="verbose level", type=int, default=0)
    # parse "fast" mode "optional"
    parser.add_argument("-f", "--fast", help="fast mode: is less accurate but much faster", action="store_true")
    # parse "output" where to save the image "optional"
    parser.add_argument("-o", "--output", help="path to output image", default='results')
    # parse "save_masks" to save the masks "optional"
    parser.add_argument("-s", "--save_masks", help="save the intermediary results", action="store_true")
    args = parser.parse_args()

    detection_model = ('hog' if args.fast else 'cnn')
    verbose = args.verbose
    image_path = args.image
    output_path = args.output
    save_masks = args.save_masks
    print('Processing image: {}'.format(image_path))
    detector = HasanDetector(detection_model, save_masks=save_masks)
    res = detector.smol_face(image_path, verbose=verbose)
    # display every the face_mask, alpha, inpainted_image and the original image using plt

    # save the result with the correct color channels
    # only get the filename without the extension
    for result in res:        
        img = res[result]
        filename = os.path.splitext(os.path.basename(image_path))[0]
        processed_name = os.path.join(output_path, filename + f'_{result}.png')
        print(f'Saving {result} at {processed_name}')
        cv2.imwrite(processed_name, cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA))


if __name__ == '__main__':
    main()