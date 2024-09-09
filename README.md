How to run the algorithm
1. Install requirements
2. Save the images you want to process in the test_images folder.
3. Update the path in garment_algorithm.py to the directory where your images are stored.
4. Two new images, named lower_body.jpg and upper_body.jpg, will be saved in the same location as garment_algorithm.py.

General information about the algorithm 
1. We use MediaPipe to detect the human pose.
2. We generate masks for each point using the Segment Anything Model (SAM), predicting on SAM with the condition of a specific point.
3. We also create masks for the entire human body using SAM, predicting with the condition of all points.
4. We filter the masks based on several criteria:
a. Size of the mask: We discard masks that are too small.
b. Overlapping masks: If masks overlap by more than 50%, we keep only the largest one.
c. Containment within the full body mask: Masks must be within the full body mask and cover less than 60% of it.
5. We select the two largest masks after filtering.
6. We determine which mask represents the upper body and which represents the lower body based on the vertical center of each mask—the one with the higher center is the upper body, and the one with the lower center is the lower body.
7. We only keep the largest contour in each mask.
8. We calculate the bounding box for each mask and crop the image at these locations, then save the results.