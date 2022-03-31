# Cars detector model
Built cars detector model using dataset of images with and without cars: <br>
1. Trained classification CNN model determining if there's a car in a picture
2. Built detection model by converting FC layers of trained CNN to convolutional layers saving trained weights
3. Implemented simple detector procedure working on output (basically heatmap) of detection model
4. Applied NMS to detections obtained by detector procedure