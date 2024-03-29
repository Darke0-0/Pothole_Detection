import numpy as np
import cv2
import os
import torch
from tqdm import tqdm
import config
from model import model
from os.path import isfile, join

# set the computation device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# load the model and the trained weights
model = model().to(device)
model.load_state_dict(torch.load('Data/checkpoints/fasterrcnn_resnet50_fpn.pth'))

DIR_TEST = config.TEST_PATH
test_images = os.listdir(DIR_TEST)
print(f"Validation instances: {len(test_images)}")

detection_threshold = config.PREDICTION_THRES
model.eval()
with torch.no_grad():
    for i, image in tqdm(enumerate(test_images), total=len(test_images)):
        orig_image = cv2.imread(f"{DIR_TEST}/{test_images[i]}", cv2.IMREAD_COLOR)
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # make the pixel range between 0 and 1
        image /= 255.0
        image = np.transpose(image, (2, 0, 1)).astype(np.float)
        image = torch.tensor(image, dtype=torch.float).cuda()
        image = torch.unsqueeze(image, 0)
        cpu_device = torch.device("cpu")
        outputs = model(image)
        
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        if len(outputs[0]['boxes']) != 0:
            for counter in range(len(outputs[0]['boxes'])):
                boxes = outputs[0]['boxes'].data.numpy()
                scores = outputs[0]['scores'].data.numpy()
                boxes = boxes[scores >= detection_threshold].astype(np.int32)
                draw_boxes = boxes.copy()
                
            for box in draw_boxes:
                cv2.rectangle(orig_image,
                            (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])),
                            (0, 0, 255), 3)
                cv2.putText(orig_image, 'PotHole', 
                            (int(box[0]), int(box[1]-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 
                            2, lineType=cv2.LINE_AA)
            cv2.imwrite(f"test_predictions/{test_images[i]}", orig_image,)
print('TEST PREDICTIONS COMPLETE')

def convert_frames_to_video(pathIn,pathOut,fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    
    #for sorting the file names properly
    for i in range(len(files)):
        filename=pathIn + files[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        #inserting the frames into an image array
        frame_array.append(img)
        if len(frame_array) == 300:
            break
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()

pathIn= 'test_predictions/'
pathOut = 'video.avi'
fps = 25.0
convert_frames_to_video(pathIn,pathOut,fps)