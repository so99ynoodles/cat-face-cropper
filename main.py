import sys
import os
import cv2

try:
    _, import_folder, export_folder, w, h = sys.argv
    width = int(w)
    height = int(h)
except ValueError as error:
    print('Please check your input: [import_folder: path] [output_folder: path] [width: number] [height: number]')
    sys.exit()


extensions = ('.jpg', '.png')
new_size = (width, height)

if not os.path.isdir(export_folder):
    os.mkdir(export_folder)

catface_cascade = cv2.CascadeClassifier('./catface_detector.xml')
pictures = list(filter(lambda x: x.endswith(extensions), os.listdir(import_folder)))

for pic_path in pictures:
    image_name = os.path.splitext(pic_path)[0]
    image = cv2.imread(f'{import_folder}/{pic_path}')
    cat_faces = catface_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=3, minSize=(75, 75))
    print(f'{len(cat_faces)} faces detected')
    for (i, (x, y, w, h)) in enumerate(cat_faces):
        face = image[y-50:y+h+50, x-50:x+w+50]
        resized = cv2.resize(face, (width, height))
        cv2.imwrite(f'{export_folder}/{image_name}.png', resized)
