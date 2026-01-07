import cv2
import math
import numpy as np
import os

from eyes_detection import detect_eye_centers_mediapipe

def check(image,title=''):
    # A simple fuction just for debugging
    cv2.imshow(str(title), image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(title+'.png',image)


def resize(image, eyes_location):
    #check(image,'raw')
    # Source eyes location
    (x1, y1), (x2, y2) = eyes_location
    x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
    # Target eyes location
    # NOTICE!! TARGET EYES LOCATION CAN BE CHANGED HERE
    xa1,ya1,xa2,ya2=140,292,310,292

    # |v1 x v2| = |v1| * |v2| * sin<v1, v2>
    # Calculate rotation angle by arcsin

    #vectors of source and target
    vx,vy=x2-x1,y2-y1
    vxa,vya=xa2-xa1,ya2-ya1
    #midpoints of source and target
    mx,my=(x1+x2)/2,(y1+y2)/2
    ma_x,ma_y=(xa1+xa2)/2,(ya1+ya2)/2
    #length of vectors of source and target
    l=math.sqrt(vx*vx+vy*vy)
    la=math.sqrt(vxa*vxa+vya*vya)

    # Rotate the image by 'angle' degrees (e.g., 90 degrees clockwise)
    angle = np.arcsin((vx*vya-vy*vxa)/(l*la))
    angle = angle/math.pi*180
    # Notice that in math counterclockwise is positive, but in cv2 clockwise is positive
    angle = -angle

    # Resize the image by a factor 'b'
    b = la/l

    center = (mx,my)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, b)
    # Fill in the background with white (255, 255, 255)
    borderValue = (255, 255, 255)
    image = cv2.warpAffine(image, rotation_matrix, image.shape[:2][::-1], borderValue=borderValue)
    #check(image,'Rotate&Rescale')

    # Define the sub-image coordinates (c, d) and (e, f)
    # NOTICE!! TARGET IMAGE SIZE CAN BE CHANGED HERE
    left = int(mx-ma_x)
    top = int(my-ma_y)
    right = left + 450
    bottom = top + 640
    #print(image.shape)
    #print(left,top,right,bottom)

    # Extract the sub-image
    image = image[top:bottom, left:right]

    # Display or save the sub-image as needed
    #check(image,'Cut')
    #cv2.imwrite('output.png',image)
    return image

def main():
    input_path = '.\input'
    output_path = '.\output'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    valid_images = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

    for filename in os.listdir(input_path):
        if not filename.lower().endswith(valid_images):
            continue

        full_path = os.path.join(input_path, filename)
        
        img = cv2.imread(full_path)
        
        if img is None:
                continue
            
        out = detect_eye_centers_mediapipe(img)
        
        if out is None:
            print(f"File name {filename} No face found.")
            #raise SystemExit
            continue

        '''
        (lx, ly), (rx, ry) = out
        print("Left eye :", (lx, ly))
        print("Right eye:", (rx, ry))

        vis = img.copy()
        cv2.circle(vis, (lx, ly), 4, (0, 255, 0), -1)
        cv2.circle(vis, (rx, ry), 4, (0, 255, 0), -1)
        cv2.imshow("eyes", vis)
        cv2.waitKey(0)
        '''

        img2 = resize(img, out)

        if img2 is None:
             print(f'File name {filename} resize failed.')
             continue

        save_path = os.path.join(output_path, filename)
        success = cv2.imwrite(save_path, img2)
        if not success:
            print(f"Failed to save {save_path}. Check if the image is valid.")
            continue
        
        '''
        vis = img2.copy()
        lx,ly,rx,ry = 140,292,310,292
        cv2.circle(vis, (lx, ly), 4, (0, 255, 0), -1)
        cv2.circle(vis, (rx, ry), 4, (0, 255, 0), -1)
        cv2.imshow("eyes", vis)
        cv2.waitKey(0)
        '''


if __name__ == '__main__':
        main()