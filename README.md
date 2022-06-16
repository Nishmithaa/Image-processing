# Image-processing
**1.Develop a program to display the grey scale image using read and write operation**<br>
import cv2<br>
img1=cv2.imread('f5.jpg',0)<br>
cv2.imshow('Folwer',img1)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
**output**<br>
![image](https://user-images.githubusercontent.com/98141713/173809043-dbc8553f-027a-449f-bc67-33151e7d226d.png)<br>

**2.Develop a program to display image by using matplotlib**<br>
from PIL import Imageimport cv2<br>
import matplotlib.pyplot as plt <br>
img=cv2.imread('f5.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
**Output**
![image](https://user-images.githubusercontent.com/98141713/173810256-c0de5dd9-a1b7-4f27-8d9b-07fec21d6a21.png)<br>

**3.Develop a program to perform linear transformation**<br>
from PIL import Image<br>
Original_Image=Image.open('f5.jpg')<br>
rotate_img1=Original_Image.rotate(180)<br>
rotate_img1.show()<br>
**Output**<br>
![image](https://user-images.githubusercontent.com/98141713/173810858-66f6b50d-8706-47b4-965c-6d187a98d7f3.png)<br>

**4.Develop a program to convert color string to RGB color value**<br>
from PIL import ImageColor<br><br>
img1=ImageColor.getrgb('yellow')<br>
print(img1)<br>
img2=ImageColor.getrgb('red')<br>
print(img2)<br>
**Output**<br>
(255, 255, 0)<br>
(255, 0, 0)<br>

**5.Develop a program to create image using colors**<br>
from PIL import Image<br>
img=Image.new('RGB',(200,600),(255,0,255))<br>
img.show()<br>
**Output**<br>
![image](https://user-images.githubusercontent.com/98141713/173812192-2b51bd26-0a78-4e2e-bf97-17c2f8740200.png)<br>

**6.Develop a program to intialize the image using various color**<br>
import cv2<br>
import matplotlib.pyplot  as plt<br>
import numpy as np<br>
img=cv2.imread('f5.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)<br>
plt.imshow(img)<br>
plt.show()<br>
img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)<br>
plt.imshow(img)<br>
plt.show()<br>
**Output**<br>
![image](https://user-images.githubusercontent.com/98141713/173816848-204b8717-4aa3-4316-811d-34edfdfada9e.png)<br>
![image](https://user-images.githubusercontent.com/98141713/173816939-6fc249c7-ed7d-4e49-a89a-8ba2f7adc460.png)<br>
![image](https://user-images.githubusercontent.com/98141713/173817080-3b823637-43cd-4408-9d19-4d9e023b4364.png)<br>

**7.Develop a program to display image atributes**<br>
from PIL import Image<br>
image=Image.open('f5.jpg')<br>
print("Filename:",image.filename)<br>
print("Format:",image.format)<br>
print("Mode:",image.mode)<br>
print("Size:",image.size)<br>
print("Width:",image.width)<br>
print("Height:",image.height)<br>
**Output**<br>
Filename: f5.jpg<br>
Format: JPEG<br>
Mode: RGB<br>
Size: (1920, 1200)<br>
Width: 1920<br>
Height: 1200<br>

**8.program to resize the original image**
import cv2
img=cv2.imread('f5.jpg')
print('Original image lentgth width',img.shape)
cv2.imshow('Original image',img)
cv2.waitKey(0)

imagesize=cv2.resize(img,(100,160))
cv2.imshow('resized image',imagesize)
print('resized image lentgh width',imagesize.shape)
cv2.waitKey(0)
**Output**
Original image lentgth width (1200, 1920, 3)
resized image lentgh width (600, 600, 3)
