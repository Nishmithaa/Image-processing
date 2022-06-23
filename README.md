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

**8.program to resize the original image**<br>
import cv2<br>
img=cv2.imread('f5.jpg')<br>
print('Original image lentgth width',img.shape)<br>
cv2.imshow('Original image',img)<br>
cv2.waitKey(0)<br>

imagesize=cv2.resize(img,(100,160))<br>
cv2.imshow('resized image',imagesize)<br>
print('resized image lentgh width',imagesize.shape)<br>
cv2.waitKey(0)<br>
**Output**<br>
Original image lentgth width (1200, 1920, 3)<br>
resized image lentgh width (600, 600, 3)<br>
**original image**<br>
![image](https://user-images.githubusercontent.com/98141713/174048111-e57ebed0-29f1-4e6d-8750-3c5176670278.png)<br>
**Resized image**<br>
![image](https://user-images.githubusercontent.com/98141713/174048354-2d9fef2f-2c2f-43db-9693-615eeb71c8bc.png)<br>

**9.Convert the original to greyscaleand then to binary**<br>
import cv2<br>
img=cv2.imread('f5.jpg')<br>
cv2.imshow("RGB",img)<br>
cv2.waitKey(0)<br>
           
img=cv2.imread('f5.jpg',0)<br>
cv2.imshow("grey",img)<br>
cv2.waitKey(0)<br>
            
ret,bw_img=cv2.threshold(img,127,255,cv2.THRESH_BINARY)<br>
cv2.imshow("Binary",bw_img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
**Output**<br>
**RGB**<br>
![image](https://user-images.githubusercontent.com/98141713/174049247-18eee5cf-eac4-4406-b2f0-19631215b50c.png)<br>

**Grey scale**<br>
![image](https://user-images.githubusercontent.com/98141713/174049558-3e80e886-068b-4246-8964-b3dfb8a6c2f6.png)<br>

**Binary**<br>
![image](https://user-images.githubusercontent.com/98141713/174050093-a738b118-e2cb-4910-aa88-f4cfbf5ea8e4.png)<br>
 


**10.Develop a program to read a real image by using URL**<br>
import cv2<br>
import matplotlib.image  as mpimg<br>
import matplotlib.pyplot  as plt<br>
img=mpimg.imread('B3.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
**Output**<br>
![image](https://user-images.githubusercontent.com/98141713/175262915-5ca7409e-a15c-42d0-9289-75acff7ecd22.png)<br>

**11.Develop a program to mask and blur the image**<br>
hsv_img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)<br>
light_orange=(1,190,200)<br>
dark_orange=(18,255,255)<br>
mask=cv2.inRange(hsv_img,light_orange,dark_orange)<br>
result=cv2.bitwise_and(img,img,mask=mask)<br>
plt.subplot(1,2,1)<br>
plt.imshow(mask,cmap="gray")<br>
plt.subplot(1,2,2)<br>
plt.imshow(result)<br>
plt.show()<br>
**Output**<br>
![image](https://user-images.githubusercontent.com/98141713/175263205-9cc6940d-f010-4407-a4c8-a85c4418e330.png)<br>

light_white=(0,0,200)<br>
dark_white=(145,60,255)<br>

mask_white=cv2.inRange(hsv_img,light_white,dark_white)<br>
result_white=cv2.bitwise_and(img,img,mask=mask_white)<br>

plt.subplot(1,2,1)<br>
plt.imshow(mask_white,cmap="gray")<br>
plt.subplot(1,2,2)<br>
plt.imshow(result_white)<br>
plt.show()<br>
**Output**<br>
![image](https://user-images.githubusercontent.com/98141713/175263355-441a6ded-08fd-44bd-82a7-d469f3827d1d.png)<br>

final_mask=mask + mask_white<br>
final_result = cv2.bitwise_and (img, img, mask=final_mask)<br>
plt.subplot(1, 2, 1)<br>
plt.imshow(final_mask, cmap="gray")<br>
plt.subplot(1, 2, 2)<br>
plt.imshow(final_result)<br>
plt.show()<br>
**Output**<br>
![image](https://user-images.githubusercontent.com/98141713/175263504-51932b9a-76a7-4cc2-9bdf-89cdd7c4bb5c.png)<br>

blur=cv2.GaussianBlur(final_result, (7, 7), 0) <br>
plt.imshow(blur) <br>
plt.show()<br>
**Output**<br>
![image](https://user-images.githubusercontent.com/98141713/175263687-125c05a5-90eb-46d9-b1a3-10f718246b5e.png)<br>

**12.Develop a program to perform a arithmetic operartions on an image**<br>
import cv2<br>
import matplotlib.image as mping<br>
import matplotlib.pyplot as plt<br>
<br>
img1=cv2.imread('B3.jpg') 
img2= cv2.imread('B33.jpg')<br>

fimg1=img1+img2 <br>
plt.imshow(fimg1)<br>
plt.show() <br>
cv2.imwrite('output.jpg', fimg1)<br>


fimg2=img1-img2<br>
plt.imshow(fimg2)<br>
plt.show()<br>
cv2.imwrite('output.jpg', fimg2)<br>

fimg3=img1*img2<br>
plt.imshow(fimg3)<br>
plt.show()<br>
cv2.imwrite('output.jpg', fimg3)<br>


fimg4=img1/img2<br>
plt.imshow(fimg4)<br>
plt.show()<br>
cv2.imwrite('output.jpg', fimg4)<br>

**Output**<br>
![image](https://user-images.githubusercontent.com/98141713/175264480-d5320748-c640-4367-9638-e1e9ecdb03ee.png)<br>
![image](https://user-images.githubusercontent.com/98141713/175264594-fd30a9fe-cb63-45f1-85d2-59969b1c36ac.png)<br>
![image](https://user-images.githubusercontent.com/98141713/175264700-5aab091c-1d95-4060-8bb3-753e189ad241.png)<br>

**Develop a program to change the image to different color spaces**<br>
import cv2<br>
img= cv2.imread("puppy2.jpg") <br>
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)<br>
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)<br>
lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)<br>
hls=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)<br>
yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)<br>
cv2.imshow("GRAY image",gray)<br>
cv2.imshow("HSV image",hsv)<br>
cv2.imshow("LAB image",lab)<br>
cv2.imshow("HLS image",hls)<br>
cv2.imshow("YUV image",yuv)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
**Output**<br>
![image](https://user-images.githubusercontent.com/98141713/175271242-d59ce738-247d-4b2b-ae18-3fa0f12703a1.png)<br>
![image](https://user-images.githubusercontent.com/98141713/175271394-7123c791-84aa-4388-8a7e-d8f2280ab4f7.png)<br>
![image](https://user-images.githubusercontent.com/98141713/175271513-3988a86e-90b3-4b57-8193-ca752be7ef55.png)<br>
![image](https://user-images.githubusercontent.com/98141713/175271701-81808ef8-016e-4c00-bba5-da5e893751d9.png)<br>
![image](https://user-images.githubusercontent.com/98141713/175271826-9852dceb-b0aa-4dee-b34e-8f1c0ea231d9.png)<br>

**Develop a program to create an image by using 2D array**<br>
import cv2 as c<br>
import numpy as np<br>
from PIL import Image<br>
array = np.zeros([100, 200, 3], dtype=np.uint8)<br>
array[:,:100]=[255, 130, 0]<br>
array[:,100:]=[0, 0, 255]<br>
img = Image.fromarray(array)<br>
img.save('image1.png')<br>
img.show()<br>
c.waitKey(0)<br>
**Output**<br>
![image](https://user-images.githubusercontent.com/98141713/175274619-3d98d06d-51e9-4f98-8756-1c7f6b9b9d01.png)<br>



