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

**13.Develop a program to change the image to different color spaces**<br>
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

**14.Develop a program to create an image by using 2D array**<br>
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

**15.Bitwise operartions on images**<br>
import cv2 <br>
import matplotlib.pyplot as plt<br>
image1=cv2.imread('d2.jpg')<br>
image2=cv2.imread('d2.jpg')<br>

ax=plt.subplots(figsize=(15,10))<br>

bitwiseAnd=cv2.bitwise_and(image1,image2)<br>
bitwiseOr=cv2.bitwise_or(image1,image2)<br>
bitwiseXor=cv2.bitwise_xor(image1,image2)<br>
bitwiseNot_img1=cv2.bitwise_not(image1)<br>
bitwiseNot_img2=cv2.bitwise_not(image2)<br>

plt.subplot(151)<br>
plt.imshow(bitwiseAnd)<br>

plt.subplot(152)<br>
plt.imshow(bitwiseOr)<br>

plt.subplot(153)<br>
plt.imshow(bitwiseXor)<br>

plt.subplot(154)<br>
plt.imshow(bitwiseNot_img1)<br>

plt.subplot(155)<br>
plt.imshow(bitwiseNot_img2)<br>

cv2.waitKey(0)<br>

**Output**<br>
![image](https://user-images.githubusercontent.com/98141713/176404696-d0d269f9-d45a-438b-910a-79734a616125.png)<br>

**16.Types of blurring on image**<br>
import cv2<br>
import numpy as np<br>
image = cv2.imread('doggo.jpg')<br>
cv2.imshow('Original Image', image)<br>
cv2.waitKey(0)<br>

Gaussian = cv2.GaussianBlur (image, (7, 7), 0) <br>
cv2.imshow('Gaussian Blurring', Gaussian)<br>
cv2.waitKey(0)<br>

median = cv2.medianBlur(image, 5)<br>
cv2.imshow('Median Blurring', median)<br>
cv2.waitKey(0)<br>

bilateral = cv2.bilateralFilter(image, 9, 75, 75)<br>
cv2.imshow('Bilateral Blurring', bilateral)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

**Output**<br>
![image](https://user-images.githubusercontent.com/98141713/176408336-f465d095-834c-48dd-bb13-faafc64adc53.png)<br>
![image](https://user-images.githubusercontent.com/98141713/176408480-09cb12a2-3325-4f74-bc8c-3522289a76d2.png)<br>
![image](https://user-images.githubusercontent.com/98141713/176408585-cff1cd4f-9dd2-4ecd-bb5e-76c0db394f07.png)<br>
![image](https://user-images.githubusercontent.com/98141713/176409028-2e68ed76-13bb-48f4-9628-e6e9d42ba87e.png)<br>

**17.Image enhancement**<br>
from PIL import Image<br>
from PIL import ImageEnhance<br>

image=Image.open('doggo.jpg')<br>
image.show()<br>

enh_bri=ImageEnhance.Brightness(image)<br>
brightness=1.5<br>
image_brightened=enh_bri. enhance (brightness) <br>
image_brightened.show()<br>

enh_col=ImageEnhance.Color(image)<br>
color= 1.5<br>
image_colored = enh_col.enhance(color)<br>
image_colored.show()<br>

enh_con=ImageEnhance.Contrast(image)<br>
contrast = 1.5<br>
image_contrasted=enh_con. enhance (contrast)<br>
image_contrasted.show()<br>

enh_sha=ImageEnhance.Sharpness(image)<br>
sharpness=3.0<br>
image_sharped = enh_sha. enhance (sharpness)<br>
image_sharped.show()<br>
**Output**<br>
![image](https://user-images.githubusercontent.com/98141713/176417178-72c0f268-08a2-43d8-8681-9c3dc597410a.png)<br>
![image](https://user-images.githubusercontent.com/98141713/176417312-47eda2a7-ed2d-47f8-9aac-0d219f695237.png)<br>
![image](https://user-images.githubusercontent.com/98141713/176417401-b0089b7d-7eff-42f7-9f3a-1cc275f18735.png)<br>
![image](https://user-images.githubusercontent.com/98141713/176417489-51094a92-5b32-4629-a175-6637a7bd0339.png)<br>
![image](https://user-images.githubusercontent.com/98141713/176417746-d671c3b7-1669-447b-9125-517aa7720907.png)<br>

**18.Image morphology**<br>
import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
from PIL import Image, ImageEnhance<br>

img = cv2.imread('d2.jpg',0) <br>
ax=plt.subplots (figsize=(20,10))<br>
kernel = np.ones((5,5), np. uint8)<br>

opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)<br>
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)<br>
erosion = cv2. erode (img, kernel, iterations = 1)<br>
dilation = cv2.dilate (img, kernel, iterations = 1)<br>
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)<br>

plt.subplot(151)<br>
plt.imshow(opening)<br>
plt.subplot(152) <br>
plt.imshow(closing)<br>
plt.subplot(153)<br>
plt.imshow(erosion)<br>
plt.subplot(154)<br>
plt.imshow(dilation)<br>
plt.subplot(155)<br>
plt.imshow(gradient)<br>
cv2.waitKey(0)<br>

**Output**<br>
![image](https://user-images.githubusercontent.com/98141713/176419716-4cdb1b3a-e575-441e-83f6-c9ea9ed204b5.png)<br>


**19.Develop a program to read the image<br>
Write or save the grayscale image<br>
Display the original image and grayscale image**<br>

import cv2<br>
OriginalImg=cv2.imread('OT77.jpg')<br>
GrayImg=cv2.imread('OT77.jpg',0)<br>
isSaved=cv2.imwrite('D:\OT77.jpg', GrayImg) <br>
cv2.imshow('Display Original Image',OriginalImg)<br>
cv2.imshow('Display Grayscale Image', GrayImg)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
if isSaved:print('The image is successfully saved.')<br>

**Output**<br>
The image is successfully saved.<br>
![image](https://user-images.githubusercontent.com/98141713/178700270-1b2dd783-9778-408c-a79e-b9dab35dbb0c.png)<br>
![image](https://user-images.githubusercontent.com/98141713/178700417-d135f555-2056-4f31-ad8c-b1bc39e8ad27.png)<br>
**Picture in drive**<br>
![image](https://user-images.githubusercontent.com/98141713/178700764-c531fb8f-03b5-4094-bd39-08f793393911.png)<br>


**20.Gray level slicing with background**<br>
import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
image=cv2.imread('Droplet.jpg',0) <br>
x,y=image.shape<br>
z=np.zeros((x,y))<br>
for i in range(0,x):<br>
    for j in range(0,y):<br>
        if(image[i][j]>50 and image[i][j]<150):<br>
         z[i][j]=255<br>
else:z[i][j]=image[i][j]<br>
equ=np.hstack((image,z))<br>
plt.title('Graylevel slicing with background')<br>
plt.imshow(equ,'gray')<br>
plt.show()<br>

**Output**<br>
![image](https://user-images.githubusercontent.com/98141713/178707961-d085ab27-503e-4c20-b593-d166ff59a40a.png)<br>

**21.Graylevel slicing without background**<br>
import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt <br>
image=cv2.imread('Droplet.jpg', 0)<br>
x,y=image.shape<br>
z=np.zeros((x,y))<br>
for i in range(0,x):<br>
    for j in range(0,y):<br>
        if(image[i][j]>50 and image[i][j]<150):<br>
                z[i][j]=255<br>
            else:<br>
                z[i][j]=0<br>
equ=np.hstack((image,z))<br>
plt.title('Graylevel slicing w/o background')<br>
plt.imshow(equ, 'gray')<br>
plt.show()<br>

**Output**<br>
![image](https://user-images.githubusercontent.com/98141713/178708052-52cb2bdc-6345-40d8-bf9d-4ccecc799dfd.png)<br>

**22.Ananlyze image data using histogram.**<br>
from skimage import io<br>
import matplotlib.pyplot as plt<br>
image = io.imread('S1.jpg')<br>

_ = plt.hist(image.ravel(), bins = 256, color = 'orange', )<br>
_ = plt.hist(image[:, :, 0].ravel(), bins = 256, color = 'red', alpha = 0.5)<br>
_ = plt.hist(image[:, :, 1].ravel(), bins = 256, color = 'Green', alpha = 0.5)<br>
_ = plt.hist(image[:, :, 2].ravel(), bins = 256, color = 'Blue', alpha = 0.5)<br>
_ = plt.xlabel('Intensity Value')<br>
_ = plt.ylabel('Count')<br>
_ = plt.legend(['Total', 'Red_Channel', 'Green_Channel', 'Blue_Channel'])<br>
plt.show()<br>

*Output*<br>
![image](https://user-images.githubusercontent.com/98141713/178956270-f7c7b498-546a-4a5b-980b-b7c216ab7225.png)<br>

2.<br><br>
import cv2<br>  
from matplotlib import pyplot as plt  <br>
img = cv2.imread('S1.jpg',0) <br>
histr = cv2.calcHist([img],[0],None,[256],[0,256]) <br>
plt.plot(histr) <br>
plt.show()<br>
*Output*<br>
![image](https://user-images.githubusercontent.com/98141713/178956958-bd65c47c-9c1b-485c-afe0-bd4c27aaf889.png)<br>

3.
import numpy as np<br>
import cv2 as cv
from matplotlib import pyplot as plt<br>
img = cv.imread('sunrise.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
img = cv.imread('sunrise.jpg',0)<br>
plt.hist(img.ravel(),256,[0,256]);<br>
plt.show()<br>

*Output*<br>
![image](https://user-images.githubusercontent.com/98141713/178957424-12b6f35a-4b44-4a2d-a865-704af1bb4452.png)<br>

**23. Program to perform basic image data analysisi using intensity transformation<br>

a)Image negetive<br>
b)log transformation<br>
c)Gamma correction**<br>

%matplotlib inline<br>
import imageio<br>
import matplotlib.pyplot as plt<br>
import warnings<br>
import matplotlib.cbook<br>
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)<br>
pic=imageio.imread('P.jpg')<br>
plt.figure(figsize=(6,6))<br>
plt.imshow(pic);<br>
plt.axis('off');<br>

**Output**<br>
![image](https://user-images.githubusercontent.com/98141713/179957464-c616d4af-5b74-4f01-b9d8-8c3970ecbe15.png)<br>

**Negetive**<br>
negetive=255-pic<br>
plt.figure(figsize=(6,6))<br>
plt.imshow(negetive);<br>
plt.axis('off');<br>

**Output**<br>
![image](https://user-images.githubusercontent.com/98141713/179957708-aa3c6639-095b-44c2-b60a-70c653ff2374.png)<br>

**Log transformation**<br>

%matplotlib inline<br>
import imageio<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
pic=imageio.imread('P.jpg')<br>
gray=lambda rgb: np.dot(rgb[...,:3], [0.299,0.587,0.114])<br>
gray=gray(pic)<br>
max_=np.max(gray)<br>

def log_transform(): return(255/np.log(1+max_))*np.log(1+gray)<br>
plt.figure(figsize=(5,5))<br>
plt.imshow(log_transform(),cmap=plt.get_cmap (name='gray'))<br>
plt.axis('off');<br>

**Output**<br>
![image](https://user-images.githubusercontent.com/98141713/179957963-f358e020-5a38-4411-9fdd-42d31906a863.png)<br>

**Gamma correction**<br>

import imageio<br>
import matplotlib.pyplot as plt<br>
pic=imageio.imread('P.jpg')<br>
gamma=2.2<br>
gamma_correction=((pic/255)**(1/gamma))<br>
plt.figure(figsize=(5,5))<br>
plt.imshow(gamma_correction)<br>
plt.axis('off');<br>

**Output**<br>
![image](https://user-images.githubusercontent.com/98141713/179958233-64f0a3d3-5f75-4f18-9c5e-0d8146579e5c.png)<br>

**24.Program to perform basic image manipulation<br>
a)Sharpness<br>
b)Flipping<br>
c)Cropping**<br>

**a)Sharping**<br>

from PIL import Image<br>
from PIL import ImageFilter<br>
import matplotlib.pyplot as plt<br>

my_image = Image.open('tea.jpg')<br>
sharp= my_image.filter(ImageFilter.SHARPEN)<br>

sharp.save('D:/image_sharpen.jpg')<br>
sharp.show()<br>
plt.imshow(sharp)<br>
plt.show() <br>

**Output**<br>
![image](https://user-images.githubusercontent.com/98141713/179958657-198fac64-5799-42c5-b8e2-6d8610ea7623.png)<br>

**Flipping**<br>
import matplotlib.pyplot as plt <br>
img=Image.open('tea.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>

flip=img.transpose(Image.FLIP_LEFT_RIGHT)<br>

flip.save('D:/image_flip.jpg')<br>
plt.imshow(flip)<br>
plt.show()<br>

**Output**<br>
![image](https://user-images.githubusercontent.com/98141713/179958999-34ad29d2-2b22-4f28-81b3-b77c4ba06c18.png)<br>

**Cropping**<br>

from PIL import Image<br>
import matplotlib.pyplot as plt<br> 
im=Image.open('tea.jpg')<br>
width, height = im.size<br>
im1=im.crop((280, 100,800,600))<br>

im1.show()<br>
plt.imshow(im1)<br>
plt.show()<br>

**Output**<br>
![image](https://user-images.githubusercontent.com/98141713/179959213-85f6e842-67b8-4f51-96c1-29ad5e5d8b4c.png)<br>

**Generate a matrix and display the image data**<br>
import matplotlib.image as image<br>
img=image.imread('sunrise.jpg')<br>
print('The Shape of the image is:',img.shape)<br>
print('The image as array is:')<br>
print(img)<br>

**Circle gradient**<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>

arr = np.zeros((256,256,3), dtype=np.uint8)<br>
imgsize = arr.shape[:2]<br>
innerColor = (255, 255, 255)<br>
outerColor = (0,0,0)<br>
for y in range(imgsize[1]):<br>
    for x in range(imgsize[0]):<br>
      distanceToCenter = np.sqrt((x - imgsize[0]//2) ** 2 + (y - imgsize[1]//2) ** 2)<br>
      distanceToCenter = distanceToCenter / (np.sqrt(2) * imgsize[0]/2)<br>
      r = outerColor[0] * distanceToCenter + innerColor[0] * (1 - distanceToCenter)<br>
        g = outerColor[1] * distanceToCenter + innerColor[1] * (1 - distanceToCenter)<br>
        b = outerColor[2] * distanceToCenter + innerColor[2] * (1 - distanceToCenter)<br>
       arr[y, x] = (int(r), int(g), int(b))<br>

plt.imshow(arr, cmap='gray')<br>
plt.show()<br>

**Output**<br>
![image](https://user-images.githubusercontent.com/98141713/180190852-4ef79f81-a6fb-40f2-9c1d-f115b535917a.png)<br>
<br>

**Generate a matrix to display an image**<br>
import matplotlib.image as image<br>
img=image.imread('sunrise.jpg')<br>
print('The Shape of the image is:',img.shape)<br>
print('The image as array is:')<br>
print(img)<br>

**Output**<br>
![image](https://user-images.githubusercontent.com/98141713/181209579-8c9da716-427a-4e33-9971-4a5f20a875e9.png)<br>

**Assignment**<br>
from numpy import asarray<br>
from PIL import Image<br>

image = Image.open('ppp.jpg')<br>
pixels = asarray(image)<br>

#print('Data Type: %s' % pixels.dtype)<br>
print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))<br>

pixels = pixels.astype('float32')<br>
# normalize to the range 0-1<br>
pixels /= 255.0<br>

print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))<br>
**Output**<br>
![image](https://user-images.githubusercontent.com/98141713/181234703-9223a19c-d242-4169-9eab-bdab5ce83ec8.png)<br>




**pixel normalization**<br>
from numpy import as array<br>
from PIL import Image<br>
image = Image.open('21.jpg')<br>
pixels = asarray(image)
#print('Data Type: %s' % pixels.dtype)<br>
print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))<br>
pixels = pixels.astype('float32')<br>
pixels /= 255.0<br>
print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))<br>


**Average**<br>

import cv2<br>
import matplotlib.pyplot as plt<br>
img=cv2.imread("21.jpg",0)<br>
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)<br>
plt.imshow(img)<br>
np.average(img)<br>


**SD**<br>

from PIL import Image,ImageStat<br>
import matplotlib.pyplot as plt<br>
im=Image.open('22.jpg')<br>
plt.imshow(im)<br>
plt.show()<br>
stat=ImageStat.Stat(im)<br>
print(stat.stddev)<br>

**Output**<br>
![image](https://user-images.githubusercontent.com/98141713/181434576-a3a26de1-626d-43f9-9c46-185b37e18ecd.png)<br>

**Max**<br>
import cv2<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
img=cv2.imread('21.jpg' )<br>
plt.imshow(img)<br>
plt.show()<br>
max_channels = np.amax([np.amax(img[:,:,0]), np.amax(img[:,:,1]),np.amax(img[:,:,2])])<br>
print(max_channels)<br>
**Output**<br>
![image](https://user-images.githubusercontent.com/98141713/181434730-bcf51d85-2911-45f5-9103-07d241600782.png)><br>



**Min**<br>
import cv2<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
img=cv2.imread('21.jpg' )<br>
plt.imshow(img)<br>
plt.show()<br>
min_channels = np.amin([np.min(img[:,:,0]), np.amin(img[:,:,1]),np.amin(img[:,:,2])])<br>
print(min_channels)<br>
**Output**<br>
![image](https://user-images.githubusercontent.com/98141713/181434843-5a4a932a-3027-445e-a48a-cf86e36cf3ca.png)<br>


**Edge detection using opencv**<br>
import cv2<br>
img = cv2.imread('man.jpg')<br>
cv2.imshow('Original', img)<br>
cv2.waitKey(0)<br>

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)<br>
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)<br>

sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)<br> 
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)<br> 
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)<br> 
cv2.imshow('Sobel X', sobelx)<br>
cv2.waitKey(0)<br>
cv2.imshow('Sobel Y', sobely)<br>
cv2.waitKey(0)<br>
cv2.imshow('Sobel X Y using Sobel() function', sobelxy)<br>
cv2.waitKey(0)<br>

 edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)<br> 
cv2.imshow('Canny Edge Detection', edges)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

**Output**<br>
**Original**<br>
![image](https://user-images.githubusercontent.com/98141713/186401229-a873103a-b124-47b7-9607-44ecbeb4c7fd.png)<br>
**Sobel X**<br>
![image](https://user-images.githubusercontent.com/98141713/186401423-102f2d2c-39ea-42d4-9cd8-65220b582bcd.png)<br>
**Sobel Y**<br>
![image](https://user-images.githubusercontent.com/98141713/186401589-ccc73cb6-5aab-4418-bda4-5ec18859cf3b.png)<br>
**Sobel X and Y using sobel fnctn**<br>
![image](https://user-images.githubusercontent.com/98141713/186401731-a2450bbf-35c6-4ef3-80c1-af8b2d54f63e.png)<br>
**Canny edge detection**<br>
![image](https://user-images.githubusercontent.com/98141713/186401851-3411cb99-66b8-48d1-8663-5a9c3d185632.png)<br>


**Basic pillow functions**<br>
from PIL import Image, ImageChops, ImageFilter <br>
from matplotlib import pyplot as plt<br>


x = Image.open("x.png")<br>
o=Image.open("o.png")<br>


print('size of the image:', x.size, 'colour mode:', x.mode)<br>
print('size of the image: ', o.size, 'colour mode:', o.mode)<br>

plt.subplot(121),plt.imshow(x)<br>
plt.axis('off') <br>
plt.subplot(122), plt.imshow(o)<br>
plt.axis('off')<br>

merged=ImageChops.multiply(x,o)<br>
add=ImageChops.add(x,o)<br>
      
greyscale=merged.convert('L')<br>
greyscale<br>

**Output**<br>
size of the image: (256, 256) colour mode: RGB<br>
size of the image:  (256, 256) colour mode: RGB<br>
![image](https://user-images.githubusercontent.com/98141713/186650405-ff63bbcd-c9ab-4369-9681-aab8c1a79ab8.png)<br>


**2**<br>
image=merged<br>
print('image size:',image.size,<br>
      '\ncolor mode:', image.mode, <br>
      '\nimage width:', image.width,'| also represented by:',image.size[0],<br>
      '\nimage height:', image.height, '| also represented by:',image.size[1],)<br>
 **Output**<br>
 image size: (256, 256) <br>
color mode: RGB <br>
image width: 256 | also represented by: 256 <br>
image height: 256 | also represented by: 256<br>

**3**<br>
pixel = greyscale.load()<br>
for row in range (greyscale.size[0]):<br>
 for column in range(greyscale.size[1]):<br>
    if pixel[row, column] != (255):<br>
      pixel[row, column] = (0)<br>
    
greyscale<br>

**Output**<br>
![image](https://user-images.githubusercontent.com/98141713/186650649-5f089a26-0809-435e-88eb-3095f4ec4dcf.png)<br>

**4**<br>
invert = ImageChops.invert(greyscale)<br>

bg=Image.new('L', (256, 256), color=(255)) <br>
subt=ImageChops. subtract (bg, greyscale)<br> 
rotate =subt.rotate(45)<br>
rotate<br>

**Output**<br>
![image](https://user-images.githubusercontent.com/98141713/186650805-d5e2bba4-ab10-4b3e-a506-3a150146ee65.png)<br>

**5**<br>
blur=greyscale.filter(ImageFilter.GaussianBlur (radius=1))<br>
edge=blur.filter(ImageFilter.FIND_EDGES)<br>
edge<br>
**Output**<br>
![image](https://user-images.githubusercontent.com/98141713/186650948-ac389f46-42bd-44eb-ada9-cd082ed323fa.png)<br>

**6**<br>
edge=edge.convert('RGB')<br>

bg_red=Image.new('RGB', (256,256), color=(255,0,0))<br>
filled_edge = ImageChops.darker(bg_red, edge)<br>
filled_edge<br>

**Output**<br>
![image](https://user-images.githubusercontent.com/98141713/186651084-e017711c-dfa4-4a4b-bd6b-0a2c9fd80fd8.png)<br>

**Image restoration**<br>

**a)restore damaged images**<br>
import numpy as np<br>
import cv2<br>
import matplotlib.pyplot as plt<br>

img=cv2.imread('dimage_damaged.png')<br>
plt.imshow(img)<br>
plt.show()<br>

mask= cv2.imread('dimage_mask.png', 0)<br>
plt.imshow(mask)<br>
plt.show()<br>

dst = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)<br>

cv2.imwrite('dimage_inpainted.png', dst)<br>
plt.imshow(dst)<br>
plt.show()<br>

**Output**<br>
![image](https://user-images.githubusercontent.com/98141713/186654385-dc88f3b8-4710-4307-a060-91dd0a7ff9b8.png)<br>
![image](https://user-images.githubusercontent.com/98141713/186654480-509778a7-8bf6-4b59-b069-6b5aeefc3d9b.png)<br>

**b)Removing logos**<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
import pandas as pd<br>
from skimage.restoration import inpaint<br>
from skimage.transform import resize<br>
from skimage import color<br>
plt.rcParams['figure.figsize'] = (10, 8)<br>

def show_image(image, title= 'Image', cmap_type='gray'):<br>
    plt.imshow(image, cmap=cmap_type)<br>
    plt.title(title)<br>
    plt.axis('off')<br>
def plot_comparison(img_original, img_filtered, img_title_filtered):<br>
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 8), sharex=True, sharey=True)<br>
    ax1.imshow(img_original, cmap=plt.cm.gray)<br>
    ax1.set_title('original')<br>
    ax1.axis('off')<br>
    
   ax2.imshow(img_filtered, cmap=plt.cm.gray)<br>
    ax2.set_title(img_title_filtered)<br>
    ax2.axis('off')<br>
    
image_with_logo= plt.imread('imlogo.png')<br>
mask= np.zeros(image_with_logo.shape[:-1])<br>
mask [210:272, 360:425] = 1<br>
image_logo_removed=inpaint.inpaint_biharmonic (image_with_logo, mask,multichannel=True)<br>
plot_comparison(image_with_logo, image_logo_removed, 'Image with logo removed')<br>

**Output**<br>
![image](https://user-images.githubusercontent.com/98141713/186654711-aad76945-355a-4f21-8b48-6f59bbc71eb0.png)<br>

**2)Noise**<br>
**a)Adding noise**<br>
from skimage.util import random_noise<br>
fruit_image = plt.imread('fruitts.jpeg')<br>

noisy_image = random_noise(fruit_image)<br>

plot_comparison(fruit_image,noisy_image,'Noisy image')<br>

**Output**<br>
![image](https://user-images.githubusercontent.com/98141713/186655007-bb809e2f-769e-4ba3-bf6b-2aad594f545f.png)<br>

**b)Redusing noise**<br>
from skimage.restoration import denoise_tv_chambolle<br>
noisy_image = plt.imread('noisy.jpg')<br>
denoised_image = denoise_tv_chambolle (noisy_image, multichannel=True)<br>
plot_comparison (noisy_image, denoised_image,'Denoised Image')<br>

**Output**<br>
![image](https://user-images.githubusercontent.com/98141713/186655276-30c383f8-4caa-4b8f-8fa2-fa8a159d978e.png)<br>

**c)Reducing noice with preserving edges**<br>
from skimage.restoration import denoise_bilateral<br>
landscape_image = plt.imread('noisy.jpg')<br>
denoised_image = denoise_bilateral (landscape_image, multichannel=True)<br>
plot_comparison (landscape_image, denoised_image, 'Denoised Image')<br>

**Output**<br>
![image](https://user-images.githubusercontent.com/98141713/186655532-007505dd-3014-4ef9-b27f-d5fa6e512809.png)<br>

**3)Segmentation**<br>
**a)Super pixel segmentation**<br>
from skimage.segmentation import slic<br>
from skimage.color import label2rgb<br>
face_image = plt.imread('face.jpg')<br>

segments = slic (face_image, n_segments=400)<br>

segmented_image = label2rgb(segments, face_image, kind='avg')<br>

plot_comparison (face_image, segmented_image, 'segmented image, 400 superpixels')<br>

**Output**<br>
![image](https://user-images.githubusercontent.com/98141713/186655808-31501688-b7d2-4506-9e07-24b068ee4374.png)<br>

**4)contours**<br>
**a)Contouring shapes**<br>
def show_image_contour(image, contours):<br>
    plt.figure() <br>
    for n, contour in enumerate (contours):<br>
        plt.plot(contour[:, 1], contour[:, 0], linewidth=3) <br>
    plt.imshow(image, interpolation='nearest', cmap='gray_r')<br>
    plt.title('Contours')<br>
    plt.axis('off')<br>
    
from skimage import measure, data<br>
horse_image = data.horse()<br>
contours = measure.find_contours (horse_image, level=0.8)<br>
show_image_contour (horse_image, contours)<br>

**Output**<br>
![image](https://user-images.githubusercontent.com/98141713/186656116-1f1deb4c-c999-428b-ad36-b099e8ccb101.png)<br>

**b)Find contous in an image that is not binary**<br>
from skimage.io import imread<br>
from skimage.filters import threshold_otsu<br>
image_dices = imread('diceimg.png')<br>
image_dices = color.rgb2gray(image_dices)<br>
thresh = threshold_otsu(image_dices)<br>
binary = image_dices > thresh<br>
contours = measure.find_contours(binary, level=0.8)<br>
show_image_contour(image_dices,contours)<br>

**output**<br>
![image](https://user-images.githubusercontent.com/98141713/186656331-221228a1-cd4d-444a-b882-2f1c88c5f250.png)<br>

**Number of dots in the contours**<br>
import numpy as np<br>
shape_contours = [cnt.shape[0] for cnt in contours]<br>

max_dots_shape = 50<br>

dots_contours = [cnt for cnt in contours if np.shape(cnt)[0] < max_dots_shape]<br>

show_image_contour (binary, contours)<br>

print('Dices dots number:{}.'.format(len(dots_contours)))<br>

**Output**<br>
![image](https://user-images.githubusercontent.com/98141713/187874092-37f67656-3b60-4384-bf1e-347b33092fd2.png)<br>

**Program to implement to perform a various edge detection**<br>
**Canny edge detection**<br>
import cv2<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
plt.style.use('seaborn')<br>

loaded_image = cv2.imread("animate.jpeg")<br>
loaded_image = cv2.cvtColor(loaded_image,cv2.COLOR_BGR2RGB)<br>
gray_image = cv2.cvtColor(loaded_image,cv2.COLOR_BGR2GRAY)<br>
edged_image= cv2.Canny(gray_image, threshold1=30, threshold2=100)<br>

plt.figure(figsize=(20,20))<br>
plt.subplot(1,3,1)<br>
plt.imshow(loaded_image,cmap="gray")<br>
plt.title("original Image")<br>
plt.axis("off")<br>

plt.subplot(1,3,2)<br>
plt.imshow(gray_image,cmap="gray")<br>
plt.axis("off")<br>

plt.title("GrayScale Image")<br>
plt.subplot(1,3,3)<br>
plt.imshow(edged_image,cmap="gray")<br>
plt.axis("off")<br>

plt.title("Canny Edge Detected Image")<br>
plt.show()<br>

**Output**<br>
![image](https://user-images.githubusercontent.com/98141713/187896522-0e7a3033-1a44-4897-8a80-32b292a3179c.png)<br>

**Edge detection schemes - the gradient (Sobel - first order derivatives)<br>
based edge detector and the Laplacian (2nd order derivative, so it is
extremely sensitive to noise) based edge detector**<br>
import cv2<br>
import numpy as np <br>
from matplotlib import pyplot as plt<br>

imge=cv2.imread('animate.jpeg')<br>
gray = cv2.cvtColor(imge, cv2.COLOR_BGR2GRAY)<br>
img = cv2.GaussianBlur (gray, (3,3),0)<br>
laplacian= cv2.Laplacian (img,cv2.CV_64F)<br>
sobelx = cv2.Sobel (img,cv2.CV_64F,1,0,ksize=5)<br>
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5) <br>
                
plt.subplot(2,2,1), plt.imshow(img,cmap = 'gray')<br>
plt.title('Original'), plt.xticks([]), plt.yticks([])<br>
plt.subplot(2,2,2), plt.imshow(laplacian, cmap = 'gray')<br>
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])<br>
plt.subplot(2,2,3), plt.imshow(sobelx, cmap = 'gray')<br>
plt.title('Sobel x'), plt.xticks([]), plt.yticks([])<br>
plt.subplot(2,2,4), plt.imshow(sobely,cmap = 'gray') <br>
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])<br>
plt.show()<br>

**Output**<br>
![image](https://user-images.githubusercontent.com/98141713/187896764-3f7b75f3-7f59-4e3c-a452-22103373071e.png)<br>

**Edge detection using prewitt operation**<br>
import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
img= cv2.imread('animate.jpeg')<br>
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) <br>
img_gaussian = cv2.GaussianBlur (gray, (3,3),0)<br>

kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])<br>
kernely=np.array([[-1,0,1],[-1,0,1],[-1,0,1]])<br>
img_prewittx= cv2.filter2D (img_gaussian, -1, kernelx)<br>
img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)<br>
cv2.imshow("Original Image", img)<br>
cv2.imshow("Prewitt x", img_prewittx)<br>
cv2.imshow("Prewitt y", img_prewitty)<br>
cv2.imshow("Prewitt", img_prewittx + img_prewitty)<br>
cv2.waitKey()<br>
cv2.destroyAllwindows()<br>

**Ouput**<br>
![image](https://user-images.githubusercontent.com/98141713/187897608-41fcd993-4fcd-442e-8491-5b425c01f094.png)<br>
![image](https://user-images.githubusercontent.com/98141713/187897751-b49c82f6-da51-4660-ad65-d46110401310.png)<br>
![image](https://user-images.githubusercontent.com/98141713/187897916-165a2da8-7207-443a-afe6-911d04e9dc55.png)<br>
![image](https://user-images.githubusercontent.com/98141713/187897976-f0e1decb-d0ed-41d5-a2dd-560656f6b37b.png)<br>

**Roberts Edge Detection- Roberts cross operator**<br>
import cv2<br>
import numpy as np <br>
from scipy import ndimage<br>
from matplotlib import pyplot as plt <br>
roberts_cross_v = np.array([[1, 0],<br>
                            [0,-1]])<br>
roberts_cross_h= np.array([[0, 1],<br>
                           [-1, 0]])<br>
img = cv2.imread("animate.jpeg",0).astype('float64')<br>
img/=255.0 <br>
vertical=ndimage.convolve( img, roberts_cross_v ) <br>
horizontal=ndimage.convolve( img, roberts_cross_h)<br>
edged_img = np.sqrt(np.square (horizontal) + np.square(vertical))<br>
edged_img*=255<br>
cv2.imwrite("Output.jpg",edged_img)<br>
cv2.imshow("OutputImage", edged_img)<br>
cv2.waitKey()<br>
cv2.destroyAllWindows()<br>

**Output**<br>
![image](https://user-images.githubusercontent.com/98141713/187898646-9f04c130-b6ad-42aa-9116-b4d97a2679b9.png)<br>








