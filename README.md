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
