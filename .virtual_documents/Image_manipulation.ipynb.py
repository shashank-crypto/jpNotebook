import numpy as np
import matplotlib.pyplot as plt
from PIL import Image , ImageOps , ImageFilter


img = np.array(Image.open('Sample_imgs/dog1.jpg'))


plt.figure(figsize = (8,8),edgecolor='green')
plt.imshow(img)


plt.figure(figsize = (8,8),edgecolor = 'yellow',frameon='False')
plt.imshow(img)


print('img-dimension: ',img.ndim,'\n')
print('img-shape: ',img.shape,'\n')
print('DType: ', img.dtype,'\n')

## Gives the R,G,B value at the point (20,20)
print(img[20,20],'\n')

# Gives the max value of pixel at channel B(Blue)
print(img[:,:,2].max())


img.shape


import matplotlib as mp


mp.rcParams["figure.edgecolor"] = 'yellow'


plt.imshow(img)


img = img.sum(2) / (255*3)


plt.imshow(img)


im = Image.open('Sample_imgs/dog3.jpg')

im_np = np.dot(np.array(Image.open('Sample_imgs/dog3.jpg')),[0.2989,0.5870,0.1140])

im_np1 = np.array(im).sum(2)/(255*3)
# plt.imshow(im_np)
plt.imshow(im_np1)
# im.show()


plt.imshow(im_np)


random_im = np.random.randint(0,190,size=(231,177,3),dtype='uint8')

print(random_im)
data = np.array(random_im)
print(data.shape)
Image.fromarray(random_im)


img = np.array(Image.open('Sample_imgs/dog4.jpg'))
git = img.copy()

git.shape


git


Image.fromarray(git)


git[1:10,1:100,:] = [0,0,0] #np.zeros(git[1:10,1:100,:].shape)


Image.fromarray(git)


git = git/255


git = git*255
git = git.astype('uint8')


git = git.astype('uint8')
Image.fromarray(git)


import skimage
from skimage import io , img_as_float, img_as_ubyte


image = io.imread('Sample_imgs/cat4.jpg')
img_float = img_as_float(image)
print(img_float)


import cv2


img = cv2.imread('Sample_imgs/cat4.jpg' , 1)
# img = cv2.IMREAD_COLOR('Sample_imgs/cat4.jpg')
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


plt.imshow(Image.open('Sample_imgs/cat4.jpg'))


(Image.open('Sample_imgs/cat4.jpg')).filter(ImageFilter.GaussianBlur(3))


import cv2
import glob


path = 'Sample_imgs/*.jpg'

for file in glob.glob(path):
    print(file)
    a = cv2.imread(file)
    c = cv2.cvtColor(a , cv2.COLOR_BGR2RGB)
    plt.imshow(Image.open(file))
    
#     plt.imshow(c)
#     cv2.imshow('Colored_image' , a)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()



