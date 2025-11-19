# first things first lets get the data we ned for the tasl
import cv2
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

iconsarray=[]
iconsLocation = Path("data_provided_for_task/IconDataset/png")
for p in iconsLocation.rglob("*.png"):
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)  # BGRA if has alpha
    iconsarray.append(img)

print(f"icons array has {len(iconsarray)} images of size {iconsarray[0].shape}")

testImagesarray=[]
testImagesLocation = Path("data_provided_for_task/images")
for p in testImagesLocation.rglob("*.png"):
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)  # BGRA if has alpha
    testImagesarray.append(img)

print(f"test images array has {len(testImagesarray)} images of size {testImagesarray[0].shape}") 

#now the data is loaded, build gaussian pyramids for the icons
class libarygaussianPyramid:
    def __init__(self, image, levels):
        self.levels = levels
        self.pyramid = [image] #as i prefer appending my pyramid is going to be upside down if it gets printed but it works the exact same
        self.build_pyramid()

    def build_pyramid(self):
        for i in range(1, self.levels):
            image = cv2.pyrDown(self.pyramid[i - 1])
            #note to self: cv2.pyrDown 
            #Applies a 5×5 Gaussian blur to the image
            #Downsamples it by a factor of 2 in width and height
            self.pyramid.append(image)
    
    def redefine_pyramid(self, image):
        self.pyramid = [image]
        self.build_pyramid()

    def get_level(self, level):
        if level < 0 or level >= self.levels:
            print("Level out of bounds")
        else:
            return self.pyramid[level]
        
    def get_pyramid(self):
        return self.pyramid
        
    def show_pyramid_test(self):
        #this function was made using github copilot. make sure to make a different version before submission as this doesnt have the functinoality i want but is used for testing
        
        rgba_images = [cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA) for img in self.pyramid]

        cols = len(rgba_images)
        plt.figure(figsize=(3 * cols, 4))

        

        for i, img in enumerate(rgba_images):
            bg_color = np.array([0, 0, (i*(255/5))], dtype=np.uint8)
            h, w = img.shape[:2]
            # Create solid background
            background = np.ones((h, w, 3), dtype=np.float32) * bg_color
            # Convert foreground to float 0–1
            fg = img[..., :3].astype(np.float32) # back to being just RGB
            alpha = img[..., 3:4].astype(np.float32) / 255.0

            # Composite: out = fg*alpha + bg*(1-alpha)

            comp = fg * alpha + background * (1 - alpha)

            ax = plt.subplot(1, cols, i + 1)
            ax.imshow(comp.astype(np.uint8))  # cast to uint8 for display
            ax.set_title(f"Level {i}")
            ax.axis("off")

        #print("test live #c")
       

    




class customGaussianPyramid(libarygaussianPyramid):
    '''the custom gaussian pyramid will use all the same methods and features as the libary version with the difference that the gaussian filtering will be done without the libary so that function alone needs to be rewritten
    '''
    def __init__(self, image, levels):
        super().__init__(image, levels)

    def build_pyramid(self):
        for i in range(1, self.levels):
            #here will be the custom gaussian
            #self.pyramid.append(image)
            break


class matchIconToImage:
    def __init__(self, iconPyramid, imagePyramid):
        self.iconPyramid = iconPyramid
        self.imagePyramid = imagePyramid
        self.matches = []
        self.match_icons_to_image()

    def match_icons_to_image(self):
        #this function will match the icons to the image at each level of the pyramid
        pass


def testEnviorment():
    testIcon = iconsarray[0]
    gp = libarygaussianPyramid(testIcon, 5)
    gp.show_pyramid_test()
    plt.tight_layout()
    plt.show()

testEnviorment()