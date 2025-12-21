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

'''
through inspection i believe the smallest size any of the icons is (proportional to the search imaage) is 0.125 or 12.5% (64/512)

the way I want to do this is have image scaled about 3 times and there for icons will be sclaed down to a max of 3 with only searching the bottom levels of the gaussian pyramid 3 levels below the level currently be ing searched

it seems like all icons are of the size 2**n or 3*2**n which reduces the amount of levels that would ened to be made

'''

#now the data is loaded, build gaussian pyramids for the icons
class libarygaussianPyramid:
    def __init__(self, image, levels=5):
        self.levels = levels
        #taking octaves out for this iteration as i dont believe they will help at all
        self.sigmaScale=0.5
        self.pyramid = [[image]] #as i prefer appending my pyramid is going to be upside down if it gets printed but it works the exact same
        self.kernal_size = (5, 5)
        self.downsample_scale = 0.5
        self.build_pyramid()
        
        #self.get_pyramid()

    def build_pyramid(self):
        
        image=self.pyramid[0][0]
        this_level = [image]
        for j in range(1, (self.octaves)): 
            sigma_value=(j)*self.sigmaScale# i was told this was a sesible way to increase the sigma value            #print(j)
            this_level.append(cv2.GaussianBlur(image, self.kernal_size, sigmaX=sigma_value, sigmaY=sigma_value))
        #print(f"Level 0 has {len(this_level)} images of size {this_level[0].shape}")
        self.pyramid = []  # Reset pyramid to empty list
        self.pyramid.append(this_level)
        #print(f"Pyramid now has {len(self.pyramid)} levels")
        
        for i in range(1, self.levels):
            #image = cv2.pyrDown(self.pyramid[i - 1][0])
            #note to self: cv2.pyrDown 
            #Applies a 5×5 Gaussian blur to the image
            #Downsamples it by a factor of 2 in width and height

            heightOfPrevious, widthOfPrevious = self.pyramid[i - 1][0].shape[:2]
            new_w = int(widthOfPrevious * self.downsample_scale)
            new_h = int(heightOfPrevious * self.downsample_scale)

            image = cv2.resize(self.pyramid[i - 1][0],(new_w, new_h),interpolation=cv2.INTER_AREA)
            this_level = [image]
            #print("test level")
            for j in range(1, (self.octaves)): 
                #print(j)
                sigma_value=j*self.sigmaScale
                this_level.append(cv2.GaussianBlur(image, self.kernal_size,  sigmaX=sigma_value, sigmaY=sigma_value))
            #print(f"Level {i} has {len(this_level)} images of size {this_level[0].shape}")
            self.pyramid.append(this_level)
            #print(f"Pyramid now has {len(self.pyramid)} levels")
          
    def redefine_pyramid(self, image):
        self.pyramid = [[image]]
        self.build_pyramid()

    def get_level(self, level):
        if level < 0 or level >= self.levels:
            print("Level out of bounds")
        else:
            return self.pyramid[level]
        
    def get_pyramid(self):
        return self.pyramid


    def show_pyramid_test(self):
        print("Levels:", len(self.pyramid))
        print("Images per level:", [len(row) for row in self.pyramid])

        rows = len(self.pyramid)
        cols = max(len(row) for row in self.pyramid)

        plt.figure(figsize=(4 * cols, 4 * rows))

        for i in range(rows):            # pyramid level
            for j in range(len(self.pyramid[i])):   # images inside level

                img = self.pyramid[i][j]
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

                # Background color depends only on the pyramid level
                bg_color = np.array([0, 0, (i * (255 / rows))], dtype=np.uint8)

                h, w = img.shape[:2]
                background = np.ones((h, w, 3), dtype=np.float32) * bg_color

                # Foreground RGB and alpha
                fg = img[..., :3].astype(np.float32)
                alpha = img[..., 3:4].astype(np.float32) / 255.0

                # Composite foreground over background
                comp = fg * alpha + background * (1 - alpha)

                ax = plt.subplot(rows, cols, i * cols + j + 1)
                ax.imshow(comp.astype(np.uint8))
                ax.set_title(f"Level {i}, Img {j}")
                ax.axis("off")



def testEnviormentA():
    #testIcon = iconsarray[0]
    #gp = libarygaussianPyramid(testIcon, levels=5, octaves=2)
    #gp.show_pyramid_test()
    #plt.tight_layout()
    #plt.show()
    #i=input("press enter to continue to next test") 
    testIcon = testImagesarray[18]
    gp = libarygaussianPyramid(testIcon, levels=5, octaves=2)
    gp.show_pyramid_test()
    plt.tight_layout()
    plt.show()