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

#print(f"icons array has {len(iconsarray)} images of size {iconsarray[0].shape}")

testImagesarray=[]
testImagesLocation = Path("data_provided_for_task/images")
for p in testImagesLocation.rglob("*.png"):
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)  # BGRA if has alpha
    testImagesarray.append(img)

#print(f"test images array has {len(testImagesarray)} images of size {testImagesarray[0].shape}") 

#now the data is loaded, build gaussian pyramids for the icons
class libarygaussianPyramid:
    def __init__(self, image, levels=5,octaves=3):
        self.levels = levels
        self.octaves = octaves
        self.sigmaScale=0.5
        self.pyramid = [[image]] #as i prefer appending my pyramid is going to be upside down if it gets printed but it works the exact same
        self.kernal_size = (9, 9)
        self.downsample_scale = 0.75
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
    def __init__(self, iconsToConsider, image):
        #old->self.iconPyramid = iconsToConsider# changed it so that the funcitons are a bit neeater
        self.iconsPyramids=[]
        for icon in iconsToConsider:
            pyrami_generator = libarygaussianPyramid(icon, levels=5, octaves=3)
            self.iconsPyramids.append(pyrami_generator.get_pyramid())
        self.image = image# originally thought we would need an image pyramid here but rereading the spec i dont believe thats what it wants
        self.matches = []
        self.match_icons_to_image()
        print(f"Matched {len(self.matches)} icons to the image, \nIconsPyramids shape: {len(self.iconsPyramids),len(self.iconsPyramids[0]),len(self.iconsPyramids[1])}   ")

    def match_icons_to_image(self):
        #this function will match the icons to the image at each level of the pyramid
        '''
        psuedo code so i dont get confused
        1 iterate over each icon pyramid
            2 iterate over each section of the test_image
                3 get the MSE between the icon at each level and the section of the image at the same level
                4 if the MSE is below the last MSE of that image then store the match
        5 return all the matches found remove all matches that are above a certain MSE threshold <-- rather than a set value we could just do that it picks the x best o

        matches should be sotred as [icon_index, top_pixel_value, left_pixel_value, bottom_pixel_value, right_pixel_value, MSE_value]
        '''
        #presteps, make vairables to be interacted with
        bestMatchOfEachIcon=[]
        #step 1
        for IndexOfIcon in range(len(self.iconsPyramids)):
            PreviousBestMSE=float('inf')#  <--- didnt know about this before but python has a representation of infinity!!!! :o
            MatchInfo=[]
            currenticonPyramid=self.iconsPyramids[IndexOfIcon]
            numberoflevels=len(currenticonPyramid)
            numberOfOctaves=len(currenticonPyramid[0])
            matchedImageSection=[]
            matchedImage=[]
            for levelIndex in range(numberoflevels):
                print(f"Matching icon {IndexOfIcon} at level {levelIndex}")
                for octaveIndex in range(numberOfOctaves):
                    print(f"  Using octave {octaveIndex}")
                    iconImage=currenticonPyramid[levelIndex][octaveIndex]
                    iconHeight=iconImage.shape[0]
                    iconWidth=iconImage.shape[1]
                    #step 2
                    for y in range(0, self.image.shape[0]-iconHeight): #max(1, iconHeight//10)): #probably dont need this but if this takes ages it could be useful
                        for x in range(0, self.image.shape[1]-iconWidth): #max(1, iconWidth//10)):
                            #step 3
                            #get the section of the image
                            imageSection=self.image[y:y+iconHeight, x:x+iconWidth]
                            try:
                                '''
                                #the icons have an alpha chanel but the images dont so we need to edit the function so that:
                                # 1 it only compares the RGB channels of the icon to the image
                                # 2 it only computes teh MSE for the pixels where the alpha channel is above a certain threshold (e.g., 128)
                                '''


                                
                                #mseValue=self.calculateMSE(iconImage, imageSection)
                                iconAlpha = iconImage[..., 3] / 255.0
                                mask = iconAlpha > 0.5

                                iconRGB = iconImage[..., :3].astype(np.int16)
                                sectionRGB = imageSection.astype(np.int16)

                                diff = (iconRGB - sectionRGB)**2
                                

                                # Apply mask to all channels
                                masked_diff = diff[mask]

                                MSE_withoutAlpha = masked_diff.sum()
                                mseValue = MSE_withoutAlpha/ np.count_nonzero(mask)  # Normalize by number of valid pixels prevents just choosing the smallest one
                                '''
                                mseValue=0 
                                for i in range(len(imageSection)):
                                    for j in range(len(imageSection[0])):
                                        if iconImage[i][j][3]>128:
                                            diff = (int(iconImage[i][j][0]) - int(imageSection[i][j][0]))**2
                                            diff += (int(iconImage[i][j][1]) - int(imageSection[i][j][1]))**2
                                            diff += (int(iconImage[i][j][2]) - int(imageSection[i][j][2]))**2
                                            mseValue += diff
                                '''
                                #step 4
                                if mseValue<PreviousBestMSE:
                                    PreviousBestMSE=mseValue
                                    MatchInfo=[IndexOfIcon, y, x, y+iconHeight, x+iconWidth, mseValue]
                                    matchedImageSection=imageSection
                                    matchedImage=iconImage
                            except Exception as e:
                                print(f"!!!!!ruh roh big error, see error below!!!!!\n{e}")
                                
            bestMatchOfEachIcon.append(MatchInfo)
            print(f"Best match for icon {IndexOfIcon}: {MatchInfo}")
            
            
            plt.figure(figsize=(6, 3))

            # Show the matched section from the test image
            plt.subplot(1, 2, 1)
            plt.imshow(matchedImageSection)
            plt.title(f"Matched Section (icon {IndexOfIcon})")
            plt.axis("off")

            # Show the icon that matched
            plt.subplot(1, 2, 2)
            # Strip alpha for display
            plt.imshow(matchedImage[..., :3])
            plt.title(f"Matched Icon {IndexOfIcon}")
            plt.axis("off")

            plt.tight_layout()
            plt.show()
        #step 5
        #for now I will set the MSE threshold to be the top 5 best matches but will set it to a better value later
        bestMatchOfEachIcon.sort(key=lambda x: x[5]) #sort by M
        self.matches=bestMatchOfEachIcon[:5]
        print(f"Best matches: {self.matches}")
        

    

def testEnviormentA():
    #testIcon = iconsarray[0]
    #gp = libarygaussianPyramid(testIcon, levels=5, octaves=2)
    #gp.show_pyramid_test()
    #plt.tight_layout()
    #plt.show()
    #i=input("press enter to continue to next test") 
    testIcon = testImagesarray[14]
    gp = libarygaussianPyramid(testIcon, levels=5, octaves=2)
    gp.show_pyramid_test()
    plt.tight_layout()
    plt.show()

def testEnviormentB():
    matchingObject=matchIconToImage(iconsarray, testImagesarray[14])# for some reason image 14 seems to have correspond to image_4
    
    #matchingObject.matches



testEnviormentB()