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
        
        for i in range(1, self.levels):
            image = cv2.pyrDown(self.pyramid[i - 1][0])
            #note to self: cv2.pyrDown 
            #Applies a 5×5 Gaussian blur to the image
            #Downsamples it by a factor of 2 in width and height
            this_level = [image]
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
            for j in range(len(self.pyramid[i])):   # images inside level  <--- with octaves removed should now just be 1 (im not getting rid of it hough incase I add octaves back)

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



class matchIconToImage:
    def __init__(self, iconsToConsider, image, levelsToIMAGEpyramid=3, IconLevelsBelowImageSizeToCheck=4):
        self.iconsToConsider=iconsToConsider
        self.image = image
        self.matches = []
        self.levelsToIMAGEpyramid=levelsToIMAGEpyramid
        self.IconLevelsBelowImageSizeToCheck=IconLevelsBelowImageSizeToCheck
        ########################################################################
        #pyrami_generator = libarygaussianPyramid(image, levels=levelsToIMAGEpyramid)
        #self.imagePyramids=(pyrami_generator.get_pyramid())
        ########################################################################
        self.iconsPyramids=[]
        for icon in iconsToConsider:
            pyrami_generator = libarygaussianPyramid(icon, levels=(levelsToIMAGEpyramid+IconLevelsBelowImageSizeToCheck))
            self.iconsPyramids.append(pyrami_generator.get_pyramid())

    def checkIndividualImageFirst(self,imageToCheck,index_of_image_pyramid,iconPyramid):
        icons_to_check=iconPyramid[(index_of_image_pyramid):(index_of_image_pyramid + self.IconLevelsBelowImageSizeToCheck)]
        #BestMatchSoFar: [error,top,left,bottom,right]
        BestMatchSoFar = [99999,0 , 0   ,1     , 1   ,0] #just place holders
        #print(f"stop0 {len(icons_to_check)} : {len(icons_to_check[0])} : {icons_to_check[0][0].shape}")
        for iconSizeIndex in range(0,len(icons_to_check)):
            iconHeight=icons_to_check[iconSizeIndex][0].shape[1]
            iconWidth=icons_to_check[iconSizeIndex][0].shape[0]
            iconImage=icons_to_check[iconSizeIndex][0]
            #print(f"stopA {imageToCheck.shape} : {iconHeight} : {iconWidth}")
            for y in range(0, (imageToCheck.shape[0]-iconHeight)):
                #print(f"_____________stopB  y:{y}")
                for x in range(0, (imageToCheck.shape[1]-iconWidth)): 
                    imageSection=imageToCheck[y:y+iconHeight, x:x+iconWidth]
                    #the main difficulty here is dealing with the fact that icon is RGBA and image is RGB
                    mseValue=0 
                    for i in range(0,iconHeight):
                        for j in range(0,iconWidth):
                            #if iconImage[i][j][3]>128:
                            diff = (int(iconImage[i][j][0]) - int(imageSection[i][j][0]))**2
                            diff += (int(iconImage[i][j][1]) - int(imageSection[i][j][1]))**2
                            diff += (int(iconImage[i][j][2]) - int(imageSection[i][j][2]))**2
                            mseValue += diff
                    mseValue=mseValue/(iconHeight*iconWidth)
                    if mseValue<BestMatchSoFar[0]:
                        BestMatchSoFar=[mseValue, y, x, y+iconHeight, x+iconWidth,iconSizeIndex]
        return BestMatchSoFar

                            
    def checkReducedAreaIndividualImage(self,imageToCheck,index_of_image_pyramid,iconPyramid,top,left,borderweidth=4,indextoCheck=0):
        '''this code is almost the exact same as the previous one but this time we know roughly where the image is so only need to check in that location +_ the border width'''
        icons_to_check=iconPyramid[(index_of_image_pyramid):(index_of_image_pyramid + self.IconLevelsBelowImageSizeToCheck)]
        #BestMatchSoFar: [error,top,left,bottom,right]
        BestMatchSoFar = [99999,0 , 0   ,1     , 1   ] #just place holders
        #print(f"stop0 ____reduced area____ {len(icons_to_check)} : {len(icons_to_check[0])} : {icons_to_check[0][0].shape}")
        iconSizeIndex=indextoCheck
        iconHeight=icons_to_check[iconSizeIndex][0].shape[1]
        iconWidth=icons_to_check[iconSizeIndex][0].shape[0]
        iconImage=icons_to_check[iconSizeIndex][0]
        #print(f"stopA {imageToCheck.shape} : {iconHeight} : {iconWidth}")
        
        for y in range(0, (2*borderweidth)):
            #to prevetn errors we will check that none of the image would be out of bounds in this search if it is then we skip this one!
            if(((y+top-borderweidth)>=0) and (y+top-borderweidth+iconHeight)<len(imageToCheck)):
                for x in range(0, (2*borderweidth)): 
                    #to prevetn errors we will check that none of the image would be out of bounds in this search if it is then we skip this one!
                    if(((x+left-borderweidth)>=0) and (x+iconWidth+left-borderweidth)<len(imageToCheck)):
                        imageSection=imageToCheck[(y+top-borderweidth):(y+top-borderweidth+iconHeight), (x+left-borderweidth):(x+iconWidth+left-borderweidth)]
                        #print(f"image section shape{imageSection.shape} bounds used {(y+top-borderweidth), (x+left-borderweidth),(y+top-borderweidth+iconHeight),(x+iconWidth+left-borderweidth)}")
                        mseValue=0 
                        for i in range(0,iconHeight):
                            for j in range(0,iconWidth):
                                #if iconImage[i][j][3]>128:
                                diff = (int(iconImage[i][j][0]) - int(imageSection[i][j][0]))**2
                                diff += (int(iconImage[i][j][1]) - int(imageSection[i][j][1]))**2
                                diff += (int(iconImage[i][j][2]) - int(imageSection[i][j][2]))**2
                                mseValue += diff
                        mseValue=mseValue/(iconHeight*iconWidth)
                        if mseValue<BestMatchSoFar[0]:
                            BestMatchSoFar=[mseValue, (y+top-borderweidth), (x+left-borderweidth), (y+top-borderweidth+iconHeight), (x+iconWidth+left-borderweidth),iconSizeIndex]
        return BestMatchSoFar
                          
                            

                    

        

    def checkIndividualIconPyramid(self,imagePyramid,iconPyramid):
        #in this section we will check one icon(pyramid) on our image(pyramid)
        #starting with the smallest pyramid (image) finding the best fit for our icon based on this 
        #then iterating back to the full scale image only checking a small section of the image to make sure we end up in the right location 
        '''first do a full image search for the icon at the smallest image size'''
        numberOfImagesInPyramid=len(imagePyramid)
        smallestImage=(imagePyramid.pop())[0]#removes it but as its local it will just be reset each time this function is called
        #print(f"image current shape{smallestImage}")
        scalledDownCoordiates=self.checkIndividualImageFirst(smallestImage,(numberOfImagesInPyramid-1),iconPyramid)
        
        count=1
        while (len(imagePyramid)>=1):
            if(scalledDownCoordiates[0]!=99999):
                topStart=scalledDownCoordiates[1]*2
                leftStart=scalledDownCoordiates[2]*2
                indexBestMatchWasFoundAt=scalledDownCoordiates[5] #this is proportional to the image size
                print(f"best coordiantes from previous: {scalledDownCoordiates}")
                smallestImage=(imagePyramid.pop())[0]
                scalledDownCoordiates=self.checkReducedAreaIndividualImage(smallestImage,(numberOfImagesInPyramid-1-count),iconPyramid,top=topStart,left=leftStart,borderweidth=4,indextoCheck=indexBestMatchWasFoundAt)
                count+=1
                print(f"best new coordiantes: {scalledDownCoordiates}")
            else:
                print(f"something went wrong and no better MSE was founnd")
        return scalledDownCoordiates#these should now be scaled up to the correct size 
                

    




def testEnviormentA():
    testIcon = iconsarray[0]
    gp = libarygaussianPyramid(testIcon, levels=5)
    gp.show_pyramid_test()
    plt.tight_layout()
    plt.show()
    i=input("press enter to continue to next test") 
    testIcon = testImagesarray[18]
    gp = libarygaussianPyramid(testIcon, levels=6)
    gp.show_pyramid_test()
    plt.tight_layout()
    plt.show()

def test_checkIndividualImage_function():
    testIcon = iconsarray[0]
    testImage = testImagesarray[18]
    testImage= cv2.pyrDown(testImage)
    testImage= cv2.pyrDown(testImage)
    pyrami_generator = libarygaussianPyramid(testIcon, levels=(6))
    testiconPyramid=pyrami_generator.get_pyramid()
    print(f"testIcon pyramid: {len(testiconPyramid)}")
    ###########################################################################
    #gp = libarygaussianPyramid(testImage, levels=1)
    #gp.show_pyramid_test()
    #plt.tight_layout()
    #plt.show()
    #i=input("press enter to continue to next test") 
    #pyrami_generator.show_pyramid_test()
    #plt.tight_layout()
    #plt.show()
    #############################################################################
    testMatchClass=matchIconToImage(iconsarray,testiconPyramid)
    result=testMatchClass.checkIndividualImageFirst(testImage,2,testiconPyramid)#01-lighthouse,257,4,385,132 -> 128.5,2,192.5,66
    print(result)
    ###################################output given [337.708740234375, 2, 128, 66, 192]  &&&&&&  [306.3955078125, 1, 64, 33, 96]
    '''
    imageSection=testImage[2:66, 128:192]
    imageSection = imageSection[:, :, ::-1]  # BGR -> RGB
    print(imageSection)
    plt.figure(figsize=(4, 4))
    plt.imshow(imageSection)
    plt.axis("off")
    plt.title("Image Section")
    plt.show()
    '''
    

def test_checkIndividualIconPyramid_function():
    testIcon = iconsarray[0]
    testImage = testImagesarray[18]
    pyrami_generator = libarygaussianPyramid(testImage , levels=(3))
    testImagePyramid=pyrami_generator.get_pyramid()
    pyrami_generator = libarygaussianPyramid(testIcon, levels=(7))
    testiconPyramid=pyrami_generator.get_pyramid()
    print(f"testIcon pyramid: {len(testiconPyramid)}")
    ###########################################################################
    #gp = libarygaussianPyramid(testImage, levels=1)
    #gp.show_pyramid_test()
    #plt.tight_layout()
    #plt.show()
    #i=input("press enter to continue to next test") 
    #pyrami_generator.show_pyramid_test()
    #plt.tight_layout()
    #plt.show()
    #############################################################################
    testMatchClass=matchIconToImage(iconsarray,testiconPyramid)
    result=testMatchClass.checkIndividualIconPyramid(testImagePyramid,testiconPyramid)#01-lighthouse,257,4,385,132 -> 128.5,2,192.5,66
    print(result)
    ###################################output given [337.708740234375, 2, 128, 66, 192]  &&&&&&  [306.3955078125, 1, 64, 33, 96]
    '''
    imageSection=testImage[2:66, 128:192]
    imageSection = imageSection[:, :, ::-1]  # BGR -> RGB
    print(imageSection)
    plt.figure(figsize=(4, 4))
    plt.imshow(imageSection)
    plt.axis("off")
    plt.title("Image Section")
    plt.show()
    '''

    
test_checkIndividualIconPyramid_function()