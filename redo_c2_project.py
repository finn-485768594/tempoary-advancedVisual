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


image_files = sorted(testImagesLocation.rglob("*.png"),key=lambda p: int(p.stem.split("_")[-1]))

for p in image_files:
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    testImagesarray.append(img)


from pathlib import Path
import csv
print(f"test images array has {len(testImagesarray)} images of size {testImagesarray[0].shape}") 

expected_array = []

csvLocation = Path("data_provided_for_task/annotations")

# sort to prevent issues that i noticced with the second half swapping with the first with images
csv_files = sorted(csvLocation.rglob("*.csv"))

for index in range(len(csv_files)):
    # extract image number, splits on file name "_" and then get second half
    image_number = int(csv_files[index].stem.split("_")[-1])

    image_annotations = [image_number]

    with open(csv_files[index], newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            #cant use index(my preffered way of doing things) as reader doesnt have a length and it throws errors
            class_id = int(row["classname"].split("-")[0])
            top=int(row["top"])
            left=int(row["left"])
            bottom=int(row["bottom"])
            right=int(row["right"])
            image_annotations.append([class_id,top,left,bottom,right])
            

    expected_array.append(image_annotations)
print(f"Loaded annotations for {len(expected_array)} images  example(8):{expected_array[18]}")




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
        self.downsample_scale = 0.75
        #self.build_pyramid()
        
        #self.get_pyramid()

    def build_pyramid_image(self):
        image=self.pyramid[0][0]
        this_level = [image]
        '''
        the pyramids and code was originaly built with them halving in size each time
        so:
        [..., X , X/2 , ...]
        but now it the size of the next two levels should be:
        [..., X , 3X/4, x/2,...]
        this means the down sampling factor between each level wont be constant as sometimes it will be 75% and sometimes 66.6%
        but it shouldnt effect the code much, this function is the images so will be the same as before only the icons will use the new pyramid structure
        '''
        for i in range(1, self.levels):
            image = cv2.pyrDown(self.pyramid[i - 1][0])
            #note to self: cv2.pyrDown 
            #Applies a 5×5 Gaussian blur to the image
            #Downsamples it by a factor of 2 in width and height
            this_level = [image]
            self.pyramid.append(this_level)
            #as well as this i want a 
        #print(f"Pyramid now has {len(self.pyramid)} levels")

    def build_pyramid_icon(self):
        image=self.pyramid[0][0]
        this_level = [image]
        '''
        the pyramids and code was originaly built with them halving in size each time
        so:
        [..., X , X/2 , ...]
        but now it the size of the next two levels should be:
        [..., X , 3X/4, x/2,...]
        this means the down sampling factor between each level wont be constant as sometimes it will be 75% and sometimes 66.6%
        but it shouldnt effect the code much, this function is the images so will be the same as before only the icons will use the new pyramid structure
        '''
        for i in range(1, self.levels):
            ''' now add the next level which is 75% of i-1'''
            imageX=self.pyramid[2*(i - 1)][0]
            image75Percent=cv2.GaussianBlur(imageX, self.kernal_size,sigmaX=1, sigmaY=1)
            heightOfPrevious, widthOfPrevious = imageX.shape[:2]
            new_w = int(widthOfPrevious * self.downsample_scale)
            new_h = int(heightOfPrevious * self.downsample_scale)
            image75Percent = cv2.resize(image75Percent,(new_w, new_h),interpolation=cv2.INTER_AREA)
            this_level = [image75Percent]
            self.pyramid.append(this_level)
            ''' now add the next level which is 50% of i-1'''#same as before
            image50Percent = cv2.pyrDown(imageX)
            #note to self: cv2.pyrDown 
            #Applies a 5×5 Gaussian blur to the image
            #Downsamples it by a factor of 2 in width and height
            this_level = [image50Percent]
            self.pyramid.append(this_level)
            ''' now add the next level which is '''
            
        
          
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
    def __init__(self, iconsToConsider, image, levelsToIMAGEpyramid=3, IconLevelsBelowImageSizeToCheck=7,maxError=1000):
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
        self.maxError=maxError
        for icon in iconsToConsider:
            pyrami_generator = libarygaussianPyramid(icon, levels=(levelsToIMAGEpyramid+IconLevelsBelowImageSizeToCheck))
            pyrami_generator.build_pyramid_icon()
            self.iconsPyramids.append(pyrami_generator.get_pyramid())

    def checkIndividualImageFirst(self,imageToCheck,index_of_image_pyramid,iconPyramid):
        icons_to_check=iconPyramid[(index_of_image_pyramid*2):((index_of_image_pyramid*2) + self.IconLevelsBelowImageSizeToCheck)]
        #BestMatchSoFar: [error,top,left,bottom,right]
        BestMatchSoFar = [99999,0 , 0   ,1     , 1   ,0] #just place holders
        #print(f"stop0a {icons_to_check,(index_of_image_pyramid),(index_of_image_pyramid + self.IconLevelsBelowImageSizeToCheck),len(iconPyramid)} ")
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
        #print(f"xxx ____reduced area____ {(index_of_image_pyramid*2),((index_of_image_pyramid*2) + self.IconLevelsBelowImageSizeToCheck)}")
        icons_to_check=iconPyramid[(index_of_image_pyramid*2):((index_of_image_pyramid*2) + self.IconLevelsBelowImageSizeToCheck)]
        #BestMatchSoFar: [error,top,left,bottom,right]
        BestMatchSoFar = [99999,0 , 0   ,1     , 1   ] #just place holders
        #print(f"stop0 ____reduced area____ {len(icons_to_check)} : {len(icons_to_check[0])} : {icons_to_check[0][0].shape} {indextoCheck}")
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
        #print(f"stopZ {scalledDownCoordiates}")
        count=1
        while (len(imagePyramid)>=1):
            if(scalledDownCoordiates[0]!=99999):
                #print(f"stopB {scalledDownCoordiates}")
                topStart=scalledDownCoordiates[1]*2
                leftStart=scalledDownCoordiates[2]*2
                indexBestMatchWasFoundAt=scalledDownCoordiates[5] #this is proportional to the image size
                #print(f"best coordiantes from previous: {scalledDownCoordiates}")
                smallestImage=(imagePyramid.pop())[0]
                #by changing the way the two gaussian pyramids work I have to convert the index of the image pyramid into the corresponding index for the image icon
                correspondingIndex=(numberOfImagesInPyramid-1-count) # the index of icon with the same size as index n can be represented as 2n-1
                scalledDownCoordiates=self.checkReducedAreaIndividualImage(smallestImage,(correspondingIndex),iconPyramid,top=topStart,left=leftStart,borderweidth=4,indextoCheck=indexBestMatchWasFoundAt)
                count+=1
                #print(f"best new coordiantes: {scalledDownCoordiates}")
            #else:
                #print(f"something went wrong and no better MSE was founnd")
        return scalledDownCoordiates#these should now be scaled up to the correct size 
                

    def checkAllIconsAgainstImage(self,iconsList,image):
        iconsList=self.iconsToConsider
        #image=self.image
        best_fit_for_each_image=[]
        
        for iconIndex in range(len(iconsList)):
            pyrami_generator_image = libarygaussianPyramid(image , levels=(self.levelsToIMAGEpyramid))
            pyrami_generator_image.build_pyramid_image()
            imagePyramid=pyrami_generator_image.get_pyramid()
            print(f"currently working on Icon {iconIndex}  ")#the current best fit for each it: {best_fit_for_each_image}")
            pyrami_generator_icon = libarygaussianPyramid(iconsList[iconIndex] , levels=(self.levelsToIMAGEpyramid+self.IconLevelsBelowImageSizeToCheck))
            pyrami_generator_icon.build_pyramid_icon()
            iconPyramid=pyrami_generator_icon.get_pyramid()
            #print(f"favraibles before going in {len(imagePyramid),len(imagePyramid[0]),imagePyramid[0][0].shape} : {len(iconPyramid),len(iconPyramid[0]),iconPyramid[0][0].shape}")
            result=self.checkIndividualIconPyramid(imagePyramid,iconPyramid)
            best_fit_for_each_image.append([iconIndex]+result)
        #we now have the best position for each icon but only some of them will have a good enough value to be found in the image
        for i in range(len(best_fit_for_each_image)):
            if (best_fit_for_each_image[i][1]<=self.maxError):
                self.matches.append(best_fit_for_each_image[i])
        return self.matches


class matchAllImagesAndIcons:
    def __init__(self,iconArray,imagesArray,answersArray,boundaryError=1000):
        self.imagesArray=imagesArray
        self.iconArray=iconArray
        self.answersArray=answersArray
        self.boundaryError=boundaryError

    
    def getIconsToImage(self,image,imagenumber):
        testMatchClass=matchIconToImage(self.iconArray,image,maxError=self.boundaryError)
        result=testMatchClass.checkAllIconsAgainstImage(self.iconArray,image)
        compactedResults=[imagenumber]
        for index in range(len(result)):
            #compactedResults.append([(result[index][0]+1),(result[index][2]),(result[index][3]),(result[index][4]),(result[index][5])]) #what i think makes sense
            '''above is how i would write the top, left, bottom,and right but the examples given do it differently so im converting to there way bellov'''
            compactedResults.append([(result[index][0]+1),(result[index][3]),(result[index][2]),(result[index][5]),(result[index][4])]) #what they want
        return compactedResults

    def getCompleteListForEachImage(self):
        result_per_image=[]
        for index in range(len(self.imagesArray)):
            print(f"Processing image {index+1} of {len(self.imagesArray)}")
            result_per_image.append(self.getIconsToImage(self.imagesArray[index],index+1))
        return result_per_image
    
    def calculateIOU(self,coordinatesA,coordinatesB):
        #first calculate the min bounding box of both the icons 
        boundingTop = max(coordinatesA[0], coordinatesB[0])
        boundingLeft = max(coordinatesA[1], coordinatesB[1])
        boundingBottom = min(coordinatesA[2], coordinatesB[2])
        boundingRight = min(coordinatesA[3], coordinatesB[3])

        if ((boundingTop>=boundingBottom)or(boundingLeft>=boundingRight)):
            return 0.0#makes everything a float
        else:
            boundingArea=(boundingBottom-boundingTop)*(boundingRight*boundingLeft)
            areaA=(coordinatesA[2]-coordinatesA[0])*(coordinatesA[3]*coordinatesA[1])
            areaB=(coordinatesB[2]-coordinatesB[0])*(coordinatesB[3]*coordinatesB[1])
            return(boundingArea/(areaA+areaB-boundingArea))
        
    def compareResultOfImageToExpected(self,compactedResult):
        comparisonImageId=compactedResult[0]
        expectedResult=[]
        for index in range(len(self.answersArray)):
            if (self.answersArray[index][0]==comparisonImageId):
                expectedResult=self.answersArray[index]
        if(expectedResult==[]):
            print(f"error expected result not recognised when searching for with ID {comparisonImageId}")
        else:

            #now we are certain that the expected result and these results are talking about the same image so by removing the identifier I can make each element consistent for easier searching
            expectedFoundImages=expectedResult[1:]
            actualFoundimages=compactedResult[1:]
            #IconID, IOU, expected coordinates, found coordinates
            matchedIcons=[]
            missedIconsInImage=[]
            #missIdentified=[]

            for indexExpected in range(len(expectedFoundImages)):
                searchIconID=expectedFoundImages[indexExpected][0]
                missed=True
                for indexActual in range(len(actualFoundimages)):
                    if (searchIconID==actualFoundimages[indexActual][0]):
                        #in this case correctly identified now we just need to work out IOU

                        coordinatesExpected=expectedFoundImages[indexExpected][1:]
                        coordinatesFound=actualFoundimages[indexActual][1:]
                        solvedIOU=self.calculateIOU(coordinatesExpected,coordinatesFound)
                        matchedIcons.append([searchIconID,solvedIOU,coordinatesExpected,coordinatesFound])
                        missed=False
                if missed:
                    missedIconsInImage.append(expectedFoundImages[indexExpected])
            return matchedIcons,missedIconsInImage
        

class matchVisualiser:
    def __init__(self):
        pass

    def showMatchedIconsOnImage(self, image, compactedResults):
        imageToShow = image.copy()

        for index in range(len(compactedResults)):
            if index != 0:  # skip image identifier row
                left   = compactedResults[index][1]
                top    = compactedResults[index][2]
                right  = compactedResults[index][3]
                bottom = compactedResults[index][4]

                class_id = compactedResults[index][0]  # or classname string

                # draw bounding box
                cv2.rectangle(imageToShow,(left, top),(right, bottom),(0, 255, 0),2)

                # text label
                label = f"ID {class_id}"


                # draw text
                cv2.putText(imageToShow,label,(left + 2, top + 12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 0),1,cv2.LINE_AA)

        imageToShow = cv2.cvtColor(imageToShow, cv2.COLOR_BGRA2RGBA)
        plt.figure(figsize=(8, 8))
        plt.imshow(imageToShow)
        plt.axis("off")
        plt.title("Matched Icons on Image")
        plt.show()
        



    
        
        

def testEnviormentA():
    testIcon = iconsarray[0]
    gp = libarygaussianPyramid(testIcon, levels=5)
    gp.build_pyramid_icon()
    gp.show_pyramid_test()
    plt.tight_layout()
    plt.show()
    pyr=gp.get_pyramid()
    for i in range(len(pyr)):
        print(f"level {i} shape {pyr[i][0].shape}")
    #i=input("press enter to continue to next test") 
    testImg = testImagesarray[7]
    gp = libarygaussianPyramid(testImg, levels=6)
    gp.build_pyramid_image()
    gp.show_pyramid_test()
    plt.tight_layout()
    plt.show()
    pyr=gp.get_pyramid()
    for i in range(len(pyr)):
        print(f"level {i} shape {pyr[i][0].shape}")

def test_checkIndividualImage_function():
    testIcon = iconsarray[0]
    testImage = testImagesarray[7]
    testImage= cv2.pyrDown(testImage)
    testImage= cv2.pyrDown(testImage)
    pyrami_generator = libarygaussianPyramid(testIcon, levels=(6))
    pyrami_generator.build_pyramid_icon()
    testiconPyramid=pyrami_generator.get_pyramid()
    #print(f"testIcon pyramid: {len(testiconPyramid)}")
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
    testImage = testImagesarray[7]
    pyrami_generator = libarygaussianPyramid(testImage , levels=(3))
    pyrami_generator.build_pyramid_image()
    testImagePyramid=pyrami_generator.get_pyramid()
    pyrami_generator = libarygaussianPyramid(testIcon, levels=(7))
    pyrami_generator.build_pyramid_icon()
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


def test_checkAllIconsAgainstImage_function():
    #testIcon = iconsarray[0]
    testImage = testImagesarray[7]
    testMatchClass=matchIconToImage(iconsarray,testImage)
    result=testMatchClass.checkAllIconsAgainstImage(iconsarray,testImage)#01-lighthouse,257,4,385,132 -> 128.5,2,192.5,66
    print(result)
    ###################################output given [337.708740234375, 2, 128, 66, 192]  &&&&&&  [306.3955078125, 1, 64, 33, 96]
    

#[[0, 293.22509765625, 4, 257, 132, 385, 4], [4, 110.629638671875, 109, 12, 301, 204, 3], [8, 588.486328125, 392, 228, 456, 292, 6], [44, 69.986328125, 149, 260, 341, 452, 3]]
#test_checkAllIconsAgainstImage_function()

def test_getIconsToImage_function():
    testClass=matchAllImagesAndIcons(iconsarray,testImagesarray,answersArray=expected_array)
    imageIndex=0
    testImage = testImagesarray[imageIndex]
    results=testClass.getIconsToImage(testImage,imagenumber=(imageIndex+1))
    print(results)#[18[1, 4, 257, 132, 385], [5, 109, 12, 301, 204], [9, 392, 228, 456, 292], [45, 149, 260, 341, 452]]
    #[18, [1, 257, 4, 385, 132], [5, 12, 109, 204, 301], [9, 228, 392, 292, 456], [45, 260, 149, 452, 341]]

def test_compareResultOfImageToExpected():
    testClass=matchAllImagesAndIcons(iconsarray,testImagesarray,answersArray=expected_array)
    testImage = testImagesarray[7]
    testImageResultExample=[8, [1, 257, 4, 385, 132], [5, 12, 109, 204, 301], [9, 228, 392, 292, 456], [45, 260, 149, 452, 341]]#just done to spped up testing!
    matchedIcons,missedIconsInImage=testClass.compareResultOfImageToExpected(testImageResultExample)
    print(f"matchedIcons: {matchedIcons} \n \nmissedIconsInImage: {missedIconsInImage}")
    
def test_showMatchedIconsOnImage_function():
    testVisualiser=matchVisualiser()
    testImage = testImagesarray[0]
    testImageResultExample=[1, [6, 104, 32, 168, 96], [35, 413, 27, 477, 91], [37, 175, 100, 431, 356], [45, 55, 241, 119, 305], [50, 12, 331, 140, 459]]#just done to spped up testing!
    testVisualiser.showMatchedIconsOnImage(testImage,testImageResultExample)

def run_all_images_tests():
    testClass=matchAllImagesAndIcons(iconsarray,testImagesarray,answersArray=expected_array,boundaryError=1000)
    completeResults=testClass.getCompleteListForEachImage()
    print(f"########################################\n\ncomplete results\n\n")
    for index in range(len(completeResults)):
        print(f"Processing image {index+1} of {len(completeResults)}")
        matchedIcons,missedIconsInImage=testClass.compareResultOfImageToExpected(completeResults[index])
        print(f"Image {index+1} \n matchedIcons: {matchedIcons} \n \nmissedIconsInImage: {missedIconsInImage}\n \n rawouput: {completeResults[index]}\n\n///////////////////////////////////////")
    
run_all_images_tests()



'''
currently working on Icon 49  
########################################

complete results


Processing image 1 of 20
Image 1
 matchedIcons: [[37, 1.0, [175, 100, 431, 356], [175, 100, 431, 356]], [6, 1.0, [104, 32, 168, 96], [104, 32, 168, 96]], [45, 1.0, [55, 241, 119, 305], [55, 241, 119, 305]], [35, 1.0, [413, 27, 477, 91], [413, 27, 477, 91]], [50, 1.0, [12, 331, 140, 459], [12, 331, 140, 459]]]

missedIconsInImage: []

 rawouput: [1, [6, 104, 32, 168, 96], [35, 413, 27, 477, 91], [37, 175, 100, 431, 356], [45, 55, 241, 119, 305], [50, 12, 331, 140, 459]]

///////////////////////////////////////
Processing image 2 of 20
Image 2
 matchedIcons: [[27, 1.0, [45, 136, 237, 328], [45, 136, 237, 328]], [3, 1.0, [363, 59, 491, 187], [363, 59, 491, 187]], [50, 1.0, [199, 363, 327, 491], [199, 363, 327, 491]], [20, 1.0, [331, 211, 395, 275], [331, 211, 395, 275]]]

missedIconsInImage: []

 rawouput: [2, [3, 363, 59, 491, 187], [20, 331, 211, 395, 275], [27, 45, 136, 237, 328], [50, 199, 363, 327, 491]]

///////////////////////////////////////
Processing image 3 of 20
Image 3
 matchedIcons: [[5, 1.0, [159, 182, 351, 374], [159, 182, 351, 374]], [31, 1.0, [67, 78, 131, 142], [67, 78, 131, 142]], [50, 1.0, [31, 406, 95, 470], [31, 406, 95, 470]], [10, 1.0, [4, 317, 68, 381], [4, 317, 68, 381]], [19, 1.0, [364, 407, 428, 471], [364, 407, 428, 471]]]

missedIconsInImage: []

 rawouput: [3, [5, 159, 182, 351, 374], [10, 4, 317, 68, 381], [19, 364, 407, 428, 471], [31, 67, 78, 131, 142], [50, 31, 406, 95, 470]]

///////////////////////////////////////
Processing image 4 of 20
Image 4
 matchedIcons: [[1, 1.0, [362, 351, 490, 479], [362, 351, 490, 479]], [3, 1.0, [60, 366, 188, 494], [60, 366, 188, 494]], [21, 1.0, [17, 242, 81, 306], [17, 242, 81, 306]]]

missedIconsInImage: [[8, 164, 19, 484, 339]]

 rawouput: [4, [1, 362, 351, 490, 479], [3, 60, 366, 188, 494], [21, 17, 242, 81, 306]]

///////////////////////////////////////
Processing image 5 of 20
Image 5
 matchedIcons: [[10, 1.0, [198, 314, 390, 506], [198, 314, 390, 506]], [8, 1.0, [53, 14, 245, 206], [53, 14, 245, 206]], [9, 1.0, [248, 190, 312, 254], [248, 190, 312, 254]], [33, 1.0, [69, 383, 133, 447], [69, 383, 133, 447]]]

missedIconsInImage: []

 rawouput: [5, [8, 53, 14, 245, 206], [9, 248, 190, 312, 254], [10, 198, 314, 390, 506], [33, 69, 383, 133, 447]]

///////////////////////////////////////
Processing image 6 of 20
Image 6
 matchedIcons: [[10, 1.0, [28, 42, 220, 234], [28, 42, 220, 234]], [25, 1.0, [253, 290, 381, 418], [253, 290, 381, 418]], [20, 1.0, [56, 358, 184, 486], [56, 358, 184, 486]], [11, 1.0, [92, 263, 156, 327], [92, 263, 156, 327]]]

missedIconsInImage: []

 rawouput: [6, [10, 28, 42, 220, 234], [11, 92, 263, 156, 327], [20, 56, 358, 184, 486], [25, 253, 290, 381, 418]]

///////////////////////////////////////
Processing image 7 of 20
Image 7
 matchedIcons: [[7, 1.0, [142, 182, 206, 246], [142, 182, 206, 246]], [41, 1.0, [259, 254, 451, 446], [259, 254, 451, 446]], [38, 1.0, [289, 58, 417, 186], [289, 58, 417, 186]], [6, 1.0, [28, 1, 156, 129], [28, 1, 156, 129]]]

missedIconsInImage: [[37, 187, 417, 251, 481]]

 rawouput: [7, [6, 28, 1, 156, 129], [7, 142, 182, 206, 246], [38, 289, 58, 417, 186], [41, 259, 254, 451, 446]]

///////////////////////////////////////
Processing image 8 of 20
Image 8
 matchedIcons: [[45, 1.0, [260, 149, 452, 341], [260, 149, 452, 341]], [1, 1.0, [257, 4, 385, 132], [257, 4, 385, 132]], [5, 1.0, [12, 109, 204, 301], [12, 109, 204, 301]], [9, 1.0, [228, 392, 292, 456], [228, 392, 292, 456]]]

missedIconsInImage: []

 rawouput: [8, [1, 257, 4, 385, 132], [5, 12, 109, 204, 301], [9, 228, 392, 292, 456], [45, 260, 149, 452, 341]]

///////////////////////////////////////
Processing image 9 of 20
Image 9
 matchedIcons: [[27, 1.0, [176, 208, 368, 400], [176, 208, 368, 400]], [34, 1.0, [410, 195, 474, 259], [410, 195, 474, 259]], [31, 1.0, [52, 408, 116, 472], [52, 408, 116, 472]], [7, 1.0, [93, 148, 157, 212], [93, 148, 157, 212]]]

missedIconsInImage: []

 rawouput: [9, [7, 93, 148, 157, 212], [27, 176, 208, 368, 400], [31, 52, 408, 116, 472], [34, 410, 195, 474, 259]]

///////////////////////////////////////
Processing image 10 of 20
Image 10
 matchedIcons: [[12, 1.0, [68, 16, 260, 208], [68, 16, 260, 208]], [29, 1.0, [245, 257, 309, 321], [245, 257, 309, 321]], [8, 1.0, [293, 342, 357, 406], [293, 342, 357, 406]], [26, 1.0, [306, 46, 434, 174], [306, 46, 434, 174]]]

missedIconsInImage: []

 rawouput: [10, [8, 293, 342, 357, 406], [12, 68, 16, 260, 208], [26, 306, 46, 434, 174], [29, 245, 257, 309, 321]]

///////////////////////////////////////
Processing image 11 of 20
Image 11
 matchedIcons: [[43, 1.0, [259, 280, 451, 472], [259, 280, 451, 472]], [47, 1.0, [26, 23, 282, 279], [26, 23, 282, 279]], [24, 1.0, [147, 294, 211, 358], [147, 294, 211, 358]], [6, 1.0, [318, 91, 382, 155], [318, 91, 382, 155]]]

missedIconsInImage: [[42, 313, 163, 377, 227]]

 rawouput: [11, [6, 318, 91, 382, 155], [24, 147, 294, 211, 358], [43, 259, 280, 451, 472], [47, 26, 23, 282, 279]]

///////////////////////////////////////
Processing image 12 of 20
Image 12
 matchedIcons: [[4, 1.0, [105, 210, 297, 402], [105, 210, 297, 402]], [7, 1.0, [142, 18, 270, 146], [142, 18, 270, 146]], [28, 1.0, [336, 107, 400, 171], [336, 107, 400, 171]], [32, 1.0, [369, 291, 497, 419], [369, 291, 497, 419]]]

missedIconsInImage: []

 rawouput: [12, [4, 105, 210, 297, 402], [7, 142, 18, 270, 146], [28, 336, 107, 400, 171], [32, 369, 291, 497, 419]]

///////////////////////////////////////
Processing image 13 of 20
Image 13
 matchedIcons: [[27, 1.0, [101, 133, 229, 261], [101, 133, 229, 261]], [24, 1.0, [361, 270, 489, 398], [361, 270, 489, 398]], [43, 1.0, [96, 314, 288, 506], [96, 314, 288, 506]], [36, 1.0, [16, 27, 80, 91], [16, 27, 80, 91]]]

missedIconsInImage: [[42, 392, 7, 456, 71]]

 rawouput: [13, [24, 361, 270, 489, 398], [27, 101, 133, 229, 261], [36, 16, 27, 80, 91], [43, 96, 314, 288, 506]]

///////////////////////////////////////
Processing image 14 of 20
Image 14
 matchedIcons: [[49, 1.0, [133, 157, 261, 285], [133, 157, 261, 285]], [43, 1.0, [320, 186, 448, 314], [320, 186, 448, 314]], [3, 1.0, [62, 10, 190, 138], [62, 10, 190, 138]], [6, 1.0, [382, 375, 510, 503], [382, 375, 510, 503]]]

missedIconsInImage: []

 rawouput: [14, [3, 62, 10, 190, 138], [6, 382, 375, 510, 503], [43, 320, 186, 448, 314], [49, 133, 157, 261, 285]]

///////////////////////////////////////
Processing image 15 of 20
Image 15
 matchedIcons: [[7, 1.0, [208, 17, 336, 145], [208, 17, 336, 145]], [43, 1.0, [183, 347, 311, 475], [183, 347, 311, 475]], [44, 1.0, [97, 431, 161, 495], [97, 431, 161, 495]], [25, 1.0, [370, 265, 498, 393], [370, 265, 498, 393]]]

missedIconsInImage: [[47, 100, 347, 164, 411]]

 rawouput: [15, [7, 208, 17, 336, 145], [25, 370, 265, 498, 393], [43, 183, 347, 311, 475], [44, 97, 431, 161, 495]]

///////////////////////////////////////
Processing image 16 of 20
Image 16
 matchedIcons: [[28, 1.0, [282, 27, 474, 219], [282, 27, 474, 219]], [6, 1.0, [74, 186, 266, 378], [74, 186, 266, 378]]]

missedIconsInImage: [[37, 186, 28, 250, 92], [42, 76, 70, 140, 134]]

 rawouput: [16, [6, 74, 186, 266, 378], [28, 282, 27, 474, 219]]

///////////////////////////////////////
Processing image 17 of 20
Image 17
 matchedIcons: [[34, 1.0, [158, 248, 414, 504], [158, 248, 414, 504]], [21, 1.0, [54, 353, 118, 417], [54, 353, 118, 417]], [15, 1.0, [379, 153, 443, 217], [379, 153, 443, 217]], [33, 1.0, [223, 89, 287, 153], [223, 89, 287, 153]]]

missedIconsInImage: []

 rawouput: [17, [15, 379, 153, 443, 217], [21, 54, 353, 118, 417], [33, 223, 89, 287, 153], [34, 158, 248, 414, 504]]

///////////////////////////////////////
Processing image 18 of 20
Image 18
 matchedIcons: [[28, 1.0, [288, 148, 416, 276], [288, 148, 416, 276]], [20, 1.0, [58, 141, 186, 269], [58, 141, 186, 269]], [36, 1.0, [341, 322, 469, 450], [341, 322, 469, 450]], [35, 1.0, [145, 403, 209, 467], [145, 403, 209, 467]]]

missedIconsInImage: []

 rawouput: [18, [20, 58, 141, 186, 269], [28, 288, 148, 416, 276], [35, 145, 403, 209, 467], [36, 341, 322, 469, 450]]

///////////////////////////////////////
Processing image 19 of 20
Image 19
 matchedIcons: [[6, 1.0, [260, 256, 452, 448], [260, 256, 452, 448]], [9, 1.0, [447, 152, 511, 216], [447, 152, 511, 216]], [12, 1.0, [56, 190, 184, 318], [56, 190, 184, 318]], [44, 1.0, [40, 9, 168, 137], [40, 9, 168, 137]]]

missedIconsInImage: []

 rawouput: [19, [6, 260, 256, 452, 448], [9, 447, 152, 511, 216], [12, 56, 190, 184, 318], [44, 40, 9, 168, 137]]

///////////////////////////////////////
Processing image 20 of 20
Image 20
 matchedIcons: [[27, 1.0, [314, 47, 506, 239], [314, 47, 506, 239]], [3, 1.0, [16, 72, 208, 264], [16, 72, 208, 264]], [23, 1.0, [134, 358, 262, 486], [134, 358, 262, 486]], [46, 1.0, [271, 313, 399, 441], [271, 313, 399, 441]], [43, 1.0, [231, 246, 295, 310], [231, 246, 295, 310]]]

missedIconsInImage: []

 rawouput: [20, [3, 16, 72, 208, 264], [23, 134, 358, 262, 486], [27, 314, 47, 506, 239], [43, 231, 246, 295, 310], [46, 271, 313, 399, 441]]

///////////////////////////////////////
'''