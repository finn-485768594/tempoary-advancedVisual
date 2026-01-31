# first things first lets get the data we ned for the tasl
import cv2
import csv
from pathlib import Path
import time
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
            #print(f"Resizing from ({widthOfPrevious}, {heightOfPrevious}) to ({new_w}, {new_h})")
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
    def __init__(self, iconsToConsider, image, levelsToIMAGEpyramid=5, IconLevelsBelowImageSizeToCheck=8,maxError=2000):
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
            pyrami_generator = libarygaussianPyramid(icon, levels=(IconLevelsBelowImageSizeToCheck))
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
            #print(f"currently working on Icon {iconIndex}  ")#the current best fit for each it: {best_fit_for_each_image}")
            pyrami_generator_icon = libarygaussianPyramid(iconsList[iconIndex] , levels=(self.IconLevelsBelowImageSizeToCheck))#self.levelsToIMAGEpyramid+self.IconLevelsBelowImageSizeToCheck))
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
    def __init__(self,iconArray,imagesArray,answersArray,boundaryError=1000,levelsToIMAGEpyramid=5, IconLevelsBelowImageSizeToCheck=8):
        self.imagesArray=imagesArray
        self.iconArray=iconArray
        self.answersArray=answersArray
        self.boundaryError=boundaryError
        self.levelsToIMAGEpyramid=levelsToIMAGEpyramid
        self.IconLevelsBelowImageSizeToCheck=IconLevelsBelowImageSizeToCheck

    def getIconsToImage(self,image,imagenumber):
        testMatchClass=matchIconToImage(self.iconArray,image,maxError=self.boundaryError,levelsToIMAGEpyramid=self.levelsToIMAGEpyramid,IconLevelsBelowImageSizeToCheck=self.IconLevelsBelowImageSizeToCheck)
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
            #print(f"Processing image {index+1} of {len(self.imagesArray)}")
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
        



    
        
        
def run_all_images_tests(boundaryError=1000,levelsToIMAGEpyramid=5, IconLevelsBelowImageSizeToCheck=8):
    start_time = time.perf_counter()
    testClass=matchAllImagesAndIcons(iconsarray,testImagesarray,answersArray=expected_array,boundaryError=boundaryError,levelsToIMAGEpyramid=levelsToIMAGEpyramid,IconLevelsBelowImageSizeToCheck=IconLevelsBelowImageSizeToCheck)
    completeResults=testClass.getCompleteListForEachImage()
    print(f"\n########################################\ncomplete results\n")
    percentgeFound=[]
    numOfImages=0
    numberImagesFound=0
    listIOUs=[]
    numberOfFalseposotives=0
    for index in range(len(completeResults)):
        print(f"Processing image {index+1} of {len(completeResults)}")
        matchedIcons,missedIconsInImage=testClass.compareResultOfImageToExpected(completeResults[index])
        print(f"Image {index+1} \n matchedIcons: {matchedIcons}\nmissedIconsInImage: {missedIconsInImage} \n rawouput: {completeResults[index]}\n///////////////////////////////////////")
        #calculate IoU:
        listIOUs=[]
        for matchIndex in range (len(matchedIcons)):
            coordinatesA=matchedIcons[matchIndex][2]
            coordinatesB=matchedIcons[matchIndex][3]
            listIOUs.append(testClass.calculateIOU(coordinatesA,coordinatesB))
        #calculate list of false positives
        for indexOficonguessed in range(1,len(completeResults[index])):
            iconguessed=completeResults[index][indexOficonguessed][0]
            if not any(iconguessed==matchedIcons[i][0] for i in range(len(matchedIcons))):
                numberOfFalseposotives+=1

        percentgeFound.append([1-(len(missedIconsInImage)/((len(matchedIcons)+len(missedIconsInImage)))),(len(missedIconsInImage)),(len(completeResults[index])-1-(len(matchedIcons)))])#percentage found, number of false negatives, number of true positives
        numOfImages+=len(missedIconsInImage)+len(matchedIcons)
        numberImagesFound+=len(matchedIcons)
    end_time = time.perf_counter() # Record end time
    calculatedElapsedTime = end_time - start_time
    print(f"percentage found per image : {percentgeFound}")
    calculatedAverageFalsePosotives=numberOfFalseposotives/len(completeResults)
    print(f"average number of false positives: {calculatedAverageFalsePosotives}")
    #print(f"average percentage found per image: {sum([percentgeFound[i][0] for i in range(len(percentgeFound))])/len(percentgeFound)}")
    calculatedIoU=(sum(listIOUs)/len(listIOUs))
    print(f"average IoU per image: {calculatedIoU}")
    print(f"average total images found: {numberImagesFound/numOfImages}     made of{numberImagesFound}/{numOfImages} true positives")
    calculatedTPR=numberImagesFound/numOfImages
    print(f"total false negatives: {sum([percentgeFound[i][1] for i in range(len(percentgeFound))])}")
    calculatedFNR=sum([percentgeFound[i][1] for i in range(len(percentgeFound))])
    print(f"Elapsed time for all images: {calculatedElapsedTime} seconds")
    calculatedAcc=calculatedTPR/(calculatedTPR+numberOfFalseposotives)
    return [calculatedElapsedTime,calculatedTPR,calculatedFNR,calculatedIoU,calculatedAverageFalsePosotives,calculatedAcc]


def checkHyperparameters():
    boundaryErrors=[500,1000,2000]
    levelsToIMAGEpyramidList=[3,4,5,6,7,8,9,10]
    IconLevelsBelowImageSizeToCheckList=[3,4,5,6,7,8,9,10]
    results=[]
    for boundaryError in boundaryErrors:
        for levelsToIMAGEpyramid in levelsToIMAGEpyramidList:
            for IconLevelsBelowImageSizeToCheck in IconLevelsBelowImageSizeToCheckList:
                try:
                    print(f"Testing boundaryError:{boundaryError} levelsToIMAGEpyramid:{levelsToIMAGEpyramid} IconLevelsBelowImageSizeToCheck:{IconLevelsBelowImageSizeToCheck}")
                    result=run_all_images_tests(boundaryError=boundaryError,levelsToIMAGEpyramid=levelsToIMAGEpyramid,IconLevelsBelowImageSizeToCheck=IconLevelsBelowImageSizeToCheck)
                    print(result)
                    results.append([boundaryError,levelsToIMAGEpyramid,IconLevelsBelowImageSizeToCheck]+result)
                except Exception as e:
                    print(f"Error encountered for boundaryError:{boundaryError} levelsToIMAGEpyramid:{levelsToIMAGEpyramid} IconLevelsBelowImageSizeToCheck:{IconLevelsBelowImageSizeToCheck} : {e}")

    print(f"Hyperparameter testing complete results:\n boundaryError,levelsToIMAGEpyramid,IconLevelsBelowImageSizeToCheck,ElapsedTime,TPR,FNR,IoU,AvgFalsePosotives,Accuracy\n {results}")


checkHyperparameters()

'''
Processing image 19 of 20
Image 19
 matchedIcons: [[6, 1.0, [260, 256, 452, 448], [260, 256, 452, 448]], [12, 1.0, [56, 190, 184, 318], [56, 190, 184, 318]]]

missedIconsInImage: [[9, 447, 152, 511, 216], [44, 40, 9, 168, 137]]

 rawouput: [19, [6, 260, 256, 452, 448], [8, 248, 351, 312, 415], [12, 56, 190, 184, 318], [28, 321, 268, 385, 332]]
 '''
#missed icon size 64, 128
#found icon size 192, 128

