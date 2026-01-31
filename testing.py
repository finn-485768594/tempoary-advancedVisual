from pathlib import Path
import csv

annotations_array = []

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
            

    annotations_array.append(image_annotations)

print(f"Loaded annotations for {len(annotations_array)} images")
print("First image annotations:")
print(annotations_array[18])



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
    testImage= cv2.pyrDown(testImage)
    testImage= cv2.pyrDown(testImage)
    #testImage= cv2.pyrDown(testImage)
    #5 8 seems to be the best soloution 69 i spossible but we start loosing stuff
    pyrami_generator = libarygaussianPyramid(testIcon, levels=(8))
    pyrami_generator.build_pyramid_icon()
    testiconPyramid=pyrami_generator.get_pyramid()
    for i in range(len(testiconPyramid)):
        print(f"level {i} shape {testiconPyramid[i][0].shape}")
    print(f"testIcon pyramid: {len(testiconPyramid)}")
    
    #############################################################################
    testMatchClass=matchIconToImage(iconsarray,testiconPyramid)
    result=testMatchClass.checkIndividualImageFirst(testImage,3,testiconPyramid)#01-lighthouse,257,4,385,132 -> 128.5,2,192.5,66
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
    #test_checkIndividualImage_function()
    testIcon = iconsarray[43]
    testImage = testImagesarray[18]
    pyrami_generator = libarygaussianPyramid(testImage , levels=(5))
    pyrami_generator.build_pyramid_image()
    testImagePyramid=pyrami_generator.get_pyramid()
    pyrami_generatorIcon = libarygaussianPyramid(testIcon, levels=(8))
    pyrami_generatorIcon.build_pyramid_icon()
    testiconPyramid=pyrami_generatorIcon.get_pyramid()
    print(f"testIcon pyramid: {len(testiconPyramid)}")

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
    imageIndex=18
    testImage = testImagesarray[imageIndex]
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
    ###############[18,[1, 257, 4, 385, 132],[5, 12, 109, 204, 301], [9, 228, 392, 292, 456], [45, 260, 149, 452, 341]]

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
    start_time = time.perf_counter()
    testClass=matchAllImagesAndIcons(iconsarray,testImagesarray,answersArray=expected_array,boundaryError=1000)
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

def run_single_image_test():
    testClass=matchIconToImage(iconsarray,testImagesarray[18],maxError=1000)
    result=testClass.checkAllIconsAgainstImage(iconsarray,testImagesarray[18])#01-lighthouse,257,4,385,132 -> 128.5,2,192.5,66
    print(result)




run_all_images_tests()

'''
Processing image 19 of 20
Image 19
 matchedIcons: [[6, 1.0, [260, 256, 452, 448], [260, 256, 452, 448]], [12, 1.0, [56, 190, 184, 318], [56, 190, 184, 318]]]

missedIconsInImage: [[9, 447, 152, 511, 216], [44, 40, 9, 168, 137]]

 rawouput: [19, [6, 260, 256, 452, 448], [8, 248, 351, 312, 415], [12, 56, 190, 184, 318], [28, 321, 268, 385, 332]]
 '''
#missed icon size 64, 128
#found icon size 192, 128