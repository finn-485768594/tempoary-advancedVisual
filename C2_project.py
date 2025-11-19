# first things first lets get the data we ned for the tasl
import cv2
from pathlib import Path

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
            self.pyramid.append(image)
    
    def redefine_pyramid(self, image):
        self.pyramid = [image]
        self.build_pyramid()

    def get_level(self, level):
        if level < 0 or level >= self.levels:
            raise ValueError("Level out of bounds")
        return self.pyramid[level]
    
    def show_pyramid(self):
        for i, level in enumerate(self.pyramid):
            cv2.imshow(f'Level {i}', level)
            cv2.waitKey(0)
        cv2.destroyAllWindows()

class customGaussianPyramid(libarygaussianPyramid):
    '''the custom gaussian pyramid will use all the same methods and features as the libary version with the difference that the gaussian filtering will be done without the libary so that function alone needs to be rewritten
    '''
    def __init__(self, image, levels):
        super().__init__(image, levels)





def testEnviorment():
    testIcon = iconsarray[0]
    gp = libarygaussianPyramid(testIcon, 5)
    gp.show_pyramid()

testEnviorment()