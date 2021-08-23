from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region
from msrest.authentication import ApiKeyCredentials
import os, time, uuid


# class used to convert a bounding box in absolute coords to a bounding box in normalized coords
class NormedBox:
    def __init__(self, img_width, img_height, absolute_box):
    # from https://docs.microsoft.com/en-us/azure/cognitive-services/custom-vision-service/quickstarts/object-detection?tabs=visual-studio&pivots=programming-language-python
        # When you tag images in object detection projects, you need to specify the region
        # of each tagged object using normalized coordinates. The following code associates
        # each of the sample images with its tagged region. The regions specify the bounding
        # box in normalized coordinates,
        # and the coordinates are given in the order: left, top, width, height.
        self.left = absolute_box['xmin'] / img_width
        self.top = absolute_box['ymin'] / img_height
        self.width = (absolute_box['xmax'] - absolute_box['xmin']) / img_width
        self.height = (absolute_box['ymax'] - absolute_box['ymin']) / img_height
        self.colorname = absolute_box['colorname']


# class used to create annotations in the format required by Azure computer vision system
class NormAnnotationSet:
    # outputs a NormAnnotationSet object for the input image
    def __init__(self, path, image_dims, bounding_box_list):
        self.path = path
        self.image_dims = image_dims
        self.bounding_box_list = bounding_box_list
        # number of columns
        self.img_width = image_dims[1]
        # number of rows
        self.img_height = image_dims[0]

        self.normed_boxes_list = []
        for box in bounding_box_list:
            curr_normed_box = NormedBox(self.img_width, self.img_height, box)
            self.normed_boxes_list.append(curr_normed_box)

    def pretty_print(self):
        print("Path:", self.path)
        for i, box in enumerate(self.normed_boxes_list):
            print("Normalized coords for box", i, " of this image")
            print("Left:", box.left)
            print("Top:", box.top)
            print("Width:", box.width)
            print("Height:", box.height)
            print("\n")

# Replace with valid values
ENDPOINT = "https://birds-detector.cognitiveservices.azure.com/"
training_key = "9bdda448fc9c4253b1a5b1d03843601f"
prediction_key = "63f7512569ab4c1c90a9574a0d44578f"
prediction_resource_id = "ee5f2585-3313-4ef9-a32b-5ea91dc1e116"

credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(ENDPOINT, credentials)
prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)

publish_iteration_name = "detectModel"

# Find the object detection domain
obj_detection_domain = next(domain for domain in trainer.get_domains() if domain.type == "ObjectDetection" and domain.name == "General")

# Create a new project
print ("Getting project...")
# Use uuid to avoid project name collisions.
project = trainer.get_project("72ca79b9-9757-4d09-a6ab-0364dc6a9847")

# Get tag for existing project
# print(trainer.get_tags(project.id)[0].id)
# fork_tag = trainer.get_tag(project.id, "6881e85b-0199-4aed-9890-39f02f1ef9a8")
yellow_tag = next(filter(lambda t: t.name == "yellow", trainer.get_tags(project.id)), None)
cyan_tag = next(filter(lambda t: t.name == "cyan", trainer.get_tags(project.id)), None)
pink_tag = next(filter(lambda t: t.name == "pink", trainer.get_tags(project.id)), None)
blue_tag = next(filter(lambda t: t.name == "blue", trainer.get_tags(project.id)), None)
plum_violet_tag = next(filter(lambda t: t.name == "plum-violet", trainer.get_tags(project.id)), None)

color_dic = {
    'yellow': yellow_tag,
    'cyan': cyan_tag,
    'pink': pink_tag,
    'blue': blue_tag,
    'plum-violet': plum_violet_tag
}

fork_image_regions = [NormAnnotationSet("fork_1", [100, 100], [{"xmax" : 50, "xmin": 1, "ymax": 50, "ymin": 1, "colorname": "yellow"}]),
                      NormAnnotationSet("fork_2", [100, 100], [{"xmax" : 50, "xmin": 1, "ymax": 50, "ymin": 1, "colorname": "cyan"}]),
                      NormAnnotationSet("fork_3", [100, 100], [{"xmax" : 50, "xmin": 1, "ymax": 50, "ymin": 1, "colorname": "pink"}]),
                      NormAnnotationSet("fork_4", [100, 100], [{"xmax" : 50, "xmin": 1, "ymax": 50, "ymin": 1, "colorname": "blue"}])
                      ]

# base_image_location = os.path.join(os.getcwd(), "Images")

base_image_location = os.path.join("/Users/cindidong/Downloads/cognitive-services-sample-data-files/CustomVision/ObjectDetection/Images", "fork")

print(base_image_location)

# Go through the data table above and create the images
print("Adding images...")

# format annotations in the way required by
# Azure custom vision object detection
# https://docs.microsoft.com/en-us/azure/cognitive-services/custom-vision-service/quickstarts/object-detection?tabs=visual-studio&pivots=programming-language-python
tagged_images_with_regions = []
for annot in fork_image_regions:
    regions = []
    path = os.path.join(base_image_location, annot.path)
    for box in annot.normed_boxes_list:
        tag_name = color_dic[box.colorname]
        regions.append(Region(tag_id=tag_name.id, left=box.left, top=box.top, width=box.width, height=box.height))
    with open(os.path.join(base_image_location, path + ".jpg"), mode="rb") as image_contents:
        tagged_images_with_regions.append(
            ImageFileCreateEntry(name=os.path.basename(path[:-4]), contents=image_contents.read(), regions=regions))

upload_result = trainer.create_images_from_files(project.id, ImageFileCreateBatch(images=tagged_images_with_regions))
if not upload_result.is_batch_successful:
    print("Image batch upload failed.")
    for image in upload_result.images:
        print("Image status: ", image.status)
    exit(-1)
