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