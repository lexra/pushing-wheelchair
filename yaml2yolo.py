import os
from glob import glob
import yaml
from tqdm import tqdm

labels = ['person', 'wheelchair', 'push_wheelchair', 'crutches', 'walking_frame']

yaml_list = glob('labels/train/*.yml') + glob('labels/test/*.yml')

for filepath in tqdm(yaml_list):
    label_list = []
    with open(filepath) as f:
        data = yaml.full_load(f)
        annotation = data['annotation']
        if 'object' in annotation:
            object_list= annotation['object']
            area_width = int(annotation['size']['width'])
            area_height= int(annotation['size']['height'])
            for obj in object_list:
                label = labels.index(obj['name'])
                bndbox = obj['bndbox'] 
                min_x = int(bndbox['xmin'])
                max_x = int(bndbox['xmax'])
                min_y = int(bndbox['ymin'])
                max_y = int(bndbox['ymax'])

                center_x = (max_x + min_x) // 2
                center_y = (max_y + min_y) // 2
                width = max_x - min_x
                height= max_y - min_y

                label_list.append([
                    label,
                    center_x / area_width,
                    center_y / area_height,
                    width / area_width,
                    height/ area_height
                ])

    savepath = filepath.replace('.yml', '.txt')
    with open(savepath, 'w') as f:
        start_new_line = False
        for label_line in label_list:
            if start_new_line:
                f.write("\n")
            else:
                start_new_line = True
            label, x_center, y_center, width, height = label_line
            f.write(f"{label} {x_center} {y_center} {width} {height}")

    os.remove(filepath)
