## Dataset of Pushing Wheelchair

### Label Conversion

```python
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
```

### Loss and mAP

#### Chart

<img src=https://github.com/lexra/pushing-wheelchair/assets/33512027/b328b9fa-5810-4a72-86a1-10fc212b2d52 width=640 />

#### mAP@0.50

```bash
 calculation mAP (mean average precision)...
 Detection layer: 121 - type = 28
 Detection layer: 130 - type = 28
1804
 detections_count = 12523, unique_truth_count = 4502
class_id = 0, name = person, ap = 62.63%         (TP = 1441, FP = 1349)
class_id = 1, name = wheelchair, ap = 3.20%      (TP = 15, FP = 101)
class_id = 2, name = push_wheelchair, ap = 96.15%        (TP = 326, FP = 163)
class_id = 3, name = crutches, ap = 93.29%       (TP = 628, FP = 290)
class_id = 4, name = walking_frame, ap = 78.55%          (TP = 710, FP = 314)

 for conf_thresh = 0.25, precision = 0.58, recall = 0.69, F1-score = 0.63
 for conf_thresh = 0.25, TP = 3120, FP = 2217, FN = 1382, average IoU = 46.91 %

 IoU threshold = 50 %, used Area-Under-Curve for each unique Recall
 mean average precision (mAP@0.50) = 0.667631, or 66.76 %
```

### Detector test

```bash
../darknet detector test \
	cfg/yolo-wheelchair.data \
	cfg/yolo-wheelchair.cfg \
	backup/yolo-wheelchair_final.weights pixmaps/push_wheelchair.jpg -ext_output -dont_show
```

```bash
 CUDA-version: 11070 (12020), cuDNN: 8.9.3, CUDNN_HALF=1, GPU count: 1
 CUDNN_HALF=1
 OpenCV version: 4.2.0
 0 : compute_capability = 890, cudnn_half = 1, GPU: NVIDIA GeForce RTX 4070 Laptop GPU
net.optimized_memory = 0
mini_batch = 1, batch = 1, time_steps = 1, train = 0
   layer   filters  size/strd(dil)      input                output
   0 Create CUDA-stream - 0
 Create cudnn-handle 0
conv      8       3 x 3/ 2    160 x 160 x   1 ->   80 x  80 x   8 0.001 BF
   1 conv      8       1 x 1/ 1     80 x  80 x   8 ->   80 x  80 x   8 0.001 BF
   2 conv      8/   8  3 x 3/ 1     80 x  80 x   8 ->   80 x  80 x   8 0.001 BF
   3 conv      4       1 x 1/ 1     80 x  80 x   8 ->   80 x  80 x   4 0.000 BF
   4 conv      8       1 x 1/ 1     80 x  80 x   4 ->   80 x  80 x   8 0.000 BF
   5 conv      8/   8  3 x 3/ 1     80 x  80 x   8 ->   80 x  80 x   8 0.001 BF
   6 conv      4       1 x 1/ 1     80 x  80 x   8 ->   80 x  80 x   4 0.000 BF
   7 dropout    p = 0.150        25600  ->   25600
   8 Shortcut Layer: 3,  wt = 0, wn = 0, outputs:  80 x  80 x   4 0.000 BF
   9 conv     24       1 x 1/ 1     80 x  80 x   4 ->   80 x  80 x  24 0.001 BF
  10 conv     24/  24  3 x 3/ 2     80 x  80 x  24 ->   40 x  40 x  24 0.001 BF
  11 conv      8       1 x 1/ 1     40 x  40 x  24 ->   40 x  40 x   8 0.001 BF
  12 conv     32       1 x 1/ 1     40 x  40 x   8 ->   40 x  40 x  32 0.001 BF
  13 conv     32/  32  3 x 3/ 1     40 x  40 x  32 ->   40 x  40 x  32 0.001 BF
  14 conv      8       1 x 1/ 1     40 x  40 x  32 ->   40 x  40 x   8 0.001 BF
  15 dropout    p = 0.150        12800  ->   12800
  16 Shortcut Layer: 11,  wt = 0, wn = 0, outputs:  40 x  40 x   8 0.000 BF
  17 conv     32       1 x 1/ 1     40 x  40 x   8 ->   40 x  40 x  32 0.001 BF
  18 conv     32/  32  3 x 3/ 1     40 x  40 x  32 ->   40 x  40 x  32 0.001 BF
  19 conv      8       1 x 1/ 1     40 x  40 x  32 ->   40 x  40 x   8 0.001 BF
  20 dropout    p = 0.150        12800  ->   12800
  21 Shortcut Layer: 16,  wt = 0, wn = 0, outputs:  40 x  40 x   8 0.000 BF
  22 conv     32       1 x 1/ 1     40 x  40 x   8 ->   40 x  40 x  32 0.001 BF
  23 conv     32/  32  3 x 3/ 2     40 x  40 x  32 ->   20 x  20 x  32 0.000 BF
  24 conv      8       1 x 1/ 1     20 x  20 x  32 ->   20 x  20 x   8 0.000 BF
  25 conv     48       1 x 1/ 1     20 x  20 x   8 ->   20 x  20 x  48 0.000 BF
  26 conv     48/  48  3 x 3/ 1     20 x  20 x  48 ->   20 x  20 x  48 0.000 BF
  27 conv      8       1 x 1/ 1     20 x  20 x  48 ->   20 x  20 x   8 0.000 BF
  28 dropout    p = 0.150        3200  ->   3200
  29 Shortcut Layer: 24,  wt = 0, wn = 0, outputs:  20 x  20 x   8 0.000 BF
  30 conv     48       1 x 1/ 1     20 x  20 x   8 ->   20 x  20 x  48 0.000 BF
  31 conv     48/  48  3 x 3/ 1     20 x  20 x  48 ->   20 x  20 x  48 0.000 BF
  32 conv      8       1 x 1/ 1     20 x  20 x  48 ->   20 x  20 x   8 0.000 BF
  33 dropout    p = 0.150        3200  ->   3200
  34 Shortcut Layer: 29,  wt = 0, wn = 0, outputs:  20 x  20 x   8 0.000 BF
  35 conv     48       1 x 1/ 1     20 x  20 x   8 ->   20 x  20 x  48 0.000 BF
  36 conv     48/  48  3 x 3/ 1     20 x  20 x  48 ->   20 x  20 x  48 0.000 BF
  37 conv     16       1 x 1/ 1     20 x  20 x  48 ->   20 x  20 x  16 0.001 BF
  38 conv     96       1 x 1/ 1     20 x  20 x  16 ->   20 x  20 x  96 0.001 BF
  39 conv     96/  96  3 x 3/ 1     20 x  20 x  96 ->   20 x  20 x  96 0.001 BF
  40 conv     16       1 x 1/ 1     20 x  20 x  96 ->   20 x  20 x  16 0.001 BF
  41 dropout    p = 0.150        6400  ->   6400
  42 Shortcut Layer: 37,  wt = 0, wn = 0, outputs:  20 x  20 x  16 0.000 BF
  43 conv     96       1 x 1/ 1     20 x  20 x  16 ->   20 x  20 x  96 0.001 BF
  44 conv     96/  96  3 x 3/ 1     20 x  20 x  96 ->   20 x  20 x  96 0.001 BF
  45 conv     16       1 x 1/ 1     20 x  20 x  96 ->   20 x  20 x  16 0.001 BF
  46 dropout    p = 0.150        6400  ->   6400
  47 Shortcut Layer: 42,  wt = 0, wn = 0, outputs:  20 x  20 x  16 0.000 BF
  48 conv     96       1 x 1/ 1     20 x  20 x  16 ->   20 x  20 x  96 0.001 BF
  49 conv     96/  96  3 x 3/ 1     20 x  20 x  96 ->   20 x  20 x  96 0.001 BF
  50 conv     16       1 x 1/ 1     20 x  20 x  96 ->   20 x  20 x  16 0.001 BF
  51 dropout    p = 0.150        6400  ->   6400
  52 Shortcut Layer: 47,  wt = 0, wn = 0, outputs:  20 x  20 x  16 0.000 BF
  53 conv     96       1 x 1/ 1     20 x  20 x  16 ->   20 x  20 x  96 0.001 BF
  54 conv     96/  96  3 x 3/ 1     20 x  20 x  96 ->   20 x  20 x  96 0.001 BF
  55 conv     16       1 x 1/ 1     20 x  20 x  96 ->   20 x  20 x  16 0.001 BF
  56 dropout    p = 0.150        6400  ->   6400
  57 Shortcut Layer: 52,  wt = 0, wn = 0, outputs:  20 x  20 x  16 0.000 BF
  58 conv     96       1 x 1/ 1     20 x  20 x  16 ->   20 x  20 x  96 0.001 BF
  59 conv     96/  96  3 x 3/ 2     20 x  20 x  96 ->   10 x  10 x  96 0.000 BF
  60 conv     24       1 x 1/ 1     10 x  10 x  96 ->   10 x  10 x  24 0.000 BF
  61 conv    136       1 x 1/ 1     10 x  10 x  24 ->   10 x  10 x 136 0.001 BF
  62 conv    136/ 136  3 x 3/ 1     10 x  10 x 136 ->   10 x  10 x 136 0.000 BF
  63 conv     24       1 x 1/ 1     10 x  10 x 136 ->   10 x  10 x  24 0.001 BF
  64 dropout    p = 0.150        2400  ->   2400
  65 Shortcut Layer: 60,  wt = 0, wn = 0, outputs:  10 x  10 x  24 0.000 BF
  66 conv    136       1 x 1/ 1     10 x  10 x  24 ->   10 x  10 x 136 0.001 BF
  67 conv    136/ 136  3 x 3/ 1     10 x  10 x 136 ->   10 x  10 x 136 0.000 BF
  68 conv     24       1 x 1/ 1     10 x  10 x 136 ->   10 x  10 x  24 0.001 BF
  69 dropout    p = 0.150        2400  ->   2400
  70 Shortcut Layer: 65,  wt = 0, wn = 0, outputs:  10 x  10 x  24 0.000 BF
  71 conv    136       1 x 1/ 1     10 x  10 x  24 ->   10 x  10 x 136 0.001 BF
  72 conv    136/ 136  3 x 3/ 1     10 x  10 x 136 ->   10 x  10 x 136 0.000 BF
  73 conv     24       1 x 1/ 1     10 x  10 x 136 ->   10 x  10 x  24 0.001 BF
  74 dropout    p = 0.150        2400  ->   2400
  75 Shortcut Layer: 70,  wt = 0, wn = 0, outputs:  10 x  10 x  24 0.000 BF
  76 conv    136       1 x 1/ 1     10 x  10 x  24 ->   10 x  10 x 136 0.001 BF
  77 conv    136/ 136  3 x 3/ 1     10 x  10 x 136 ->   10 x  10 x 136 0.000 BF
  78 conv     24       1 x 1/ 1     10 x  10 x 136 ->   10 x  10 x  24 0.001 BF
  79 dropout    p = 0.150        2400  ->   2400
  80 Shortcut Layer: 75,  wt = 0, wn = 0, outputs:  10 x  10 x  24 0.000 BF
  81 conv    136       1 x 1/ 1     10 x  10 x  24 ->   10 x  10 x 136 0.001 BF
  82 conv    136/ 136  3 x 3/ 2     10 x  10 x 136 ->    5 x   5 x 136 0.000 BF
  83 conv     48       1 x 1/ 1      5 x   5 x 136 ->    5 x   5 x  48 0.000 BF
  84 conv    224       1 x 1/ 1      5 x   5 x  48 ->    5 x   5 x 224 0.001 BF
  85 conv    224/ 224  3 x 3/ 1      5 x   5 x 224 ->    5 x   5 x 224 0.000 BF
  86 conv     48       1 x 1/ 1      5 x   5 x 224 ->    5 x   5 x  48 0.001 BF
  87 dropout    p = 0.150        1200  ->   1200
  88 Shortcut Layer: 83,  wt = 0, wn = 0, outputs:   5 x   5 x  48 0.000 BF
  89 conv    224       1 x 1/ 1      5 x   5 x  48 ->    5 x   5 x 224 0.001 BF
  90 conv    224/ 224  3 x 3/ 1      5 x   5 x 224 ->    5 x   5 x 224 0.000 BF
  91 conv     48       1 x 1/ 1      5 x   5 x 224 ->    5 x   5 x  48 0.001 BF
  92 dropout    p = 0.150        1200  ->   1200
  93 Shortcut Layer: 88,  wt = 0, wn = 0, outputs:   5 x   5 x  48 0.000 BF
  94 conv    224       1 x 1/ 1      5 x   5 x  48 ->    5 x   5 x 224 0.001 BF
  95 conv    224/ 224  3 x 3/ 1      5 x   5 x 224 ->    5 x   5 x 224 0.000 BF
  96 conv     48       1 x 1/ 1      5 x   5 x 224 ->    5 x   5 x  48 0.001 BF
  97 dropout    p = 0.150        1200  ->   1200
  98 Shortcut Layer: 93,  wt = 0, wn = 0, outputs:   5 x   5 x  48 0.000 BF
  99 conv    224       1 x 1/ 1      5 x   5 x  48 ->    5 x   5 x 224 0.001 BF
 100 conv    224/ 224  3 x 3/ 1      5 x   5 x 224 ->    5 x   5 x 224 0.000 BF
 101 conv     48       1 x 1/ 1      5 x   5 x 224 ->    5 x   5 x  48 0.001 BF
 102 dropout    p = 0.150        1200  ->   1200
 103 Shortcut Layer: 98,  wt = 0, wn = 0, outputs:   5 x   5 x  48 0.000 BF
 104 conv    224       1 x 1/ 1      5 x   5 x  48 ->    5 x   5 x 224 0.001 BF
 105 conv    224/ 224  3 x 3/ 1      5 x   5 x 224 ->    5 x   5 x 224 0.000 BF
 106 conv     48       1 x 1/ 1      5 x   5 x 224 ->    5 x   5 x  48 0.001 BF
 107 dropout    p = 0.150        1200  ->   1200
 108 Shortcut Layer: 103,  wt = 0, wn = 0, outputs:   5 x   5 x  48 0.000 BF
 109 max                3x 3/ 1      5 x   5 x  48 ->    5 x   5 x  48 0.000 BF
 110 route  108                                            ->    5 x   5 x  48
 111 max                5x 5/ 1      5 x   5 x  48 ->    5 x   5 x  48 0.000 BF
 112 route  108                                            ->    5 x   5 x  48
 113 max                9x 9/ 1      5 x   5 x  48 ->    5 x   5 x  48 0.000 BF
 114 route  113 111 109 108                        ->    5 x   5 x 192
 115 conv     96       1 x 1/ 1      5 x   5 x 192 ->    5 x   5 x  96 0.001 BF
 116 conv     96/  96  5 x 5/ 1      5 x   5 x  96 ->    5 x   5 x  96 0.000 BF
 117 conv     96       1 x 1/ 1      5 x   5 x  96 ->    5 x   5 x  96 0.000 BF
 118 conv     96/  96  5 x 5/ 1      5 x   5 x  96 ->    5 x   5 x  96 0.000 BF
 119 conv     96       1 x 1/ 1      5 x   5 x  96 ->    5 x   5 x  96 0.000 BF
 120 conv     30       1 x 1/ 1      5 x   5 x  96 ->    5 x   5 x  30 0.000 BF
 121 yolo
[yolo] params: iou loss: ciou (4), iou_norm: 0.07, obj_norm: 1.00, cls_norm: 1.00, delta_norm: 1.00, scale_x_y: 1.00
nms_kind: greedynms (1), beta = 0.600000
 122 route  115                                            ->    5 x   5 x  96
 123 upsample                 2x     5 x   5 x  96 ->   10 x  10 x  96
 124 route  123 80                                 ->   10 x  10 x 120
 125 conv    120/ 120  5 x 5/ 1     10 x  10 x 120 ->   10 x  10 x 120 0.001 BF
 126 conv    120       1 x 1/ 1     10 x  10 x 120 ->   10 x  10 x 120 0.003 BF
 127 conv    120/ 120  5 x 5/ 1     10 x  10 x 120 ->   10 x  10 x 120 0.001 BF
 128 conv    120       1 x 1/ 1     10 x  10 x 120 ->   10 x  10 x 120 0.003 BF
 129 conv     30       1 x 1/ 1     10 x  10 x 120 ->   10 x  10 x  30 0.001 BF
 130 yolo
[yolo] params: iou loss: ciou (4), iou_norm: 0.07, obj_norm: 1.00, cls_norm: 1.00, delta_norm: 1.00, scale_x_y: 1.00
nms_kind: greedynms (1), beta = 0.600000
Total BFLOPS 0.055
avg_outputs = 15225
 Allocate additional workspace_size = 134.22 MB
Loading weights from backup/yolo-wheelchair_final.weights...
 seen 64, trained: 5120 K-images (80 Kilo-batches_64)
Done! Loaded 131 layers from weights-file
 Detection layer: 121 - type = 28
 Detection layer: 130 - type = 28
pixmaps/push_wheelchair.jpg: Predicted in 5.980000 milli-seconds.
push_wheelchair: 100%   (left_x:  136   top_y:   29   width:  181   height:  388)
wheelchair: 57% (left_x:  142   top_y:  100   width:  167   height:  311)
```

<img src=https://github.com/lexra/pushing-wheelchair/assets/33512027/9b607f43-0fc5-4116-b1ac-813a56e29d41 width=480 />

### Using Darknet

```bash
https://github.com/hank-ai/darknet
```

```bash
https://www.ccoderun.ca/programming/darknet_faq
```

```bash
https://github.com/Tianxiaomo/pytorch-YOLOv4
```


