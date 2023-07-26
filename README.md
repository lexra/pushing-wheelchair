## Dataset of Pushing Wheelchair

### Loss and mAP chart

<img src=https://github.com/lexra/pushing-wheelchair/assets/33512027/b328b9fa-5810-4a72-86a1-10fc212b2d52 width=640 />


### Detector test

```bash
../darknet detector test \
	cfg/yolo-wheelchair.data \
	cfg/yolo-wheelchair.cfg \
	backup/yolo-wheelchair_final.weights pixmaps/push_wheelchair.jpg -dont_show
```

<img src=https://github.com/lexra/pushing-wheelchair/assets/33512027/9b607f43-0fc5-4116-b1ac-813a56e29d41 width=480 />
