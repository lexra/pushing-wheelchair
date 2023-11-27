#!/bin/bash -e

sudo echo ''

##############################
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m'

##############################
NAME=yolo-fastest
CFG="cfg/${NAME}.cfg"
GPUS="-gpus 0"
WEIGHTS=""

WIDTH=$(cat ${CFG} | grep "^width" | awk -F '=' '{print $2}')
HEIGHT=$(cat ${CFG} | grep "^height" | awk -F '=' '{print $2}')

##############################
if [ ! -e downloads/Images_RGB.zip ]; then
	curl -L http://mobility-aids.informatik.uni-freiburg.de/dataset/Images_RGB.zip -o downloads/Images_RGB.zip
	curl -L http://mobility-aids.informatik.uni-freiburg.de/dataset/Annotations_RGB.zip -o downloads/Annotations_RGB.zip
	curl -L http://mobility-aids.informatik.uni-freiburg.de/dataset/Annotations_RGB_TestSet2.zip -o downloads/Annotations_RGB_TestSet2.zip
fi

##############################
sudo rm -rf images labels train.txt test.txt anchors.txt counters_per_class.txt data
mkdir -p images
mkdir -p labels
unzip -o downloads/Annotations_RGB.zip -d labels | pv -l > /dev/null
unzip -o downloads/Annotations_RGB_TestSet2.zip -d labels | pv -l > /dev/null
unzip -o downloads/Images_RGB.zip -d images | pv -l > /dev/null

chmod -x images/Images_RGB
find images/Images_RGB -name "*.png" | xargs sudo chmod 666
mv -fv images/Images_RGB images/train
chmod 775 images/train
mv -fv labels/Annotations_RGB labels/train
mv -fv labels/Annotations_RGB_TestSet2 labels/test

##############################
python3 yaml2yolo.py
mkdir -p images/train && cp -Rpf labels/train/*.txt images/train

##############################
mkdir -p images/test && cp -Rpf labels/test/*.txt images/test
for P in `ls labels/test | awk -F '.txt' '{print $1}'`; do mv -f images/train/${P}.png images/test ; done

##############################
for J in $(ls images/train | grep txt | awk -F '.txt' '{print $1}'); do echo "$(pwd)/images/train/${J}.png" ; done | tee train.txt
for J in $(ls images/test | grep txt | awk -F '.txt' '{print $1}'); do echo "$(pwd)/images/test/${J}.png" ; done | tee test.txt

##############################
sed "s|/work/Yolo-Fastest/pushing-wheelchair|`pwd`|" -i cfg/${NAME}.data

echo '' && echo -e "${YELLOW} echo '' | ../darknet detector calc_anchors cfg/${NAME}.data -num_of_clusters 6 -width ${WIDTH} -height ${HEIGHT} -dont_show ${NC}"
echo '' | ../darknet detector calc_anchors cfg/${NAME}.data -num_of_clusters 6 -width ${WIDTH} -height ${HEIGHT} -dont_show
[ 0 -ne $(cat ${CFG} | grep "^anchors" | awk -F '=' '{print $2}' | wc -l) ] && cat ${CFG} | grep "^anchors" | awk -F '=' '{print $2}' | tail -1 > cfg/${NAME}.anchors

##############################
[ "$TERM" == "xterm" ] && GPUS="${GPUS} -dont_show"
[ -e ../data/labels/100_0.png ] && ln -sf ../data .
mkdir -p backup

[ -e backup/${NAME}_last.weights ] && WEIGHTS=backup/${NAME}_last.weights
echo ""
echo -e "${YELLOW} ../darknet detector train cfg/${NAME}.data ${CFG} ${WEIGHTS} ${GPUS} -mjpeg_port 8090 -map ${NC}"
../darknet detector train cfg/${NAME}.data ${CFG} ${WEIGHTS} ${GPUS} -mjpeg_port 8090 -map

##############################
if [ -e ../keras-YOLOv3-model-set/tools/model_converter/fastest_1.1_160/convert.py ]; then
	git -C ../keras-YOLOv3-model-set checkout tools/model_converter/fastest_1.1_160/post_train_quant_convert_demo.py
	sed "s|model_input_shape = \"160x160\"|model_input_shape = \"${WIDTH}x${HEIGHT}\"|" -i ../keras-YOLOv3-model-set/tools/model_converter/fastest_1.1_160/post_train_quant_convert_demo.py
	echo ""
	echo -e "${YELLOW}python3 ../keras-YOLOv3-model-set/tools/model_converter/fastest_1.1_160/convert.py --config_path cfg/${NAME}.cfg --weights_path backup/${NAME}_final.weights --output_path backup/${NAME}.h5 ${NC}"
	rm -rf backup/${NAME}.h5
	python3 ../keras-YOLOv3-model-set/tools/model_converter/fastest_1.1_160/convert.py \
		--config_path cfg/${NAME}.cfg \
		--weights_path backup/${NAME}_final.weights \
		--output_path backup/${NAME}.h5 || true
	echo ""
	echo -e "${YELLOW}python3 ../keras-YOLOv3-model-set/tools/model_converter/fastest_1.1_160/post_train_quant_convert_demo.py --keras_model_file backup/${NAME}.h5 --annotation_file train.txt --output_file backup/${NAME}.tflite${NC}"
        [ -e backup/${NAME}.h5 ] && \
		python3 ../keras-YOLOv3-model-set/tools/model_converter/fastest_1.1_160/post_train_quant_convert_demo.py \
		--keras_model_file backup/${NAME}.h5 \
		--annotation_file train.txt \
		--output_file backup/${NAME}.tflite
        echo ""
	echo -e "${YELLOW} xxd -i backup/${NAME}.tflite > backup/${NAME}.cc ${NC}"
	[ -e backup/${NAME}.tflite ] && \
		xxd -i backup/${NAME}.tflite > backup/${NAME}.cc
fi

##############################
echo ""
echo -e "${YELLOW} ../darknet detector map cfg/${NAME}.data cfg/${NAME}.cfg backup/${NAME}_final.weights -iou_thresh 0.5 ${NC}"
../darknet detector map cfg/${NAME}.data cfg/${NAME}.cfg backup/${NAME}_final.weights -iou_thresh 0.5 | grep -v '\-points'

##############################
# g++ tests/opencv-camera/opencv-camera.cpp -o tests/opencv-camera/opencv-camera `pkg-config --cflags --libs opencv4`
echo ""
echo -e "${YELLOW} Detector Test: ${NC}"
echo -e "${YELLOW} ../darknet detector test cfg/${NAME}.data cfg/${NAME}.cfg backup/${NAME}_final.weights pixmaps/push_wheelchair.jpg -ext_output -dont_show ${NC}"
echo ""
exit 0
