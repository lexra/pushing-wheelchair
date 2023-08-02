#!/bin/bash -e

##############################
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m'

NAME=yolo-wheelchair
CFG="cfg/${NAME}.cfg"
GPUS="-gpus 0"
WEIGHTS=""

##############################
if [ ! -e Images_RGB.zip ]; then
	curl -L http://mobility-aids.informatik.uni-freiburg.de/dataset/Images_RGB.zip -o Images_RGB.zip
	curl -L http://mobility-aids.informatik.uni-freiburg.de/dataset/Annotations_RGB.zip -o Annotations_RGB.zip
	curl -L http://mobility-aids.informatik.uni-freiburg.de/dataset/Annotations_RGB_TestSet2.zip -o Annotations_RGB_TestSet2.zip
fi

##############################
sudo rm -rf images labels
mkdir -p images
mkdir -p labels
unzip -o Annotations_RGB.zip -d labels | pv -l >/dev/null
unzip -o Annotations_RGB_TestSet2.zip -d labels | pv -l >/dev/null
unzip -o Images_RGB.zip -d images | pv -l >/dev/null

sudo chmod -x images/Images_RGB
find images/Images_RGB -name "*.png" | xargs sudo chmod 666
mv -fv images/Images_RGB images/train
chmod 775 images/train
mv -fv labels/Annotations_RGB labels/train
mv -fv labels/Annotations_RGB_TestSet2 labels/test

##############################
python3 yaml2yolo.py
cp -Rpf labels/train/*.txt images/train

##############################
mkdir -p images/test
cp -Rpf labels/test/*.txt images/test
for P in `ls labels/test | awk -F '.txt' '{print $1}'`; do
	#cp -Rpf images/train/${P}.png images/test
	mv -f images/train/${P}.png images/test
done

##############################
W=`cat cfg/yolo-wheelchair.cfg | grep width | awk -F '=' '{print $2}'`
H=`cat cfg/yolo-wheelchair.cfg | grep height | awk -F '=' '{print $2}'`

if [ 1 -eq `cat cfg/${NAME}.cfg | grep channels | awk -F '=' '{print $2}'` ]; then
	pushd images/train
	for N in `ls *.txt | awk -F '.txt' '{print $1}'`; do convert ${N}.png -colorspace gray tmp.png && mv -fv tmp.png ${N}.png ; done
	popd
	pushd images/test
	for N in `ls *.txt | awk -F '.txt' '{print $1}'`; do convert ${N}.png -colorspace gray tmp.png && mv -fv tmp.png ${N}.png ; done
	popd
fi

##############################
for J in $(ls images/train | grep txt | awk -F '.txt' '{print $1}'); do echo "$(pwd)/images/train/${J}.png" ; done | tee train.txt
for J in $(ls images/test | grep txt | awk -F '.txt' '{print $1}'); do echo "$(pwd)/images/test/${J}.png" ; done | tee test.txt

##############################
sed "s|/work/himax/Yolo-Fastest/pushing-wheelchair|`pwd`|" -i cfg/${NAME}.data

##############################
[ "$TERM" == "xterm" ] && GPUS="${GPUS} -dont_show"
[ -e ../data/labels/100_0.png ] && ln -sf ../data .
mkdir -p backup

[ -e backup/${NAME}_last.weights ] && WEIGHTS=backup/${NAME}_last.weights
../darknet detector train cfg/${NAME}.data ${CFG} ${WEIGHTS} ${GPUS} -mjpeg_port 8090 -map

##############################
if [ -e ../keras-YOLOv3-model-set/tools/model_converter/fastest_1.1_160/convert.py ]; then
	git -C ../keras-YOLOv3-model-set checkout tools/model_converter/fastest_1.1_160/post_train_quant_convert_demo.py
	sed "s|model_input_shape = \"160x160\"|model_input_shape = \"${W}x${H}\"|" -i ../keras-YOLOv3-model-set/tools/model_converter/fastest_1.1_160/post_train_quant_convert_demo.py

	python3 ../keras-YOLOv3-model-set/tools/model_converter/fastest_1.1_160/convert.py \
		--config_path cfg/${NAME}.cfg \
		--weights_path backup/${NAME}_best.weights \
		--output_path backup/${NAME}.h5

	python3 ../keras-YOLOv3-model-set/tools/model_converter/fastest_1.1_160/post_train_quant_convert_demo.py --keras_model_file backup/${NAME}.h5 --annotation_file train.txt --output_file backup/${NAME}.tflite
	xxd -i backup/${NAME}.tflite > backup/${NAME}-$(date +'%Y%m%d').cc
fi

##############################
../darknet detector map cfg/${NAME}.data cfg/${NAME}.cfg backup/${NAME}_final.weights -iou_thresh 0.5 | grep -v '\-points'

##############################
# g++ tests/opencv-camera/opencv-camera.cpp -o tests/opencv-camera/opencv-camera `pkg-config --cflags --libs opencv4`
echo ""
echo -e "${YELLOW} Detector Test: ${NC}"
echo -e "${YELLOW} ../darknet detector test cfg/yolo-wheelchair.data cfg/yolo-wheelchair.cfg backup/yolo-wheelchair_final.weights pixmaps/push_wheelchair.jpg -ext_output -dont_show ${NC}"
echo ""
exit 0
