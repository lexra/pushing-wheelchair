#!/bin/bash -e

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
#unzip -o Images_RGB.zip -d images

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
	cp -Rpf images/train/${P}.png images/test
done

##############################
for J in $(ls images/train | grep txt | awk -F '.txt' '{print $1}'); do echo "$(pwd)/images/train/${J}.png" ; done | tee train.txt
for J in $(ls images/test | grep txt | awk -F '.txt' '{print $1}'); do echo "$(pwd)/images/test/${J}.png" ; done | tee test.txt

##############################
sed "s|/work/Yolo-Fastest/pushing-wheehchair|`pwd`|" -i cfg/${NAME}.data

##############################
[ "$TERM" == "xterm" ] && GPUS="${GPUS} -dont_show"
[ -e ../data/labels/100_0.png ] && ln -sf ../data .
mkdir -p backup

[ -e backup/${NAME}_last.weights ] && WEIGHTS=backup/${NAME}_last.weights
../darknet detector train cfg/${NAME}.data ${CFG} ${WEIGHTS} ${GPUS} -mjpeg_port 8090 -map

##############################
if [ -e ../keras-YOLOv3-model-set/tools/model_converter/fastest_1.1_160/convert.py ]; then
	python3 ../keras-YOLOv3-model-set/tools/model_converter/fastest_1.1_160/convert.py \
                --config_path cfg/${NAME}.cfg \
                --weights_path backup/${NAME}_final.weights \
                --output_path backup/${NAME}.h5
        python3 ../keras-YOLOv3-model-set/tools/model_converter/fastest_1.1_160/post_train_quant_convert_demo.py --keras_model_file backup/${NAME}.h5 --annotation_file train.txt --output_file backup/${NAME}.tflite
	xxd -i backup/${NAME}.tflite > backup/${NAME}-$(date +'%Y%m%d').cc
fi

##############################
#ls -l backup/${NAME}*
echo ""
echo "../darknet detector test cfg/yolo-wheelchair.data cfg/yolo-wheelchair.cfg backup/yolo-wheelchair_final.weights pixmaps/push_wheelchair.jpg -dont_show"
exit 0
