#!/bin/bash

# تفعيل البيئة الافتراضية
source ../../my_env/bin/activate

# تعريف متغيرات Jenkins حتى ما يقتل الخدمة
export BUILD_ID=dontKillMe
export JENKINS_NODE_COOKIE=dontKillMe

# قراءة مسار أفضل نموذج من ملف best_model.txt
path_model=$(cat best_model.txt)

# تشغيل الخدمة على البورت 5003
mlflow models serve -m $path_model -p 5003 --no-conda &
