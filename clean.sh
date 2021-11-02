#!/bin/bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
cd $SCRIPTPATH
rm -rf ext/build/
rm -rf ext/torch_extension.cpython-36m-x86_64-linux-gnu.so
rm -rf ext/torch_extension.cpython-37m-x86_64-linux-gnu.so
rm -rf ext/torch_extension.egg-info/
rm -rf od/modeling/head/ThunderNet/_C.cpython-36m-x86_64-linux-gnu.so
rm -rf od/modeling/head/ThunderNet/_C.cpython-37m-x86_64-linux-gnu.so
rm -rf od/modeling/head/ThunderNet/build/
rm -rf od/modeling/head/ThunderNet/thundernet.egg-info/
rm -rf od/modeling/head/centernet/DCNv2/DCNv2.egg-info/
rm -rf od/modeling/head/centernet/DCNv2/_ext.cpython-36m-x86_64-linux-gnu.so
rm -rf od/modeling/head/centernet/DCNv2/_ext.cpython-37m-x86_64-linux-gnu.so
rm -rf od/modeling/head/centernet/DCNv2/build/
rm -rf od/modeling/head/fcos/build/
rm -rf od/modeling/head/fcos/fcos_nms/_C.cpython-36m-x86_64-linux-gnu.so
rm -rf od/modeling/head/fcos/fcos_nms/_C.cpython-37m-x86_64-linux-gnu.so
rm -rf od/modeling/head/fcos/uninet_layers.egg-info/