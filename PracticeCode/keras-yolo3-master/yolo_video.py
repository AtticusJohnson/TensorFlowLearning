import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image

'''
--image 进入图片检测模式

--model 指定权重文件的位置，默认是 model_data/yolo.h5

--anchors 指定anchors文件位置，默认是 model_data/yolo_anchors.txt

--classes_path 指定类别文件位置, 默认是 model_data/coco_classes.txt

'''


def detect_img(yolo):
    '''
    :param yolo:
    :return:
    function: 打开并检测图片，展示检测后的图片
    '''
    while True:  # 循环执行
        img = input('Input image filename:')  # 在cmd中提示用户输入图片路径
        try:
            image = Image.open(img)  # 尝试打开图片
        except:
            print('Open Error! Try again!')  # 如果不行，抛出“打开图像错误”
            continue
        else:
            r_image = yolo.detect_image(image)  # 如果是图片，执行yolo检测算法
            r_image.show()  # 展示检测后的图片
    yolo.close_session()  # 关闭yolo会话


FLAGS = None  # 标记为None


if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    # argparse.ArgumentParser 是命令行解析函数
    # argument_default定义解析函数的默认值：全局禁止parse_args()调用时进行属性创建

    '''
    Command line options
    '''

    # 向命令行函数中添加参数
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str, required=False, default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()  # 解析命令行添加参数

    if FLAGS.image:  # 检测图片：cmd中输入-image 则为True
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:  # cmd中输入-input "路径",则为True
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))

    elif "input" in FLAGS:  # 检测视频：cmd中输入-input "路径",则为True
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)

    else:
        print("Must specify at least video_input_path.  See usage with --help.")
