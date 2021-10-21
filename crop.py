"""
在华为Ascend设备与传统X86设备上的图像应用开发存在明显模式差异。主要体现在：
 - 设备初始化与去初始化
 - 内存申请与释放
 - 图像格式
 - 各种操作限制

以下内容以图像裁切为例，展示了在Ascend设备上完整的图像处理过程。包括四个部分：
 1. 设备与环境初始化
 2. 读取图像文件
 3. 裁切图像
 4. 收尾，释放资源

希望通过这个最小代码例子，你能建立在Ascend设备上图像开发的直观感受。
让我们开始吧！
"""
import acl
import cv2
import numpy as np

import atlas.common.atlas_utils.constants as constants
import atlas.common.atlas_utils.utils as utils


def setup_desc(desc, width, height, w_align=16, h_align=2):
    """Setup the image description object.

    The image should be in YUV420SP format.

    Args:
        desc: the image description.
        width: the width of the image.
        height: the height of the image.
        w_align: the width align factor.
        h_align: the height align factor.

    Returns:
        ptr_on_device: the pointer of the image on the device.
    """
    # Calculate buffer size.
    width_stride = ((width + w_align - 1) // w_align) * w_align
    height_stride = ((height + h_align - 1) // h_align) * h_align
    buf_size = (width_stride * height_stride * 3) // 2

    # Allocate memory
    ptr_on_device, ret = acl.media.dvpp_malloc(buf_size)
    utils.check_ret("acl.media.dvpp_malloc", ret)

    # Setup the image.
    acl.media.dvpp_set_pic_desc_data(desc, ptr_on_device)
    acl.media.dvpp_set_pic_desc_size(desc, buf_size)
    acl.media.dvpp_set_pic_desc_format(
        desc, constants.PIXEL_FORMAT_YUV_SEMIPLANAR_420)
    acl.media.dvpp_set_pic_desc_width(desc, width)
    acl.media.dvpp_set_pic_desc_height(desc, height)
    acl.media.dvpp_set_pic_desc_width_stride(desc, width_stride)
    acl.media.dvpp_set_pic_desc_height_stride(desc, height_stride)

    return ptr_on_device


def main():
    # 第一步：初始化

    # 初始化。Ascend设备上运行至少需要初始化4个对象：ACL、Device、Context和Stream。这个过
    # 程对于Python开发者可能比较陌生。好在这个过程是一次性的。

    # ACL初始化。ACL是操作Ascend设备的Python库，有点像OpenCV。
    ret = acl.init()
    assert ret == 0, "初始化ACL失败。"

    # Device初始化。一张Atlas加速卡上可能有多个Ascend芯片可以使用。在这里指定你要选用的芯片
    # 序号。
    device_id = 0
    ret = acl.rt.set_device(device_id)
    assert ret == 0, "初始化Device失败。"

    # Context初始化。Context可以看做是程序运行的小环境，它起着隔离作用。
    context, ret = acl.rt.create_context(device_id)
    assert ret == 0, "初始化Context失败。"

    # Stream初始化。Stream多用作程序执行的同步。
    stream, ret = acl.rt.create_stream()
    assert ret == 0, "初始化Stream失败。"

    # 成功执行到这一步，初始化完成。

    # 第二步：图像读入

    # ACL中专门为图像相关操作提供了硬件加速模块。这些加速有一个前提条件：图像需要为YUV格式。
    # 另外请注意，OpenCV暂时无法将BGR图像转换为ACL支持的YUV420SP格式。所以我们将直接读入一
    # 张YUV图像。

    # 读入一张YUV图像。
    img_file = "wood_rabbit_1024_1068_nv12.yuv"
    org_yuv = np.fromfile(img_file, dtype=np.byte)

    # 为了验证读取结果，将读入的YUV图像转换为你熟悉的OpenCV图像：NumpyArray。并在窗口中查看
    # 图像内容。
    org_yuv = np.reshape(
        org_yuv, (int((1072)*3/2), int(1024))).astype(np.uint8)
    org_bgr = cv2.cvtColor(org_yuv, cv2.COLOR_YUV2BGR_NV12)

    # 第三步：图像裁切

    # 如果你熟悉OpenCV的话，图像裁切可以通过数组slice的方式一行代码搞定。但是在ACL中，图像并
    # 不是以NumpyArray的形式存储。所以你无法通过slice的方式裁切图像，而只能依赖ACL提供的API。
    # 完成一次裁切分为以下几步：

    # 1. 准备待裁切图像

    # 虽然你已经将YUV图像读入内存，但是硬件加速的裁切操作需要在Ascend芯片上进行，这意味着你需
    # 要把图像数据从主机(host)一侧拷贝到Ascend芯片(Device)一侧。另外，Device一侧的数据并非
    # Numpy数组，在这里生存的图像有自己独特的“描述方式”。

    # 构建Device一侧的图像描述。
    input_desc = acl.media.dvpp_create_pic_desc()
    assert input_desc is not None

    # 图像描述对象构建之后，使用真实要处理的图像参数填充该描述，计算内存占用大小，并将具体的像
    # 数值拷贝到Device一侧。
    h, w, _ = org_bgr.shape
    intput_ptr = setup_desc(input_desc, w, h)
    in_buffer_size = org_yuv.itemsize * org_yuv.size
    np_yuv_ptr = acl.util.numpy_to_ptr(org_yuv)
    ret = acl.rt.memcpy(intput_ptr, in_buffer_size,
                        np_yuv_ptr, in_buffer_size,
                        constants.ACL_MEMCPY_HOST_TO_DEVICE)
    assert ret == 0, "图像内存拷贝失败。"

    # 至此，待裁切图像准备完成。

    # 2. 设定裁切区域。

    # 由于输入图像不是Numpy数组，你只能使用ACL提供的API来设定裁切区域。另外，Ascend设备上的
    # 开发有诸多限制。例如裁切区域[x0, x1, y0, y1]必须满足[偶数、奇数、偶数、奇数]的约束。
    crop_w = 320
    crop_h = 420
    x = 480
    y = 360
    crop_area = acl.media.dvpp_create_roi_config(
        x, x + crop_w-1, y, y + crop_h-1)

    # 3. 构建输出图像。

    # 与输入图像类似，输出图像默认生存在Device一侧，你需要提前为它准备好内存空间。这个过程与输
    # 入图像的构建过程非常相似。
    output_desc = acl.media.dvpp_create_pic_desc()
    assert output_desc is not None

    # 设置输出图片描述。输出图像的宽度与高度为截取后的大小。
    setup_desc(output_desc, crop_w, crop_h)

    # 4. 开始裁切

    # 在正式执行裁切指令前，还需要为裁切操作创建“数据处理通道”，这也是ACL开发的不同之处。
    dvpp_channel_desc = acl.media.dvpp_create_channel_desc()
    ret = acl.media.dvpp_create_channel(dvpp_channel_desc)

    # 一切就绪，执行裁切指令。
    ret = acl.media.dvpp_vpc_crop_async(dvpp_channel_desc, input_desc,
                                        output_desc, crop_area, stream)

    # 裁切指令为异步执行，所以需要等待它执行完毕才能获得裁切结果。
    ret = acl.rt.synchronize_stream(stream)

    # 至此裁切结束。

    # 5. 获取结果。
    # 别忘了裁切后的图像生存在Device一侧，并且是YUV420SP格式。我们需要将它拷贝到主机(host)
    # 一侧，然后转换成熟悉的Numpy数组。

    # 获取裁切后的图像信息。
    data = acl.media.dvpp_get_pic_desc_data(output_desc)
    data_size = acl.media.dvpp_get_pic_desc_size(output_desc)
    ret_code = acl.media.dvpp_get_pic_desc_ret_code(output_desc)
    utils.check_ret("acl.media.dvpp_get_pic_desc_ret_code", ret_code)

    # 拷贝数据到主机一侧并转换为Numpy数组。
    np_pic = np.zeros(data_size, dtype=np.byte)
    np_pic_ptr = acl.util.numpy_to_ptr(np_pic)
    ret = acl.rt.memcpy(np_pic_ptr, data_size,
                        data, data_size,
                        constants.ACL_MEMCPY_DEVICE_TO_HOST)
    utils.check_ret("acl.rt.memcpy", ret)

    # 6. 查阅结果。

    # 最后，在窗口中查阅裁切前后的对比。这一步需要将YUV420SP转换为BGR格式。
    result_yuv = np.reshape(
        np_pic, (int((crop_h)*3/2), int(crop_w))).astype(np.uint8)
    result_bgr = cv2.cvtColor(result_yuv, cv2.COLOR_YUV2BGR_NV12)
    cv2.imshow('Original', org_bgr)
    cv2.imshow('Cropped', result_bgr)
    cv2.waitKey()

    # 第四步：收尾、释放资源

    # 解码结束后，释放资源，包括输入/输出图片的描述信息、输入/输出内存、通道描述信息、通道等
    acl.media.dvpp_destroy_pic_desc(input_desc)
    acl.media.dvpp_destroy_pic_desc(output_desc)

    # 释放Device侧内存
    ret = acl.media.dvpp_free(intput_ptr)

    # 释放操作通道
    if dvpp_channel_desc:
        ret = acl.media.dvpp_destroy_channel(dvpp_channel_desc)
        assert ret == 0
        ret = acl.media.dvpp_destroy_channel_desc(dvpp_channel_desc)
        assert ret == 0

    # 释放运行管理资源
    ret = acl.rt.destroy_stream(stream)
    ret = acl.rt.destroy_context(context)
    ret = acl.rt.reset_device(device_id)
    ret = acl.finalize()

    print('结束。')


if __name__ == "__main__":
    main()
