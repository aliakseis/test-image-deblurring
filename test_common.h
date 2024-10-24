/*
 * test_common.h - test common
 *
 *  Copyright (c) 2015 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Wind Yuan <feng.yuan@intel.com>
 * Author: John Ye <john.ye@intel.com>
 */

#ifndef XCAM_TEST_COMMON_H
#define XCAM_TEST_COMMON_H

//#include <unistd.h>
//#include <getopt.h>
#include <string>

#define TEST_CAMERA_POSITION_OFFSET_X 2000

#undef CHECK_DECLARE
#undef CHECK
#undef CHECK_CONTINUE

#define CHECK_DECLARE(level, exp, statement, msg, ...) \
    if (!(exp)) {        \
        XCAM_LOG_##level (msg, ## __VA_ARGS__);   \
        statement;                              \
    }

#define CHECK(ret, msg, ...)  \
    CHECK_DECLARE(ERROR, (ret) == XCAM_RETURN_NO_ERROR, return -1, msg, ## __VA_ARGS__)

#define CHECK_STATEMENT(ret, statement, msg, ...)  \
    CHECK_DECLARE(ERROR, (ret) == XCAM_RETURN_NO_ERROR, statement, msg, ## __VA_ARGS__)

#define CHECK_CONTINUE(ret, msg, ...)  \
    CHECK_DECLARE(WARNING, (ret) == XCAM_RETURN_NO_ERROR, , msg, ## __VA_ARGS__)

#define CHECK_EXP(exp, msg, ...) \
    CHECK_DECLARE(ERROR, exp, return -1, msg, ## __VA_ARGS__)

#define CAPTURE_DEVICE_VIDEO "/dev/video3"
#define CAPTURE_DEVICE_STILL "/dev/video0"
#define DEFAULT_CAPTURE_DEVICE CAPTURE_DEVICE_VIDEO

#define DEFAULT_EVENT_DEVICE   "/dev/v4l-subdev6"
#define DEFAULT_CPF_FILE       "/etc/atomisp/imx185.cpf"
#define DEFAULT_SAVE_FILE_NAME "capture_buffer"
#define DEFAULT_DYNAMIC_3A_LIB "/usr/lib/xcam/plugins/3a/libxcam_3a_aiq.so"
#define DEFAULT_HYBRID_3A_LIB "/usr/lib/xcam/plugins/3a/libxcam_3a_hybrid.so"
#define DEFAULT_SMART_ANALYSIS_LIB_DIR "/usr/lib/xcam/plugins/smart"

#define FISHEYE_CONFIG_PATH "./calib_params/"
#define FISHEYE_CONFIG_ENV_VAR "FISHEYE_CONFIG_PATH"

#define FPS_CALCULATION(objname, count) XCAM_STATIC_FPS_CALCULATION(objname, count)

#define PROFILING_START(name)  XCAM_STATIC_PROFILING_START(name)
#define PROFILING_END(name, times_of_print) XCAM_STATIC_PROFILING_END(name, times_of_print)

#endif  // XCAM_TEST_COMMON_H
