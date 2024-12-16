/*
 * test-image-deblurring.cpp - test image deblurring
 *
 *  Copyright (c) 2017 Intel Corporation
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
 * Author: Andrey Parfenov <a1994ndrey@gmail.com>
 * Author: Wind Yuan <feng.yuan@intel.com>
 */

#include "test_common.h"
//#include "test_inline.h"

//#include <unistd.h>
//#include <getopt.h>
//#include <image_file.h>
#include "cv_image_sharp.h"
#include "cv_wiener_filter.h"
#include "cv_image_deblurring.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>

using namespace XCam;

/*
static void
usage (const char* arg0)
{
    printf ("Usage: %s --input file --output file\n"
            "\t--input,    input image(RGB)\n"
            "\t--output,   output image(RGB) PREFIX\n"
            "\t--blind,    optional, blind or non-blind deblurring, default true; select from [true/false]\n"
            "\t--save,     optional, save file or not, default true; select from [true/false]\n"
            "\t--help,     usage\n",
            arg0);
}
*/

/*
static void
blind_deblurring (cv::Mat &input_image, cv::Mat &output_image)
{
    auto image_deblurring = std::make_unique<CVImageDeblurring>();
    cv::Mat kernel;
    image_deblurring->blind_deblurring(input_image, output_image, kernel);// , -1, -1, false);
}
*/

static void
blind_deblurring(cv::Mat& input_image, cv::Mat& output_image)
{
    const auto CELL_SIZE = 128;

    const int numXSteps = std::max((input_image.cols + CELL_SIZE / 2) / CELL_SIZE, 2);
    const int numYSteps = std::max((input_image.rows + CELL_SIZE / 2) / CELL_SIZE, 2);

    const double xStep = double(input_image.cols - 1) / numXSteps;
    const double yStep = double(input_image.rows - 1) / numYSteps;

    auto comp = [](const cv::Rect& left, const cv::Rect& right) {
        return std::tie(left.x, left.y, left.width, left.height)
            < std::tie(right.x, right.y, right.width, right.height);
        };

    std::map < cv::Rect, cv::Mat, decltype(comp)> cache(comp);

    auto image_deblurring = std::make_unique<CVImageDeblurring>();

    auto cacheLam = [&input_image, &cache, &image_deblurring](const cv::Rect& rect)->cv::Mat
        {
            if (rect.x < 0 || rect.y < 0
                || rect.x + rect.width > input_image.cols
                || rect.y + rect.height > input_image.rows)
            {
                return cv::Mat();
            }

            auto it = cache.find(rect);
            if (it != cache.end())
                return it->second;
            cv::Mat src = input_image(rect);
            cv::Mat kernel;
            cv::Mat output_image;
            try {
                image_deblurring->blind_deblurring(src, output_image, kernel);// , -1, -1, false);
            }
            catch (...) {
                output_image = src;
            }
            return cache.insert({ rect, output_image }).first->second;
        };

    auto calcLam = [&input_image, cacheLam, xStep, yStep](const cv::Rect& rect, int x, int y, double centerX, double centerY) -> cv::Vec3b {
        auto mat = cacheLam(rect);
        cv::Vec3b val = mat.empty() ? input_image.at<cv::Vec3b>(y, x) : mat.at<cv::Vec3b>(y - rect.y, x - rect.x);
        double weightX = (1 + cos((x - centerX) * M_PI / (xStep)));
        double weightY = (1 + cos((y - centerY) * M_PI / (yStep)));
        return val * (weightX * weightY / 4);
        };

    output_image = cv::Mat(input_image.size(), CV_8UC3);

    for (int y = 0; y < input_image.rows; ++y)
        for (int x = 0; x < input_image.cols; ++x)
        {
            auto xSteps = static_cast<int>(x / xStep);
            auto ySteps = static_cast<int>(y / yStep);

            const auto leftBegin = (xSteps - 1) * xStep;
            const auto rightBegin = xSteps * xStep;
            const auto leftEnd = (xSteps + 1) * xStep;
            const auto rightEnd = (xSteps + 2) * xStep;

            const auto topBegin = (ySteps - 1) * yStep;
            const auto bottomBegin = ySteps * yStep;
            const auto topEnd = (ySteps + 1) * yStep;
            const auto bottomEnd = (ySteps + 2) * yStep;

            cv::Rect leftTop(cv::Point(leftBegin, topBegin), cv::Point(leftEnd + 1, topEnd + 1));
            cv::Rect rightTop(cv::Point(rightBegin, topBegin), cv::Point(rightEnd + 1, topEnd + 1));
            cv::Rect leftBottom(cv::Point(leftBegin, bottomBegin), cv::Point(leftEnd + 1, bottomEnd + 1));
            cv::Rect rightBottom(cv::Point(rightBegin, bottomBegin), cv::Point(rightEnd + 1, bottomEnd + 1));

            cv::Vec3b res = (calcLam(leftTop, x, y, rightBegin, bottomBegin)
                + calcLam(rightTop, x, y, leftEnd, bottomBegin)
                + calcLam(leftBottom, x, y, rightBegin, topEnd)
                + calcLam(rightBottom, x, y, leftEnd, topEnd));

            output_image.at<cv::Vec3b>(y, x) = res;
        }
}


static void
non_blind_deblurring (cv::Mat &input_image, cv::Mat &output_image)
{
    auto wiener_filter = std::make_unique<CVWienerFilter>();
    cv::cvtColor (input_image, input_image, cv::COLOR_BGR2GRAY);
    // use simple motion blur kernel
    int kernel_size = 13;
    cv::Mat kernel = cv::Mat::zeros (kernel_size, kernel_size, CV_32FC1);
    for (int i = 0; i < kernel_size; i++)
    {
        kernel.at<float> ((kernel_size - 1) / 2, i) = 1.0;
    }
    kernel /= kernel_size;
    //flip kernel to perform convolution
    cv::Mat conv_kernel;
    cv::flip (kernel, conv_kernel, -1);
    cv::Mat blurred;
    cv::filter2D (input_image, blurred, CV_32FC1, conv_kernel, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
    // restore the image
    cv::Mat median_blurred;
    medianBlur (blurred, median_blurred, 3);
    auto helpers = std::make_unique<CVImageProcessHelper>();
    float noise_power = 1.0f / helpers->get_snr (blurred, median_blurred);
    wiener_filter->wiener_filter (blurred, kernel, output_image, noise_power);
}

int main (int argc, char *argv[])
{
    //const char *file_in_name = NULL;
    //const char *file_out_name = NULL;

    //bool need_save_output = true;
    //bool blind = true;

    //const struct option long_opts[] = {
    //    {"input", required_argument, NULL, 'i'},
    //    {"output", required_argument, NULL, 'o'},
    //    {"blind", required_argument, NULL, 'b'},
    //    {"save", required_argument, NULL, 's'},
    //    {"help", no_argument, NULL, 'H'},
    //    {0, 0, 0, 0},
    //};

    //int opt = -1;
    //while ((opt = getopt_long (argc, argv, "", long_opts, NULL)) != -1)
    //{
    //    switch (opt) {
    //    case 'i':
    //        file_in_name = optarg;
    //        break;
    //    case 'o':
    //        file_out_name = optarg;
    //        break;
    //    case 'b':
    //        blind = (strcasecmp (optarg, "false") == 0 ? false : true);
    //        break;
    //    case 's':
    //        need_save_output = (strcasecmp (optarg, "false") == 0 ? false : true);
    //        break;
    //    case 'H':
    //        usage (argv[0]);
    //        return 0;
    //    default:
    //        printf ("getopt_long return unknown value:%c\n", opt);
    //        usage (argv[0]);
    //        return -1;
    //    }
    //}

    //if (optind < argc || argc < 2)
    //{
    //    printf ("unknown option %s\n", argv[optind]);
    //    usage (argv[0]);
    //    return -1;
    //}


    cv::CommandLineParser parser(argc, argv,
        "{help h||show this message}"
        "{@input | |input file name}"
        "{@output | |output file name}"
        "{blind b|true|blind mode (true/false)}"
        //"{save s|false|save output (true/false)}"
    );

    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    std::string file_in_name = parser.get<std::string>(0);
    std::string file_out_name = parser.get<std::string>(1);
    bool blind = parser.get<bool>("blind");
    bool need_save_output = true;// parser.get<bool>("save");

    if (!parser.check()) {
        parser.printErrors();
        return -1;
    }



    if (file_in_name.empty() || file_out_name.empty())
    {
        XCAM_LOG_ERROR ("input/output path is NULL");
        return -1;
    }

    printf ("Description-----------\n");
    printf ("input image file:%s\n", file_in_name.c_str());
    printf ("output file :%s\n", file_out_name.c_str());
    printf ("blind deblurring:%s\n", blind ? "true" : "false");
    printf ("need save file:%s\n", need_save_output ? "true" : "false");
    printf ("----------------------\n");

    auto sharp = std::make_unique<CVImageSharp>();
    cv::Mat input_image = cv::imread (file_in_name, cv::IMREAD_COLOR);
    cv::Mat output_image;
    if (input_image.empty ())
    {
        XCAM_LOG_ERROR ("input file read error");
        return -1;
    }
    if (blind)
    {
        blind_deblurring (input_image, output_image);
    }
    else
    {
        non_blind_deblurring (input_image, output_image);
    }
    float input_sharp = sharp->measure_sharp (input_image);
    float output_sharp = sharp->measure_sharp (output_image);
    if (need_save_output)
    {
        cv::imwrite (file_out_name, output_image);
    }
    //XCAM_ASSERT (output_sharp > input_sharp);

    return 0;
}

