/*
 * Copyright (c) 2018 Wei OUYANG
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * Filter implementing image super-resolution using deep convolutional networks.
 * https://arxiv.org/abs/1501.00092
 * https://arxiv.org/abs/1609.05158
 */
#include "avfilter.h"
#include "formats.h"
#include "internal.h"
#include "libavutil/opt.h"
#include "libavformat/avio.h"
#include "libswscale/swscale.h"
#include "libavutil/pixdesc.h"
#include "libavutil/common.h"

#include "dnn_interface.h"

typedef struct Im2ImContext {
    const AVClass *class;
    char *model_filename;
    DNNModule *dnn_module;
    DNNModel *model;
    DNNData input, output;
    int split_channels;
    int convert_ch_in;
    int convert_ch_out;
    struct SwsContext *sws_contexts[3];
    int sws_slice_h, sws_input_linesize, sws_output_linesize;
    uint8_t *input_conv_buffer;
    uint8_t *output_conv_buffer;
    enum AVPixelFormat out_pix_fmt;
} Im2ImContext;

#define OFFSET(x) offsetof(Im2ImContext, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM

static const AVOption im2im_options[] = {
    { "model", "path to model file specifying network architecture and its parameters", OFFSET(model_filename), AV_OPT_TYPE_STRING, {.str=NULL}, 0, 0, FLAGS },
    { "split_channels",  "obtain channels by splitting single channel frames (pix_fmt=gray*) vertically", OFFSET(split_channels), AV_OPT_TYPE_INT, {.i64 = 1}, 1, INT_MAX, FLAGS },
    { "pix_fmt", NULL, OFFSET(out_pix_fmt), AV_OPT_TYPE_PIXEL_FMT, {.i64 = AV_PIX_FMT_GRAYF32}, AV_PIX_FMT_NONE, INT_MAX, FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(im2im);

static av_cold int init(AVFilterContext *context)
{
    Im2ImContext *im2im_context = context->priv;
    im2im_context->dnn_module = ff_get_dnn_module(DNN_TF);
    if (!im2im_context->dnn_module){
        av_log(context, AV_LOG_ERROR, "could not create DNN module for requested backend\n");
        return AVERROR(ENOMEM);
    }
    if (!im2im_context->model_filename){
        av_log(context, AV_LOG_ERROR, "model file for network was not specified\n");
        return AVERROR(EIO);
    }
    else{
        im2im_context->model = (im2im_context->dnn_module->load_model)(im2im_context->model_filename);
    }
    if (!im2im_context->model){
        av_log(context, AV_LOG_ERROR, "could not load DNN model\n");
        return AVERROR(EIO);
    }

    im2im_context->sws_contexts[0] = NULL;
    im2im_context->sws_contexts[1] = NULL;
    im2im_context->sws_contexts[2] = NULL;
    im2im_context->convert_ch_in = 0;
    im2im_context->convert_ch_out = 0;
    im2im_context->input_conv_buffer = NULL;
    im2im_context->output_conv_buffer = NULL;
    return 0;
}

static int query_formats(AVFilterContext *context)
{
    const enum AVPixelFormat pixel_formats[] = {AV_PIX_FMT_RGB32, AV_PIX_FMT_BGR32,
                                                AV_PIX_FMT_GRAY8, AV_PIX_FMT_GRAY16, AV_PIX_FMT_GRAYF32,
                                                AV_PIX_FMT_RGB24, AV_PIX_FMT_BGR24,
                                                AV_PIX_FMT_RGBA, AV_PIX_FMT_BGRA,
                                                AV_PIX_FMT_NONE};
    AVFilterFormats *formats_list;

    formats_list = ff_make_format_list(pixel_formats);
    if (!formats_list){
        av_log(context, AV_LOG_ERROR, "could not create formats list\n");
        return AVERROR(ENOMEM);
    }

    return ff_set_common_formats(context, formats_list);
}

static int config_props(AVFilterLink *inlink)
{
    AVFilterContext *context = inlink->dst;
    Im2ImContext *im2im_context = context->priv;
    AVFilterLink *outlink = context->outputs[0];
    DNNReturnType result;
    int split_channels = im2im_context->split_channels;
    int inlink_ch = 1;
    int outlink_ch = 1;

    const AVPixFmtDescriptor *desc_in = av_pix_fmt_desc_get(inlink->format);
    if(desc_in == NULL){
        av_log(context, AV_LOG_ERROR, "unsupported input pixel format\n");
        return AVERROR(EIO);
    }
    inlink_ch = desc_in->nb_components;

    if(split_channels > 1 && inlink_ch > 1){
        av_log(context, AV_LOG_ERROR, "split_channels can only be set when the input pixel format is gray8, gray16 or grayf32\n");
        return AVERROR(EIO);
    }
    else if(split_channels > 1 && inlink_ch == 1){
        im2im_context->convert_ch_in = 1;
        im2im_context->input.width = inlink->w;
        im2im_context->input.height = inlink->h / split_channels; // split channel vertically
        im2im_context->input.channels = split_channels;
    }
    else{
        im2im_context->input.width = inlink->w;
        im2im_context->input.height = inlink->h;
        im2im_context->input.channels = inlink_ch;
    }

    result = (im2im_context->model->set_input_output)(im2im_context->model->model, &im2im_context->input, &im2im_context->output);
    if (result != DNN_SUCCESS){
        av_log(context, AV_LOG_ERROR, "could not set input and output for the model\ninput_h:%d, input_w:%d, input_ch:%d\n",
               (int)im2im_context->input.height, (int)im2im_context->input.width, (int)im2im_context->input.channels);
        return AVERROR(EIO);
    }
    else{
        outlink->h = im2im_context->output.height;
        outlink->w = im2im_context->output.width;
        outlink->format = im2im_context->out_pix_fmt;
        int output_channels = im2im_context->output.channels;

        im2im_context->sws_input_linesize = inlink->w << 2;
        const AVPixFmtDescriptor *desc_out = av_pix_fmt_desc_get(outlink->format);
        if(desc_out == NULL){
            av_log(context, AV_LOG_ERROR, "unsupported output pixel format\n");
            return AVERROR(EIO);
        }
        outlink_ch = desc_out->nb_components;

        // multi-channel: get channles from the input frame
        if(split_channels == 1 && inlink_ch > 1){
            im2im_context->sws_contexts[1] = sws_getContext(inlink->w, inlink->h, AV_PIX_FMT_GRAY8,
                                                         inlink->w, inlink->h, AV_PIX_FMT_GRAYF32,
                                                         0, NULL, NULL, NULL);
        }
        // multi-channel: get channels by splitting the frame with a single channel (vertically)
        else if(split_channels > 1 && inlink_ch == 1){
            im2im_context->sws_contexts[1] = sws_getContext(inlink->w, inlink->h, inlink->format,
                                                         inlink->w, inlink->h, AV_PIX_FMT_GRAYF32,
                                                         0, NULL, NULL, NULL);
            im2im_context->input_conv_buffer = av_malloc((inlink->w * inlink->h * 4) + 32);
            if (!im2im_context->input_conv_buffer)
                return AVERROR(ENOMEM);
        }
        // single channel input
        else{
            im2im_context->sws_contexts[1] = sws_getContext(inlink->w, inlink->h, inlink->format,
                                                         im2im_context->input.width, im2im_context->input.height, AV_PIX_FMT_GRAYF32,
                                                         0, NULL, NULL, NULL);
        }
        im2im_context->sws_output_linesize = im2im_context->output.width << 2;
        if(output_channels > 1){
            // multi-channel: combine channels vertically into frames with a single channel
            if(outlink_ch == 1){
                im2im_context->convert_ch_out = 1;
                outlink->h = im2im_context->output.height * output_channels;
                im2im_context->sws_contexts[2] = sws_getContext(im2im_context->output.width, im2im_context->output.height*output_channels, AV_PIX_FMT_GRAYF32,
                                                             im2im_context->output.width, im2im_context->output.height*output_channels, outlink->format,
                                                             0, NULL, NULL, NULL);
                im2im_context->output_conv_buffer = av_malloc(im2im_context->output.width * im2im_context->output.height * output_channels * 4 + 32);
                if (!im2im_context->output_conv_buffer)
                    return AVERROR(ENOMEM);
            }
            // multi-channel: merge output channels to the output frame
            else if(outlink_ch == output_channels){
                im2im_context->sws_contexts[2] = sws_getContext(im2im_context->output.width, im2im_context->output.height, AV_PIX_FMT_GRAYF32,
                                                             im2im_context->output.width, im2im_context->output.height, AV_PIX_FMT_GRAY8,
                                                             0, NULL, NULL, NULL);
            }
            else{
                av_log(context, AV_LOG_ERROR, "The number of channels from the model output (%d) does not match the output pixel format (ch=%d)\n", output_channels, outlink_ch);
                return AVERROR(EIO);
            }
        }
        // single channel output
        else{
            im2im_context->sws_contexts[2] = sws_getContext(im2im_context->output.width, im2im_context->output.height, AV_PIX_FMT_GRAYF32,
                                                         outlink->w, outlink->h, outlink->format,
                                                         0, NULL, NULL, NULL);
        }

        if (!im2im_context->sws_contexts[1] || !im2im_context->sws_contexts[2]){
            av_log(context, AV_LOG_ERROR, "could not create SwsContext for conversions\noutput_h:%d, output_w:%d, output_ch:%d\n",
                   (int)im2im_context->output.height, (int)im2im_context->output.width, (int)im2im_context->output.channels);
            return AVERROR(ENOMEM);
        }

        return 0;
    }
}

static int filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext *context = inlink->dst;
    Im2ImContext *im2im_context = context->priv;
    AVFilterLink *outlink = context->outputs[0];
    AVFrame *out = ff_get_video_buffer(outlink, outlink->w, outlink->h);
    DNNReturnType dnn_result;
    int output_channels = im2im_context->output.channels;
    int input_channels = im2im_context->input.channels;
    int i;
    long j;

    if (!out){
        av_log(context, AV_LOG_ERROR, "could not allocate memory for output frame\n");
        av_frame_free(&in);
        return AVERROR(ENOMEM);
    }
    av_frame_copy_props(out, in);
    if( input_channels > 1){
        float *in_data;
        long channel_pixels = im2im_context->input.height * im2im_context->input.width;
        // multi-channel: get channels by splitting the frame with a single channel (vertically)
        if(im2im_context->convert_ch_in > 0){
            in_data = (float *) im2im_context->input.data;
            sws_scale(im2im_context->sws_contexts[1], (const uint8_t *const *)in->data, in->linesize,
                      0, in->height, (uint8_t *const *)(&im2im_context->input_conv_buffer), &im2im_context->sws_input_linesize);
            long offset = 0;
            float *out_buffer;
            for(i=0;i<input_channels;i++){
                offset = channel_pixels * 4 * i;
                out_buffer = (float *) (im2im_context->input_conv_buffer + offset);
                // converting data according to tensorflow NHWC memory layout
                for(j=0;j<channel_pixels;j++){
                    in_data[j*input_channels+i] = *(out_buffer+j);
                }
            }
        }
        // multi-channel: get channles from the input frame
        else {
            uint8_t *in_img = (uint8_t *)(in->data[0]);
            for(i=0;i<input_channels;i++){
                in_data = (float *)im2im_context->input.data;
                // converting data according to tensorflow NHWC memory layout
                for(j=0;j<channel_pixels;j++){
                    in_data[j*input_channels+i] = ((float) in_img[j*input_channels+i])/255.0;
                }
            }
        }

    }
    // single channel input
    else{
          sws_scale(im2im_context->sws_contexts[1], (const uint8_t *const *)in->data, in->linesize,
                    0, in->height, (uint8_t *const *)(&im2im_context->input.data), &im2im_context->sws_input_linesize);
    }

    av_frame_free(&in);

    dnn_result = (im2im_context->dnn_module->execute_model)(im2im_context->model);
    if (dnn_result != DNN_SUCCESS){
        av_log(context, AV_LOG_ERROR, "failed to execute loaded model\n");
        return AVERROR(EIO);
    }

    if(output_channels > 1){
        float *out_data;
        long channel_pixels = im2im_context->output.height * im2im_context->output.width;
        // multi-channel: convert channels to a frame with a single channel
        if(im2im_context->convert_ch_out > 0){
            out->height = im2im_context->output.height * output_channels;
            out->width = im2im_context->output.width;
            out->format = im2im_context->out_pix_fmt;
            out_data = (float *) im2im_context->output.data;
            long offset = 0;
            float *out_buffer;
            for(i=0;i<output_channels;i++){
                offset = channel_pixels * 4 * i;
                out_buffer = (float *) (im2im_context->output_conv_buffer + offset);
                for(j=0;j<channel_pixels;j++){
                    out_buffer[j] = out_data[j * output_channels + i];
                }
            }
            sws_scale(im2im_context->sws_contexts[2], (const uint8_t *const *)(&im2im_context->output_conv_buffer), &im2im_context->sws_output_linesize,
                      0, out->height, (uint8_t *const *)out->data, out->linesize);
        }
        // multi-channel: convert channels to a frame with multiple channels
        else {
            out->height = im2im_context->output.height;
            out->width = im2im_context->output.width;
            out->format = im2im_context->out_pix_fmt;
            // consider endianness ?
            out_data = (float *) im2im_context->output.data;
            uint8_t *out_img = (uint8_t *)(out->data[0]);
            for(i=0;i<output_channels;i++){
                for(j=0;j<channel_pixels;j++){
                    out_img[j*output_channels+i] = (uint8_t) (av_clipf(out_data[j*output_channels+i], 0.0, 1.0)*255.0);
                }
            }
        }
    }
    // single channel mode
    else{
        out->height = im2im_context->output.height;
        out->width = im2im_context->output.width;
        out->format = im2im_context->out_pix_fmt;
        sws_scale(im2im_context->sws_contexts[2], (const uint8_t *const *)(&im2im_context->output.data), &im2im_context->sws_output_linesize,
                  0, out->height, (uint8_t *const *)out->data, out->linesize);
    }

    return ff_filter_frame(outlink, out);
}

static av_cold void uninit(AVFilterContext *context)
{
    int i;
    Im2ImContext *im2im_context = context->priv;

    if (im2im_context->dnn_module){
        (im2im_context->dnn_module->free_model)(&im2im_context->model);
        av_freep(&im2im_context->dnn_module);
    }
    if(im2im_context->input_conv_buffer){
        av_freep(&im2im_context->input_conv_buffer);
    }
    if(im2im_context->output_conv_buffer){
        av_freep(&im2im_context->output_conv_buffer);
    }
    for (i = 0; i < 3; ++i){
        if (im2im_context->sws_contexts[i]){
            sws_freeContext(im2im_context->sws_contexts[i]);
        }
    }
}

static const AVFilterPad im2im_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = config_props,
        .filter_frame = filter_frame,
    },
    { NULL }
};

static const AVFilterPad im2im_outputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
    },
    { NULL }
};

AVFilter ff_vf_im2im = {
    .name          = "im2im",
    .description   = NULL_IF_CONFIG_SMALL("Apply deep neural networks based image-to-image (im2im) translation with Tensorflow."),
    .priv_size     = sizeof(Im2ImContext),
    .init          = init,
    .uninit        = uninit,
    .query_formats = query_formats,
    .inputs        = im2im_inputs,
    .outputs       = im2im_outputs,
    .priv_class    = &im2im_class,
    .flags         = AVFILTER_FLAG_SUPPORT_TIMELINE_GENERIC |
                     AVFILTER_FLAG_SLICE_THREADS,
};
