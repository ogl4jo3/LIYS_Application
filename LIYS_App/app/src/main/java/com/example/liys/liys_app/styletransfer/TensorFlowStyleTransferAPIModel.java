/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.example.liys.liys_app.styletransfer;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.os.Trace;
import android.util.Log;

import com.example.liys.liys_app.env.Logger;
import com.example.liys.liys_app.segmentation.Segmentation;

import org.tensorflow.Graph;
import org.tensorflow.Operation;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Vector;

public class TensorFlowStyleTransferAPIModel {
    private static final Logger LOGGER = new Logger();

    // Config values.
    private String inputName;
    private int inputWidth;
    private int inputHeight;

    // Pre-allocated buffers.
    private int[] intValues;
    //TODO:transform model with input int8 and output int8 value
    //    private byte[] byteValues;
    private float[] byteValues;

    private String[] outputNames;
    //    private byte[] outputValues;
    private float[] outputValues;

    private boolean logStats = false;

    private TensorFlowInferenceInterface inferenceInterface;

    /**
     * Initializes a native TensorFlow session for classifying images.
     *
     * @param assetManager  The asset manager to be used to load assets.
     * @param modelFilename The filepath of the model GraphDef protocol buffer.
     */
    public static TensorFlowStyleTransferAPIModel create(
            final AssetManager assetManager,
            final String modelFilename,
            final int inputWidth,
            final int inputHeight) throws IOException {
        LOGGER.w("Tensorflow PB Style Transfer");
        final TensorFlowStyleTransferAPIModel d = new TensorFlowStyleTransferAPIModel();

        d.inferenceInterface = new TensorFlowInferenceInterface(assetManager, modelFilename);

        final Graph g = d.inferenceInterface.graph();

        d.inputName = "ImageTensor";
        // The inputName node has a shape of [N, H, W, C], where
        // N is the batch size
        // H = W are the height and width
        // C is the number of channels (3 for our purposes - RGB)
        final Operation inputOp = g.operation(d.inputName);
        if (inputOp == null) {
            throw new RuntimeException("Failed to find input Node '" + d.inputName + "'");
        }

        d.inputWidth = inputWidth;
        d.inputHeight = inputHeight;

        // Pre-allocate buffers.
        d.outputNames = new String[]{"Prediction/add"};

        d.intValues = new int[d.inputWidth * d.inputHeight];
//        d.byteValues = new byte[d.inputWidth * d.inputHeight * 3];
//        d.outputValues = new byte[d.inputWidth * d.inputHeight * 3];
        d.byteValues = new float[d.inputWidth * d.inputHeight * 3];
        d.outputValues = new float[d.inputWidth * d.inputHeight * 3];
        return d;
    }

    private TensorFlowStyleTransferAPIModel() {
    }

    public int[] styleImage(final Bitmap bitmap) {
        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("style transfer image");

        Trace.beginSection("preprocessBitmap");
        // Preprocess the image data to extract R, G and B bytes from int of form 0x00RRGGBB
        // on the provided parameters.
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        for (int i = 0; i < intValues.length; ++i) {
            byteValues[i * 3 + 2] = (byte) (intValues[i] & 0xFF);
            byteValues[i * 3 + 1] = (byte) ((intValues[i] >> 8) & 0xFF);
            byteValues[i * 3 + 0] = (byte) ((intValues[i] >> 16) & 0xFF);
        }
        Trace.endSection(); // preprocessBitmap


        final long startTime = SystemClock.uptimeMillis();
        // Copy the input data into TensorFlow.
        Trace.beginSection("feed");
        inferenceInterface.feed(inputName, byteValues, 1, inputHeight, inputWidth, 3);
        Trace.endSection();

        // Run the inference call.
        Trace.beginSection("run");
        inferenceInterface.run(outputNames, logStats);
        Trace.endSection();

        // Copy the output Tensor back into the output array.
        Trace.beginSection("fetch");
        inferenceInterface.fetch(outputNames[0], outputValues);
        Trace.endSection();
        final long endTime = SystemClock.uptimeMillis();
        LOGGER.d("style transfer image model cost: " + (endTime - startTime) + "ms");

        int[] rltValues = new int[intValues.length];

        for (int i = 0; i < intValues.length; ++i) {
            rltValues[i] = convertToRGBInt(outputValues[i * 3], outputValues[i * 3 + 1], outputValues[i * 3 + 2]);
        }

        return rltValues;
    }

    public void close() {
        inferenceInterface.close();
    }

    private static int convertToInt(float data) {
        if (data < 0) {
            data = 0;
        } else if (data > 255) {
            data = 255;
        }
        return (255 - (byte) data);
    }

    private static int convertToRGBInt(float red, float green, float blue) {
        int rlt = 0;
        rlt = convertToInt(red);
        rlt = (rlt << 8) + convertToInt(green);
        rlt = -((rlt << 8) + convertToInt(blue) + 1);
        return rlt;
    }

}
