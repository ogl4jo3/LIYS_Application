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

package com.example.liys.liys_app.segmentation;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Trace;

import org.tensorflow.Graph;
import org.tensorflow.Operation;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import com.example.liys.liys_app.env.Logger;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Vector;

/**
 * Wrapper for frozen detection models trained using the Tensorflow Object Detection API:
 * github.com/tensorflow/models/tree/master/research/object_detection
 */
public class TensorFlowSegmentationAPIModel implements Segmentation {
    private static final Logger LOGGER = new Logger();

    // Only return this many results.
    private static final int MAX_RESULTS = 100;

    // Config values.
    private String inputName;
    private int inputWidth;
    private int inputHeight;

    // Pre-allocated buffers.
    private Vector<String> labels = new Vector<String>();
    private int[] intValues;
    private byte[] byteValues;

    private String[] outputNames;
    private int[] outputValues;

    private boolean logStats = false;

    private TensorFlowInferenceInterface inferenceInterface;

    /**
     * Initializes a native TensorFlow session for classifying images.
     *
     * @param assetManager  The asset manager to be used to load assets.
     * @param modelFilename The filepath of the model GraphDef protocol buffer.
     * @param labelFilename The filepath of label file for classes.
     */
    public static Segmentation create(
            final AssetManager assetManager,
            final String modelFilename,
            final String labelFilename,
            final int inputWidth,
            final int inputHeight) throws IOException {
        LOGGER.w("Tensorflow PB Segmentation");
        final TensorFlowSegmentationAPIModel d = new TensorFlowSegmentationAPIModel();

        InputStream labelsInput;
        String actualFilename = labelFilename.split("file:///android_asset/")[1];
        labelsInput = assetManager.open(actualFilename);
        BufferedReader br;
        br = new BufferedReader(new InputStreamReader(labelsInput));
        String line;
        while ((line = br.readLine()) != null) {
            LOGGER.w(line);
            d.labels.add(line);
        }
        br.close();


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
        d.outputNames = new String[]{"SemanticPredictions"};

        d.intValues = new int[d.inputWidth * d.inputHeight];
        d.byteValues = new byte[d.inputWidth * d.inputHeight * 3];
        d.outputValues = new int[d.inputWidth * d.inputHeight];
        return d;
    }

    private TensorFlowSegmentationAPIModel() {
    }

    @Override
    public int[] segImage(final Bitmap bitmap) {
        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("recognizeImage");

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

        // Copy the input data into TensorFlow.
        Trace.beginSection("feed");
        inferenceInterface.feed(inputName, byteValues, 1, inputWidth, inputHeight, 3);
        Trace.endSection();

        // Run the inference call.
        Trace.beginSection("run");
        inferenceInterface.run(outputNames, logStats);
        Trace.endSection();

        // Copy the output Tensor back into the output array.
        Trace.beginSection("fetch");
        outputValues = new int[intValues.length];
        inferenceInterface.fetch(outputNames[0], outputValues);
        Trace.endSection();

        return outputValues;
    }

    @Override
    public void enableStatLogging(final boolean logStats) {
        this.logStats = logStats;
    }

    @Override
    public String getStatString() {
        return inferenceInterface.getStatString();
    }

    @Override
    public void close() {
        inferenceInterface.close();
    }

}
