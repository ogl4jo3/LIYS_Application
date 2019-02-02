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

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.os.Trace;
import android.util.Log;

import com.example.liys.liys_app.env.Logger;

import org.tensorflow.lite.Delegate;
import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Vector;

/**
 * Wrapper for frozen detection models trained using the Tensorflow Object Detection API:
 * github.com/tensorflow/models/tree/master/research/object_detection
 */
public class TFLiteSegmentationModel {
    private static final Logger LOGGER = new Logger();

    // Config values.
    private String inputName;
    private int inputWidth;
    private int inputHeight;

    // Pre-allocated buffers.
    private int[] intValues;
    private ByteBuffer imgData = null;

    private String[] outputNames;
    private long[][][] outputValues;

    /**
     * Options for configuring the Interpreter.
     */
    private final Interpreter.Options tfliteOptions = new Interpreter.Options();

    /**
     * The loaded TensorFlow Lite model.
     */
    private MappedByteBuffer tfliteModel;

    /**
     * An instance of the driver class to run model inference with Tensorflow Lite.
     */
    protected Interpreter tflite;

    /**
     * holds a gpu delegate
     */
    Delegate gpuDelegate = null;

    /**
     * Initializes a native TensorFlow session for classifying images.
     *
     * @param assetManager  The asset manager to be used to load assets.
     * @param modelFilename The filepath of the model GraphDef protocol buffer.
     */
    public static TFLiteSegmentationModel create(
            final AssetManager assetManager,
            final String modelFilename,
            final int inputWidth,
            final int inputHeight) throws IOException {
        LOGGER.i("Tensorflow Lite Segmentation");
        final TFLiteSegmentationModel d = new TFLiteSegmentationModel();

        d.tfliteModel = loadModelFile(assetManager, modelFilename);
        d.tflite = new Interpreter(d.tfliteModel, d.tfliteOptions);

        d.inputName = "sub_7";
        d.inputWidth = inputWidth;
        d.inputHeight = inputHeight;

        // Pre-allocate buffers.
        d.outputNames = new String[]{"ArgMax"};

        d.imgData = ByteBuffer.allocateDirect(4 * d.inputWidth * d.inputHeight * 3);
        d.imgData.order(ByteOrder.nativeOrder());

        d.intValues = new int[d.inputWidth * d.inputHeight];
        d.outputValues = new long[1][d.inputWidth][d.inputHeight];
        return d;
    }

    private TFLiteSegmentationModel() {
    }

    public int[] segImage(final Bitmap bitmap) {
        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("segmentImage");

        Trace.beginSection("preprocessBitmap");
        // Preprocess the image data to extract R, G and B bytes from int of form 0x00RRGGBB
        // on the provided parameters.
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        if (imgData != null) {
            imgData.rewind();
        }


        for (int i = 0; i < intValues.length; ++i) {
            imgData.putFloat((float) (((intValues[i] >> 16) & 0xFF) * 0.00784313771874 - 1));
            imgData.putFloat((float) (((intValues[i] >> 8) & 0xFF) * 0.00784313771874 - 1));
            imgData.putFloat((float) (((intValues[i]) & 0xFF) * 0.00784313771874 - 1));

        }
        Trace.endSection(); // preprocessBitmap

        // Run the inference call.
        Trace.beginSection("run");
        long startTime = SystemClock.uptimeMillis();

        tflite.run(imgData, outputValues);

        long lastNativeTimeMs = SystemClock.uptimeMillis() - startTime;
        Log.d("TF Lite", "- TF Lite(Native) Time: " + lastNativeTimeMs + "ms");
        Trace.endSection();

        int[] outputReshape = new int[inputWidth * inputHeight];

        for (int i = 0; i < inputWidth; i++) {
            for (int j = 0; j < inputHeight; j++) {
                outputReshape[i * inputHeight + j] = (int) outputValues[0][i][j];
            }
        }


        return outputReshape;
    }

    /**
     * Memory-map the model file in Assets.
     */
    private static MappedByteBuffer loadModelFile(AssetManager assetManager, String modelFileName) throws IOException {
        String actualModelname = modelFileName.split("file:///android_asset/")[1];
        AssetFileDescriptor fileDescriptor = assetManager.openFd(actualModelname);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }


    private void recreateInterpreter() {
        if (tflite != null) {
            tflite.close();
            // TODO(b/120679982)
            // gpuDelegate.close();
            tflite = new Interpreter(tfliteModel, tfliteOptions);
        }
    }

    public void useGpu() {
        if (gpuDelegate == null && GpuDelegateHelper.isGpuDelegateAvailable()) {
            gpuDelegate = GpuDelegateHelper.createGpuDelegate();
            tfliteOptions.addDelegate(gpuDelegate);
            recreateInterpreter();
        }
    }

    public void useCPU() {
        tfliteOptions.setUseNNAPI(false);
        recreateInterpreter();
    }

    public void useNNAPI() {
        tfliteOptions.setUseNNAPI(true);
        recreateInterpreter();
    }

    public void setNumThreads(int numThreads) {
        tfliteOptions.setNumThreads(numThreads);
        recreateInterpreter();
    }

    /* Closes tflite to release resources. */
    public void close() {
        tflite.close();
        tflite = null;
        tfliteModel = null;
    }

}
