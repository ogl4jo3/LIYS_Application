/*
 * Copyright 2016 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.example.liys.liys_app.segmentation;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.hardware.Camera;
import android.media.ImageReader;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.util.Base64;
import android.util.Log;
import android.util.Size;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import com.example.liys.liys_app.CameraActivity;
import com.example.liys.liys_app.R;
import com.example.liys.liys_app.env.ImageUtils;
import com.example.liys.liys_app.env.Logger;
import com.example.liys.liys_app.styletransfer.TensorFlowStyleTransferAPIModel;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.SocketTimeoutException;
import java.net.URL;

/**
 * Segmentation and Style transfer camera frame.
 */
public class SegmentationActivity extends CameraActivity implements OnImageAvailableListener {
    private static final Logger LOGGER = new Logger();
    private static final String TAG = "SegmentationActivity";


    private static final int TF_SEG_API_INPUT_SIZE = 513;// 513 , 256
    private static final String TF_SEG_API_MODEL_FILE =
            "file:///android_asset/frozen_inference_graph.pb";
    private static final String TFLITE_SEG_MODEL_FILE =
//            "file:///android_asset/seg_tflite.tflite";
//            "file:///android_asset/deeplab.tflite";
//            "file:///android_asset/deeplab_dm30.tflite";
            "file:///android_asset/seg_dm30_550000.tflite";
//                "file:///android_asset/deeplab_256_dm50.tflite";
//                "file:///android_asset/deeplab_256_dm30.tflite";


    private static final String TF_SEG_API_LABELS_FILE = "file:///android_asset/seg_labels_list.txt";

    private static final boolean MAINTAIN_ASPECT = false;

    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);

    private static final boolean SAVE_PREVIEW_BITMAP = false;

    private Integer sensorOrientation;

    private boolean useTFLite = true;
    private Segmentation segmentation;
    private TFLiteSegmentationModel tfLiteSegmentationModel;

    private long lastProcessingTimeMs;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private Bitmap segMap = null;

    private boolean computingDetection = false;

    private long timestamp = 0;

    private Matrix frameToCropTransform;

    //    private String[] labelName = new String[]{
//            "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
//            "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
//            "person", "pottedplant", "sheep", "sofa", "train", "tv"};

    private String[] labelName = new String[]{
            "background", "person"};

    private int[] COLOR_MAP;
    private boolean[] segClass;

    private ImageView segMapView;
    private LinearLayout allLabelsLL;
    private Button btnSeg;
    private Button btnSty;
    private Button btnLiys;
    private boolean isSegOn = false;
    private boolean isStyOn = false;
    private boolean isLiysOn = false;
    private TextView tvDebugInfo;

    private float resize_ratio;
    private int targetWidth;
    private int targetHeight;

    // style transfer
    private static final String TF_STY_API_MODELS_PATH = "file:///android_asset/style_transfer/";
    private String styModelName = "frozen_inference_graph-la_muse.pb";
    private String styModelPath = "";
    private TensorFlowStyleTransferAPIModel styModel;
    private float styResizeRatio = 0.6f;
    private int styInputWidth;
    private int styInputHeight;

    // post to cloud
    private String postUrlStr = "http://34.80.179.216:5000/LIYS";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        if (useTFLite) {
            labelName = new String[]{"background", "person"};
        } else {
            labelName = new String[]{
                    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
                    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
                    "person", "pottedplant", "sheep", "sofa", "train", "tv"};
        }

        LinearLayout llCameraViewV1 = findViewById(R.id.ll_camera_view_v1);
        llCameraViewV1.setVisibility(View.VISIBLE);

        allLabelsLL = findViewById(R.id.labels);
        tvDebugInfo = findViewById(R.id.tv_debug_info);
        btnSeg = findViewById(R.id.btn_seg);
        btnSty = findViewById(R.id.btn_sty);
        btnLiys = findViewById(R.id.btn_liys);
        COLOR_MAP = COLOR_MAP();
        segClass = new boolean[labelName.length];
        setSegLabels();

        btnSeg.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (isSegOn) {
                    stopSeg();
                } else {
                    startSeg();
                }

            }
        });

        btnSty.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (isStyOn) {
                    stopSty();
                } else {
                    startSty();
                }

            }
        });

        btnLiys.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (isLiysOn) {
                    stopLiys();
                } else {
                    startLiys();
                }
            }
        });

    }

    private Handler segHandler;
    private HandlerThread segHandlerThread;

    private Handler styHandler;
    private HandlerThread styHandlerThread;

    @Override
    public synchronized void onResume() {
        super.onResume();

        segHandlerThread = new HandlerThread("segmentation handler thread");
        segHandlerThread.start();
        segHandler = new Handler(segHandlerThread.getLooper());

        styHandlerThread = new HandlerThread("style transfer handler thread");
        styHandlerThread.start();
        styHandler = new Handler(styHandlerThread.getLooper());
    }

    @Override
    public synchronized void onPause() {
        styHandlerThread.quitSafely();
        segHandlerThread.quitSafely();
        try {
            segHandler.removeCallbacksAndMessages(null);
            styHandler.removeCallbacksAndMessages(null);

            segHandlerThread.join();
            segHandlerThread = null;
            segHandler = null;

            styHandlerThread.join();
            styHandlerThread = null;
            styHandler = null;
        } catch (final InterruptedException e) {
            LOGGER.e(e, "Exception!");
        }

        super.onPause();
    }

    @Override
    public synchronized void onDestroy() {
        super.onDestroy();
    }

    private void startSeg() {
        if (isStyOn) stopSty();
        if (isLiysOn) stopLiys();
        isSegOn = true;
        btnSeg.setText(R.string.btn_stop_segmentation);
        segMapView.setAlpha(0.5f);
        segMapView.setVisibility(View.VISIBLE);
        segMapView.setImageBitmap(null);
        allLabelsLL.setVisibility(View.VISIBLE);
        timestamp = 0;
        setSegLabels();
    }

    private void stopSeg() {
        isSegOn = false;
        tvDebugInfo.setText("");
        btnSeg.setText(R.string.btn_start_segmentation);
        segMapView.setAlpha(1.0f);
        segMapView.setVisibility(View.GONE);
        allLabelsLL.setVisibility(View.GONE);
    }

    private void startSty() {
        if (isSegOn) stopSeg();
        if (isLiysOn) stopLiys();
        isStyOn = true;
        btnSty.setText(R.string.btn_stop_style_transfer);
        segMapView.setAlpha(0.7f);
        segMapView.setVisibility(View.VISIBLE);
        segMapView.setImageBitmap(null);
        timestamp = 0;
    }

    private void stopSty() {
        isStyOn = false;
        tvDebugInfo.setText("");
        btnSty.setText(R.string.btn_start_style_transfer);
        segMapView.setAlpha(1.0f);
    }

    private void startLiys() {
        if (isSegOn) stopSeg();
        if (isStyOn) stopSty();
        isLiysOn = true;
        btnLiys.setText(R.string.btn_stop_liys);
        segMapView.setVisibility(View.VISIBLE);
        segMapView.setImageBitmap(null);
//        allLabelsLL.setVisibility(View.VISIBLE);
        timestamp = 0;
        setSegLabels();
        lastUpdateTime = SystemClock.uptimeMillis();
    }

    private void stopLiys() {
        isLiysOn = false;
        tvDebugInfo.setText("");
        btnLiys.setText(R.string.btn_start_liys);
        segMapView.setVisibility(View.GONE);
        allLabelsLL.setVisibility(View.GONE);
        segHandler.removeCallbacksAndMessages(null);
        styHandler.removeCallbacksAndMessages(null);
    }

    private void setSegLabels() {
        allLabelsLL.removeAllViews();
        LinearLayout llLine = new LinearLayout(getApplicationContext());
        LinearLayout.LayoutParams llLineLP = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT);
        llLineLP.setMargins(4, 2, 4, 2);

        llLine.setOrientation(LinearLayout.HORIZONTAL);
        llLine.setLayoutParams(llLineLP);

        LinearLayout.LayoutParams tvLP = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT);
        tvLP.setMargins(2, 2, 2, 2);

        for (int i = 0; i < labelName.length; i++) {
            TextView labelTV = new TextView(getApplicationContext());
            labelTV.setTextColor(getResources().getColor(R.color.colorWhite));
            labelTV.setBackgroundColor(COLOR_MAP[i]);
            labelTV.setLayoutParams(tvLP);
            labelTV.setPadding(2, 2, 2, 2);
            labelTV.setText(labelName[i]);
            llLine.addView(labelTV);

            if (i % 7 == 6 || i == labelName.length - 1) {
                allLabelsLL.addView(llLine);
                llLine = new LinearLayout(getApplicationContext());
                llLine.setOrientation(LinearLayout.HORIZONTAL);
                llLine.setLayoutParams(llLineLP);
            }

        }
    }

    private int[] COLOR_MAP() {
        int[] colorMap = new int[labelName.length];
        for (int i = 0; i < colorMap.length; i++) {
            colorMap[i] = -16777216 * (i) / colorMap.length;
        }
        return colorMap;
    }

    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        Log.d(TAG, "postUrlStr: " + postUrlStr);

        segMapView = findViewById(R.id.segMap);
        previewWidth = size.getWidth();
        previewHeight = size.getHeight();
        styValues = new int[previewWidth * previewHeight];

        ViewGroup.LayoutParams segMapLP = segMapView.getLayoutParams();
//        segMapLP.width = px2dp(previewWidth);
        segMapLP.height = ImageUtils.px2dp(getApplicationContext(), previewHeight);
        segMapView.setLayoutParams(segMapLP);

        int cropSize = TF_SEG_API_INPUT_SIZE;

        try {
            if (useTFLite) {
                tfLiteSegmentationModel = TFLiteSegmentationModel.create(getAssets(), TFLITE_SEG_MODEL_FILE
                        , TF_SEG_API_INPUT_SIZE, TF_SEG_API_INPUT_SIZE);
//                tfLiteSegmentationModel.useNNAPI();
                tfLiteSegmentationModel.setNumThreads(10);
            } else {
                segmentation = TensorFlowSegmentationAPIModel.create(getAssets(), TF_SEG_API_MODEL_FILE,
                        TF_SEG_API_LABELS_FILE, TF_SEG_API_INPUT_SIZE, TF_SEG_API_INPUT_SIZE);
            }
            cropSize = TF_SEG_API_INPUT_SIZE;
            setStyModel();
        } catch (final IOException e) {
            LOGGER.e("Exception initializing classifier!", e);
            LOGGER.e(e.toString());
            Toast toast =
                    Toast.makeText(
                            getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }

        sensorOrientation = rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        LOGGER.i("crop size %d", cropSize);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

        segMap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

        resize_ratio = 1.0f * TF_SEG_API_INPUT_SIZE /
                (previewWidth > previewHeight ? previewWidth : previewHeight);
        targetHeight = (int) (previewWidth * resize_ratio);
        targetWidth = (int) (previewHeight * resize_ratio);

        LOGGER.i("targetHeight %d", targetHeight);
        LOGGER.i("targetWidth %d", targetWidth);

        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        targetWidth, targetHeight,
                        targetWidth, targetHeight,
                        sensorOrientation, MAINTAIN_ASPECT);

    }

    @Override
    public void onPreviewFrame(byte[] bytes, Camera camera) {
        super.onPreviewFrame(bytes, camera);

        if (isLiysOn) {
//            processImage();
            processImageV2();
//            postImage();

        } else if (isSegOn) {
            segImage();
        } else if (isStyOn) {
            styImage();
        } else {
            readyForNextImage();
        }
    }

    @Override
    public void onImageAvailable(ImageReader reader) {
        super.onImageAvailable(reader);
        //TODO:check camera2 API
        if (isSegOn) {
            processImage();
        } else {
            readyForNextImage();
        }
    }

    @Override
    protected void processImage() {
        ++timestamp;
        final long currTimestamp = timestamp;

        // No mutex needed as this method is not reentrant.
        if (computingDetection) {
            readyForNextImage();
            return;
        }
        computingDetection = true;
        LOGGER.i("Preparing image " + currTimestamp + " for LIYS in bg thread.");

        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);
        final Bitmap adjustedFrameBitmap = adjustFrame(rgbFrameBitmap, sensorOrientation, true);
        final Bitmap inputMap = Bitmap.createScaledBitmap(adjustedFrameBitmap, targetWidth, targetHeight, false);

        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawColor(0xFF808080);
        canvas.drawBitmap(inputMap, new Matrix(), null);
        readyForNextImage();
        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap);
        }

        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {
                        final long processingStartTime = SystemClock.uptimeMillis();
                        Bitmap styInputBmp = Bitmap.createScaledBitmap(adjustedFrameBitmap, styInputWidth, styInputHeight, false);
                        long styStartTime = SystemClock.uptimeMillis();
                        int[] styResults = new int[styInputWidth * styInputHeight];
                        try {
                            styResults = styModel.styleImage(styInputBmp);
                        } catch (Exception e) {
                            e.printStackTrace();
                            Toast.makeText(getApplicationContext(), "error, image size too large or others!"
                                    , Toast.LENGTH_LONG).show();
                        }
                        // style tranfer result
                        Bitmap styRltBmp = Bitmap.createBitmap(styResults, styInputWidth, styInputHeight, Bitmap.Config.ARGB_8888);
                        final Bitmap styOutputBmp = Bitmap.createScaledBitmap(styRltBmp, previewWidth, previewHeight, false);
                        final int[] styRltValues = new int[previewWidth * previewHeight];
                        styOutputBmp.getPixels(styRltValues, 0, styOutputBmp.getWidth(), 0, 0, styOutputBmp.getWidth(), styOutputBmp.getHeight());
                        long styEndTime = SystemClock.uptimeMillis();
                        Log.d(TAG, "style transfer image cost: " + (styEndTime - styStartTime) + "ms");

                        final long startTime = SystemClock.uptimeMillis();
                        final int[] results;

                        if (useTFLite) {
                            results = tfLiteSegmentationModel.segImage(croppedBitmap);
                        } else {
                            results = segmentation.segImage(croppedBitmap);
                        }

                        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
                        LOGGER.d("Running segment one frame cost: " + lastProcessingTimeMs);

                        resetSegClass();
                        final int[] colorRlt = label2color(results);

                        Bitmap segRltBmp = Bitmap.createBitmap(colorRlt, TF_SEG_API_INPUT_SIZE, TF_SEG_API_INPUT_SIZE, Config.ARGB_8888);
                        Bitmap resizedBmp = Bitmap.createBitmap(segRltBmp, 0, 0, targetWidth, targetHeight);
                        segMap = Bitmap.createScaledBitmap(resizedBmp, previewWidth, previewHeight, false);

                        int[] segRltResized = new int[previewWidth * previewHeight];
                        segMap.getPixels(segRltResized, 0, segMap.getWidth(), 0, 0, segMap.getWidth(), segMap.getHeight());

                        for (int i = 0; i < previewHeight; i++) {
                            for (int j = 0; j < previewWidth; j++) {
                                if (segRltResized[i * previewWidth + j] != COLOR_MAP[0]) {
                                    styRltValues[i * previewWidth + j] = Color.TRANSPARENT;
                                }
                            }
                        }
                        lastProcessingTimeMs = SystemClock.uptimeMillis() - processingStartTime;
                        final Bitmap rltBmp = Bitmap.createBitmap(styRltValues, previewWidth, previewHeight, Config.ARGB_8888);

                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                displaySegClass();
                                segMapView.setImageBitmap(rltBmp);
                                LOGGER.d("segment and style transfer one frame cost: " + lastProcessingTimeMs + " ms");
                                if (isLiysOn) {
                                    tvDebugInfo.setText("segment and style transfer one frame cost: " + lastProcessingTimeMs + " ms");
                                }

                            }
                        });

                        computingDetection = false;
                    }
                });
    }


    private long lastUpdateTime;
    private int[] styValues;
    private boolean isFirstStyDone = false;

    /**
     * segment and style transfer with multi thread
     */
    private void processImageV2() {
        ++timestamp;
        final long currTimestamp = timestamp;

        // No mutex needed as this method is not reentrant.
        if (computingDetection) {
            readyForNextImage();
            return;
        }
        computingDetection = true;
        LOGGER.i("Preparing image " + currTimestamp + " for segmentation in bg thread.");

        int[] rgbFrameBytes = getRgbBytes();
        rgbFrameBitmap.setPixels(rgbFrameBytes, 0, previewWidth, 0, 0, previewWidth, previewHeight);

        final Bitmap adjustedFrameBitmap = adjustFrame(rgbFrameBitmap, sensorOrientation, true);

        final Bitmap inputMap = Bitmap.createScaledBitmap(adjustedFrameBitmap, targetWidth, targetHeight, false);

        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawColor(0xFF808080);
        canvas.drawBitmap(inputMap, new Matrix(), null);
        readyForNextImage();
        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap);
        }

        segInBackground(
                new Runnable() {
                    @Override
                    public void run() {
                        Log.e(TAG, Thread.currentThread().getName() + " run: " + SystemClock.uptimeMillis());
                        final long startTime = SystemClock.uptimeMillis();
                        final int[] results;
                        results = tfLiteSegmentationModel.segImage(croppedBitmap);
                        resetSegClass();
                        final int[] colorRlt = label2color(results);

                        Bitmap segRltBmp = Bitmap.createBitmap(colorRlt, TF_SEG_API_INPUT_SIZE, TF_SEG_API_INPUT_SIZE, Config.ARGB_8888);
                        Bitmap resizedBmp = Bitmap.createBitmap(segRltBmp, 0, 0, targetWidth, targetHeight);
                        segMap = Bitmap.createScaledBitmap(resizedBmp, previewWidth, previewHeight, false);

                        int[] segRltResized = new int[previewWidth * previewHeight];
                        segMap.getPixels(segRltResized, 0, segMap.getWidth(), 0, 0, segMap.getWidth(), segMap.getHeight());
//                        synchronized (styValues) {
                        for (int i = 0; i < previewHeight; i++) {
                            for (int j = 0; j < previewWidth; j++) {
                                if (segRltResized[i * previewWidth + j] == COLOR_MAP[0]) {
                                    segRltResized[i * previewWidth + j] = styValues[i * previewWidth + j];
                                } else {
                                    segRltResized[i * previewWidth + j] = Color.TRANSPARENT;
                                }
                            }
                        }
//                        }

                        final Bitmap rltBmp = Bitmap.createBitmap(segRltResized, previewWidth, previewHeight, Config.ARGB_8888);

                        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
                        LOGGER.d("Running segment one frame cost: " + lastProcessingTimeMs);

                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
//                                segMapView.setImageBitmap(segMap);
                                if (isFirstStyDone) {
                                    segMapView.setImageBitmap(rltBmp);
                                }
                                long now = SystemClock.uptimeMillis();
                                LOGGER.d("now: " + now + ", lastUpdateTime: " + lastUpdateTime + " ms");
                                LOGGER.d("segment update: " + (now - lastUpdateTime) + " ms");
                                if (isLiysOn) {
                                    tvDebugInfo.setText("segment update: " + (now - lastUpdateTime) + " ms");
                                }
                                lastUpdateTime = SystemClock.uptimeMillis();

                            }
                        });

                        computingDetection = false;
                    }
                });
        styInBackground(new Runnable() {
            @Override
            public void run() {
                Log.e(TAG, Thread.currentThread().getName() + " run: " + SystemClock.uptimeMillis());
                final long startTime = SystemClock.uptimeMillis();
                Bitmap styInputBmp = Bitmap.createScaledBitmap(adjustedFrameBitmap, styInputWidth, styInputHeight, false);

                int[] styResults = new int[styInputWidth * styInputHeight];
                try {
                    styResults = styModel.styleImage(styInputBmp);
                } catch (Exception e) {
                    e.printStackTrace();
                    Toast.makeText(getApplicationContext(), "error, image size too large or others!"
                            , Toast.LENGTH_LONG).show();
                }
                // style tranfer result
                Bitmap styRltBmp = Bitmap.createBitmap(styResults, styInputWidth, styInputHeight, Bitmap.Config.ARGB_8888);
                final Bitmap styOutputBmp = Bitmap.createScaledBitmap(styRltBmp, previewWidth, previewHeight, false);
//                synchronized (styValues) {
                styOutputBmp.getPixels(styValues, 0, previewWidth, 0, 0, previewWidth, previewHeight);
//                }
                lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
                LOGGER.d("style transfer cost: " + lastProcessingTimeMs + " ms");
                isFirstStyDone = true;

            }
        });
    }

    private synchronized void segInBackground(final Runnable r) {
        if (segHandler != null) {
            segHandler.post(r);
        }
    }

    private synchronized void styInBackground(final Runnable r) {
        if (styHandler != null) {
            styHandler.post(r);
        }
    }

    private void postImage() {
        ++timestamp;
        final long currTimestamp = timestamp;

        // No mutex needed as this method is not reentrant.
        if (computingDetection) {
            readyForNextImage();
            return;
        }
        computingDetection = true;
        LOGGER.i("Preparing image " + currTimestamp + " for LIYS in bg thread.");

        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);
        final Bitmap adjustedFrameBitmap = adjustFrame(rgbFrameBitmap, sensorOrientation, true);
        final Bitmap inputMap = Bitmap.createScaledBitmap(adjustedFrameBitmap, targetWidth, targetHeight, false);

        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawColor(0xFF808080);
        canvas.drawBitmap(inputMap, new Matrix(), null);
        readyForNextImage();
        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap);
        }

        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {
                        final long processingStartTime = SystemClock.uptimeMillis();
                        long encodeStartTime = SystemClock.uptimeMillis();
                        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
                        adjustedFrameBitmap.compress(Bitmap.CompressFormat.JPEG, 100, byteArrayOutputStream);
                        byte[] byteArray = byteArrayOutputStream.toByteArray();
                        String styInputBase64 = Base64.encodeToString(byteArray, Base64.DEFAULT);
                        //Log.d(TAG, "encode cost: " + (SystemClock.uptimeMillis() - encodeStartTime) + "ms");
                        postLIYS(styInputBase64);
                        computingDetection = false;
                    }
                });
    }

    private void segImage() {
        ++timestamp;
        final long currTimestamp = timestamp;

        // No mutex needed as this method is not reentrant.
        if (computingDetection) {
            readyForNextImage();
            return;
        }
        computingDetection = true;
        LOGGER.i("Preparing image " + currTimestamp + " for segmentation in bg thread.");

        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);
        final Bitmap adjustedFrameBitmap = adjustFrame(rgbFrameBitmap, sensorOrientation, true);
        final Bitmap inputMap = Bitmap.createScaledBitmap(adjustedFrameBitmap, targetWidth, targetHeight, false);

        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawColor(0xFF808080);
        canvas.drawBitmap(inputMap, new Matrix(), null);
        readyForNextImage();
        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap);
        }

        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {
                        final long startTime = SystemClock.uptimeMillis();
                        final int[] results;

                        if (useTFLite) {
                            results = tfLiteSegmentationModel.segImage(croppedBitmap);
                        } else {
                            results = segmentation.segImage(croppedBitmap);
                        }

                        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
                        LOGGER.d("Running segment one frame cost: " + lastProcessingTimeMs);

                        resetSegClass();
                        final int[] colorRlt = label2color(results);

                        Bitmap segRltBmp = Bitmap.createBitmap(colorRlt, TF_SEG_API_INPUT_SIZE, TF_SEG_API_INPUT_SIZE, Config.ARGB_8888);
                        Bitmap resizedBmp = Bitmap.createBitmap(segRltBmp, 0, 0, targetWidth, targetHeight);
                        segMap = Bitmap.createScaledBitmap(resizedBmp, previewWidth, previewHeight, false);

                        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                displaySegClass();
                                segMapView.setImageBitmap(segMap);
                                LOGGER.d("segment one frame cost: " + lastProcessingTimeMs + " ms");
                                if (isSegOn) {
                                    tvDebugInfo.setText("segment one frame cost: " + lastProcessingTimeMs + " ms");
                                }

                            }
                        });

                        computingDetection = false;
                    }
                });
    }

    private void styImage() {
        ++timestamp;
        final long currTimestamp = timestamp;

        // No mutex needed as this method is not reentrant.
        if (computingDetection) {
            readyForNextImage();
            return;
        }
        computingDetection = true;
        LOGGER.i("Preparing image " + currTimestamp + " for style transfer in bg thread.");

        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);
        final Bitmap adjustedFrameBitmap = adjustFrame(rgbFrameBitmap, sensorOrientation, true);

        readyForNextImage();
        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(rgbFrameBitmap);
        }

        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {
                        final long startTime = SystemClock.uptimeMillis();
                        Bitmap styInputBmp = Bitmap.createScaledBitmap(adjustedFrameBitmap, styInputWidth, styInputHeight, false);

                        int[] styResults = new int[styInputWidth * styInputHeight];
                        try {
                            styResults = styModel.styleImage(styInputBmp);
                        } catch (Exception e) {
                            e.printStackTrace();
                            Toast.makeText(getApplicationContext(), "error, image size too large or others!"
                                    , Toast.LENGTH_LONG).show();
                        }
                        // style tranfer result
                        Bitmap styRltBmp = Bitmap.createBitmap(styResults, styInputWidth, styInputHeight, Bitmap.Config.ARGB_8888);
                        final Bitmap styOutputBmp = Bitmap.createScaledBitmap(styRltBmp, previewWidth, previewHeight, false);
                        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                segMapView.setImageBitmap(styOutputBmp);
                                LOGGER.d("style transfer one frame cost: " + lastProcessingTimeMs + " ms");
                                if (isStyOn) {
                                    tvDebugInfo.setText("style transfer one frame cost: " + lastProcessingTimeMs + " ms");
                                }


                            }
                        });

                        computingDetection = false;
                    }
                });
    }

    @Override
    protected int getLayoutId() {
        return R.layout.camera_connection_fragment_segmentation;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    private int[] label2color(int[] rltLabel) {
        int[] rltImage = new int[rltLabel.length];

        for (int i = 0; i < rltLabel.length; i++) {
            rltImage[i] = COLOR_MAP[rltLabel[i]];
            segClass[rltLabel[i]] = true;
        }

        return rltImage;
    }

    private void resetSegClass() {
        for (int i = 0; i < segClass.length; i++)
            segClass[i] = false;
    }

    private void displaySegClass() {
        allLabelsLL.removeAllViews();

        int segClassNum = 0;
        for (boolean seg : segClass) {
            if (seg) {
                segClassNum++;
            }
        }

        LinearLayout llLine = new LinearLayout(getApplicationContext());
        LinearLayout.LayoutParams llLineLP = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT);
        llLineLP.setMargins(4, 4, 4, 4);

        llLine.setOrientation(LinearLayout.HORIZONTAL);
        llLine.setLayoutParams(llLineLP);

        LinearLayout.LayoutParams tvLP = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT);
        tvLP.setMargins(4, 4, 4, 4);

        for (int i = 0, j = 0; i < labelName.length; i++) {
            if (!segClass[i]) {
                continue;
            }
            TextView labelTV = new TextView(getApplicationContext());
            labelTV.setTextColor(getResources().getColor(R.color.colorWhite));
            labelTV.setBackgroundColor(COLOR_MAP[i]);
            labelTV.setLayoutParams(tvLP);
            labelTV.setPadding(4, 4, 4, 4);
            labelTV.setText(labelName[i]);
            llLine.addView(labelTV);

            if (j % 5 == 4 || i == labelName.length - 1 || j == segClassNum - 1) {
                allLabelsLL.addView(llLine);
                llLine = new LinearLayout(getApplicationContext());
                llLine.setOrientation(LinearLayout.HORIZONTAL);
                llLine.setLayoutParams(llLineLP);
            }
            j++;

        }
    }

    private void setStyModel() {
        styModelPath = TF_STY_API_MODELS_PATH + styModelName; //default model
        styInputWidth = (int) (previewHeight * styResizeRatio);
        styInputHeight = (int) (previewWidth * styResizeRatio);
        //style transfer model ,width and height must be divisible by 8 .
        styInputWidth = ((styInputWidth / 8) + (styInputWidth % 8 == 0 ? 0 : 1)) * 8;
        styInputHeight = ((styInputHeight / 8) + (styInputHeight % 8 == 0 ? 0 : 1)) * 8;
        Log.d(TAG, "style transfer input image resize ratio: " + styResizeRatio);
        Log.d(TAG, "style transfer input image width: " + styInputWidth + ", height: " + styInputHeight);
        try {
            styModel = TensorFlowStyleTransferAPIModel.create(getAssets(), styModelPath,
                    styInputWidth, styInputHeight);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private Bitmap adjustFrame(Bitmap srcBmp, float rotate, boolean horizontallyFlip) {
        Matrix matrix = new Matrix();
        matrix.postRotate(rotate);
        if (horizontallyFlip) {
            matrix.postScale(-1, 1, srcBmp.getWidth() / 2f, srcBmp.getHeight() / 2f);
        }
        return Bitmap.createBitmap(srcBmp, 0, 0, srcBmp.getWidth(), srcBmp.getHeight(),
                matrix, true);
    }


    private void postLIYS(String styInputBase64) {
        final long postStartTime = SystemClock.uptimeMillis();
        HttpURLConnection conn = null;
        StringBuilder response = new StringBuilder();

        try {
            URL url = new URL(postUrlStr);
            conn = (HttpURLConnection) url.openConnection();
            conn.setRequestProperty("Content-Type", "application/json; charset=UTF-8");
            conn.setRequestProperty("Accept", "application/json");
            conn.setRequestMethod("POST");
            conn.setConnectTimeout(15000);
            conn.setReadTimeout(15000);
            conn.setDoInput(true);
            conn.setDoOutput(true);
            conn.setUseCaches(false);

            OutputStream os = conn.getOutputStream();
            DataOutputStream writer = new DataOutputStream(os);
            JSONObject jsonObj = new JSONObject();
            jsonObj.put("inputs", styInputBase64);
            String jsonString = jsonObj.toString();
            //Log.d(TAG, "request: " + jsonString);
            writer.writeBytes(jsonString);
            writer.flush();
            writer.close();
            os.close();
            //Get Response
            InputStream is = conn.getInputStream();
            BufferedReader reader = new BufferedReader(new InputStreamReader(is));
            String line;
            while ((line = reader.readLine()) != null) {
                response.append(line);
                response.append('\r');
            }
            reader.close();
            //Log.d(TAG, "response: " + response.toString());
            JSONObject responseJSONObj = new JSONObject(response.toString());
            String rltBase64 = responseJSONObj.getString("result");
            //Log.d(TAG, "rltBase64: " + rltBase64);

            long decodeStartTime = SystemClock.uptimeMillis();
            byte[] decodedString = Base64.decode(rltBase64, Base64.URL_SAFE);
            final Bitmap rltImageBmp = BitmapFactory.decodeByteArray(decodedString, 0, decodedString.length);
            Log.d(TAG, "decode cost: " + (SystemClock.uptimeMillis() - decodeStartTime) + "ms");

            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    long endTime = SystemClock.uptimeMillis();
                    segMapView.setImageBitmap(rltImageBmp);
                    LOGGER.d("one frame cose: " + (endTime - postStartTime) + " ms");
                    if (isLiysOn) {
                        tvDebugInfo.setText("one frame cose: " + (endTime - postStartTime) + " ms");
                    }

                }
            });
        } catch (SocketTimeoutException se) {
            se.printStackTrace();
            Toast.makeText(getApplicationContext(), "SocketTimeoutException!"
                    , Toast.LENGTH_SHORT).show();
        } catch (IOException ioe) {
            ioe.printStackTrace();
            Toast.makeText(getApplicationContext(), "IOException!"
                    , Toast.LENGTH_SHORT).show();
        } catch (JSONException je) {
            je.printStackTrace();
            Toast.makeText(getApplicationContext(), "JSONException!"
                    , Toast.LENGTH_SHORT).show();
        } finally {
            if (conn != null) {
                conn.disconnect();
            }
        }

    }


}
