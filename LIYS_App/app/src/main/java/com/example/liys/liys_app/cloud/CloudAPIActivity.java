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

package com.example.liys.liys_app.cloud;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.BitmapFactory;
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
import android.view.KeyEvent;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.HorizontalScrollView;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import com.example.liys.liys_app.CameraActivity;
import com.example.liys.liys_app.R;
import com.example.liys.liys_app.env.ImageUtils;
import com.example.liys.liys_app.env.Logger;

import org.json.JSONArray;
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
 * Segmentation and Style transfer camera frame. post frame to server, then return inference frame.
 */
public class CloudAPIActivity extends CameraActivity implements OnImageAvailableListener {
    private static final Logger LOGGER = new Logger();
    private static final String TAG = "SegmentationActivity";


    private static final int TF_SEG_API_INPUT_SIZE = 513;
    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);

    private static final boolean SAVE_PREVIEW_BITMAP = false;

    private Integer sensorOrientation;
    private long lastProcessingTimeMs;
    private Bitmap rgbFrameBitmap = null;

    private boolean computingDetection = false;

    private long timestamp = 0;

    private HorizontalScrollView hsvCameraViewV2;
    private LinearLayout llStyles;
    private LinearLayout.LayoutParams styleBtnLP;

    private EditText etIPAddr;
    private ImageView segMapView;
    private TextView tvDebugInfo;

    private int targetWidth;
    private int targetHeight;

    HandlerThread handlerThread;
    Handler handler;

    // post to cloud
    private String ipAddr = "http://34.80.179.216:5000";
    private String postUrlStr = "/LIYS";
    private final static String INIT_STYLE_MODEL_URL = "/initStyleModel";
    private final static String LIYS_URL = "/LIYS";
    private final static String GET_STYLE_URL = "/getStyleInfo";
    private final static String SEG_URL = "/segmentation";

    private static final String NONE_STYLE = "NONE_STYLE";
    private static final String SEG_STYLE = "SEG_STYLE";
    private String[] styles;
    private String currStyle = NONE_STYLE;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        hsvCameraViewV2 = findViewById(R.id.hsv_camera_view_v2);
        hsvCameraViewV2.setVisibility(View.VISIBLE);
        etIPAddr = findViewById(R.id.et_ip_address);
        etIPAddr.setVisibility(View.VISIBLE);
        llStyles = findViewById(R.id.ll_style_images);
        tvDebugInfo = findViewById(R.id.tv_debug_info);

        etIPAddr.setText(ipAddr);
        etIPAddr.setOnKeyListener(new View.OnKeyListener() {
            public boolean onKey(View v, int keyCode, KeyEvent event) {
                if ((event.getAction() == KeyEvent.ACTION_DOWN) && (keyCode == KeyEvent.KEYCODE_ENTER)) {
                    ipAddr = etIPAddr.getText().toString();
                    getStyleInfo();

                    Log.d(TAG, "ipAddr: " + ipAddr);
                    Toast.makeText(CloudAPIActivity.this, "ipAddr: " + ipAddr
                            , Toast.LENGTH_SHORT).show();

                    return true;
                }
                return false;
            }
        });

        handlerThread = new HandlerThread("network get thread");
        handlerThread.start();
        handler = new Handler(handlerThread.getLooper());

        styleBtnLP = new LinearLayout.LayoutParams(
                ImageUtils.px2dp(this, 80), ImageUtils.px2dp(this, 80));
        styleBtnLP.setMargins(ImageUtils.px2dp(this, 2), ImageUtils.px2dp(this, 2)
                , ImageUtils.px2dp(this, 2), ImageUtils.px2dp(this, 2));

        Button origBtn = new Button(this);
        origBtn.setLayoutParams(styleBtnLP);
        origBtn.setText("None");
        origBtn.setAllCaps(false);
        origBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                postUrlStr = "";
                currStyle = NONE_STYLE;
                setStyle(currStyle);
            }
        });
        llStyles.addView(origBtn);

        Button segBtn = new Button(this);
        segBtn.setLayoutParams(styleBtnLP);
        segBtn.setText("Seg");
        segBtn.setAllCaps(false);
        segBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                currStyle = SEG_STYLE;
                postUrlStr = SEG_URL;
                segMapView.setVisibility(View.VISIBLE);
                setStyle(currStyle);
            }
        });
        llStyles.addView(segBtn);
        getStyleInfo();

    }

    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        Log.d(TAG, "postUrlStr: " + postUrlStr);

        segMapView = findViewById(R.id.segMap);
        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        ViewGroup.LayoutParams segMapLP = segMapView.getLayoutParams();
//        segMapLP.width = px2dp(previewWidth);
        segMapLP.height = ImageUtils.px2dp(getApplicationContext(), previewHeight);
        segMapView.setLayoutParams(segMapLP);

        sensorOrientation = rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        LOGGER.i("crop size %d", TF_SEG_API_INPUT_SIZE);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
        float resize_ratio = 1.0f * TF_SEG_API_INPUT_SIZE /
                (previewWidth > previewHeight ? previewWidth : previewHeight);
        targetHeight = (int) (previewWidth * resize_ratio);
        targetWidth = (int) (previewHeight * resize_ratio);

        LOGGER.i("targetHeight %d", targetHeight);
        LOGGER.i("targetWidth %d", targetWidth);


    }

    @Override
    public void onPreviewFrame(byte[] bytes, Camera camera) {
        super.onPreviewFrame(bytes, camera);

        if (!currStyle.equals(NONE_STYLE)) {
//            processImage();
            postImage();

        } else {
            readyForNextImage();
        }
    }

    @Override
    public void onImageAvailable(ImageReader reader) {
        super.onImageAvailable(reader);
        //TODO:check camera2 API
    }

    @Override
    protected void processImage() {
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

        readyForNextImage();
        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(inputMap);
        }

        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {
                        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
                        inputMap.compress(Bitmap.CompressFormat.JPEG, 100, byteArrayOutputStream);
                        byte[] byteArray = byteArrayOutputStream.toByteArray();
                        String styInputBase64 = Base64.encodeToString(byteArray, Base64.DEFAULT);
                        postLIYS(styInputBase64);
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
        HttpURLConnection conn = null;
        StringBuilder response = new StringBuilder();

        try {
            URL url = new URL(ipAddr + postUrlStr);
            Log.d(TAG, "post url: " + ipAddr + postUrlStr);
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
            jsonObj.put("style", currStyle);
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

            byte[] decodedString = Base64.decode(rltBase64, Base64.URL_SAFE);
            final Bitmap rltImageBmp = BitmapFactory.decodeByteArray(decodedString, 0, decodedString.length);
            final Bitmap scaledBmp = Bitmap.createScaledBitmap(rltImageBmp, previewWidth, previewHeight, false);

            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    long endTime = SystemClock.uptimeMillis();
                    segMapView.setImageBitmap(scaledBmp);
//                    segMapView.setImageBitmap(rltImageBmp);

                    LOGGER.d("one frame cost: " + (endTime - lastProcessingTimeMs) + " ms");
                    if (!currStyle.equals(NONE_STYLE)) {
                        tvDebugInfo.setText("one frame cost: " + (endTime - lastProcessingTimeMs) + " ms");
                    }
                    lastProcessingTimeMs = SystemClock.uptimeMillis();

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

    private void getStyleInfo() {
        handler.removeCallbacksAndMessages(null);

        final int childCount = llStyles.getChildCount();
        llStyles.removeViews(2, childCount - 2);

        handler.post(new Runnable() {
            public void run() {
                HttpURLConnection conn = null;
                StringBuilder response = new StringBuilder();

                try {
                    String urlStr = ipAddr + GET_STYLE_URL;
                    Log.d(TAG, "urlStr: " + urlStr);
                    URL url = new URL(urlStr);
                    conn = (HttpURLConnection) url.openConnection();
                    conn.setRequestMethod("GET");
                    conn.setConnectTimeout(2000);
                    conn.setReadTimeout(10000);
                    conn.connect();

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
                    JSONArray stylesString = responseJSONObj.getJSONArray("styles");
                    JSONArray imagesString = responseJSONObj.getJSONArray("images");
                    int styleNum = stylesString.length();
                    styles = new String[styleNum];
                    for (int i = 0; i < styleNum; i++) {
                        final String style = stylesString.getString(i);
                        styles[i] = style;
                        String styleImage = imagesString.getString(i);
                        final Bitmap styleBmp = decodeBase64toBmp(styleImage);
                        final Bitmap scaledBmp = Bitmap.createScaledBitmap(styleBmp, 80, 80, false);
                        //Log.d(TAG, style + ":" + styleImage);

                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                ImageButton ibImage = new ImageButton(CloudAPIActivity.this);
                                ibImage.setImageBitmap(scaledBmp);
                                ibImage.setLayoutParams(styleBtnLP);
                                ibImage.setBackgroundColor(Color.TRANSPARENT);
                                ibImage.setPadding(ImageUtils.px2dp(CloudAPIActivity.this, 2), ImageUtils.px2dp(CloudAPIActivity.this, 2)
                                        , ImageUtils.px2dp(CloudAPIActivity.this, 2), ImageUtils.px2dp(CloudAPIActivity.this, 2));
                                ibImage.setAdjustViewBounds(true);
                                ibImage.setOnClickListener(new View.OnClickListener() {
                                    @Override
                                    public void onClick(View v) {
                                        Log.d(TAG, "style: " + style);
                                        postUrlStr = LIYS_URL;
                                        setStyle(style);
                                    }
                                });
                                llStyles.addView(ibImage);
                            }
                        });
                    }

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
        });

    }

    private void setStyle(String style) {
        currStyle = style;
        if (style.equals(NONE_STYLE)) {
            tvDebugInfo.setText("");
            segMapView.setVisibility(View.GONE);
            segMapView.setImageBitmap(null);
        } else if (style.equals(SEG_STYLE)) {
            segMapView.setVisibility(View.VISIBLE);
        } else {
            initStyleModel();
            segMapView.setVisibility(View.VISIBLE);
        }
        timestamp = 0;
        Toast.makeText(getApplicationContext(), "current style :" + currStyle
                , Toast.LENGTH_SHORT).show();
    }

    private void initStyleModel() {
        handler.post(new Runnable() {
            public void run() {
                HttpURLConnection conn = null;
                StringBuilder response = new StringBuilder();

                try {
                    URL url = new URL(ipAddr + INIT_STYLE_MODEL_URL);
                    Log.d(TAG, "init style model url: " + ipAddr + INIT_STYLE_MODEL_URL);
                    conn = (HttpURLConnection) url.openConnection();
                    conn.setRequestProperty("Content-Type", "application/json; charset=UTF-8");
                    conn.setRequestProperty("Accept", "application/json");
                    conn.setRequestMethod("POST");
                    conn.setConnectTimeout(5000);
                    conn.setReadTimeout(5000);
                    conn.setDoInput(true);
                    conn.setDoOutput(true);
                    conn.setUseCaches(false);

                    OutputStream os = conn.getOutputStream();
                    DataOutputStream writer = new DataOutputStream(os);
                    JSONObject jsonObj = new JSONObject();
                    jsonObj.put("style", currStyle);
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
                    Log.d(TAG, "response: " + response.toString());
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
        });
    }


    private Bitmap decodeBase64toBmp(String base64Str) {
        byte[] decodedString = Base64.decode(base64Str, Base64.URL_SAFE);
        return BitmapFactory.decodeByteArray(decodedString, 0, decodedString.length);
    }

}
