package com.example.liys.liys_app.liysimage;

import android.app.AlertDialog;
import android.app.ProgressDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.res.AssetManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Base64;
import android.util.Log;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import com.example.liys.liys_app.R;
import com.example.liys.liys_app.env.ImageUtils;
import com.example.liys.liys_app.segmentation.TFLiteSegmentationModel;
import com.example.liys.liys_app.styletransfer.TensorFlowStyleTransferAPIModel;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.SocketTimeoutException;
import java.net.URL;


public class LIYSImageActivity extends AppCompatActivity {

    private static final String TAG = "LIYSImageActivity";
    private static final boolean DEBUG = true;

    private static final int REQUEST_IMAGE_INPUT = 1;
    private static final int REQUEST_IMAGE_CAPTURE = 2;

    //Style transfer
    private static final String TF_STY_API_MODELS_PATH = "file:///android_asset/style_transfer/";
    private String styModelName = "frozen_inference_graph-la_muse.pb";
    private String styModelPath = "";
    private int styleNum;
    private static final String[] styles = new String[]{
            "la_muse", "rain_princess", "scream", "wreck"
//            , "udnie", "wave"//TODO:fix model
    };

    private static final int MAX_LEN = 2048;
    private TensorFlowStyleTransferAPIModel styModel;
    private float styResizeRatio = 1.0f;

    //Segmentation
    private TFLiteSegmentationModel tfLiteSegmentationModel;
    private static final int TF_SEG_API_INPUT_SIZE = 513;
    private static final String TFLITE_SEG_MODEL_FILE =
//            "file:///android_asset/seg_tflite.tflite";
            "file:///android_asset/deeplab.tflite";
    //            "file:///android_asset/deeplab_dm30.tflite";
    private static final int BG_COLOR = Color.TRANSPARENT;
    private static final int PERSON_COLOR = Color.RED;

    //post image
    private static final boolean TWO_POST_REQ = false;
    private String postUrlStr = "http://10.17.6.148:5000/LIYS";
    private int[] rspSegValues;
    private int[] rspStyValues;


    private ImageView ivRltImage;
    private Bitmap origImageBmp;
    private Bitmap styInputBmp;
    private Bitmap rltImageBmp;
    private LinearLayout llStyles;
    private LinearLayout.LayoutParams styleBtnLP;

    private Spinner spResizeRatio;
    ArrayAdapter<CharSequence> resizeRatioAdapter;
    private TextView tvImageInfo;

    private Uri origImageUri;
    private int origWidth;
    private int origHeight;
    private int inputWidth;
    private int inputHeight;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_liysimage);
        ivRltImage = findViewById(R.id.iv_rlt_image);
        llStyles = findViewById(R.id.ll_style_images);

        spResizeRatio = findViewById(R.id.sp_resize_ratio);
        resizeRatioAdapter = ArrayAdapter.createFromResource(this,
                R.array.resize_ratio_array, android.R.layout.simple_spinner_item);
        resizeRatioAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spResizeRatio.setAdapter(resizeRatioAdapter);
        styResizeRatio = Float.valueOf(spResizeRatio.getSelectedItem().toString());
        spResizeRatio.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                styResizeRatio = Float.valueOf(parent.getItemAtPosition(position).toString());

                try {
                    if (origImageUri == null) return;
                    final InputStream imageStream = getContentResolver().openInputStream(origImageUri);
                    final Bitmap selectedImage = BitmapFactory.decodeStream(imageStream);
                    setInputBmp(selectedImage);
                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                }

            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {

            }
        });


        tvImageInfo = findViewById(R.id.tv_image_info);
        if (DEBUG) {
            tvImageInfo.setVisibility(View.VISIBLE);
        }

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
                ivRltImage.setImageBitmap(origImageBmp);
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
                segImage();
            }
        });
        llStyles.addView(segBtn);

        Button postBtn = new Button(this);
        postBtn.setLayoutParams(styleBtnLP);
        postBtn.setText("Post");
        postBtn.setAllCaps(false);
        postBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (styInputBmp == null) {
                    Toast.makeText(getApplicationContext(), "choose original image"
                            , Toast.LENGTH_LONG).show();
                    return;
                }

                final long startTime = SystemClock.uptimeMillis();
                final ProgressDialog dialog = ProgressDialog.show(LIYSImageActivity.this, "",
                        "Loading. Please wait...", true);

                final Object request1Lock = new Object();
                final Object request2Lock = new Object();

                HandlerThread handlerThread = new HandlerThread("network thread");
                handlerThread.start();
                Handler handler = new Handler(handlerThread.getLooper());

                if (TWO_POST_REQ) {
                    rspSegValues = new int[inputWidth * inputHeight];
                    rspStyValues = new int[inputWidth * inputHeight];
                }

                handler.post(new Runnable() {
                    public void run() {
                        synchronized (request1Lock) {
                            if (TWO_POST_REQ) {
                                long encodeStartTime = SystemClock.uptimeMillis();
                                ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
                                styInputBmp.compress(Bitmap.CompressFormat.JPEG, 100, byteArrayOutputStream);
                                byte[] byteArray = byteArrayOutputStream.toByteArray();
                                final String styInputBase64 = Base64.encodeToString(byteArray, Base64.DEFAULT);
                                Log.d(TAG, "encode cost: " + (SystemClock.uptimeMillis() - encodeStartTime) + "ms");
                                postSeg(styInputBase64);
                            } else {
                                postLIYS();
                            }

                            Log.d(TAG, Thread.currentThread().getName() + " end!");
                        }
                    }
                });

                HandlerThread handlerThread2 = new HandlerThread("network thread 2");
                handlerThread2.start();
                Handler handler2 = new Handler(handlerThread2.getLooper());

                handler2.post(new Runnable() {
                    public void run() {
                        synchronized (request2Lock) {
                            if (TWO_POST_REQ) {
                                long encodeStartTime = SystemClock.uptimeMillis();
                                ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
                                styInputBmp.compress(Bitmap.CompressFormat.JPEG, 100, byteArrayOutputStream);
                                byte[] byteArray = byteArrayOutputStream.toByteArray();
                                final String styInputBase64 = Base64.encodeToString(byteArray, Base64.DEFAULT);
                                Log.d(TAG, "encode cost: " + (SystemClock.uptimeMillis() - encodeStartTime) + "ms");
                                postSty(styInputBase64);
                            }

                            Log.d(TAG, Thread.currentThread().getName() + " done!");
                        }
                    }
                });

                new Thread(new Runnable() {
                    @Override
                    public void run() {
                        Log.d(TAG, "wait thread run: ");
                        synchronized (request1Lock) {
                            synchronized (request2Lock) {
                                if (TWO_POST_REQ) {
                                    long mergeStartTime = SystemClock.uptimeMillis();

                                    int[] origImageValues = new int[inputWidth * inputHeight];
                                    styInputBmp.getPixels(origImageValues, 0, styInputBmp.getWidth(), 0, 0, styInputBmp.getWidth(), styInputBmp.getHeight());

                                    for (int i = 0; i < inputHeight; i++) {
                                        for (int j = 0; j < inputWidth; j++) {
                                            if (rspSegValues[i * inputWidth + j] == Color.BLACK) {
                                                origImageValues[i * inputWidth + j] = rspStyValues[i * inputWidth + j];
                                            }
                                        }
                                    }
                                    rltImageBmp = Bitmap.createBitmap(origImageValues, inputWidth, inputHeight
                                            , Bitmap.Config.ARGB_8888);
                                    ivRltImage.setImageBitmap(rltImageBmp);
                                    Log.d(TAG, "merge image cost: " + (SystemClock.uptimeMillis() - mergeStartTime) + "ms");
                                    Log.d(TAG, "total cost: " + (SystemClock.uptimeMillis() - startTime) + "ms");
                                }
                                Log.d(TAG, "all done, dialog dismiss!");
                                dialog.dismiss();
                            }

                        }
                    }
                }).start();

            }
        });
        llStyles.addView(postBtn);

        styleNum = styles.length;
        createStyleImageBtn(this);

        styModelPath = TF_STY_API_MODELS_PATH + styModelName; //default model

    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.menu_liys_image, menu);
        // return true so that the menu pop up is opened
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        int id = item.getItemId();
        if (id == R.id.menu_take_picture) {
            dispatchTakePictureIntent();
        } else if (id == R.id.menu_input_orig_image) {
            inputImageIntent();
        } else if (id == R.id.menu_set_post_url) {
            AlertDialog.Builder alertDialog = new AlertDialog.Builder(LIYSImageActivity.this);
            alertDialog.setTitle("URL");
            final EditText input = new EditText(LIYSImageActivity.this);
            LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    LinearLayout.LayoutParams.MATCH_PARENT);
            input.setLayoutParams(lp);
            input.setText(postUrlStr);
            alertDialog.setView(input);

            alertDialog.setPositiveButton("YES",
                    new DialogInterface.OnClickListener() {
                        public void onClick(DialogInterface dialog, int which) {
                            postUrlStr = input.getText().toString();
                        }
                    });

            alertDialog.setNegativeButton("NO",
                    new DialogInterface.OnClickListener() {
                        public void onClick(DialogInterface dialog, int which) {
                            dialog.cancel();
                        }
                    });

            alertDialog.show();
        }

        return super.onOptionsItemSelected(item);
    }

    private void dispatchTakePictureIntent() {
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
            startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE);
        }
    }

    private void inputImageIntent() {
        Intent photoPickerIntent = new Intent(Intent.ACTION_PICK
                , MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(photoPickerIntent, REQUEST_IMAGE_INPUT);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
            Bundle extras = data.getExtras();
            Bitmap origBmp = (Bitmap) extras.get("data");
            Log.d(TAG, "take picture image width: " + origBmp.getWidth() +
                    ", height: " + origBmp.getHeight());
            setInputBmp(origBmp);

        } else if (requestCode == REQUEST_IMAGE_INPUT && resultCode == RESULT_OK) {
            try {
                origImageUri = data.getData();
                final InputStream imageStream = getContentResolver().openInputStream(origImageUri);
                final Bitmap selectedImage = BitmapFactory.decodeStream(imageStream);
//                Log.d(TAG, "input image width: " + selectedImage.getWidth() +
//                        ", height: " + selectedImage.getHeight());

                int orientation = getOrientation(this, origImageUri);
                Log.d(TAG, "orientation: " + orientation);
                Bitmap rotatedBitmap = orientation == -1 ?
                        selectedImage : ImageUtils.rotateBitmap(selectedImage, orientation);

                setInputBmp(rotatedBitmap);
            } catch (FileNotFoundException e) {
                e.printStackTrace();
                Toast.makeText(LIYSImageActivity.this, "Something went wrong", Toast.LENGTH_LONG).show();
            }
        }
    }

    private void setInputBmp(Bitmap origBmp) {
        Log.d(TAG, "orig image width: " + origBmp.getWidth() + ", height: " + origBmp.getHeight());

        origWidth = origBmp.getWidth();
        origHeight = origBmp.getHeight();
        setInputSize();
        styInputBmp = Bitmap.createScaledBitmap(origBmp, inputWidth, inputHeight, false);
        origImageBmp = styInputBmp;
        ivRltImage.setImageBitmap(origImageBmp);
    }


    private void createStyleImageBtn(Context context) {
        for (int i = 0; i < styleNum; ++i) {
//            Log.d(TAG, "Creating item " + i);
            String filePath = "style_thumbnails/" + styles[i] + ".jpg";
            final Bitmap bm = getBitmapFromAsset(context, filePath);

            ImageButton ibImage = new ImageButton(this);
            final String styleImageKey = styles[i];
            ibImage.setImageBitmap(bm);
            ibImage.setLayoutParams(styleBtnLP);
            ibImage.setBackgroundColor(Color.TRANSPARENT);
            ibImage.setPadding(ImageUtils.px2dp(this, 2), ImageUtils.px2dp(this, 2)
                    , ImageUtils.px2dp(this, 2), ImageUtils.px2dp(this, 2));
            ibImage.setAdjustViewBounds(true);
            ibImage.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    styModelPath = TF_STY_API_MODELS_PATH + "frozen_inference_graph-" + styleImageKey + ".pb";
                    transformImage();
                }
            });
            llStyles.addView(ibImage);
        }
    }

    private Bitmap getBitmapFromAsset(final Context context, final String filePath) {
        final AssetManager assetManager = context.getAssets();

        Bitmap bitmap = null;
        try {
            final InputStream inputStream = assetManager.open(filePath);
            bitmap = BitmapFactory.decodeStream(inputStream);
        } catch (final IOException e) {
            Log.e(TAG, "Error opening bitmap!", e);
        }

        return Bitmap.createScaledBitmap(bitmap, 100, 100, false);
    }

    private void setStyModel() {
        try {
            styModel = TensorFlowStyleTransferAPIModel.create(getAssets(), styModelPath,
                    inputWidth, inputHeight);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void setSegModel() {
        try {
            tfLiteSegmentationModel = TFLiteSegmentationModel.create(getAssets(), TFLITE_SEG_MODEL_FILE
                    , TF_SEG_API_INPUT_SIZE, TF_SEG_API_INPUT_SIZE);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void setInputSize() {
        Log.d(TAG, "resize ratio: " + styResizeRatio);

        inputWidth = (int) (origWidth * styResizeRatio);
        inputHeight = (int) (origHeight * styResizeRatio);

        if (inputWidth > MAX_LEN || inputHeight > MAX_LEN) {
            Log.d(TAG, "image size too large, resize to" + MAX_LEN);
            float resizeRatio = (float) MAX_LEN / (inputWidth > inputHeight ? inputWidth : inputHeight);
            inputWidth = (int) (inputWidth * resizeRatio);
            inputHeight = (int) (inputHeight * resizeRatio);
        }

        //style transfer model ,width and height must be divisible by 8 .
        inputWidth = ((inputWidth / 8) + (inputWidth % 8 == 0 ? 0 : 1)) * 8;
        inputHeight = ((inputHeight / 8) + (inputHeight % 8 == 0 ? 0 : 1)) * 8;
        Log.d(TAG, "style transfer input image width: " + inputWidth + ", height: " + inputHeight);

        tvImageInfo.setText("original image width: " + origWidth + ", height: " + origHeight +
                "\nstyle transfer input image width: " + inputWidth + ", height: " + inputHeight);
    }

    private int[] label2color(int[] rltLabel) {
        int[] COLOR_MAP = new int[]{BG_COLOR, PERSON_COLOR};
        int[] rltImage = new int[rltLabel.length];
        for (int i = 0; i < rltLabel.length; i++) {
            rltImage[i] = COLOR_MAP[rltLabel[i]];
        }
        return rltImage;
    }

    private int getOrientation(Context context, Uri photoUri) {
        /* it's on the external media. */
        Cursor cursor = null;
        try {
            cursor = context.getContentResolver().query(photoUri,
                    new String[]{MediaStore.Images.ImageColumns.ORIENTATION}, null, null, null);

            if (cursor.getCount() != 1) {
                return -1;
            }

            cursor.moveToFirst();
            return cursor.getInt(0);
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                if (cursor != null) {
                    cursor.close();
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        return -1;
    }

    private void transformImage() {
        if (styInputBmp == null) {
            Toast.makeText(getApplicationContext(), "choose original image"
                    , Toast.LENGTH_LONG).show();
            return;
        }
        final ProgressDialog dialog = ProgressDialog.show(LIYSImageActivity.this, "",
                "Loading. Please wait...", true);
        Handler handler = new Handler();
        handler.postDelayed(new Runnable() {
            public void run() {
                long transformStartTime = SystemClock.uptimeMillis();
                int[] origImageValues = new int[inputWidth * inputHeight];
                styInputBmp.getPixels(origImageValues, 0, styInputBmp.getWidth(), 0, 0, styInputBmp.getWidth(), styInputBmp.getHeight());

                // style transfer
                setStyModel();
                long startTime = SystemClock.uptimeMillis();
                int[] results = new int[inputWidth * inputHeight];
                try {
                    results = styModel.styleImage(styInputBmp);
                } catch (Exception e) {
                    e.printStackTrace();
                    Toast.makeText(getApplicationContext(), "error, image size too large or others!"
                            , Toast.LENGTH_LONG).show();
                }
                // style transfer result
                //Bitmap outputBmp = Bitmap.createBitmap(results, inputWidth, inputHeight, Bitmap.Config.ARGB_8888);
                //ivRltImage.setImageBitmap(outputBmp);
                long endTime = SystemClock.uptimeMillis();
                Log.d(TAG, "style transfer image cost: " + (endTime - startTime) + "ms");
//                        Toast.makeText(getApplicationContext(), "style transfer image cost: "
//                                + (endTime - startTime) + "ms", Toast.LENGTH_SHORT).show();
                styModel.close();

                // segmentation
                setSegModel();
                startTime = SystemClock.uptimeMillis();
                float resize_ratio = 1.0f * TF_SEG_API_INPUT_SIZE /
                        (inputWidth > inputHeight ? inputWidth : inputHeight);
                int targetWidth = (int) (inputWidth * resize_ratio);
                int targetHeight = (int) (inputHeight * resize_ratio);


                final Bitmap inputMap = Bitmap.createScaledBitmap(styInputBmp, targetWidth, targetHeight, false);
                Bitmap croppedBitmap = Bitmap.createBitmap(TF_SEG_API_INPUT_SIZE, TF_SEG_API_INPUT_SIZE, Bitmap.Config.ARGB_8888);

                final Canvas canvas = new Canvas(croppedBitmap);
                canvas.drawColor(0xFF808080);
                canvas.drawBitmap(inputMap, new Matrix(), null);
                int[] segResults = tfLiteSegmentationModel.segImage(croppedBitmap);

                int[] colorRlt = label2color(segResults);
                Bitmap segRltBmp = Bitmap.createBitmap(colorRlt, TF_SEG_API_INPUT_SIZE, TF_SEG_API_INPUT_SIZE, Bitmap.Config.ARGB_8888);
                Bitmap resizedBmp = Bitmap.createBitmap(segRltBmp, 0, 0, targetWidth, targetHeight);
                Bitmap segMap = Bitmap.createScaledBitmap(resizedBmp, inputWidth, inputHeight, false);
                //ivRltImage.setImageBitmap(segMap); //segmentation result
                int[] segRltResized = new int[inputWidth * inputHeight];
                segMap.getPixels(segRltResized, 0, segMap.getWidth(), 0, 0, segMap.getWidth(), segMap.getHeight());

                for (int i = 0; i < inputHeight; i++) {
                    for (int j = 0; j < inputWidth; j++) {
                        if (segRltResized[i * inputWidth + j] != BG_COLOR) {
                            results[i * inputWidth + j] = origImageValues[i * inputWidth + j];
                        }
                    }
                }
                rltImageBmp = Bitmap.createBitmap(results, inputWidth, inputHeight
                        , Bitmap.Config.ARGB_8888);
                ivRltImage.setImageBitmap(rltImageBmp);

                endTime = SystemClock.uptimeMillis();
                Log.d(TAG, "segment image cost: " + (endTime - startTime) + "ms");
//                        Toast.makeText(getApplicationContext(), "segment image cost: "
//                                + (endTime - startTime) + "ms", Toast.LENGTH_SHORT).show();

                tfLiteSegmentationModel.close();

                long transformEndTime = SystemClock.uptimeMillis();
                Log.d(TAG, "transform image cost: " + (transformEndTime - transformStartTime) + "ms");
                Toast.makeText(getApplicationContext(), "transform image cost: "
                        + (transformEndTime - transformStartTime) + "ms", Toast.LENGTH_SHORT).show();
                dialog.dismiss();
            }
        }, 50);
    }


    private void segImage() {
        if (styInputBmp == null) {
            Toast.makeText(getApplicationContext(), "choose original image"
                    , Toast.LENGTH_LONG).show();
            return;
        }
        final ProgressDialog dialog = ProgressDialog.show(LIYSImageActivity.this, "",
                "Loading. Please wait...", true);
        Handler handler = new Handler();
        handler.postDelayed(new Runnable() {
            public void run() {
                int[] origImageValues = new int[inputWidth * inputHeight];
                styInputBmp.getPixels(origImageValues, 0, styInputBmp.getWidth(), 0, 0, styInputBmp.getWidth(), styInputBmp.getHeight());

                // segmentation
                setSegModel();
                long startTime = SystemClock.uptimeMillis();
                float resize_ratio = 1.0f * TF_SEG_API_INPUT_SIZE /
                        (inputWidth > inputHeight ? inputWidth : inputHeight);
                int targetWidth = (int) (inputWidth * resize_ratio);
                int targetHeight = (int) (inputHeight * resize_ratio);

                final Bitmap inputMap = Bitmap.createScaledBitmap(styInputBmp, targetWidth, targetHeight, false);
                Bitmap croppedBitmap = Bitmap.createBitmap(TF_SEG_API_INPUT_SIZE, TF_SEG_API_INPUT_SIZE, Bitmap.Config.ARGB_8888);

                final Canvas canvas = new Canvas(croppedBitmap);
                canvas.drawColor(0xFF808080);
                canvas.drawBitmap(inputMap, new Matrix(), null);
                int[] segResults = tfLiteSegmentationModel.segImage(croppedBitmap);

                int[] colorRlt = label2color(segResults);
                Bitmap segRltBmp = Bitmap.createBitmap(colorRlt, TF_SEG_API_INPUT_SIZE, TF_SEG_API_INPUT_SIZE, Bitmap.Config.ARGB_8888);
                Bitmap resizedBmp = Bitmap.createBitmap(segRltBmp, 0, 0, targetWidth, targetHeight);
                Bitmap segMap = Bitmap.createScaledBitmap(resizedBmp, inputWidth, inputHeight, false);
                int[] segRltResized = new int[inputWidth * inputHeight];
                segMap.getPixels(segRltResized, 0, segMap.getWidth(), 0, 0, segMap.getWidth(), segMap.getHeight());

                for (int i = 0; i < inputHeight; i++) {
                    for (int j = 0; j < inputWidth; j++) {
                        if (segRltResized[i * inputWidth + j] == BG_COLOR) {
                            origImageValues[i * inputWidth + j] = Color.TRANSPARENT;
                        }
                    }
                }
                rltImageBmp = Bitmap.createBitmap(origImageValues, inputWidth, inputHeight
                        , Bitmap.Config.ARGB_8888);
                ivRltImage.setImageBitmap(rltImageBmp);

                long endTime = SystemClock.uptimeMillis();
                Log.d(TAG, "segment image cost: " + (endTime - startTime) + "ms");

                tfLiteSegmentationModel.close();

                Toast.makeText(getApplicationContext(), "segment image cost: "
                        + (endTime - startTime) + "ms", Toast.LENGTH_SHORT).show();
                dialog.dismiss();
            }
        }, 50);
    }

    private void postLIYS() {
        Log.d(TAG, Thread.currentThread().getName() + " run: ");
        final long startTime = SystemClock.uptimeMillis();
        HttpURLConnection conn = null;
        StringBuilder response = new StringBuilder();

        Log.d(TAG, "postUrlStr: " + postUrlStr);
        long encodeStartTime = SystemClock.uptimeMillis();
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        styInputBmp.compress(Bitmap.CompressFormat.JPEG, 100, byteArrayOutputStream);
        byte[] byteArray = byteArrayOutputStream.toByteArray();
        String styInputBase64 = Base64.encodeToString(byteArray, Base64.DEFAULT);
        Log.d(TAG, "encode cost: " + (SystemClock.uptimeMillis() - encodeStartTime) + "ms");


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
                    Log.d(TAG, "total cost: " + (endTime - startTime) + "ms");
                    Toast.makeText(getApplicationContext(), "cost: " + (endTime - startTime) + "ms"
                            , Toast.LENGTH_SHORT).show();
                    ivRltImage.setImageBitmap(rltImageBmp);

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

    private void postSeg(String base64Image) {
        Log.d(TAG, Thread.currentThread().getName() + " run: ");
        final long startTime = SystemClock.uptimeMillis();
        HttpURLConnection conn = null;
        StringBuilder response = new StringBuilder();
        String postSegUrlStr = postUrlStr.substring(0, postUrlStr.lastIndexOf('/') + 1) + "segmentation";
        Log.d(TAG, "postSegUrlStr: " + postSegUrlStr);

        try {
            URL url = new URL(postSegUrlStr);
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
            jsonObj.put("inputs", base64Image);
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

            //long decodeStartTime = SystemClock.uptimeMillis();
            byte[] decodedString = Base64.decode(rltBase64, Base64.URL_SAFE);
            final Bitmap rltImageBmp = BitmapFactory.decodeByteArray(decodedString, 0, decodedString.length);
            //Log.d(TAG, "decode cost: " + (SystemClock.uptimeMillis() - decodeStartTime) + "ms");

            Bitmap scaledBmp = Bitmap.createScaledBitmap(rltImageBmp, inputWidth, inputHeight, false);
            scaledBmp.getPixels(rspSegValues, 0, scaledBmp.getWidth(), 0, 0, scaledBmp.getWidth(), scaledBmp.getHeight());

            long endTime = SystemClock.uptimeMillis();
            Log.d(TAG, "segmentation cost: " + (endTime - startTime) + "ms");

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

    private void postSty(String base64Image) {
        Log.d(TAG, Thread.currentThread().getName() + " run: ");
        final long startTime = SystemClock.uptimeMillis();
        HttpURLConnection conn = null;
        StringBuilder response = new StringBuilder();
        String postSegUrlStr = postUrlStr.substring(0, postUrlStr.lastIndexOf('/') + 1) + "styleTransfer";
        Log.d(TAG, "postStyUrlStr: " + postSegUrlStr);

        try {
            URL url = new URL(postSegUrlStr);
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
            jsonObj.put("inputs", base64Image);
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

            //long decodeStartTime = SystemClock.uptimeMillis();
            byte[] decodedString = Base64.decode(rltBase64, Base64.URL_SAFE);
            final Bitmap rltImageBmp = BitmapFactory.decodeByteArray(decodedString, 0, decodedString.length);
            //Log.d(TAG, "decode cost: " + (SystemClock.uptimeMillis() - decodeStartTime) + "ms");

            Bitmap scaledBmp = Bitmap.createScaledBitmap(rltImageBmp, inputWidth, inputHeight, false);
            scaledBmp.getPixels(rspStyValues, 0, scaledBmp.getWidth(), 0, 0, scaledBmp.getWidth(), scaledBmp.getHeight());

            long endTime = SystemClock.uptimeMillis();
            Log.d(TAG, "style transfer cost: " + (endTime - startTime) + "ms");

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
