package com.example.liys.liys_app;

import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import com.example.liys.liys_app.cloud.CloudAPIActivity;
import com.example.liys.liys_app.liysimage.LIYSImageActivity;
import com.example.liys.liys_app.segmentation.SegmentationActivity;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";

    private Button btnCamera;
    private Button btnImage;
    private Button btnCloud;

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Log.d(TAG, "test jni : " + stringFromJNI());

        btnCamera = findViewById(R.id.btn_camera);
        btnImage = findViewById(R.id.btn_image);
        btnCloud = findViewById(R.id.btn_cloud);

        btnCamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent cameraIntent = new Intent(MainActivity.this, SegmentationActivity.class);
                startActivity(cameraIntent);
            }
        });

        btnImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent imageIntent = new Intent(MainActivity.this, LIYSImageActivity.class);
                startActivity(imageIntent);

            }
        });

        btnCloud.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent cloudIntent = new Intent(MainActivity.this, CloudAPIActivity.class);
                startActivity(cloudIntent);

            }
        });

    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public native String stringFromJNI();
}
