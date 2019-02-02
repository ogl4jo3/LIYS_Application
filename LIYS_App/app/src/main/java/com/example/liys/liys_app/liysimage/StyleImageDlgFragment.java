package com.example.liys.liys_app.liysimage;


import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.support.v4.app.DialogFragment;
import android.support.v4.app.Fragment;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.LinearLayout;
import android.widget.Toast;

import com.example.liys.liys_app.R;

import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;

/**
 * choose style image dialog
 */
public class StyleImageDlgFragment extends DialogFragment {
    private static final String TAG = "StyleImageDlgFragment";
    private static final String ARG_PARAM1 = "param1";

    private String mParam1;


    private onStyleImageListener mListener;

    public interface onStyleImageListener {

        public void chooseStyImage(String styleImageName, Bitmap styleImage);
    }


    private static final String[] styles = new String[]{
            "la_muse", "rain_princess", "scream", "udnie",
            "wave", "wreck"
    };
    private int styleNum;

    private Map<String, Bitmap> styleImageMap;


    public StyleImageDlgFragment() {
        // Required empty public constructor
    }

    public static StyleImageDlgFragment newInstance(String param1) {
        StyleImageDlgFragment fragment = new StyleImageDlgFragment();
        Bundle args = new Bundle();
        args.putString(ARG_PARAM1, param1);
        fragment.setArguments(args);
        return fragment;
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        if (getArguments() != null) {
            mParam1 = getArguments().getString(ARG_PARAM1);
        }
        styleNum = styles.length;

        mListener = (onStyleImageListener) getActivity();
        createStyleImage(getContext());

    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_style_image_dlg, container, false);
        getDialog().setTitle(R.string.title_style_image_dlg);

        LinearLayout llStyleImages = view.findViewById(R.id.ll_style_images);
//        llStyleImages.removeAllViews();
        LinearLayout llLine = new LinearLayout(getContext());
        LinearLayout.LayoutParams llLineLP = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT);
        llLineLP.setMargins(2, 2, 2, 2);

        llLine.setOrientation(LinearLayout.HORIZONTAL);
        llLine.setLayoutParams(llLineLP);

        LinearLayout.LayoutParams ibLP = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT);
        ibLP.setMargins(20, 10, 20, 10);

        for (int i = 0; i < styleNum; i++) {

            ImageButton ibImage = new ImageButton(getContext());
            final String styleImageKey = styles[i];
            ibImage.setImageBitmap(styleImageMap.get(styleImageKey));
            ibImage.setLayoutParams(ibLP);
            ibImage.setPadding(2, 2, 2, 2);
            ibImage.setAdjustViewBounds(true);
            ibImage.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    mListener.chooseStyImage(styleImageKey, styleImageMap.get(styleImageKey));
                    dismiss();
                }
            });
            llLine.addView(ibImage);

            if (i % 2 == 1 || i == styleNum - 1) {
                llStyleImages.addView(llLine);
                llLine = new LinearLayout(getContext());
                llLine.setOrientation(LinearLayout.HORIZONTAL);
                llLine.setLayoutParams(llLineLP);
            }
        }


        return view;
    }

    private void createStyleImage(Context context) {
        styleImageMap = new HashMap<>();

        for (int i = 0; i < styleNum; ++i) {
//            Log.d(TAG, "Creating item " + i);
            String filePath = "style_thumbnails/" + styles[i] + ".jpg";
            final Bitmap bm = getBitmapFromAsset(context, filePath);
            styleImageMap.put(styles[i], bm);
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

        return bitmap;
    }

}
