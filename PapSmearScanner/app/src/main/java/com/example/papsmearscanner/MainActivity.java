package com.example.papsmearscanner;

import android.content.Context;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {
    private static int RESULT_LOAD_IMAGE = 1;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button load_button = (Button) findViewById(R.id.load);
        Button detect_button = (Button) findViewById(R.id.detect);
        //request permission for external files
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestPermissions(new String[]  {android.Manifest.permission.READ_EXTERNAL_STORAGE}, 1);
        }

        load_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                TextView result_text = findViewById(R.id.result_text);
                result_text.setText("");
                Intent i = new Intent(
                        Intent.ACTION_PICK,
                        MediaStore.Images.Media.EXTERNAL_CONTENT_URI
                );

                startActivityForResult(i, RESULT_LOAD_IMAGE);

            }
        });

        detect_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                Bitmap bitmap = null;
                Module module = null;

                //Getting image from the image view
                ImageView curr_image = (ImageView) findViewById(R.id.curr_image);
                try {
                    bitmap = ((BitmapDrawable)curr_image.getDrawable()).getBitmap();
                    bitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true);

                    // loading serialized torchscript module from packaged into app android asset model.pt,
                    module = Module.load(assetFilePath(MainActivity.this, "resnet_loaded.pt"));
                } catch (IOException e) {
                    Log.e("PytorchHelloWorld", "Error reading assets", e);
                    finish();
                }
                //okay, we cannot do normalization, since we haven't done that in training.
                float[] MYVISION_NORM_MEAN_RGB = new float[] {0.0f, 0.0f, 0.0f};
                float[] MYVISION_NORM_STD_RGB = new float[] {1.0f, 1.0f, 1.0f};

                //Input Tensor
                /*
                final Tensor input = TensorImageUtils.bitmapToFloat32Tensor(
                        bitmap,
                        TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                        TensorImageUtils.TORCHVISION_NORM_STD_RGB
                );
                */
                final Tensor input = TensorImageUtils.bitmapToFloat32Tensor(
                        bitmap,
                        MYVISION_NORM_MEAN_RGB,
                        MYVISION_NORM_STD_RGB
                );

                //Calling the forward of the model to run our input
                final Tensor output = module.forward(IValue.from(input)).toTensor();

                final float[] scores = output.getDataAsFloatArray();
                Log.i("PapSmearScanner", "scores: " + scores[0]
                                                    + ", " + scores[1]
                                                    + ", " + scores[2]
                                                    + ", " + scores[3]
                                                    + ", " + scores[4]);
                // searching for the index with maximum score
                float maxScore = -Float.MAX_VALUE;
                int maxScoreIdx = -1;
                for (int i = 0; i < scores.length; i++) {
                    if (scores[i] > maxScore) {
                        maxScore = scores[i];
                        maxScoreIdx = i;
                    }
                }
                Log.i("PapSmearScanner", "maxScoreIdx: " + maxScoreIdx);

                String detectedCancerClass = CancerClassifications.CANCER_SIPAKMED_CLASSES[maxScoreIdx];

                //Writing the detected class in to the text view of the layout
                TextView textView = findViewById(R.id.result_text);
                textView.setText(detectedCancerClass);
            }
        });

    }

    // Call Back method  to get the Message form other Activity
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        //This functions return the selected image from gallery
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == RESULT_LOAD_IMAGE && resultCode == RESULT_OK && null != data) {
            Uri selectedImage = data.getData();
            String[] filePathColumn = { MediaStore.Images.Media.DATA };

            Cursor cursor = getContentResolver().query(selectedImage,
                    filePathColumn, null, null, null);
            assert cursor != null : "cursor is null";
            cursor.moveToFirst();

            int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
            String picturePath = cursor.getString(columnIndex);
            cursor.close();

            ImageView curr_image = (ImageView) findViewById(R.id.curr_image);
            curr_image.setImageBitmap(BitmapFactory.decodeFile(picturePath));

            //Setting the URI so we can read the Bitmap from the image
            curr_image.setImageURI(null);
            curr_image.setImageURI(selectedImage);
        }
    }

    /**
     * Copies specified asset to the file in /files app directory and returns this file absolute path.
     *
     * @return absolute file path
     */
    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }
}
