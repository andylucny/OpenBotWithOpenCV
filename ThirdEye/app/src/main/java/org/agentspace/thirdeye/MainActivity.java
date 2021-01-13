package org.agentspace.thirdeye;

import android.content.Context;
import android.content.res.AssetManager;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.WindowManager;

import androidx.annotation.RequiresApi;

import org.agentspace.thirdeye.cnn.CNNExtractorService;
import org.agentspace.thirdeye.cnn.impl.CNNExtractorServiceImpl;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;

import org.openbot.env.UsbConnection;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Locale;


public class MainActivity extends CameraActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = MainActivity.class.getName();

    private static final String IMAGENET_CLASSES = "imagenet_classes.txt";
    private static final String MODEL_FILE = "pytorch_mobilenet.onnx";

    private CameraBridgeViewBase mOpenCvCameraView;
    private Net opencvNet;

    private CNNExtractorService cnnService;

    private UsbConnection usbConnection;
    private int baudRate = 115200;
    private int speedMultiplier = 131; //101 //192
    private String last_message = "";

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully!");
                    mOpenCvCameraView.enableView();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    @Override
    public void onResume() {
        super.onResume();
        // OpenCV manager initialization
        OpenCVLoader.initDebug();
        mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
    }

    public static final class ControlSignal {
        private final float left;
        private final float right;

        public ControlSignal(float left, float right) {
            this.left = Math.max(-1.f, Math.min(1.f, left));
            this.right = Math.max(-1.f, Math.min(1.f, right));
        }

        public float getLeft() {
            return left;
        }

        public float getRight() {
            return right;
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    private void connectUsb() {
        usbConnection = new UsbConnection(this, baudRate);
        usbConnection.startUsbConnection();
    }

    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    //@RequiresApi(api = Build.VERSION_CODES.KITKAT)
    private void disconnectUsb() {
        if (usbConnection != null) {
            sendControlToVehicle(new ControlSignal(0, 0));
            usbConnection.stopUsbConnection();
            usbConnection = null;
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    //@RequiresApi(api = Build.VERSION_CODES.KITKAT)
    protected void sendControlToVehicle(ControlSignal vehicleControl) { // ControlSignal (-1..0..1,-1..0..1)
        if (usbConnection == null || !usbConnection.isOpen()) {
            connectUsb();
        }
        if ((usbConnection != null) && usbConnection.isOpen() && !usbConnection.isBusy()) {
            String message =
                    String.format(
                            Locale.US,
                            "c%d,%d\r\n",
                            (int) (vehicleControl.getLeft() * speedMultiplier),
                            (int) (vehicleControl.getRight() * speedMultiplier));
            if (!message.equals(last_message)) {
                usbConnection.send(message);
                last_message = message;
            }
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    //@RequiresApi(api = Build.VERSION_CODES.KITKAT)
    protected void sendIndicatorToVehicle(int vehicleIndicator) {  // 1 0 or -1
        if (usbConnection == null || !usbConnection.isOpen()) {
            connectUsb();
        }
        if (usbConnection != null && usbConnection.isOpen() && !usbConnection.isBusy()) {
            String message = String.format(Locale.US, "i%d\r\n", vehicleIndicator);
            if (!message.equals(last_message)) {
                usbConnection.send(message);
                last_message = message;
            }
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);

        // initialize implementation of CNNExtractorService
        this.cnnService = new CNNExtractorServiceImpl();
        // configure camera listener
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.CameraView);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        // start USB connection
        // connectUsb();
    }


    @Override
    protected List<? extends CameraBridgeViewBase> getCameraViewList() {
        return Collections.singletonList(mOpenCvCameraView);
    }

    public void onCameraViewStarted(int width, int height) {
        // obtaining converted network
        String onnxModelPath = getPath(MODEL_FILE, this);
        if (onnxModelPath.trim().isEmpty()) {
            Log.i(TAG, "Failed to get model file");
            return;
        }
        opencvNet = cnnService.getConvertedNet(onnxModelPath, TAG);
    }


    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    //@RequiresApi(api = Build.VERSION_CODES.KITKAT)
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat frame = inputFrame.rgba();
        String classesPath = getPath(IMAGENET_CLASSES, this);
        String predictedClass = cnnService.getPredictedLabel(frame, opencvNet, classesPath);

        // place the connection status on the image
        boolean connection = (usbConnection != null && usbConnection.isOpen());
        Imgproc.putText(frame, connection ? "connected" : "disconnected", new Point(200, 200), Imgproc.FONT_HERSHEY_SIMPLEX, 2, new Scalar(255, 121, 0), 2);

        // place the predicted label on the image
        Imgproc.putText(frame, predictedClass, new Point(200, 100), Imgproc.FONT_HERSHEY_SIMPLEX, 2, new Scalar(255, 121, 0), 3);

        // control
        if (predictedClass.contains("ruler")) {
            sendControlToVehicle(new ControlSignal(-1, 1));
        }
        else if (predictedClass.contains("bottle")) {
            sendControlToVehicle(new ControlSignal(1, -1));
        }
        else if (predictedClass.contains("ball")) {
            sendControlToVehicle(new ControlSignal(1, 1));
        }
        else if (predictedClass.contains("screw") || predictedClass.contains("nail")) {
            //sendControlToVehicle(new ControlSignal(-1, -1));
        }
        else {
            sendControlToVehicle(new ControlSignal(0, 0));
        }

        return frame;
    }

    public void onCameraViewStopped() {
    }

    private static String getPath(String file, Context context) {
        AssetManager assetManager = context.getAssets();
        BufferedInputStream inputStream;
        try {
            // read the defined data from assets
            inputStream = new BufferedInputStream(assetManager.open(file));
            byte[] data = new byte[inputStream.available()];
            inputStream.read(data);
            inputStream.close();
            // Create copy file in storage.
            File outFile = new File(context.getFilesDir(), file);
            FileOutputStream os = new FileOutputStream(outFile);
            os.write(data);
            os.close();
            // Return a path to file which may be read in common way.
            return outFile.getAbsolutePath();
        } catch (IOException ex) {
            Log.i(TAG, "Failed to upload a file");
        }
        return "";
    }

}