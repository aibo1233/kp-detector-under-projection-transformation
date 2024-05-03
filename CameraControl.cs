using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using OpenCVForUnity;
using OpenCVForUnity.CoreModule;
using System;
using System.Threading;

public class CameraControl : MonoBehaviour
{
    // Camera orientation
    private Vector3 m_camRot;
    // camera Transform
    private Transform m_camTransform;
    // Determine if the camera has moved, used to update the transformation matrix
    private bool moved = false;
    // Initial camera pose
    Vector3 start_position;
    // Camera position reset button
    private Button btn_reset;
    // Screenshot button
    private Button btn_screen_shot;
    // Execute projection transformation button
    private Button btn_change_angle;
    // Automatically execute all projection transformations button
    private Button btn_get_all_data;
    // Transform matrix textbox
    private Text txt_matrix;
    // The coordinates of the reference point in the initial perspective, 
    // used to calculate the transformation matrix
    float[] screen_position_src = new float[8];
    // Camera distance from the center of the image(Camera moves on a sphere with a fixed radius from the center of the image)
    float distance = 1000;


    // initialization
    void Start()
    {
        // Obtain camera related parameters
        m_camTransform = Camera.main.transform;
        m_camRot = Camera.main.transform.eulerAngles;
        start_position = transform.position;

        //Button event binding
        btn_reset = GameObject.Find("btn_reset").GetComponent<Button>();
        btn_reset.onClick.AddListener(cameraReset);
        btn_screen_shot = GameObject.Find("btn_screen_shot").GetComponent<Button>();
        btn_screen_shot.onClick.AddListener(screen_shot);
        btn_change_angle = GameObject.Find("btn_change_angle").GetComponent<Button>();
        btn_change_angle.onClick.AddListener(change_angle);
        btn_get_all_data = GameObject.Find("btn_get_all_data").GetComponent<Button>();
        btn_get_all_data.onClick.AddListener(change_angle_all);

        //Get matrix text box
        txt_matrix = GameObject.Find("txt_matrix").GetComponent<Text>();

        //Obtain the coordinates of the four points on the screen in the initial state
        screen_position_src = getScreenPosition();
    }


    // Update frame
    void Update()
    {
        //If the camera moves, recalculate the transformation matrix and display it
        if (moved)
        {
            float[] screen_position_dst = getScreenPosition();
            Mat matrix = opencvtest(screen_position_src, screen_position_dst);
            txt_matrix.text = matrix.dump();
            Debug.Log(matrix.dump());

            moved = false;
        }

    }


    //Take a screenshot of the current screen
    private void screen_shot()
    {
        UnityEngine.ScreenCapture.CaptureScreenshot("test.png");
        Debug.Log("Screenshot");
    }


    // Perform transformations based on the direction and angle of input
    private void change_angle()
    {
        Text txt_direction = GameObject.Find("txt_direction").GetComponent<Text>();
        String str_direction = txt_direction.text;
        Text txt_angle = GameObject.Find("txt_angle").GetComponent<Text>();
        String str_angle = txt_angle.text;

        int direction = int.Parse(str_direction);
        float angle = float.Parse(str_angle);
        Debug.Log(direction);
        Debug.Log(angle);
        camera_set_angle(90 - angle, direction);
    }


    // Automatically generate all transformation matrices and save them to a file
    private void change_angle_all()
    {
        // The array defines the degree and direction of each transformation
        float[] angles = { 5, 10, 15, 20, 25, 30, 35, 40, 45 };
        int[] directions = { 0, 1, 2, 3, 4, 5, 6, 7 };
        String[] res = new String[72];
        int index = 0;

        foreach (int direction in directions)
        {
            foreach (float angle in angles)
            {
                Debug.Log("direction:" + direction + "angle:" + angle);
                camera_set_angle(90 - angle, direction);
                // Thread.Sleep(1000);


                // Calculate transformation matrix
                float[] screen_position_dst = getScreenPosition();
                Mat matrix = opencvtest(screen_position_src, screen_position_dst);
                txt_matrix.text = matrix.dump();
                // Debug.Log(matrix.dump());


                String str_mtx = "[";
                for (int i = 0; i < 3; i++)
                {
                    str_mtx += "[";
                    for (int j = 0; j < 3; j++)
                    {
                        double[] temp = matrix.get(i, j);
                        str_mtx += temp[0].ToString();
                        if (j != 2)
                        {
                            str_mtx += ",";
                        }
                    }
                    str_mtx += "]";
                    if (i != 2)
                    {
                        str_mtx += ",";
                    }
                }
                str_mtx += "]";
                res[index] = str_mtx;
                index++;
                // Debug.Log(str_mtx);
            }
        }

        //write file
        using (System.IO.StreamWriter file =
            new System.IO.StreamWriter(@"output_1000.txt", false))
        {
            foreach (string item in res)
            {
                file.WriteLine(item);
            }
        }

        Debug.Log("all data!");
    }


    // Camera returns to initial position
    private void cameraReset()
    {
        // Return to the initial position
        Vector3 temp_p = new Vector3(0, 0, -1000);
        transform.position = start_position;
        // Rotating camera
        m_camRot.x = 0;
        m_camRot.y = 0;
        m_camRot.z = 0;
        m_camTransform.eulerAngles = m_camRot;

        moved = true;
    }


    //Calculate camera position based on parameters
    //_angle is at the angle to the plane，_direction indicating eight directions（from 0 to 7
    private void camera_set_angle(float _angle, int _direction)
    {
        float z = -(float)(distance * Math.Sin(Math.PI * _angle / 180));
        float x = 0;
        float y = 0;

        float distance_h = (float)(distance * Math.Cos(Math.PI * _angle / 180));    //0,2,4,6
        float distance_x_y = (float)(distance * Math.Cos(Math.PI * _angle / 180) * Math.Cos(Math.PI * 45 / 180));   //1,3,5,7

        //0,2,4,6
        if (_direction % 2 == 0)
        {
            if (_direction == 0)
            {
                x = distance_h;
                y = 0;
            }
            else if (_direction == 2)
            {
                x = 0;
                y = distance_h;
            }
            else if (_direction == 4)
            {
                x = -distance_h;
                y = 0;
            }
            else if (_direction == 6)
            {
                x = 0;
                y = -distance_h;
            }
        }
        else
        {
            //1,3,5,7
            if (_direction == 1)
            {
                x = distance_x_y;
                y = distance_x_y;
            }
            else if (_direction == 3)
            {
                x = -distance_x_y;
                y = distance_x_y;
            }
            else if (_direction == 5)
            {
                x = -distance_x_y;
                y = -distance_x_y;

            }
            else if (_direction == 7)
            {
                x = distance_x_y;
                y = -distance_x_y;
            }
        }


        // Update camera position
        transform.position = new Vector3(x, y, z);
        // The camera center points towards the center point of the image
        GameObject targetObj = GameObject.FindWithTag("cube1");
        transform.LookAt(targetObj.transform.position);


        // Update flag bits, update transformation matrix
        moved = true;
    }


    // Calculate the transformation matrix based on four pairs of points
    private Mat opencvtest(float[] arr_src, float[] arr_dst)
    {
        // Should maintain the order of p1. x, p1. y, p2. x, p2. y
        Mat mat_src = new Mat(4, 1, CvType.CV_32FC2);
        mat_src.put(0, 0, arr_src[0], arr_src[1], arr_src[2], arr_src[3], arr_src[4], arr_src[5], arr_src[6], arr_src[7]);
        Mat mat_dst = new Mat(4, 1, CvType.CV_32FC2);
        mat_dst.put(0, 0, arr_dst[0], arr_dst[1], arr_dst[2], arr_dst[3], arr_dst[4], arr_dst[5], arr_dst[6], arr_dst[7]);

        // Calculate transformation matrix
        Mat res = OpenCVForUnity.ImgprocModule.Imgproc.getPerspectiveTransform(mat_src, mat_dst);
        return res;
    }


    //Obtain the pixel coordinates of points in the 3D world on the screen
    float[] getScreenPosition()
    {
        // Screen size,set the screen to the same size
        float[] screen_size = { 1000, 1000 };
        // Image size(The size of the image to be transformed through these matrices)
        float[] image_size = { 2000, 2000 };
        // 3D coordinates (any four points taken)
        Vector3 p1 = new Vector3(-500, -500, 0);
        Vector3 p2 = new Vector3(500, -500, 0);
        Vector3 p3 = new Vector3(500, 500, 0);
        Vector3 p4 = new Vector3(-500, 500, 0);


        // 2D screen coordinates (bottom left corner as origin)
        Vector3 screenPosition1 = Camera.main.WorldToScreenPoint(p1);
        Vector3 screenPosition2 = Camera.main.WorldToScreenPoint(p2);
        Vector3 screenPosition3 = Camera.main.WorldToScreenPoint(p3);
        Vector3 screenPosition4 = Camera.main.WorldToScreenPoint(p4);



        // Change to the top left corner as the origin
        float[] res = { screenPosition1[0], screen_size[1] - screenPosition1[1], screenPosition2[0], screen_size[1] - screenPosition2[1], screenPosition3[0], screen_size[1] - screenPosition3[1], screenPosition4[0], screen_size[1] - screenPosition4[1] };
        for (int i = 0; i < 8; i++)
        {
            res[i] += (image_size[0] - screen_size[0]) / 2;
        }
        return res;
    }
}