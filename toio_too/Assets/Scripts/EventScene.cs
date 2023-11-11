using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using toio;
using System.IO;



// The file name and class name must match.
public class EventScene : MonoBehaviour
{
    public enum ExperimentMode: byte
    {
        NoMove = 0, StraightMove = 1
    }

    public ConnectType connectType = ConnectType.Real; 
    public ExperimentMode expMode = ExperimentMode.NoMove;
    //public int moveSpeed = 0;
    public int moveDurationMs = 1000;
    public int moveMaxSpeed = 100;
    float intervalTime = 0.05f;
    float elapsedTime = 0;
    CubeManager cm;
    Cube cube;
    bool showId = true;
    bool connected = false;
    bool updated = false;
    int onMat = 1;
    int collided = 0;
    float lastMoveTime = 0;

    public string file_label = "test";
    StreamWriter writer;

    async void Start()
    {
        string path = file_label + System.DateTime.UtcNow.ToString() + ".csv";
        writer = new StreamWriter(path, true);
        writer.WriteLine("Time,Euler0,Euler1,Euler2,ShakeLevel,PosX, PosY,OnMat,MotorSpeedL,MotorSpeedR,Collided");

        cm = new CubeManager(connectType);
        await cm.SingleConnect();
        
        cube = cm.cubes[0];

        cube.buttonCallback.AddListener("EventScene", OnPressButton);
        cube.slopeCallback.AddListener("EventScene", OnSlope);
        cube.collisionCallback.AddListener("EventScene", OnCollision);
        cube.idCallback.AddListener("EventScene", OnUpdateID);
        cube.standardIdCallback.AddListener("EventScene", OnUpdateStandardID);
        cube.idMissedCallback.AddListener("EventScene", OnMissedID);
        cube.standardIdMissedCallback.AddListener("EventScene", OnMissedStandardID);
        cube.poseCallback.AddListener("EventScene", OnPose);
        cube.doubleTapCallback.AddListener("EventScene", OnDoubleTap);
        cube.shakeCallback.AddListener("EventScene", OnShake);
        cube.motorSpeedCallback.AddListener("EventScene", OnMotorSpeed);
        cube.magnetStateCallback.AddListener("EventScene", OnMagnetState);
        cube.magneticForceCallback.AddListener("EventScene", OnMagneticForce);
        cube.attitudeCallback.AddListener("EventScene", OnAttitude);

        // Enable sensors
        await cube.ConfigMotorRead(true);
        await cube.ConfigAttitudeSensor(Cube.AttitudeFormat.Eulers, 10, Cube.AttitudeNotificationType.OnChanged);
        await cube.ConfigMagneticSensor(Cube.MagneticMode.MagneticForce, 20, Cube.MagneticNotificationType.OnChanged);
        await cube.ConfigIDNotification(10, Cube.IDNotificationType.OnChanged);
        await cube.ConfigIDMissedNotification(10);
        cube.ConfigCollisionThreshold(1);

        Debug.Log("Connected!");
        connected = true;
    }

    void FixedUpdate()
    {
        if(connected)
        {
            writer.WriteLine(
                (Time.time).ToString()+","+
                (cube.eulers).ToString()[1..^1]+","+
                (cube.shakeLevel).ToString() +","+
                (cube.pos).ToString()[1..^1]+","+
                (onMat).ToString()+","+
                (cube.leftSpeed).ToString()+","+
                (cube.rightSpeed).ToString()+","+
                collided.ToString()
            );
            if(collided > 0)
            {
                collided -= 1;
            }
        }
    }

    void Update()
    {
        if(connected)
        {
            if(Time.time - lastMoveTime > moveDurationMs/1000)
            {
                if(expMode == ExperimentMode.StraightMove)
                {
                    Debug.Log("asdf");
                    int speed = Random.Range(8,moveMaxSpeed);
                    cube.Move(speed,speed,moveDurationMs);
                    lastMoveTime = Time.time;
                }
            }
        }
    }

    void OnApplicationQuit()
    {
        Debug.Log("Application ending after " + Time.time + " seconds");
        string path = file_label + System.DateTime.UtcNow.ToString() + ".csv";
        Debug.Log("Output written to " + path);
        writer.Close();
    }

    void OnCollision(Cube c)
    {
        Debug.Log("Collision");
        collided = 100;
    }

    void OnSlope(Cube c)
    {
        Debug.Log("Slope");
    }

    void OnPressButton(Cube c)
    {
        if (c.isPressed)
        {
            showId = !showId;
        }
        Debug.Log("Button Pressed");
    }

    void OnUpdateID(Cube c)
    {
        if (showId)
        {
            Debug.LogFormat("pos=(x:{0}, y:{1}), angle={2}", c.pos.x, c.pos.y, c.angle);
        }
        onMat = 1;
    }

    void OnUpdateStandardID(Cube c)
    {
        if (showId)
        {
            Debug.LogFormat("standardId:{0}, angle={1}", c.standardId, c.angle);
        }
    }

    void OnMissedID(Cube cube)
    {
        Debug.LogFormat("Position ID Missed.");
        onMat = 0;
    }

    void OnMissedStandardID(Cube c)
    {
        Debug.LogFormat("Standard ID Missed.");
    }

    void OnPose(Cube c)
    {
        // Debug.Log($"pose = {c.pose.ToString()}");
    }

    void OnDoubleTap(Cube c)
    {
        c.PlayPresetSound(3);
    }

    void OnShake(Cube c)
    {
        // Debug.Log(c.shakeLevel);
        updated = false;
    }

    void OnMotorSpeed(Cube c)
    {
        // Debug.Log($"motor speed: left={c.leftSpeed}, right={c.rightSpeed}");
    }

    void OnMagnetState(Cube c)
    {
        // Debug.Log($"magnet state: {c.magnetState.ToString()}");
    }

    void OnMagneticForce(Cube c)
    {
        Debug.Log($"magnetic force = {c.magneticForce}");
    }

    void OnAttitude(Cube c)
    {
        // Debug.Log($"attitude = {c.eulers}");
        updated = false;
    }
}