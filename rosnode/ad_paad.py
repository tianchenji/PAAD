import cv2
import torch
import rospy
import numpy as np

from cv_bridge import CvBridge
from PIL import Image, ImageDraw
from torchvision import transforms
from nets.PAAD import PAAD

from fpn_msgs.msg import Terrabot
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import CompressedImage, LaserScan

# anomaly detection pub
ad_pub = rospy.Publisher('/ad/prob_of_failure', Float32MultiArray, queue_size=1)

# define anomaly detection frequency
loop_hertz = 10 #5

class Anomaly_Detector:

    def __init__(self):

        # parameters for the front-camera view
        # image size in pixels
        self.w_ego, self.h_ego = 320, 240
        # focal length in pixels
        self.f = 460
        # camera height in meters
        self.Y = - 0.23

        # discount factor for the probability of failure in the future
        self.gamma              = 0.95
        self.ad_horizon         = 10
        self.pred_score_weights = self.get_weights()

        # filtering in time
        self.lookback_win = 3
        self.failure_bin  = [0] * self.lookback_win
        self.failure_tol  = 3

        p_horizon       = 20
        self.lidar_clip = 1.85
        self.bridge     = CvBridge()

        # memory for sensor data
        self.image       = np.zeros((240, 320, 3), dtype=np.uint8)
        self.point_cloud = np.zeros(1081)
        self.pred_xs     = np.zeros(p_horizon - 1)
        self.pred_ys     = np.zeros(p_horizon - 1)
        self.nav_mode    = 1

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.paad = PAAD(
            device=self.device,
            freeze_features=True,
            pretrained_file="nets/VisionNavNet_state_hd.pth.tar",
            horizon=self.ad_horizon).to(self.device)

        # load network parameters
        PATH = './nets/paad.pth'
        self.paad.load_state_dict(torch.load(PATH))

        self.paad.eval()

    def get_weights(self):

        alpha = (1 - self.gamma) / (1 - self.gamma**self.ad_horizon)

        weights = np.array([alpha * self.gamma**i for i in range(self.ad_horizon)])

        return weights

    def get_pred_traj_ego(self, pred_Zs, pred_Xs):
        '''
        compute the predicted trajectory in front-camera view
        '''

        # project 3D points onto the image plane
        xs = self.f * pred_Xs / pred_Zs
        ys = self.f * self.Y / pred_Zs

        # convert 2D points into the image coordinate
        xs = - xs + self.w_ego / 2
        ys = - ys + self.h_ego / 2

        pred_traj = [(x, y) for x, y in zip(xs, ys)]

        return pred_traj

    def predict(self):
        '''
        Proactive anomaly detection
        ---------------------------

        Inputs:  image         - RGB camera data (dimension of 240*320)
                 point_cloud   - 2D LiDAR data (dimenison of 1081)
                 pred_traj_img - planned trajectory in front-camera view (dimension of 128*320)

        Outputs: pred_score - probability of failure per time step in the prediction horizon
        '''

        # nav_mode is equal to 2 in autonomous mode and 1 in manual mode
        if self.nav_mode != 2:
            self.failure_bin = [0] * self.lookback_win

        elif self.nav_mode == 2 and len(self.pred_xs) > 0 and len(self.pred_ys) > 0:

            with torch.no_grad():

                # prepare camera data
                image_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                image = image_transform(self.image)

                # prepare lidar data
                point_cloud = np.clip(self.point_cloud,
                                      a_min=0, a_max=self.lidar_clip)/self.lidar_clip
                point_cloud = torch.as_tensor(point_cloud, dtype=torch.float32)

                # prepare mpc data
                pred_traj = self.get_pred_traj_ego(self.pred_xs, self.pred_ys)

                pred_traj_img = Image.new(mode="L", size=(self.w_ego, self.h_ego))
                traj_draw     = ImageDraw.Draw(pred_traj_img)
                traj_draw.line(pred_traj, fill="white", width=6, joint="curve")

                pred_traj_img = pred_traj_img.crop((0, 112, 320, 240))

                pred_traj_transform = transforms.Compose([transforms.ToTensor()])
                pred_traj_img = pred_traj_transform(pred_traj_img)

                image.unsqueeze_(0)
                point_cloud.unsqueeze_(0)
                pred_traj_img.unsqueeze_(0)

                image, pred_traj_img = image.to(self.device), pred_traj_img.to(self.device)
                point_cloud          = point_cloud.to(self.device)

                # make inference
                _, _, _, pred_score = self.paad(image, pred_traj_img, point_cloud)

                pred_score.squeeze_()
                pred_score = pred_score.cpu().numpy()

            pred_score_seq     = list(pred_score.round(2))
            pred_score_seq_msg = Float32MultiArray(data=pred_score_seq)
            ad_pub.publish(pred_score_seq_msg)

            pred_score_sum = (self.pred_score_weights * pred_score).sum().round(2)
            self.failure_bin.pop(0)
            self.failure_bin.append(pred_score_sum > 0.5)
            failure_flag = sum(self.failure_bin) >= self.failure_tol
            
            print("Failure status:", failure_flag)

    def camera_update(self, msg):

        cv_img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')

        # convert from BGR to RGB
        self.image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

    def lidar_update(self, msg):

        self.point_cloud = np.array(msg.ranges)

    def mpc_update(self, msg):

        self.pred_xs = abs(np.array([data.position.x for data in msg.poses]))[:9]
        self.pred_ys = np.array([data.position.y for data in msg.poses])[:9]

    def nav_mode_update(self, msg):

        self.nav_mode = msg.drive_mode


if __name__ == "__main__":

    anomaly_detector = Anomaly_Detector()
    rospy.init_node('anomaly_detector', anonymous=True)
    rospy.Subscriber("/terrasentia/usb_cam_node/rotated/compressed", CompressedImage,
                     anomaly_detector.camera_update, queue_size=1)
    rospy.Subscriber("/terrasentia/scan", LaserScan, anomaly_detector.lidar_update, queue_size=1)
    rospy.Subscriber("/terrasentia/mpc_node/mpc_pred_vals", PoseArray,
                     anomaly_detector.mpc_update, queue_size=1)
    rospy.Subscriber("/terrasentia/nav_mode", Terrabot,
                     anomaly_detector.nav_mode_update, queue_size=1)

    rate = rospy.Rate(loop_hertz)
    while not rospy.is_shutdown():

        anomaly_detector.predict()

        rate.sleep()