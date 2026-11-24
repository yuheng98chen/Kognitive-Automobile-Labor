#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>

/**
 * @class ImageUndistortDownscaleNode
 * @brief ROS node for efficient undistortion and downscaling of color and depth
 * images.
 */
class ImageUndistortDownscaleNode {
public:
  ImageUndistortDownscaleNode(ros::NodeHandle &nh) : it_(nh) {
    // Camera parameters from YAML
    D_ = (cv::Mat_<double>(1, 5) << -0.055714257061481476, 0.06405200064182281,
          -2.1757679860456847e-05, -1.4761384591110982e-05,
          -0.020222660154104233);
    K_ = (cv::Mat_<double>(3, 3) << 642.3306274414062, 0.0, 648.546630859375,
          0.0, 641.7576293945312, 359.7590637207031, 0.0, 0.0, 1.0);

    width_ = 1280;
    height_ = 720;
    nh.param("downscale_factor", downscale_factor_, 2);

    // Pre-compute output dimensions
    output_width_ = width_ / downscale_factor_;
    output_height_ = height_ / downscale_factor_;

    // Pre-compute undistort + resize maps for maximum efficiency
    cv::Mat new_K = K_.clone();
    new_K.at<double>(0, 0) /= downscale_factor_; // fx
    new_K.at<double>(1, 1) /= downscale_factor_; // fy
    new_K.at<double>(0, 2) /= downscale_factor_; // cx
    new_K.at<double>(1, 2) /= downscale_factor_; // cy

    cv::initUndistortRectifyMap(K_, D_, cv::Mat(), new_K,
                                cv::Size(output_width_, output_height_),
                                CV_32FC1, map1_, map2_);

    // Store the adjusted K matrix for camera_info publishing
    K_adjusted_ = new_K.clone();

    // Pre-allocate output matrices
    undistorted_downscaled_.create(output_height_, output_width_, CV_8UC3);
    depth_downscaled_.create(output_height_, output_width_, CV_16UC1);

    // Initialize camera_info_pub_ before calling setupCameraInfo
    camera_info_pub_ = nh.advertise<sensor_msgs::CameraInfo>(
        "/camera_front/processed/camera_info", 1, true);

    setupCameraInfo();

    // Use queue size of 1 to drop frames if processing can't keep up
    image_sub_ = it_.subscribe("/camera_front/color/image_raw", 1,
                               &ImageUndistortDownscaleNode::processColorImage,
                               this, image_transport::TransportHints("raw"));
    image_pub_ = it_.advertise("/camera_front/color/processed", 1);

    depth_sub_ = it_.subscribe("/camera_front/aligned_depth_to_color/image_raw", 1,
                               &ImageUndistortDownscaleNode::processDepthImage,
                               this, image_transport::TransportHints("raw"));
    depth_pub_ = it_.advertise("/camera_front/depth/processed", 1);
  }

private:
  void processColorImage(const sensor_msgs::ImageConstPtr &msg) {
    // Skip processing if no subscribers
    if (image_pub_.getNumSubscribers() == 0)
      return;

    cv_bridge::CvImageConstPtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::RGB8);
    } catch (cv_bridge::Exception &e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    // Single-step undistort + resize using pre-computed maps
    cv::remap(cv_ptr->image, undistorted_downscaled_, map1_, map2_,
              cv::INTER_LINEAR, cv::BORDER_CONSTANT);

    // Create output message
    sensor_msgs::ImagePtr out_msg = boost::make_shared<sensor_msgs::Image>();
    out_msg->header = msg->header;
    out_msg->height = output_height_;
    out_msg->width = output_width_;
    out_msg->encoding = "rgb8";
    out_msg->is_bigendian = false;
    out_msg->step = output_width_ * 3;

    size_t data_size = out_msg->step * output_height_;
    out_msg->data.resize(data_size);
    std::memcpy(&out_msg->data[0], undistorted_downscaled_.data, data_size);

    image_pub_.publish(out_msg);

    // Publish updated camera_info with same timestamp
    camera_info_msg_.header = msg->header;
    camera_info_pub_.publish(camera_info_msg_);
  }

  void processDepthImage(const sensor_msgs::ImageConstPtr &msg) {

    cv_bridge::CvImageConstPtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvShare(msg);
    } catch (cv_bridge::Exception &e) {
      ROS_ERROR("cv_bridge exception (depth): %s", e.what());
      return;
    }

    cv::resize(cv_ptr->image, depth_downscaled_,
               cv::Size(output_width_, output_height_), 0, 0,
               cv::INTER_NEAREST);

    sensor_msgs::ImagePtr out_msg = boost::make_shared<sensor_msgs::Image>();
    out_msg->header = msg->header;
    out_msg->height = output_height_;
    out_msg->width = output_width_;
    out_msg->encoding = msg->encoding;
    out_msg->is_bigendian = msg->is_bigendian;
    out_msg->step = output_width_ * (msg->step / msg->width);

    size_t data_size = out_msg->step * output_height_;
    out_msg->data.resize(data_size);
    std::memcpy(&out_msg->data[0], depth_downscaled_.data, data_size);

    depth_pub_.publish(out_msg);
  }

  void setupCameraInfo() {
    camera_info_msg_.header.frame_id = "camera_front_optical_frame";
    camera_info_msg_.width = output_width_;
    camera_info_msg_.height = output_height_;

    // Copy adjusted K matrix
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        camera_info_msg_.K[i * 3 + j] = K_adjusted_.at<double>(i, j);
      }
    }

    // Copy D matrix
    camera_info_msg_.D.resize(5);
    for (int i = 0; i < 5; ++i) {
      camera_info_msg_.D[i] = D_.at<double>(0, i);
    }

    // P matrix (projection matrix)
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        camera_info_msg_.P[i * 4 + j] = K_adjusted_.at<double>(i, j);
      }
      camera_info_msg_.P[i * 4 + 3] = 0.0;
    }

    camera_info_msg_.distortion_model = "plumb_bob";
    camera_info_msg_.binning_x = downscale_factor_;
    camera_info_msg_.binning_y = downscale_factor_;
    camera_info_msg_.roi.x_offset = 0;
    camera_info_msg_.roi.y_offset = 0;
    camera_info_msg_.roi.height = 0;
    camera_info_msg_.roi.width = 0;
    camera_info_msg_.roi.do_rectify = false;

    camera_info_pub_.publish(camera_info_msg_);
  }

  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;
  image_transport::Subscriber depth_sub_;
  image_transport::Publisher depth_pub_;
  ros::Publisher camera_info_pub_;

  cv::Mat K_, D_, K_adjusted_;
  cv::Mat map1_, map2_;
  cv::Mat undistorted_downscaled_, depth_downscaled_;
  sensor_msgs::CameraInfo camera_info_msg_;

  int width_, height_;
  int output_width_, output_height_;
  int downscale_factor_;
};

int main(int argc, char **argv) {
  ros::init(argc, argv, "image_undistort_downscale");
  ros::NodeHandle nh("~");
  ImageUndistortDownscaleNode node(nh);
  ros::spin();
  return 0;
}