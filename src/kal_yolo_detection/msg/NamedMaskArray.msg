# Named masks for segmentation results
# Header timestamp/frame matches source image
std_msgs/Header header
# Keys identifying each mask, e.g. "road", "pylon_0"
string[] keys
# Corresponding masks as mono8 images (0 background, 255 foreground)
sensor_msgs/Image[] masks
