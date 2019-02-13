#include <ros/ros.h>

#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

ros::Publisher filtered_pointcloud_pub;
pcl::VoxelGrid<pcl::PCLPointCloud2> filter;

void pointcloud_callback(const PointCloud::ConstPtr& msg) {
    PointCloud::Ptr cloud_filtered(new PointCloud);
    pcl::PCLPointCloud2::Ptr cloud2(new pcl::PCLPointCloud2), filtered_cloud2(new pcl::PCLPointCloud2);
    pcl::toPCLPointCloud2(*msg, *cloud2);

    filter.setInputCloud(cloud2);
    filter.filter(*filtered_cloud2);

    pcl::fromPCLPointCloud2(*filtered_cloud2, *cloud_filtered);

    cloud_filtered->header.frame_id = "guidance";
    pcl_conversions::toPCL(ros::Time::now(), cloud_filtered->header.stamp);
    filtered_pointcloud_pub.publish(cloud_filtered);
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "plane_segmentation_node");
    ros::NodeHandle nh;
    ros::Subscriber sub = nh.subscribe<PointCloud>("/guidance/points2", 1, pointcloud_callback);
    filtered_pointcloud_pub = nh.advertise<PointCloud>("/guidance/filtered_points", 1);
    filter.setLeafSize(0.01f, 0.01f, 0.01f);
    ros::spin();
    return 0;
}
