#ifndef __PLANE_SEGMENTATION_H__
#define __PLANE_SEGMENTATION_H__

#include <ros/ros.h>

#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>

#include <string>

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

class PlaneSegmentation 
{
public:
    PlaneSegmentation(void);
    ~PlaneSegmentation(void);
    void run(void);
private:
    void pointcloud_callback(const PointCloud::ConstPtr& msg);
    bool new_pointcloud;

    ros::NodeHandle nh;
    ros::Subscriber pointcloud_sub;
    ros::Publisher plane_pointcloud_pub;

    pcl::VoxelGrid<pcl::PCLPointCloud2> voxel_filter;
    pcl::SACSegmentation<pcl::PointXYZ> plane_segmenter;
    pcl::ExtractIndices<pcl::PointXYZ> extracter;
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kd_tree;
};
#endif
