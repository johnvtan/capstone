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
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/common/geometry.h>

#include <string>
#include <deque>

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

class PlaneSegmentation 
{
public:
    PlaneSegmentation(void);
    ~PlaneSegmentation(void);
    void run(void);
private:
    void pointcloud_callback(const PointCloud::ConstPtr& msg);
    pcl::PointXYZ compute_centroid(const PointCloud::Ptr& pointcloud);
    double get_rectangular_area(const PointCloud::Ptr& pointcloud);
    bool near_previous_centroid(const pcl::PointXYZ& centroid_point);
    PointCloud::Ptr filter_by_std_dev(const PointCloud::Ptr& pointcloud,
                                      const pcl::ModelCoefficients::Ptr coefficients,
                                      unsigned int num_dev);
    double get_deviation(const PointCloud::Ptr& pointcloud,
                         const pcl::ModelCoefficients::Ptr& coefficients,
                         double* mean_error);

    PointCloud::Ptr extract_cloud(const PointCloud::Ptr& pointcloud, const pcl::PointIndices::Ptr& indices, bool remove);

    uint32_t colors[100];
    std::deque< std::vector<pcl::PointXYZ>> centroid_buffer;
    int frame_count;
    ros::NodeHandle nh;
    ros::Subscriber pointcloud_sub;
    ros::Publisher plane_pointcloud_pub;
    ros::Publisher colored_plane_pub;

    pcl::VoxelGrid<pcl::PCLPointCloud2> voxel_filter;
    pcl::SACSegmentation<pcl::PointXYZ> plane_segmenter;
    pcl::ExtractIndices<pcl::PointXYZ> extracter;
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kd_tree;
};
#endif
