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
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/gp3.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <string>
#include <deque>

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

struct plane_probability {
    pcl::PointXYZ centroid;
    double probability;
};

struct persistent_plane {
    pcl::PointXYZ centroid;
    PointCloud::Ptr plane;
    int lifespan;
};

class PlaneSegmentation 
{
public:
    PlaneSegmentation(void);
    ~PlaneSegmentation(void);
    void run(void);
private:
    void pointcloud_callback(const PointCloud::ConstPtr& msg);
    PointCloud::Ptr remove_outliers(const PointCloud::Ptr& pointcloud);
    pcl::PointXYZ compute_centroid(const PointCloud::Ptr& pointcloud);
    double get_rectangular_area(const PointCloud::Ptr& pointcloud);
    bool near_previous_centroid(const pcl::PointXYZ& centroid_point, const PointCloud::Ptr& pointcloud);
    PointCloud::Ptr filter_by_std_dev(const PointCloud::Ptr& pointcloud,
                                      const pcl::ModelCoefficients::Ptr coefficients,
                                      unsigned int num_dev);
    double get_deviation(const PointCloud::Ptr& pointcloud,
                         const pcl::ModelCoefficients::Ptr& coefficients,
                         double* mean_error);
    double get_planar_probability(const pcl::PointXYZ& centroid_point);
    double get_surface_area(const PointCloud::Ptr& pointcloud);
    double find_polygon_area(const PointCloud::Ptr& pointcloud, const pcl::PolygonMesh& polygons);
    PointCloud::Ptr extract_cloud(const PointCloud::Ptr& pointcloud, const pcl::PointIndices::Ptr& indices, bool remove);

    double get_depth_confidence_score(const PointCloud::Ptr pointcloud, double max_distance, double min_distance,
                                      double& min_depth_confidence_score, double& max_depth_confidence_score);
    double get_flatness_score(const PointCloud::Ptr& pointcloud, const pcl::ModelCoefficients::Ptr coefficients,
                              double& min_distance, double& max_distance);
    double get_steepness_score(const PointCloud::Ptr& pointcloud, double& min_score, double& max_score);

    PointCloud::Ptr downsample_organized(const PointCloud::ConstPtr& pointcloud, int scale);

    uint32_t colors[100];
    std::deque< std::vector<pcl::PointXYZ>> centroid_buffer;
    std::vector<struct plane_probability> plane_probabilities;
    std::vector<struct persistent_plane> existing_planes;

    int frame_count;
    ros::NodeHandle nh;
    ros::Subscriber pointcloud_sub;
    ros::Publisher plane_pointcloud_pub;
    ros::Publisher colored_plane_pub;
    ros::Publisher mesh_pub;

    pcl::VoxelGrid<pcl::PCLPointCloud2> voxel_filter;
    pcl::SACSegmentation<pcl::PointXYZ> plane_segmenter;
};
#endif
