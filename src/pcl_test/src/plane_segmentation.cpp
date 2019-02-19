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

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

ros::Publisher filtered_pointcloud_pub;
pcl::VoxelGrid<pcl::PCLPointCloud2> filter;
pcl::SACSegmentation<pcl::PointXYZ> seg;
pcl::ExtractIndices<pcl::PointXYZ> extract;
pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
pcl::search::KdTree<pcl::PointXYZ>::Ptr kd_tree(new pcl::search::KdTree<pcl::PointXYZ>());

void pointcloud_callback(const PointCloud::ConstPtr& msg) {
    PointCloud::Ptr cloud_filtered(new PointCloud);
    pcl::PCLPointCloud2::Ptr cloud2(new pcl::PCLPointCloud2), filtered_cloud2(new pcl::PCLPointCloud2);
    pcl::toPCLPointCloud2(*msg, *cloud2);

    // downsample cloud using voxel filter
    filter.setInputCloud(cloud2);
    filter.filter(*filtered_cloud2);

    // convert to PointCloud
    pcl::fromPCLPointCloud2(*filtered_cloud2, *cloud_filtered);

    // do plane segmentation lol
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices); 
    
    int num_points = (int)cloud_filtered->points.size();
    PointCloud::Ptr removed(new PointCloud);
    PointCloud::Ptr plane(new PointCloud);
    pcl::PointCloud<pcl::Normal>::Ptr plane_normals(new pcl::PointCloud<pcl::Normal>());
    PointCloud::Ptr all_planar(new PointCloud);
    while (cloud_filtered->points.size() > 0.3 * num_points) {
        seg.setInputCloud(cloud_filtered);
        seg.segment(*inliers, *coefficients);
        if (inliers->indices.size() == 0) {
            ROS_WARN("Could not find plane in this model");
            break;
        }
        extract.setInputCloud(cloud_filtered);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*plane);

        // TODO - filter out planar segments by normal vector. The normal should point directly toward/away from the camera
        normal_estimator.setInputCloud(plane);
        normal_estimator.setSearchMethod(kd_tree);

        // What's a good search radius? Too small = lots of NaN values
        normal_estimator.setRadiusSearch(10.0);
        normal_estimator.compute(*plane_normals);

        // plane_normals contains a normal vector for every point in the cloud
        // try average over the entire cloud
        pcl::CentroidPoint<pcl::Normal> avg;
        for (int i = 0; i < plane_normals->points.size(); i++) {
            avg.add(plane_normals->points[i]);
        }
        pcl::Normal surface_normal;
        avg.get(surface_normal);
        //std::cout << surface_normal << std::endl;

        // try filtering by planes w/ normals close to (0, 0, 1) or (0, 0, -1)?
        // Z axis points towards/away from viewpoint
        // normal.normal_x, normal.normal_y, normal.normal_z
        // TODO - figure out the normal vector of planes pointing towards the viewpoint
        if ((surface_normal.normal_x > -0.3 && surface_normal.normal_x < 0.3) &&
            (surface_normal.normal_y > -0.3 && surface_normal.normal_y < 0.3) &&
            (surface_normal.normal_z > -1.2 && surface_normal.normal_z < -0.8)) {

            // add this plane to all planes
            std::cout << "good" << std::endl;
            *all_planar += *plane;
        }

        extract.setNegative(true);
        extract.filter(*removed);
        cloud_filtered.swap(removed);
    }

    all_planar->header.frame_id = "guidance";
    pcl_conversions::toPCL(ros::Time::now(), all_planar->header.stamp);
    filtered_pointcloud_pub.publish(all_planar);
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "plane_segmentation_node");
    ros::NodeHandle nh;
    ros::Subscriber sub = nh.subscribe<PointCloud>("/guidance/points2", 1, pointcloud_callback);
    filtered_pointcloud_pub = nh.advertise<PointCloud>("/guidance/filtered_points", 1);

    // voxel filter settings
    filter.setLeafSize(0.1f, 0.1f, 0.1f);

    // plane segmentation settings
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(1000);
    seg.setDistanceThreshold(0.1f);
    ros::spin();
    return 0;
}
