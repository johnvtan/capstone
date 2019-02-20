#include <pcl_test/plane_segmentation.h>

int main(int argc, char **argv) {
    ros::init(argc, argv, "plane_segmentation_node");
    PlaneSegmentation plane_segmenter;
    plane_segmenter.run();
    return 0;
}

PlaneSegmentation::PlaneSegmentation(void) : 
    nh(), 
    kd_tree(new pcl::search::KdTree<pcl::PointXYZ>()) 
{
    // TODO - boost shit to bind class to callback
    pointcloud_sub = nh.subscribe<PointCloud>("/guidance/points2", 1, &PlaneSegmentation::pointcloud_callback, this);
    plane_pointcloud_pub = nh.advertise<PointCloud>("/guidance/filtered_points", 1);

    // voxel filter settings
    voxel_filter.setLeafSize(0.1f, 0.1f, 0.1f);

    // plane segmentation settings
    plane_segmenter.setOptimizeCoefficients(true);
    plane_segmenter.setModelType(pcl::SACMODEL_PLANE);
    plane_segmenter.setMethodType(pcl::SAC_RANSAC);
    plane_segmenter.setMaxIterations(1000);
    plane_segmenter.setDistanceThreshold(0.1f);
}

PlaneSegmentation::~PlaneSegmentation(void) {

}

void PlaneSegmentation::run(void) {
    ros::spin();
}

void PlaneSegmentation::pointcloud_callback(const PointCloud::ConstPtr& msg) 
{
    PointCloud::Ptr cloud_filtered(new PointCloud);
    pcl::PCLPointCloud2::Ptr cloud2(new pcl::PCLPointCloud2), filtered_cloud2(new pcl::PCLPointCloud2);
    pcl::toPCLPointCloud2(*msg, *cloud2);

    // downsample cloud using voxel filter
    voxel_filter.setInputCloud(cloud2);
    voxel_filter.filter(*filtered_cloud2);

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
    std::cout << "Nearest (?) depth is: " << cloud_filtered->points.at(0).z << std::endl;
    while (cloud_filtered->points.size() > 0.3 * num_points) {
        plane_segmenter.setInputCloud(cloud_filtered);
        plane_segmenter.segment(*inliers, *coefficients);
        if (inliers->indices.size() == 0) {
            ROS_WARN("Could not find plane in this model");
            break;
        }
        extracter.setInputCloud(cloud_filtered);
        extracter.setIndices(inliers);
        extracter.setNegative(false);
        extracter.filter(*plane);

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
        if ((surface_normal.normal_x > -0.5 && surface_normal.normal_x < 0.5) &&
            (surface_normal.normal_y > -0.5 && surface_normal.normal_y < 0.5)) {
            if ((surface_normal.normal_z > -1.5 && surface_normal.normal_z < -0.5) ||
                (surface_normal.normal_z > 0.5 && surface_normal.normal_z < 1.5)) {

                std::cout << "good" << std::endl;
                // add this plane to all planes
                *all_planar += *plane;
            }
        }

        extracter.setNegative(true);
        extracter.filter(*removed);
        cloud_filtered.swap(removed);
    }

    all_planar->header.frame_id = "guidance";
    pcl_conversions::toPCL(ros::Time::now(), all_planar->header.stamp);
    plane_pointcloud_pub.publish(all_planar);
}
