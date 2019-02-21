#include <pcl_test/plane_segmentation.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    srand(100);
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
    colored_plane_pub = nh.advertise<pcl::PointCloud<pcl::PointXYZRGB>>("/guidance/planes_colored", 1);

    // voxel filter settings
    voxel_filter.setLeafSize(0.1f, 0.1f, 0.1f);

    // plane segmentation settings
    plane_segmenter.setOptimizeCoefficients(true);
    plane_segmenter.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
    plane_segmenter.setAxis(Eigen::Vector3f(0.0f, 0.0f, 1.0f));
    plane_segmenter.setEpsAngle(0.17f); // 15 degrees in radians
    plane_segmenter.setMethodType(pcl::SAC_RANSAC);
    plane_segmenter.setMaxIterations(1000);
    plane_segmenter.setDistanceThreshold(0.1f);

    for (int i = 0; i < 100; i++) {
        colors[i] = rand();
    }
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
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr all_planar_colored(new pcl::PointCloud<pcl::PointXYZRGB>);

    std::vector<pcl::PointXYZ> plane_centroids;
    int plane_count = 0;
    while (cloud_filtered->points.size() > 0.5 * num_points) {
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

        pcl::CentroidPoint<pcl::PointXYZ> curr_centroid;
        for (int i = 0; i < plane->points.size(); i++) {
            pcl::PointXYZRGB colored_point;
            pcl::PointXYZ point = plane->points.at(i);
            colored_point.x = point.x;
            colored_point.y = point.y;
            colored_point.z = point.z;

            colored_point.rgb = colors[plane_count]; 
            all_planar_colored->points.push_back(colored_point);
            curr_centroid.add(point);
        }

        auto max_x_iter = std::max_element(plane->points.begin(), plane->points.end(), 
                                           [](pcl::PointXYZ a, pcl::PointXYZ b) -> bool
                                           {
                                                return a.x < b.x;
                                           });
        auto  max_y_iter = std::max_element(plane->points.begin(), plane->points.end(), 
                                            [](pcl::PointXYZ a, pcl::PointXYZ b) -> bool
                                            {
                                                 return a.y < b.y;
                                            });
        auto min_x_iter = std::min_element(plane->points.begin(), plane->points.end(), 
                                           [](pcl::PointXYZ a, pcl::PointXYZ b) -> bool
                                           {
                                                return a.x < b.x;
                                           });
        auto min_y_iter = std::min_element(plane->points.begin(), plane->points.end(), 
                                           [](pcl::PointXYZ a, pcl::PointXYZ b) -> bool
                                           {
                                                return a.y < b.y;
                                           });
        
        pcl::PointXYZ centroid_point;
        curr_centroid.get(centroid_point);
        bool result = false;
        for (int i = 0; i < centroid_buffer.size(); i++) {
            for (int j = 0; j < centroid_buffer[i].size(); j++) {
                pcl::PointXYZ prev_point = centroid_buffer[i][j];
                double squ_dist = std::pow((centroid_point.x - prev_point.x), 2) +
                                  std::pow((centroid_point.y - prev_point.y), 2) + 
                                  std::pow((centroid_point.z - prev_point.z), 2);

                if (std::sqrt(squ_dist) < 1.0) {
                    result = true;
                    break;
                }
            }
        }
           

        ROS_INFO("Plane size: %lu; Deviation = %lf", plane->points.size(), get_deviation(plane, coefficients));
        double area = ((*max_x_iter).x - (*min_x_iter).x) * ((*max_y_iter).y - (*min_y_iter).y);
        std::cout << "Area is: " << area << std::endl;
        
        double density = double(plane->points.size()) / area;
        std::cout << "Density is: " << density << std::endl;

        if (density > 70 && result || centroid_buffer.size() == 0) {
            plane_count++;
            plane_centroids.push_back(centroid_point);
            *all_planar += *plane;
        }

        extracter.setNegative(true);
        extracter.filter(*removed);
        cloud_filtered.swap(removed);
        plane_count++;
    }

    centroid_buffer.push_back(plane_centroids);
    if (centroid_buffer.size() > 3) {
        centroid_buffer.pop_front();
    }
    std::cout << "-----------------------------------Plane count is: " << plane_count << std::endl;
    all_planar->header.frame_id = "guidance";
    pcl_conversions::toPCL(ros::Time::now(), all_planar->header.stamp);
    plane_pointcloud_pub.publish(all_planar);

    all_planar_colored->header.frame_id = "guidance";
    pcl_conversions::toPCL(ros::Time::now(), all_planar_colored->header.stamp);
    colored_plane_pub.publish(all_planar_colored);
}

double PlaneSegmentation::get_deviation(const PointCloud::Ptr& pointcloud,
                                        const pcl::ModelCoefficients::Ptr& coefficients) {

    std::vector<double> errors;
    double mean_error = 0;
    double min_error = 100000;
    double max_error = 0;
    for (int i = 0; i < pointcloud->points.size(); i++) {
        // get point
        pcl::PointXYZ point = pointcloud->points.at(i);

        // compute distance
        double c1 = coefficients->values[0];
        double c2 = coefficients->values[1];
        double c3 = coefficients->values[2];
        double c4 = coefficients->values[3];

        double d = pcl::pointToPlaneDistance<pcl::PointXYZ>(point, c1, c2, c3, c4); // in meters
        //std::cout << point << std::endl;
        //ROS_INFO("Distance to plane: %lf", d);

        // update statistics
        errors.push_back(d);
        mean_error += d;
        if (d > max_error) {
            max_error = d;
        }
        if (d < min_error) {
            min_error = d;
        }
    }
    mean_error /= pointcloud->points.size();

    double deviation = 0;
    for (int i = 0; i < pointcloud->points.size(); i++) {
        deviation += std::pow(errors[i] - mean_error, 2);
    }

    return std::sqrt(deviation / pointcloud->points.size());
}
