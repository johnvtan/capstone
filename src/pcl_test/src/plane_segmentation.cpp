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

    // do plane segmentation
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices); 
    
    int num_points = (int)cloud_filtered->points.size();
    PointCloud::Ptr removed(new PointCloud);
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

        PointCloud::Ptr plane_raw = extract_cloud(cloud_filtered, inliers, false);

        // extract points within standard deviation
        PointCloud::Ptr plane = filter_by_std_dev(plane_raw, coefficients, 2);
       
        // Find rectangular area and density of pointcloud
        double rect_area = get_rectangular_area(plane);
        double density = double(plane->points.size()) / rect_area;

        // Try to find surface area of the pointcloud
        
        // dump info to console
        std::cout << "-----------------NEW PLANE: " << plane_count << " ---------------" << std::endl;
        std::cout << "Plane size: " <<  plane->points.size() << std::endl;
        std::cout << "Area is: " << rect_area << std::endl;
        std::cout << "Density is: " << density << std::endl;
         
        // Filter by density and comparison to previous centroids
        pcl::PointXYZ centroid_point = compute_centroid(plane);
        if (density > 70 && near_previous_centroid(centroid_point) || centroid_buffer.size() == 0) {
            std::cout << "**************PUBLISHED**************" << std::endl;
            plane_count++;
            plane_centroids.push_back(centroid_point);
            *all_planar += *plane;
        }

        // create colored pointcloud
        for (int i = 0; i < plane->points.size(); i++) {
            pcl::PointXYZRGB colored_point;
            pcl::PointXYZ point = plane->points.at(i);
            colored_point.x = point.x;
            colored_point.y = point.y;
            colored_point.z = point.z;

            colored_point.rgb = colors[plane_count]; 
            all_planar_colored->points.push_back(colored_point);
        }

        // remove this plane from the rest of the pointcloud
        removed = extract_cloud(cloud_filtered, inliers, true);
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

pcl::PointXYZ PlaneSegmentation::compute_centroid(const PointCloud::Ptr& pointcloud) {
    // calculate centroid of the plane
    pcl::CentroidPoint<pcl::PointXYZ> curr_centroid;
    for (int i = 0; i < pointcloud->points.size(); i++) {
        pcl::PointXYZ point = pointcloud->points.at(i);
        curr_centroid.add(point);
    }
    pcl::PointXYZ centroid_point;
    curr_centroid.get(centroid_point);
    return centroid_point;
}

double PlaneSegmentation::get_rectangular_area(const PointCloud::Ptr& pointcloud) {
    // Find (min_x, min_y), (max_x, max_y) to get a rectangular area of the pointcloud
    auto max_x_iter = std::max_element(pointcloud->points.begin(), pointcloud->points.end(), 
                                       [](pcl::PointXYZ a, pcl::PointXYZ b) -> bool
                                       {
                                            return a.x < b.x;
                                       });
    auto  max_y_iter = std::max_element(pointcloud->points.begin(), pointcloud->points.end(), 
                                        [](pcl::PointXYZ a, pcl::PointXYZ b) -> bool
                                        {
                                             return a.y < b.y;
                                        });
    auto min_x_iter = std::min_element(pointcloud->points.begin(), pointcloud->points.end(), 
                                       [](pcl::PointXYZ a, pcl::PointXYZ b) -> bool
                                       {
                                            return a.x < b.x;
                                       });
    auto min_y_iter = std::min_element(pointcloud->points.begin(), pointcloud->points.end(), 
                                       [](pcl::PointXYZ a, pcl::PointXYZ b) -> bool
                                       {
                                            return a.y < b.y;
                                       });
 
    return ((*max_x_iter).x - (*min_x_iter).x) * ((*max_y_iter).y - (*min_y_iter).y);
}

bool PlaneSegmentation::near_previous_centroid(const pcl::PointXYZ& centroid_point) {
    // compare the centroid of the current cloud to centroids in the buffer. 
    // Only consider this plane if it's centroid was close (<1m) from a centroid seen within the last 3 frames 
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
    return result;
}

PointCloud::Ptr PlaneSegmentation::filter_by_std_dev(const PointCloud::Ptr& pointcloud,
                                                     const pcl::ModelCoefficients::Ptr coefficients,
                                                     unsigned int num_dev)
{
    // filter out any points more than two standard deviations from the cloud
    pcl::PointIndices::Ptr points_within_n_std_dev(new pcl::PointIndices);
    double mean_distance = 0;
    double dev = get_deviation(pointcloud, coefficients, &mean_distance);
    for (int i = 0; i < pointcloud->points.size(); i++) {
        pcl::PointXYZ point = pointcloud->points.at(i);

        // compute distance
        double c1 = coefficients->values[0];
        double c2 = coefficients->values[1];
        double c3 = coefficients->values[2];
        double c4 = coefficients->values[3];

        double d = pcl::pointToPlaneDistance<pcl::PointXYZ>(point, c1, c2, c3, c4); // in meters
        if (std::abs(d - mean_distance) < num_dev * dev) {
            // then we keep this point since we're within two std devs
            points_within_n_std_dev->indices.push_back(i); 
        }
    }

    return extract_cloud(pointcloud, points_within_n_std_dev, false);
}

double PlaneSegmentation::get_deviation(const PointCloud::Ptr& pointcloud,
                                        const pcl::ModelCoefficients::Ptr& coefficients,
                                        double* mean_error) {

    std::vector<double> errors;
    *mean_error = 0;
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
        *mean_error += d;
        if (d > max_error) {
            max_error = d;
        }
        if (d < min_error) {
            min_error = d;
        }
    }
    *mean_error /= pointcloud->points.size();

    double deviation = 0;
    for (int i = 0; i < pointcloud->points.size(); i++) {
        deviation += std::pow(errors[i] - *mean_error, 2);
    }

    return std::sqrt(deviation / pointcloud->points.size());
}

PointCloud::Ptr PlaneSegmentation::extract_cloud(const PointCloud::Ptr& pointcloud,
                                                 const pcl::PointIndices::Ptr& indices,
                                                 bool remove)
{
    PointCloud::Ptr return_cloud(new PointCloud);
    pcl::ExtractIndices<pcl::PointXYZ> extracter;

    extracter.setInputCloud(pointcloud);
    extracter.setIndices(indices);
    extracter.setNegative(remove);
    extracter.filter(*return_cloud);
    return return_cloud;
}
