#include <pcl_test/plane_segmentation.h>
#include <stdlib.h>
#include <cmath>

int main(int argc, char **argv) {
    srand(100);
    ros::init(argc, argv, "plane_segmentation_node");
    PlaneSegmentation plane_segmenter;
    plane_segmenter.run();
    return 0;
}

PlaneSegmentation::PlaneSegmentation(void) : 
    nh()
{
    // TODO - boost shit to bind class to callback
    pointcloud_sub = nh.subscribe<PointCloud>("/guidance/points2", 1, &PlaneSegmentation::pointcloud_callback, this);
    plane_pointcloud_pub = nh.advertise<PointCloud>("/guidance/filtered_points", 1);
    mesh_pub = nh.advertise<PointCloud>("/guidance/mesh_pub", 1);
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

    voxel_filter.setInputCloud(cloud2);
    voxel_filter.filter(*filtered_cloud2);

    // convert to PointCloud
    pcl::fromPCLPointCloud2(*filtered_cloud2, *cloud_filtered);

    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices); 
    
    int num_points = (int)cloud_filtered->points.size();
    PointCloud::Ptr removed(new PointCloud);
    PointCloud::Ptr all_planar(new PointCloud);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr all_planar_colored(new pcl::PointCloud<pcl::PointXYZRGB>);

    std::vector<pcl::PointXYZ> plane_centroids;
    int plane_count = 0;

    pcl::PointXYZ max_distance_pt =  *std::max_element(cloud_filtered->points.begin(), cloud_filtered->points.end(), 
                                           [](pcl::PointXYZ a, pcl::PointXYZ b) -> bool
                                           {
                                                return a.z < b.z;
                                           });
    pcl::PointXYZ min_distance_pt = *std::min_element(cloud_filtered->points.begin(), cloud_filtered->points.end(), 
                                           [](pcl::PointXYZ a, pcl::PointXYZ b) -> bool
                                           {
                                                return a.z < b.z;
                                           });
    double max_distance = max_distance_pt.z;
    double min_distance = min_distance_pt.z;

    int count = 0;
    std::vector<double> depth_scores;
    std::vector<double> flatness_scores; 
    std::vector<double> steepness_scores;

    std::vector<PointCloud::Ptr> extracted_planes;
    while (cloud_filtered->points.size() > 0.5 * num_points) {

        // get plane model
        plane_segmenter.setInputCloud(cloud_filtered);
        plane_segmenter.segment(*inliers, *coefficients);
        if (inliers->indices.size() == 0) {
            ROS_WARN("Could not find plane in this model");
            break;
        }

        PointCloud::Ptr plane_raw = extract_cloud(cloud_filtered, inliers, false);

        // extract points within standard deviation
        //PointCloud::Ptr plane = filter_by_std_dev(plane_raw, coefficients, 2);
        PointCloud::Ptr plane = remove_outliers(plane_raw);
        extracted_planes.push_back(plane);

        // get depth confidence score of centroid as like an average
        pcl::PointXYZ centroid_point = compute_centroid(plane);

        std::cout << "plane # " << count++ << std::endl;
        double depth_confidence_score = get_depth_confidence_score(centroid_point, max_distance, min_distance);

        double flatness_score = get_flatness_score(plane, coefficients);

        double steepness_score = get_steepness_score(plane);
        depth_scores.push_back(depth_confidence_score);
        flatness_scores.push_back(flatness_score);
        steepness_scores.push_back(steepness_score);

        // remove this plane from the rest of the pointcloud
        removed = extract_cloud(cloud_filtered, inliers, true);
        cloud_filtered.swap(removed);
    }

    if (extracted_planes.size() == 0) {
        return;
    }

    // find min and max values in all the scores
    double max_depth_score = *std::max_element(depth_scores.begin(), depth_scores.end());
    double min_depth_score = *std::min_element(depth_scores.begin(), depth_scores.end());

    double max_flatness_score = *std::max_element(flatness_scores.begin(), flatness_scores.end());
    double min_flatness_score = *std::min_element(flatness_scores.begin(), flatness_scores.end());

    double max_steepness_score = *std::max_element(steepness_scores.begin(), steepness_scores.end());
    double min_steepness_score = *std::min_element(steepness_scores.begin(), steepness_scores.end());

    for (int i = 0; i < extracted_planes.size(); i++) {
        PointCloud::Ptr plane = extracted_planes.at(i);

        // Find rectangular area and density of pointcloud
        double rect_area = get_rectangular_area(plane);
        double density = double(plane->points.size()) / rect_area;

        // Try to find surface area of the pointcloud
        double surface_area = get_surface_area(plane);

        // Check if this plane was close to one in the previous 3 frames
        // TODO - register point with nearest neighbor if under a threshold, otherwise create a new point
        // remove points that haven't been registered with in like 3 frames or something - give them a lifespan
        // Figure out an analogous steepness/flatness score for the sites
        // Flatness as average distance to plane model?
        // Steepness is the same metric as described in the paper
        // Add depth score as well
        // Can use surface area as a score as well potentially
        
        pcl::PointXYZ centroid_point = compute_centroid(plane);
        //bool near_prev = near_previous_centroid(centroid_point);
        bool near_prev = true;
        
        double depth_confidence_score = (depth_scores.at(i) - min_depth_score) / (max_depth_score - min_depth_score);
        double flatness_score = 1 - (flatness_scores.at(i) - min_flatness_score) / (max_flatness_score - min_flatness_score);
        //double flatness_score = 1 - flatness_scores.at(i) / max_flatness_score;
        double steepness_score = (steepness_scores.at(i) - min_steepness_score) / (max_steepness_score - min_steepness_score);
        double total_score = 0.2 * depth_confidence_score + 0.4 * flatness_score + 0.4 * steepness_score;
        
        // dump info to console
        std::cout << "-----------------NEW PLANE: " << plane_count << " ---------------" << std::endl;
        std::cout << "Plane size: " <<  plane->points.size() << std::endl;
        std::cout << "Rectangular Area is: " << rect_area << std::endl;
        std::cout << "Density is: " << density << std::endl;
        std::cout << "Surface area is: " << surface_area << std::endl;
        std::cout << "Near previous? " << near_prev << std::endl;
        std::cout << "Depth conf score: " << depth_confidence_score << std::endl;
        std::cout << "Flatness score: " << flatness_score << std::endl;
        std::cout << "Steepness score: " << steepness_score << std::endl;
        std::cout << "Total score: " << total_score << std::endl;

        // Filter by density and comparison to previous centroids
        if (total_score > 0.75 && surface_area > 0.75) {
            std::cout << "**************PUBLISHED**************" << std::endl;
            plane_count++;
            plane_centroids.push_back(centroid_point);
            *all_planar += *plane;
        }

        plane_count++;

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

    }

    centroid_buffer.push_back(plane_centroids);
    if (centroid_buffer.size() > 3) {
        centroid_buffer.pop_front();
    }

    all_planar_colored->header.frame_id = "guidance";
    pcl_conversions::toPCL(ros::Time::now(), all_planar_colored->header.stamp);
    colored_plane_pub.publish(all_planar_colored);

    std::cout << "-----------------------------------Plane count is: " << plane_count << std::endl;
    all_planar->header.frame_id = "guidance";
    pcl_conversions::toPCL(ros::Time::now(), all_planar->header.stamp);
    plane_pointcloud_pub.publish(all_planar);
}

PointCloud::Ptr PlaneSegmentation::remove_outliers(const PointCloud::Ptr& pointcloud) {
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> remover;
    PointCloud::Ptr filtered(new PointCloud);
    remover.setInputCloud(pointcloud);
    remover.setMeanK(50);
    remover.setStddevMulThresh(1.0);
    remover.filter(*filtered);
    return filtered;
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
    auto max_y_iter = std::max_element(pointcloud->points.begin(), pointcloud->points.end(), 
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
    int count = 0;
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

double PlaneSegmentation::get_surface_area(const PointCloud::Ptr& pointcloud) {
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kd_tree(new pcl::search::KdTree<pcl::PointXYZ>);

    // First, estimate normals
    kd_tree->setInputCloud(pointcloud);
    normal_estimator.setInputCloud(pointcloud);
    normal_estimator.setSearchMethod(kd_tree);
    normal_estimator.setKSearch(20);
    normal_estimator.compute(*normals);

    // concatenate XYZ and normal fields
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>);
    pcl::concatenateFields(*pointcloud, *normals, *cloud_with_normals);

    // initialize search tree and objects for greedy triangulation
    pcl::search::KdTree<pcl::PointNormal>::Ptr normal_kd_tree(new pcl::search::KdTree<pcl::PointNormal>);
    normal_kd_tree->setInputCloud(cloud_with_normals);
    pcl::GreedyProjectionTriangulation<pcl::PointNormal> greedy_projection;
    pcl::PolygonMesh triangles;

    // greedy projection parameters
    greedy_projection.setSearchRadius(0.25);
    greedy_projection.setMu(2.5);
    greedy_projection.setMaximumNearestNeighbors(100);
    greedy_projection.setMaximumSurfaceAngle(M_PI / 4);
    greedy_projection.setMinimumAngle(M_PI/18);
    greedy_projection.setMaximumAngle(2 * M_PI / 3);
    greedy_projection.setNormalConsistency(false);

    greedy_projection.setInputCloud(cloud_with_normals);
    greedy_projection.setSearchMethod(normal_kd_tree);
    greedy_projection.reconstruct(triangles);

    // convert mesh to pointcloud

    // publish mesh_pub for debug
    /*
    mesh_cloud->header.frame_id = "guidance";
    pcl_conversions::toPCL(ros::Time::now(), mesh_cloud->header.stamp);
    mesh_pub.publish(mesh_cloud);
    */
    return find_polygon_area(pointcloud, triangles);
 }

double PlaneSegmentation::find_polygon_area(const PointCloud::Ptr& cloud, const pcl::PolygonMesh& mesh) {
   // then calculate the area of the mesh cloud
   /*
    double area = 0.0;
    int num_points = pointcloud->points.size();
    int j = 0;
    Eigen::Vector3f va, vb, res; 
    res(0) = res(1) = res(2) = 0.0f;

    for (int i = 0; i < num_points; ++i) {
        j = (i + 1) % num_points;
        va = pointcloud->points.at(i).getVector3fMap();
        vb = pointcloud->points.at(j).getVector3fMap();
        res += va.cross(vb);
    }
    area = res.norm();
    return area * 0.5;
    */
    int index1, index2, index3;
    double x1, x2, x3, y1, y2, y3, z1, z2, z3;
    double a, b, c, q;
    double area = 0;
    for (int i = 0; i < mesh.polygons.size(); ++i) {
        index1 = mesh.polygons[i].vertices[0];
        index2 = mesh.polygons[i].vertices[1];
        index3 = mesh.polygons[i].vertices[2];
        
        x1 = cloud->points[index1].x;
        y1 = cloud->points[index1].y;
        z1 = cloud->points[index1].z;

        x2 = cloud->points[index2].x;
        y2 = cloud->points[index2].y;
        z2 = cloud->points[index2].z;

        x3 = cloud->points[index3].x;
        y3 = cloud->points[index3].y;
        z3 = cloud->points[index3].z;

        // heron's formula
        a=sqrt(std::pow((x1-x2),2)+std::pow((y1-y2),2)+std::pow((z1-z2),2));
        b=sqrt(std::pow((x1-x3),2)+std::pow((y1-y3),2)+std::pow((z1-z3),2));
        c=sqrt(std::pow((x3-x2),2)+std::pow((y3-y2),2)+std::pow((z3-z2),2));
        q=(a+b+c)/2;

        area=area+std::sqrt(q*(q-a)*(q-b)*(q-c));
    }
    return area;
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

double PlaneSegmentation::get_depth_confidence_score(const pcl::PointXYZ& point, double max_distance, double min_distance) {
    return 1 - ((std::pow(point.z, 2) - std::pow(min_distance, 2)) / (std::pow(max_distance, 2))); 
}

double PlaneSegmentation::get_flatness_score(const PointCloud::Ptr& pointcloud, const pcl::ModelCoefficients::Ptr coefficients) {
    double mean_distance = 0;
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

        mean_distance += d;
    }
    mean_distance /= pointcloud->points.size();
    return mean_distance;
}

// TODO not really working? Figure out why later
double PlaneSegmentation::get_steepness_score(const PointCloud::Ptr& pointcloud) {
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    ne.setRadiusSearch(0.5);
    ne.setSearchMethod(tree);
    ne.setInputCloud(pointcloud);
    ne.setViewPoint(0, 0, 0);

    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
    ne.compute(*normals);
       
    double avg_steepness_score = 0;
    double avg_theta = 0;
    int count = 0;
    for (int i = 0; i < normals->points.size(); i++) {
        double z_component = normals->points.at(i).normal[2];
        if (!std::isnan(z_component)) {
            if (z_component < 0) {
                z_component *= -1;
            }
            // use curvature?
            double theta = acos(z_component);
            double curr_score = exp(-1 * std::pow(theta, 2) / (2 * std::pow(15, 2)));
            avg_theta += theta;
            avg_steepness_score += curr_score;
            count++;
        }
    }
    std::cout << "Theta: " << avg_theta / count << " avg score: " << avg_steepness_score / count << std::endl;
    return avg_steepness_score / count;
}
